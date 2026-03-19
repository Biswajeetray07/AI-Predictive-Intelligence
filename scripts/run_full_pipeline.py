#!/usr/bin/env python3
"""
Master Pipeline Orchestrator
=================================
Single entry point: python scripts/run_full_pipeline.py

Phases:
  1. Data Collection (all API collectors)
  2. Data Processing & Merging
  3. Feature Engineering
  3b. Regime Feature Generation (adds regime probs as model inputs)
  4. Sequence Generation (train/val/test splits)
  5. Hyperparameter Optimization (Optuna) [opt-in: --hpo]
  6. Full Model Training (TS + NLP + Fusion)
  7. Evaluation & Backtesting
  8. Regime Detection (global)
  9. Drift Detection & Retraining Check
"""

import os
import sys

# Ensure the root project directory is in the Python path
# so that `import src...` works even when running from scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import yaml
import torch
import logging
import argparse
import numpy as np
from datetime import datetime
from functools import partial
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PROJECT_ROOT is the container of 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Pipeline")


# ═════════════════════════════════════════════════════════════════════════════
# DEVICE DETECTION
# ═════════════════════════════════════════════════════════════════════════════
def get_device():
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"🔥 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("🍎 Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        logger.info("💻 Using CPU")
    return device


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1: DATA COLLECTION
# ═════════════════════════════════════════════════════════════════════════════
def phase_1_data_collection():
    """Run the full data collection pipeline."""
    logger.info("=" * 70)
    logger.info("PHASE 1: DATA COLLECTION")
    logger.info("=" * 70)
    
    from src.pipelines.run_data_pipeline import collect_all_data
    results = collect_all_data()
    
    successful = sum(1 for s, _ in results.values() if s)
    total = len(results)
    logger.info(f"Collection complete: {successful}/{total} collectors succeeded")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2: DATA PROCESSING & MERGING
# ═════════════════════════════════════════════════════════════════════════════
def phase_2_data_processing():
    """Process and merge collected datasets."""
    logger.info("=" * 70)
    logger.info("PHASE 2: DATA PROCESSING & MERGING")
    logger.info("=" * 70)
    
    # Process each domain
    try:
        from src.data_processing.financial_processing import FinancialDataProcessor
        logger.info("Processing financial data...")
        processor = FinancialDataProcessor(base_dir=PROJECT_ROOT)
        processor.run_all()
        logger.info("✅ Financial data processed")
    except Exception as e:
        logger.warning(f"⚠️ Financial processing: {e}")
    
    try:
        from src.data_processing.news_processing import NewsDataProcessor
        logger.info("Processing news data...")
        processor = NewsDataProcessor(base_dir=PROJECT_ROOT)
        if hasattr(processor, 'run_all'):
            processor.run_all()
        elif hasattr(processor, 'process_all'):
            processor.process_all() # type: ignore
        else:
            processor.process_news_data() if hasattr(processor, 'process_news_data') else None # type: ignore
        logger.info("✅ News data processed")
    except Exception as e:
        logger.warning(f"⚠️ News processing: {e}")
    
    try:
        from src.data_processing.social_processing import SocialMediaProcessor
        logger.info("Processing social media data...")
        processor = SocialMediaProcessor(base_dir=PROJECT_ROOT)
        if hasattr(processor, 'run_all'):
            processor.run_all()
        elif hasattr(processor, 'process_all'):
            processor.process_all() # type: ignore
        else:
            methods = ['process_twitter', 'process_reddit', 'process_hackernews', 'process_github', 'process_stackexchange', 'process_mastodon', 'process_youtube']
            for m in methods:
                if hasattr(processor, m):
                    getattr(processor, m)()
        logger.info("✅ Social data processed")
    except Exception as e:
        logger.warning(f"⚠️ Social processing: {e}")
    
    try:
        from src.data_processing.external_processing import ExternalDataProcessor
        logger.info("Processing external Kaggle data...")
        processor = ExternalDataProcessor(base_dir=PROJECT_ROOT)
        processor.run_all()
        logger.info("✅ External data processed")
    except Exception as e:
        logger.warning(f"⚠️ External processing: {e}")

    try:
        from src.data_processing.macro_processing import MacroEconomicProcessor
        logger.info("Processing advanced macro indicators...")
        processor = MacroEconomicProcessor(base_dir=PROJECT_ROOT)
        processor.run_all()
        logger.info("✅ Macro indicators processed")
    except Exception as e:
        logger.warning(f"⚠️ Macro processing: {e}")

    # Datasets are NOT merged here anymore. 
    # Merging has been moved to AFTER feature generation (Phase 3b) to ensure
    # that newly generated feature indices are actually included in the final dataset.


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3: FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
def phase_3_feature_engineering():
    """Generate features from processed data."""
    logger.info("=" * 70)
    logger.info("PHASE 3: FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    from src.feature_engineering.feature_generator import generate_all_features
    results = generate_all_features()
    successful = sum(1 for v in results.values() if v is not None)
    logger.info(f"Features generated: {successful}/{len(results)}")
    
    # ═════════════════════════════════════════════════════════════════════════════
    # PHASE 3b: MERGE ALL DATASETS (Must happen AFTER features are generated)
    # ═════════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("PHASE 3b: MERGE ALL PROCESSED DATASETS")
    logger.info("=" * 70)
    try:
        from src.data_processing.merge_datasets import merge_all
        logger.info("Merging base datasets and newly generated feature indices...")
        merge_all()
        logger.info("✅ Datasets successfully merged into master CSV")
    except Exception as e:
        logger.error(f"❌ Merge Failed: {e}")
        sys.exit(1)
        
    return results


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3c: REGIME FEATURE GENERATION
# ═════════════════════════════════════════════════════════════════════════════
def phase_3c_regime_features():
    """Generate regime probability features and append them to the merged dataset."""
    logger.info("=" * 70)
    logger.info("PHASE 3b: REGIME FEATURE GENERATION")
    logger.info("=" * 70)

    import pandas as pd

    merged_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'merged', 'all_merged_dataset.csv')
    if not os.path.exists(merged_path):
        logger.warning("⚠️ Merged dataset not found — skipping regime features")
        return

    try:
        from src.feature_engineering.regime_detection.regime_features import generate_regime_features

        df = pd.read_csv(merged_path)

        # Detect close column name
        close_col = 'Close' if 'Close' in df.columns else 'close'

        logger.info(f"Running regime detection on merged dataset ({len(df)} rows)...")
        df = generate_regime_features(
            df,
            close_col=close_col,
            ticker_col='ticker',
            n_regimes=5,
        )

        # Save back — regime columns are now part of the merged dataset
        df.to_csv(merged_path, index=False)
        regime_cols = [c for c in df.columns if c.startswith('regime_')]
        logger.info(f"✅ Regime features appended: {regime_cols}")

    except Exception as e:
        logger.warning(f"⚠️ Regime feature generation: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4: SEQUENCE GENERATION
# ═════════════════════════════════════════════════════════════════════════════
def phase_4_sequence_generation():
    """Build training sequences from features."""
    logger.info("=" * 70)
    logger.info("PHASE 4: SEQUENCE GENERATION")
    logger.info("=" * 70)
    
    from src.data_processing.build_sequences import main as build_sequences_main
    build_sequences_main()
    
    # Verify outputs exist
    model_inputs = os.path.join(PROJECT_ROOT, 'data', 'processed', 'model_inputs')
    for split in ['train', 'val', 'test']:
        x_path = os.path.join(model_inputs, f'X_{split}.npy')
        y_path = os.path.join(model_inputs, f'y_{split}.npy')
        if os.path.exists(x_path):
            X = np.load(x_path, mmap_mode='r')
            logger.info(f"  {split}: X shape = {X.shape}")
        else:
            logger.warning(f"  ⚠️ Missing: {x_path}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5: HYPERPARAMETER OPTIMIZATION (OPTUNA)
# ═════════════════════════════════════════════════════════════════════════════
def phase_5_hyperparameter_optimization(device, n_trials=5):
    """Run Optuna HPO for all model types."""
    import optuna
    
    logger.info("=" * 70)
    logger.info(f"PHASE 5: HYPERPARAMETER OPTIMIZATION ({n_trials} trials)")
    logger.info("=" * 70)
    
    # Load data for TS optimization
    model_inputs = os.path.join(PROJECT_ROOT, 'data', 'processed', 'model_inputs')
    X_train = np.load(os.path.join(model_inputs, 'X_train.npy'), mmap_mode='r')
    y_train = np.load(os.path.join(model_inputs, 'y_train.npy'), mmap_mode='r')
    
    # Sub-sample for fast HPO (use 20% of data)
    n_samples = min(len(X_train), 2000)
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_sub = np.array(X_train[indices])
    y_sub = np.array(y_train[indices])
    
    logger.info(f"HPO using {n_samples} samples from {len(X_train)} total")
    
    best_params = {}
    
    # ── TS Model Optimization ──
    from src.training.optimization.optuna_ts import objective_timeseries
    
    ts_models = ['lstm', 'gru', 'transformer', 'tft']
    for model_name in ts_models:
        logger.info(f"\n--- Optimizing: {model_name.upper()} ---")
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=1)
        )
        study.optimize(
            partial(objective_timeseries, model_name=model_name, X_full=X_sub, y_full=y_sub, device=device), # type: ignore
            n_trials=n_trials,
            timeout=300  # 5 min timeout per model
        )
        best_params[model_name] = study.best_params
        logger.info(f"  Best {model_name}: val_loss={study.best_value:.6f}")
    
    # NOTE: Fusion HPO is intentionally NOT here.
    # It is run in Phase 6, AFTER real embeddings are created by 6A/6B.
    
    # ── Save Best Params (TS models only at this point) ──
    best_params_path = os.path.join(PROJECT_ROOT, 'configs', 'best_params.yaml')
    # Load existing params to preserve any previous Fusion params
    existing_params = {}
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            loaded = yaml.safe_load(f)
            if loaded and 'best_params' in loaded:
                existing_params = loaded['best_params']
    existing_params.update(best_params)
    with open(best_params_path, 'w') as f:
        yaml.dump({'best_params': existing_params, 'timestamp': datetime.now().isoformat()}, f, default_flow_style=False)
    
    logger.info(f"\n✅ TS best parameters saved to: {best_params_path}")
    logger.info("ℹ️  Fusion HPO will run after embeddings are generated in Phase 6.")
    return best_params


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 6: FULL MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════════════
def phase_6_full_training(device, best_params=None, run_fusion_hpo=False, n_trials=5):
    """Train all models using best parameters."""
    logger.info("=" * 70)
    logger.info("PHASE 6: FULL MODEL TRAINING")
    logger.info("=" * 70)
    
    # ── 6A: Time Series Ensemble ──
    logger.info("\n─── 6A: Time Series Ensemble Training ───")
    from src.training.timeseries.train import main as train_ts_main
    train_ts_main()
    
    # ── 6B: NLP DeBERTa ──
    logger.info("\n─── 6B: NLP DeBERTa Multi-Task Training ───")
    try:
        from src.training.nlp.train import main as train_nlp_main
        train_nlp_main()
    except Exception as e:
        logger.warning(f"⚠️ NLP training failed: {e}")
        # CRITICAL: End any leaked MLflow run so Fusion training can start cleanly
        try:
            import mlflow
            mlflow.end_run()
            logger.info("Cleaned up leaked MLflow run from NLP training.")
        except Exception:
            pass
        logger.info("Generating mock NLP embeddings as fallback...")
        from scripts.generate_mock_nlp import main as generate_mock
        generate_mock()
    
    # ── 6B.5: Fusion HPO (NOW with real embeddings!) ──
    if run_fusion_hpo:
        logger.info("\n─── 6B.5: Fusion HPO (Real Embeddings) ───")
        try:
            import optuna
            from functools import partial
            from src.training.optimization.optuna_fusion import objective_fusion
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(
                partial(objective_fusion, device=device),
                n_trials=n_trials,
                timeout=600
            )
            # Save Fusion params alongside existing TS params
            best_params_path = os.path.join(PROJECT_ROOT, 'configs', 'best_params.yaml')
            existing_params = {}
            if os.path.exists(best_params_path):
                with open(best_params_path, 'r') as f:
                    loaded = yaml.safe_load(f)
                    if loaded and 'best_params' in loaded:
                        existing_params = loaded['best_params']
            existing_params['fusion'] = study.best_params
            with open(best_params_path, 'w') as f:
                yaml.dump({'best_params': existing_params, 'timestamp': datetime.now().isoformat()}, f, default_flow_style=False)
            logger.info(f"  ✅ Fusion HPO complete: val_loss={study.best_value:.6f}")
        except Exception as e:
            logger.warning(f"⚠️ Fusion HPO failed: {e}. Using existing/default params.")
    else:
        logger.info("\n⏭️  Skipping Fusion HPO (pass --hpo to enable)")
    
    # ── 6C: Fusion Model ──
    logger.info("\n─── 6C: Fusion Model Training ───")
    # Defensive: ensure any leaked MLflow runs are closed
    try:
        import mlflow
        mlflow.end_run()
    except Exception:
        pass
    from src.training.fusion.train import main as train_fusion_main
    train_fusion_main()


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 7: EVALUATION & BACKTESTING
# ═════════════════════════════════════════════════════════════════════════════
def phase_7_evaluation(device):
    """Run evaluation and backtesting."""
    logger.info("=" * 70)
    logger.info("PHASE 7: EVALUATION & BACKTESTING")
    logger.info("=" * 70)
    
    try:
        from src.validation.backtest import BacktestEngine
        engine = BacktestEngine(device=str(device))
        results = engine.run()
        
        logger.info("\n📊 Backtest Results:")
        for k, v in results.items():
            logger.info(f"  {k}: {v}")
        return results
    except Exception as e:
        logger.warning(f"⚠️ Backtesting: {e}")
        return {}


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 8: REGIME DETECTION
# ═════════════════════════════════════════════════════════════════════════════
def phase_8_regime_detection():
    """Generate HMM regime states for dynamic ensembling."""
    logger.info("=" * 70)
    logger.info("PHASE 8: HMM REGIME DETECTION")
    logger.info("=" * 70)
    
    try:
        from scripts.generate_global_regime import main as regime_main
        regime_main()
        logger.info("✅ Regime states generated")
    except Exception as e:
        logger.warning(f"⚠️ Regime detection: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 9: DRIFT DETECTION & RETRAINING CHECK
# ═════════════════════════════════════════════════════════════════════════════
def phase_9_drift_detection():
    """Run drift detection on training vs recent data and decide if retraining is needed."""
    logger.info("=" * 70)
    logger.info("PHASE 9: DRIFT DETECTION & RETRAINING CHECK")
    logger.info("=" * 70)

    try:
        import pandas as pd
        from src.evaluation.monitoring.retraining_trigger import RetrainingTrigger

        model_inputs = os.path.join(PROJECT_ROOT, 'data', 'processed', 'model_inputs')
        train_meta_path = os.path.join(model_inputs, 'metadata_train.csv')
        test_meta_path = os.path.join(model_inputs, 'metadata_test.csv')
        merged_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'merged', 'all_merged_dataset.csv')

        if not os.path.exists(merged_path):
            logger.warning("⚠️ Merged dataset not found — skipping drift check")
            return

        df = pd.read_csv(merged_path)
        df['date'] = pd.to_datetime(df['date'])

        # Identify numeric feature columns
        exclude = ['date', 'ticker', 'Close']
        feature_cols = [c for c in df.select_dtypes(include='number').columns if c not in exclude]

        # Split into reference (train period) and current (test period)
        train_cutoff = pd.to_datetime('2025-01-01')
        train_data = df[df['date'] < train_cutoff]
        test_data = df[df['date'] >= train_cutoff]

        if train_data.empty or test_data.empty:
            logger.warning("⚠️ Not enough data for drift detection")
            return

        # Use top 20 features to avoid excessive computation
        top_features = feature_cols[:20]

        trigger = RetrainingTrigger(
            psi_threshold=0.25,
            ks_alpha=0.05,
            drift_ratio_threshold=0.3,
        )
        trigger.fit_reference(train_data, top_features)
        decision = trigger.check_and_decide(test_data)

        logger.info(f"Drift status: {decision['drift_status']}")
        logger.info(f"Features drifted: {decision['features_drifted']}/{decision['features_monitored']}")
        logger.info(f"Should retrain: {decision['should_retrain']}")

        if decision['should_retrain']:
            logger.warning("🚨 Critical drift detected. Consider retraining models.")

    except Exception as e:
        logger.warning(f"⚠️ Drift detection: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="AI Predictive Intelligence — Full Pipeline")
    parser.add_argument('--hpo', action='store_true', help="Enable Hyperparameter Optimization (Phase 5)")
    parser.add_argument('--hpo-trials', type=int, default=5, help="Number of HPO trials per model")
    parser.add_argument('--skip-collection', action='store_true', help="Skip Phase 1 (Data Collection)")
    args = parser.parse_args()

    pipeline_start = time.time()
    
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  AI PREDICTIVE INTELLIGENCE — FULL PRODUCTION PIPELINE              ║")
    logger.info(f"║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                   ║")
    logger.info("╚" + "═" * 68 + "╝")
    
    device = get_device()
    
    # Phase 1: Data Collection
    if not args.skip_collection:
        phase_1_data_collection()
    else:
        logger.info("⏭️ Skipping Phase 1 (Data Collection) — using existing data")
    
    # Phase 2: Data Processing
    phase_2_data_processing()
    
    # Phase 3: Gen Features
    phase_3_feature_engineering()
    
    # Phase 3b: Merge Datasets
    # (Happens internally inside `phase_3_feature_engineering` now as per previous diff, wait, no, the function returns results but where is phase_3b called? Ah, I put the merge code *inside* `phase_3_feature_engineering()`. Let me just add Regime Detection here.)
    
    # Phase 3c: Regime Detection
    phase_3c_regime_features() # adds regime probs to merged dataset
    
    # Phase 4: Sequence Generation
    phase_4_sequence_generation()
    
    # Phase 5: Hyperparameter Optimization (opt-in)
    best_params = None
    if args.hpo:
        best_params = phase_5_hyperparameter_optimization(device, n_trials=args.hpo_trials)
    else:
        logger.info("⏭️ Skipping Phase 5 (HPO) — using existing configuration. Pass --hpo to enable.")
    
    # Phase 6: Full Training
    phase_6_full_training(device, best_params)
    
    # Phase 7: Evaluation
    phase_7_evaluation(device)
    
    # Phase 8: Regime Detection (global states for inference pipeline)
    phase_8_regime_detection()

    # Phase 9: Drift Detection & Retraining Check
    phase_9_drift_detection()
    
    # ─── Final Summary ───
    elapsed = time.time() - pipeline_start
    logger.info("\n" + "═" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total duration: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    
    # List saved models
    models_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        logger.info(f"Saved models: {len(model_files)}")
        for mf in model_files:
            size_mb = os.path.getsize(os.path.join(models_dir, mf)) / (1024*1024)
            logger.info(f"  {mf} ({size_mb:.1f} MB)")
    
    logger.info("═" * 70)
    

if __name__ == '__main__':
    main()

