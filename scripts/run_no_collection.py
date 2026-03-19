#!/usr/bin/env python3
"""
AI Predictive Intelligence — Processing & Training Orchestrator
==============================================================
Runs the full pipeline skipping only Phase 1 (Data Collection).
Includes:
 - Processing & Merging
 - Feature Engineering
 - Regime Feature Injection
 - Sequence Generation
 - Hyperparameter Optimization (Optuna)
 - Model Training (TS, NLP, Fusion)
 - Evaluation & Drift Detection
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_full_pipeline import (
    get_device,
    phase_2_data_processing,
    phase_3_feature_engineering,
    phase_3b_regime_features,
    phase_4_sequence_generation,
    phase_5_hyperparameter_optimization,
    phase_6_full_training,
    phase_7_evaluation,
    phase_8_regime_detection,
    phase_9_drift_detection
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("NoCollectionPipeline")

def main():
    parser = argparse.ArgumentParser(description="AI Pipeline (No Collection)")
    parser.add_argument('--hpo', action='store_true', help="Enable Hyperparameter Optimization")
    parser.add_argument('--trials', type=int, default=10, help="Number of HPO trials")
    args = parser.parse_args()

    pipeline_start = time.time()
    
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  AI PREDICTIVE INTELLIGENCE — DATA & TRAINING PIPELINE              ║")
    logger.info(f"║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                   ║")
    logger.info("║  (Skipping Data Collection)                                         ║")
    logger.info("╚" + "═" * 68 + "╝")
    
    device = get_device()
    
    # 1. Processing
    phase_2_data_processing()
    
    # 2. Features
    phase_3_feature_engineering()
    phase_3b_regime_features()
    
    # 3. Sequences
    phase_4_sequence_generation()
    
    # 4. HPO (Optional)
    if args.hpo:
        phase_5_hyperparameter_optimization(device, n_trials=args.trials)
    else:
        logger.info("⏭️ Skipping HPO - Using existing or default parameters")
    
    # 5. Training
    phase_6_full_training(device)
    
    # 6. Evaluation
    phase_7_evaluation(device)
    
    # 7. Post-Training Metrics & Monitoring
    phase_8_regime_detection()
    phase_9_drift_detection()
    
    elapsed = time.time() - pipeline_start
    logger.info("\n" + "═" * 70)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info(f"Total duration: {elapsed/3600:.2f} hours")
    logger.info("═" * 70)

if __name__ == '__main__':
    main()
