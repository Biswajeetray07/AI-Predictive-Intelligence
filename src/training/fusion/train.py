"""
Training script for the Deep Fusion Model.

Pipeline:
    1. Load NLP embeddings from features/nlp_embeddings.npy
    2. Load Time Series predictions/features from saved TS models
    3. Align by date
    4. Train the Attention-based Fusion model
    5. Save to saved_models/fusion_model.pt
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import logging

from src.utils.seed import set_seed

try:
    from src.utils.mlflow_utils import setup_experiment, start_run, log_epoch_metrics, log_final_metrics, end_run  # type: ignore[import, no-redef]
    MLFLOW_TRACKING = True
except ImportError:
    MLFLOW_TRACKING = False
    # Provide dummy functions with correct signatures to avoid type errors
    def setup_experiment(*args, **kwargs): return None
    def start_run(*args, **kwargs): return None
    def log_epoch_metrics(*args, **kwargs):
        pass
    def log_final_metrics(*args, **kwargs):
        pass
    def end_run(*args, **kwargs):
        pass

from src.models.fusion.multi_horizon_fusion import MultiHorizonFusionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logging.getLogger().setLevel(logging.INFO)


def load_config():
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'training_config.yaml')
    best_params_path = os.path.join(PROJECT_ROOT, 'configs', 'best_params.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f).get('fusion_model', {})
    else:
        config = {
            'batch_size': 32, 'epochs': 30, 'learning_rate': 5e-4,
            'nlp_embedding_dim': 768, 'attention_heads': 4,
            'mlp_hidden': [512, 256, 128], 'dropout': 0.3, 'patience': 8,
        }

    if os.path.exists(best_params_path):
        try:
            with open(best_params_path, 'r') as f:
                best = yaml.safe_load(f).get('best_params', {}).get('fusion', {})
            if best:
                config.update(best)
                logging.info("Loaded HPO best_params for Fusion model")
        except Exception as e:
            logging.warning(f"Could not load best_params for Fusion: {e}")
    
    return config


class FusionDataset(Dataset):
    """Combines NLP embeddings with TS features aligned by date."""

    def __init__(self, nlp_embeddings, ts_features, targets):
        self.nlp = torch.tensor(nlp_embeddings, dtype=torch.float32)
        self.ts = torch.tensor(ts_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.nlp)

    def __getitem__(self, idx):
        return self.nlp[idx], self.ts[idx], self.targets[idx]


def load_fusion_data(project_root):
    """
    Load and align NLP embeddings with time series features.
    Builds a 3D (batch, seq_len, 768) tensor of NLP embeddings leading up to the target date.
    Returns aligned arrays ready for training.
    
    Falls back to generated embeddings if real ones don't exist yet.
    """
    features_dir = os.path.join(project_root, 'data', 'features')
    model_inputs = os.path.join(project_root, 'data', 'processed', 'model_inputs')

    # Paths
    nlp_emb_path = os.path.join(features_dir, 'nlp_embeddings.npy')
    nlp_meta_path = os.path.join(features_dir, 'nlp_embedding_dates.csv')
    ts_emb_path = os.path.join(features_dir, 'ts_embeddings.npy')
    ts_meta_path = os.path.join(model_inputs, 'metadata_val.csv')
    ts_targets_path = os.path.join(model_inputs, 'y_multi_val.npy')

    # Try to load real data first
    real_embeddings_exist = (
        os.path.exists(nlp_emb_path) and 
        os.path.exists(nlp_meta_path) and 
        os.path.exists(ts_emb_path) and 
        os.path.exists(ts_meta_path) and
        os.path.exists(ts_targets_path)
    )
    
    if real_embeddings_exist:
        logging.info("Loading real embeddings from data pipeline...")
        logging.info("Building 3D NLP sequences from TS metadata...")
        nlp_2d = np.load(nlp_emb_path)
        nlp_meta = pd.read_csv(nlp_meta_path)
        nlp_meta['date'] = pd.to_datetime(nlp_meta['date'])
        
        ts_meta = pd.read_csv(ts_meta_path)
        ts_meta['date'] = pd.to_datetime(ts_meta['date'])
        
        ts_features = np.load(ts_emb_path)
        targets = np.load(ts_targets_path)
        # Vectorized mapping of NLP sequences to TS targets
        seq_len = 60
        dim = nlp_2d.shape[1]
        batch_size = len(ts_meta)
        nlp_embeddings = np.zeros((batch_size, seq_len, dim), dtype=np.float32)
        
        # Create a combined lookup index dataframe
        nlp_meta['idx'] = np.arange(len(nlp_meta))
        
        # Build sequence of dates for each target
        all_dates = []
        all_tickers = []
        all_batch_idx = []
        all_seq_idx = []
        
        for i, row in ts_meta.iterrows():
            end_date = row['date'].date()
            ticker = row['ticker']
            dates = pd.date_range(end=end_date, periods=seq_len, freq='B')
            
            all_dates.extend(dates.date)
            all_tickers.extend([ticker] * seq_len)
            all_batch_idx.extend([i] * seq_len)
            all_seq_idx.extend(range(seq_len))
            
        seq_df = pd.DataFrame({
            'date': pd.to_datetime(all_dates),
            'ticker': all_tickers,
            'batch_idx': all_batch_idx,
            'seq_idx': all_seq_idx
        }).sort_values('date')
        
        # ─── GLOBAL BROADCAST & NEAREST MATCH ALIGNMENT ───
        # Filter NLP features to MACRO (or whatever global signal is present)
        # and sort by date for nearest-date matching
        nlp_global = nlp_meta[nlp_meta['ticker'] == 'MACRO'].copy()
        if nlp_global.empty and not nlp_meta.empty:
            nlp_global = nlp_meta[nlp_meta['ticker'] == nlp_meta['ticker'].iloc[0]].copy()
            
        nlp_global = nlp_global.sort_values('date')
        
        if not nlp_global.empty:
            # Match each TS date to the numerically closest available NLP date
            merged = pd.merge_asof(
                seq_df, 
                nlp_global[['date', 'idx']], 
                on='date', 
                direction='nearest'
            )
        else:
            merged = seq_df
            merged['idx'] = np.nan
        
        # Restore the original sequence order
        merged = merged.sort_values(['batch_idx', 'seq_idx'])
        
        # Extract valid indices
        valid_mask = sum(merged['idx'].notna())
        logging.info(f"Found NLP context for {valid_mask}/{len(merged)} sequence steps (using Nearest Date Alignment)")
        
        valid = merged.dropna(subset=['idx'])
        b_idx = valid['batch_idx'].values.astype(int)
        s_idx = valid['seq_idx'].values.astype(int)
        n_idx = valid['idx'].values.astype(int)
        
        # Bulk assignment
        nlp_embeddings[b_idx, s_idx, :] = nlp_2d[n_idx]
                    
    else:
        logging.error("❌ CRITICAL: Required embeddings/metadata not found in data pipeline.")
        logging.error("Cannot train Fusion model without REAL NLP and TimeSeries embeddings.")
        logging.error("Check Phase 6 (NLP and TS training) to ensure they completed successfully.")
        raise FileNotFoundError("Missing real embeddings or metadata for Fusion training. Mock fallback disabled.")

    logging.info(f"Fusion data: NLP={nlp_embeddings.shape}, TS={ts_features.shape}, Targets={targets.shape}")
    return nlp_embeddings, ts_features, targets

def train_fusion():
    set_seed(42)
    config = load_config()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logging.info(f"Device: {device}")

    # MLflow Experiment Tracking
    if MLFLOW_TRACKING:
        # Defensive: end any previously leaked run before starting a new one
        try:
            import mlflow as _mlflow
            _mlflow.end_run()
        except Exception:
            pass
        setup_experiment('fusion_training')
        start_run('deep_fusion_training', params=config)

    model_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'fusion_model.pt')
    
    # Load data
    nlp_emb, ts_feat, targets = load_fusion_data(PROJECT_ROOT)

    # Split temporally
    n = len(nlp_emb)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_ds = FusionDataset(nlp_emb[:train_end], ts_feat[:train_end], targets[:train_end])
    val_ds = FusionDataset(nlp_emb[train_end:val_end], ts_feat[train_end:val_end], targets[train_end:val_end])
    test_ds = FusionDataset(nlp_emb[val_end:], ts_feat[val_end:], targets[val_end:])

    bs = config.get('batch_size', 32)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    # Build model
    model = MultiHorizonFusionModel(
        nlp_dim=config.get('nlp_embedding_dim', 768),
        ts_dim=ts_feat.shape[-1],
        attention_heads=config.get('attention_heads', 4),
        mlp_hidden=config.get('mlp_hidden', [512, 256, 128]),
        dropout=config.get('dropout', 0.3),
    ).to(device)
    
    wd = config.get('weight_decay', 1e-4)
    l1_lambda = config.get('l1_lambda', 0.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('learning_rate', 5e-4), weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)
    criterion = nn.MSELoss()

    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'fusion_checkpoint.pt')

    best_val = float('inf')
    patience = config.get('patience', 8)
    patience_counter = 0
    start_epoch = 0
    
    # Setup AMP (Automatic Mixed Precision) with compatibility layer
    use_amp = device.type == 'cuda' and config.get('use_amp', True)
    scaler = None
    if use_amp:
        try:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
        except ImportError:
            logging.warning("GradScaler not available, disabling AMP")
            use_amp = False

    # ── Load Checkpoint ──
    if os.path.exists(checkpoint_path):
        logging.info("Found Fusion checkpoint. Resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val = checkpoint['best_val']
        patience_counter = checkpoint['patience_counter']
        logging.info(f"Resuming Fusion from epoch {start_epoch+1}")

    # Training loop with early stopping
    for epoch in range(start_epoch, config.get('epochs', 30)):
        model.train()
        train_loss = 0
        for nlp, ts, y in train_loader:
            nlp, ts, y = nlp.to(device), ts.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Use autocast context manager with compatibility
            if use_amp:
                try:
                    from torch.cuda.amp import autocast
                except ImportError:
                    autocast = None
                
                if autocast:
                    with autocast():
                        preds = model(nlp, ts)
                        pred_1d, pred_5d, pred_30d = preds['1d'], preds['5d'], preds['30d']
                        
                        loss_1d = criterion(pred_1d, y[:, 0:1])
                        loss_5d = criterion(pred_5d, y[:, 1:2])
                        loss_30d = criterion(pred_30d, y[:, 2:3])
                        main_loss = loss_1d + loss_5d + loss_30d
                        
                        # Add L1 Penalty
                        l1_penalty = sum(p.abs().sum() for p in model.parameters()) if l1_lambda > 0 else 0
                        loss = main_loss + l1_lambda * l1_penalty
                    
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                else:
                    # Fallback without autocast
                    preds = model(nlp, ts)
                    pred_1d, pred_5d, pred_30d = preds['1d'], preds['5d'], preds['30d']
                    
                    loss_1d = criterion(pred_1d, y[:, 0:1])
                    loss_5d = criterion(pred_5d, y[:, 1:2])
                    loss_30d = criterion(pred_30d, y[:, 2:3])
                    main_loss = loss_1d + loss_5d + loss_30d
                    
                    l1_penalty = sum(p.abs().sum() for p in model.parameters()) if l1_lambda > 0 else 0
                    loss = main_loss + l1_lambda * l1_penalty
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            else:
                preds = model(nlp, ts)
                pred_1d, pred_5d, pred_30d = preds['1d'], preds['5d'], preds['30d']
                
                loss_1d = criterion(pred_1d, y[:, 0:1])
                loss_5d = criterion(pred_5d, y[:, 1:2])
                loss_30d = criterion(pred_30d, y[:, 2:3])
                main_loss = loss_1d + loss_5d + loss_30d
                
                l1_penalty = sum(p.abs().sum() for p in model.parameters()) if l1_lambda > 0 else 0
                loss = main_loss + l1_lambda * l1_penalty
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for nlp, ts, y in val_loader:
                nlp, ts, y = nlp.to(device), ts.to(device), y.to(device)
                preds = model(nlp, ts)
                pred_1d, pred_5d, pred_30d = preds['1d'], preds['5d'], preds['30d']
                
                loss_1d = criterion(pred_1d, y[:, 0:1])
                loss_5d = criterion(pred_5d, y[:, 1:2])
                loss_30d = criterion(pred_30d, y[:, 2:3])
                val_loss += (loss_1d + loss_5d + loss_30d).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            logging.info(f"  [Fusion] New best validation loss: {best_val:.6f}")
        else:
            patience_counter += 1

        # Save Checkpoint after every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val': best_val,
            'patience_counter': patience_counter,
        }, checkpoint_path)

        if (epoch + 1) % 1 == 0:
            logging.info(f"Epoch {epoch+1} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Best: {best_val:.6f}")
            if MLFLOW_TRACKING:
                log_epoch_metrics(epoch, {'fusion_train_loss': train_loss, 'fusion_val_loss': val_loss})

        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break

    # Final Save
    torch.save(model.state_dict(), model_path)
    logging.info(f"Fusion model saved to {model_path}")

    # Test evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for nlp, ts, y in test_loader:
            nlp, ts, y = nlp.to(device), ts.to(device), y.to(device)
            preds = model(nlp, ts)
            pred_1d, pred_5d, pred_30d = preds['1d'], preds['5d'], preds['30d']
            
            loss_1d = criterion(pred_1d, y[:, 0:1])
            loss_5d = criterion(pred_5d, y[:, 1:2])
            loss_30d = criterion(pred_30d, y[:, 2:3])
            test_loss += (loss_1d + loss_5d + loss_30d).item()
    test_loss /= len(test_loader)
    logging.info(f"Test MSE: {test_loss:.6f}")

    if MLFLOW_TRACKING:
        log_final_metrics({'fusion_test_mse': float(test_loss), 'fusion_best_val': float(best_val)})
        end_run()

    logging.info("\nFUSION TRAINING PIPELINE COMPLETED SUCCESSFULLY.")


def main():
    """Alias for train_fusion() — called by the pipeline orchestrator."""
    train_fusion()


if __name__ == "__main__":
    train_fusion()
