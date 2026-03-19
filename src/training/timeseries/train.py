"""
Training script for Time Series Forecasting models.

Supports training multiple architectures (LSTM, GRU, Transformer, TFT),
early stopping, model registry saving, and weighted ensemble prediction.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from collections import defaultdict

from src.utils.seed import set_seed

try:
    from src.utils.mlflow_utils import setup_experiment, start_run, log_epoch_metrics, log_final_metrics, end_run  # type: ignore[import, no-redef]
    MLFLOW_TRACKING = True
except ImportError:
    MLFLOW_TRACKING = False
    def setup_experiment(*args, **kwargs): return None
    def start_run(*args, **kwargs): return None
    def log_epoch_metrics(*args, **kwargs): pass
    def log_final_metrics(*args, **kwargs): pass
    def end_run(*args, **kwargs): pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.training.timeseries.dataset import create_dataloaders
from src.models.timeseries.lstm import LSTMForecaster
from src.models.timeseries.gru import GRUForecaster
from src.models.timeseries.transformer import TransformerForecaster
from src.models.timeseries.tft import TFTForecaster

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config():
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'training_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f).get('timeseries_model', {})
    else:
        config = {
            'batch_size': 64, 'epochs': 50, 'learning_rate': 1e-3,
            'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.2, 'patience': 10,
            'models': ['lstm', 'gru', 'transformer', 'tft'],
            'ensemble': {'method': 'weighted_average', 'weights': [0.25, 0.20, 0.30, 0.25]},
        }

    # ── Merge HPO best_params if available ──
    best_params_path = os.path.join(PROJECT_ROOT, 'configs', 'best_params.yaml')
    if os.path.exists(best_params_path):
        try:
            with open(best_params_path, 'r') as f:
                best = yaml.safe_load(f).get('best_params', {})
            if best:
                config['_best_params'] = best
                logging.info(f"Loaded HPO best_params for: {list(best.keys())}")
        except Exception as e:
            logging.warning(f"Could not load best_params.yaml: {e}")

    return config


# ─── Model Factory ───────────────────────────────────────────────────────────

def build_model(model_name: str, input_dim: int, config: dict) -> nn.Module:
    # Use per-model HPO best_params if available, falling back to global config
    best = config.get('_best_params', {}).get(model_name, {})
    hidden = best.get('hidden_dim', config.get('hidden_dim', 128))
    layers = best.get('num_layers', config.get('num_layers', 2))
    drop = best.get('dropout', config.get('dropout', 0.2))

    if best:
        logging.info(f"  Using HPO best_params for {model_name}: hidden={hidden}, layers={layers}, dropout={drop:.4f}")

    if model_name == 'lstm':
        return LSTMForecaster(input_dim, hidden, layers, drop)
    elif model_name == 'gru':
        return GRUForecaster(input_dim, hidden, layers, drop)
    elif model_name == 'transformer':
        return TransformerForecaster(input_dim, d_model=hidden, nhead=4, num_layers=layers, dropout=drop)
    elif model_name == 'tft':
        return TFTForecaster(input_dim, hidden, num_heads=4, num_layers=layers, dropout=drop)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ─── Training & Evaluation ───────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, grad_accum_steps=1, l1_lambda=0.0):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    use_amp = scaler is not None and device.type == 'cuda'
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            # AMP: Mixed precision forward pass (CUDA only)
            try:
                from torch.cuda.amp import autocast
                with autocast():
                    pred, _ = model(x)
                    main_loss = criterion(pred, y)
                    
                    # Add L1 Penalty to AMP path
                    l1_penalty = sum(p.abs().sum() for p in model.parameters()) if l1_lambda > 0 else 0
                    loss = main_loss + l1_lambda * l1_penalty
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            except ImportError:
                # Fall back to standard training if AMP not available
                pred, _ = model(x)
                main_loss = criterion(pred, y)
                l1_penalty = sum(p.abs().sum() for p in model.parameters()) if l1_lambda > 0 else 0
                loss = main_loss + l1_lambda * l1_penalty
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        else:
            # Standard forward/backward (MPS / CPU)
            pred, _ = model(x)
            main_loss = criterion(pred, y)
            
            # Add L1 Penalty
            l1_penalty = sum(p.abs().sum() for p in model.parameters()) if l1_lambda > 0 else 0
            loss = main_loss + l1_lambda * l1_penalty
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        total_loss += loss.item()
        
        if (batch_idx + 1) % 1000 == 0:
            logging.info(f"    Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.6f}")
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred, _ = model(x) # Unpack out, context
        loss = criterion(pred, y)
        total_loss += loss.item()
    return total_loss / len(loader)


def train_model(model_name, model, train_loader, val_loader, config, device):
    """Train a single model with early stopping and checkpointing."""
    # Prioritize model-specific HPO results if available
    best = config.get('_best_params', {}).get(model_name, {})
    epochs = best.get('epochs', config.get('epochs', 25))
    patience = best.get('patience', config.get('patience', 10))
    lr = best.get('learning_rate', config.get('learning_rate', 1e-3))
    wd = best.get('weight_decay', config.get('weight_decay', 1e-4))
    l1_lambda = best.get('l1_lambda', config.get('l1_lambda', 0.0))
    
    checkpoint_dir = os.path.join(PROJECT_ROOT, 'saved_models', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_checkpoint.pt')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0
    # Only use GradScaler on CUDA; None for MPS/CPU
    try:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if device.type == 'cuda' else None
    except ImportError:
        scaler = None

    # ── Load Checkpoint if exists ──
    if os.path.exists(checkpoint_path):
        logging.info(f"  [{model_name}] Found checkpoint. Resuming...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            patience_counter = checkpoint['patience_counter']
            logging.info(f"  [{model_name}] Resuming from epoch {start_epoch+1}")
        except Exception as e:
            logging.warning(f"  [{model_name}] Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            best_val_loss = float('inf')
            patience_counter = 0

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, grad_accum_steps=2, l1_lambda=l1_lambda)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        if device.type == 'mps':
            torch.mps.empty_cache()
            
        scheduler.step(val_loss)

        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1

        if MLFLOW_TRACKING:
            log_epoch_metrics(epoch, {
                f"{model_name}_train_loss": float(train_loss),
                f"{model_name}_val_loss": float(val_loss),
                f"{model_name}_best_val": float(best_val_loss),
            })

        # Save Checkpoint after every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
        }, checkpoint_path)

        if (epoch + 1) % 1 == 0: # Log every epoch for resume visibility
            gpu_mem = ""
            if device.type == 'cuda':
                allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
                reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
                gpu_mem = f" | GPU: {allocated:.1f}GB alloc / {reserved:.1f}GB reserved"
            logging.info(f"  [{model_name}] Epoch {epoch+1}/{epochs} | "
                         f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                         f"Best: {best_val_loss:.6f} | Patience: {patience_counter}/{patience}{gpu_mem}")

        if patience_counter >= patience:
            logging.info(f"  [{model_name}] Early stopping at epoch {epoch+1}")
            break

    # Clean up checkpoint after successful full training
    # if os.path.exists(checkpoint_path):
    #     os.remove(checkpoint_path)

    return best_val_loss


@torch.no_grad()
def ensemble_predict(models, weights, loader, device):
    """Generate weighted ensemble predictions."""
    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        batch_preds = []
        for model in models:
            model.eval()
            pred, _ = model(x)
            pred = pred.cpu().numpy()
            batch_preds.append(pred)

        # Weighted average
        weighted = np.zeros_like(batch_preds[0])
        for pred, w in zip(batch_preds, weights):
            weighted += w * pred
        all_preds.append(weighted)
        all_targets.append(y.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


@torch.no_grad()
def get_ensemble_embeddings(models, weights, loader, device):
    """Extract weighted ensemble latent context vectors."""
    all_embeddings = []
    for x, _ in loader:
        x = x.to(device)
        batch_embs = []
        for model in models:
            model.eval()
            _, context = model(x)
            batch_embs.append(context.cpu().numpy())
        
        # Weighted average of context vectors
        weighted = np.zeros_like(batch_embs[0])
        for emb, w in zip(batch_embs, weights):
            weighted += w * emb
        all_embeddings.append(weighted)
    
    return np.concatenate(all_embeddings)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    config = load_config()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")

    # MLflow Experiment Tracking
    if MLFLOW_TRACKING:
        setup_experiment('timeseries_training')
        start_run('ts_ensemble_training', params=config)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    data_dir = os.path.join(project_root, 'data', 'processed', 'model_inputs')
    batch_size = config.get('batch_size', 64)

    if not os.path.exists(data_dir):
        logging.error(f"Data directory missing: {data_dir}. Ensure build_sequences.py worked.")
        return

    try:
        train_loader, val_loader, test_loader = create_dataloaders(data_dir, batch_size)
    except Exception as e:
        logging.error(f"Failed to create dataloaders: {e}")
        return

    # Get input dimension from a sample batch
    input_dim = None
    for x, y in train_loader:
        input_dim = x.shape[-1]
        logging.info(f"Input dimension: {input_dim}, Sequence length: {x.shape[1]}")
        break
    
    if input_dim is None:
        logging.error("Could not determine input dimension from train_loader (empty dataset?)")
        return

    model_names = config.get('models', ['lstm', 'gru', 'transformer', 'tft'])
    model_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    os.makedirs(model_dir, exist_ok=True)

    trained_models = []
    val_losses = {}

    for name in model_names:
        save_path = os.path.join(model_dir, f'{name}_model.pt')
        
        # Build model regardless to keep trained_models list intact
        model = build_model(name, input_dim, config).to(device)
        
        load_success = False
        if os.path.exists(save_path):
            logging.info(f"\n[{name.upper()}] Model already trained and saved. Loading weights...")
            try:
                model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
                val_losses[name] = 0.0 # Placeholder for completed models
                load_success = True
            except Exception as e:
                logging.warning(f"\n[{name.upper()}] Error loading weights: {e}. Will retrain.")
                load_success = False

        if not load_success:
            logging.info(f"\n{'='*50}")
            logging.info(f"Training: {name.upper()}")
            logging.info(f"{'='*50}")

            best_val = train_model(name, model, train_loader, val_loader, config, device)
            val_losses[name] = best_val

            # Save to model registry
            torch.save(model.state_dict(), save_path)
            logging.info(f"  Saved {name} model to {save_path}")

        trained_models.append(model)

    # ── Ensemble Evaluation ──
    logging.info(f"\n{'='*50}")
    logging.info("ENSEMBLE EVALUATION")
    logging.info(f"{'='*50}")

    ensemble_config = config.get('ensemble', {})
    weights = ensemble_config.get('weights', [1.0 / len(trained_models)] * len(trained_models))

    preds, targets = ensemble_predict(trained_models, weights, test_loader, device)
    test_mse = np.mean((preds - targets) ** 2)
    test_mae = np.mean(np.abs(preds - targets))

    logging.info(f"Individual model validation losses: {val_losses}")
    logging.info(f"Ensemble Test MSE: {test_mse:.6f}")
    logging.info(f"Ensemble Test MAE: {test_mae:.6f}")

    # Log final metrics to MLflow
    if MLFLOW_TRACKING:
        log_final_metrics({
            'ensemble_test_mse': float(test_mse),
            'ensemble_test_mae': float(test_mae),
            **{f'val_loss_{k}': float(v) for k, v in val_losses.items()},
        })
        end_run()

    # ── Export Latent Embeddings for Fusion ──
    logging.info("\nExtracting TS latent embeddings for Fusion model...")
    # We export embeddings for the Validation set because the Fusion model trains on VAL 
    # (to prevent target leakage from the TS models train set)
    ts_embs = get_ensemble_embeddings(trained_models, weights, val_loader, device)
    
    features_dir = os.path.join(PROJECT_ROOT, 'data', 'features')
    os.makedirs(features_dir, exist_ok=True)
    emb_path = os.path.join(features_dir, 'ts_embeddings.npy')
    np.save(emb_path, ts_embs)
    logging.info(f"TS Embeddings saved: {ts_embs.shape} to {emb_path}")

    logging.info("\nTIME SERIES TRAINING PIPELINE COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
