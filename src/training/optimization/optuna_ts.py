import os
import sys
import optuna
import mlflow
import torch
import torch.nn as nn
import numpy as np
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.training.optimization.walk_forward import WalkForwardSplitter, get_subset_dataloaders
from src.training.timeseries.dataset import TimeSeriesDataset
from src.training.timeseries.train import build_model, train_one_epoch, evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def objective_timeseries(trial, model_name, X_full, y_full, device):
    """
    Optuna objective function for a specific Time-Series model (LSTM, GRU, Transformer, TFT).
    Uses Walk-Forward Validation and logs to MLflow.
    """
    # 1. Suggest Hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 192, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.7)
    lr = trial.suggest_float('learning_rate', 1e-5, 2e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    l1_lambda = trial.suggest_float('l1_lambda', 1e-7, 1e-3, log=True)
    batch_size = 64  # Fixed as per user request
    
    config = {
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'l1_lambda': l1_lambda,
        'batch_size': batch_size
    }
    
    # Enforced max epochs and early stopping patience
    max_epochs = 50
    patience = 10
    
    # 2. Setup MLflow Tracking
    mlflow.set_experiment(f"Hyperopt_{model_name.upper()}")
    fold_losses = []
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(config)
        
        # 3. Walk-Forward CV
        full_dataset = torch.utils.data.TensorDataset(torch.tensor(X_full, dtype=torch.float32), 
                                                      torch.tensor(y_full, dtype=torch.float32))
        
        splitter = WalkForwardSplitter(n_splits=3, min_train_size_ratio=0.4)
        splits = splitter.split(len(full_dataset))
        
        input_dim = X_full.shape[-1]
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            train_loader, val_loader = get_subset_dataloaders(full_dataset, train_idx, val_idx, batch_size)
            
            # Re-initialize model per fold
            model = build_model(model_name, input_dim, config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.MSELoss()
            
            # MPS/CPU compatible: only use GradScaler on CUDA
            use_amp = device.type == 'cuda'
            try:
                from torch.cuda.amp import GradScaler
                scaler = GradScaler() if use_amp else None
            except ImportError:
                scaler = None  # AMP not available
            
            best_val_loss = float('inf')
            early_stop_counter = 0
            
            for epoch in range(max_epochs):
                # Standard training
                model.train()
                epoch_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    
                    outputs, _ = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    # Add L1 Penalty
                    l1_penalty = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + l1_lambda * l1_penalty
                    
                    loss.backward()
                    optimizer.step()
                
                val_loss = evaluate(model, val_loader, criterion, device)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0  # Reset patience if improved
                else:
                    early_stop_counter += 1
                
                # Optuna pruning based on averge across folds
                step = fold * max_epochs + epoch
                trial.report(val_loss, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
                # Early Stopping
                if early_stop_counter >= patience:
                    logging.info(f"Fold {fold} early stopping triggered at epoch {epoch}")
                    break
                    
            fold_losses.append(best_val_loss)
            mlflow.log_metric(f"fold_{fold}_loss", best_val_loss)

        avg_loss = np.mean(fold_losses)
        mlflow.log_metric("avg_val_loss", float(avg_loss))  # Convert to float
        
    return avg_loss
