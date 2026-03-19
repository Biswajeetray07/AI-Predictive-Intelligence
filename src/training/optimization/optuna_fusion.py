import os
import sys
import optuna
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.models.fusion.multi_horizon_fusion import MultiHorizonFusionModel
from src.training.fusion.train import load_fusion_data, FusionDataset

def objective_fusion(trial, device):
    """
    Optuna objective function for the MultiHorizonFusionModel.
    """
    # 1. Hyperparameters
    attention_heads = trial.suggest_categorical('attention_heads', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.6)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    l1_lambda = trial.suggest_float('l1_lambda', 1e-7, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    
    # MLP Layer depth and size
    num_layers = trial.suggest_int('mlp_layers', 1, 3)
    mlp_hidden = []
    in_dim = trial.suggest_categorical('hidden_dim_L1', [128, 256, 512])
    mlp_hidden.append(in_dim)
    for i in range(1, num_layers):
        dim = trial.suggest_categorical(f'hidden_dim_L{i+1}', [64, 128, 256])
        mlp_hidden.append(dim)
        
    config = {
        'attention_heads': attention_heads,
        'dropout': dropout,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'l1_lambda': l1_lambda,
        'batch_size': batch_size,
        'mlp_hidden': mlp_hidden
    }
    
    # 2. Data Loader
    nlp_emb, ts_feat, targets = load_fusion_data(PROJECT_ROOT)
    
    n = len(nlp_emb)
    train_end = int(n * 0.8)
    
    train_ds = FusionDataset(nlp_emb[:train_end], ts_feat[:train_end], targets[:train_end])
    val_ds = FusionDataset(nlp_emb[train_end:], ts_feat[train_end:], targets[train_end:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # 3. Setup MLflow
    mlflow.set_experiment("Hyperopt_FUSION")
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(config)
        
        model = MultiHorizonFusionModel(
            nlp_dim=nlp_emb.shape[-1],
            ts_dim=ts_feat.shape[-1],
            attention_heads=attention_heads,
            mlp_hidden=mlp_hidden,
            dropout=dropout
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        best_val = float('inf')
        max_epochs = 10
        
        for epoch in range(max_epochs):
            # Train
            model.train()
            for nlp, ts, y in train_loader:
                nlp, ts, y = nlp.to(device), ts.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(nlp, ts)
                out_1d, out_5d, out_30d = outputs['1d'], outputs['5d'], outputs['30d']
                
                main_loss = criterion(out_1d, y[:, 0:1]) + criterion(out_5d, y[:, 1:2]) + criterion(out_30d, y[:, 2:3])
                
                # Consistent L1 regularization
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss = main_loss + l1_lambda * l1_penalty
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for nlp, ts, y in val_loader:
                    nlp, ts, y = nlp.to(device), ts.to(device), y.to(device)
                    outputs = model(nlp, ts)
                    out_1d, out_5d, out_30d = outputs['1d'], outputs['5d'], outputs['30d']
                    v_loss = criterion(out_1d, y[:, 0:1]) + criterion(out_5d, y[:, 1:2]) + criterion(out_30d, y[:, 2:3])
                    val_loss += v_loss.item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val:
                best_val = val_loss
                
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        mlflow.log_metric("val_loss", best_val)
        
    return best_val

