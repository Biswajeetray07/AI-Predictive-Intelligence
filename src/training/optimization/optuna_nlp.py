import os
import sys
import optuna
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.models.nlp.model import MultiTaskNLPModel

def objective_nlp(trial, dataset, device):
    """
    Optuna objective function for the DeBERTa-v3 NLP model.
    """
    # 1. Hyperparameters (Memory-safe for 24GB VRAM)
    lr = trial.suggest_float('learning_rate', 1e-6, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16]) # Kept small for memory
    max_length = trial.suggest_categorical('max_seq_length', [128, 256])
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.2)
    weight_decay = trial.suggest_float('weight_decay', 0.01, 0.1)
    
    # Task weights for multi-task loss
    w_sent = trial.suggest_float('w_sentiment', 0.5, 1.5)
    w_evt = trial.suggest_float('w_events', 0.5, 1.5)
    w_top = trial.suggest_float('w_topics', 0.1, 1.0)
    w_ent = trial.suggest_float('w_entities', 0.1, 1.0)
    
    config = {
        'learning_rate': lr,
        'batch_size': batch_size,
        'max_seq_length': max_length,
        'warmup_ratio': warmup_ratio,
        'weight_decay': weight_decay,
        'w_sentiment': w_sent,
        'w_events': w_evt,
        'w_topics': w_top,
        'w_entities': w_ent
    }
    
    # Short optimization cycles
    max_epochs = 2
    
    # Sub-sample the dataset for optimization (e.g., 20% of the data)
    subset_size = int(len(dataset) * 0.2)
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    
    # Split into 80/20 train/val for the subset
    train_size = int(0.8 * subset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = Subset(dataset, train_indices.tolist())  # Convert to list
    val_dataset = Subset(dataset, val_indices.tolist())  # Convert to list
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Setup MLflow
    mlflow.set_experiment("Hyperopt_NLP_DeBERTa")
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(config)
        
        # Build Model
        model = MultiTaskNLPModel(freeze_encoder_layers=10) # Heavy freeze to speed up search
        model = model.to(device)
        
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=lr, weight_decay=weight_decay)
                          
        total_steps = len(train_loader) * max_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        sentiment_criterion = nn.CrossEntropyLoss()
        event_criterion = nn.CrossEntropyLoss()
        topic_criterion = nn.BCEWithLogitsLoss()
        entity_criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        
        for epoch in range(max_epochs):
            # Train
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                
                outputs = model(
                    batch['input_ids'].to(device), 
                    batch['attention_mask'].to(device), 
                    batch['source_ids'].to(device), 
                    batch['day_of_week'].to(device), 
                    batch['month'].to(device)
                )
                
                l_s = sentiment_criterion(outputs['sentiment'], batch['sentiment_label'].to(device))
                l_v = event_criterion(outputs['events'], batch['event_label'].to(device))
                l_t = topic_criterion(outputs['topics'], batch['topic_label'].to(device))
                l_e = entity_criterion(outputs['entities'], batch['entity_label'].to(device))
                
                loss = (w_sent * l_s) + (w_evt * l_v) + (w_top * l_t) + (w_ent * l_e)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
            # Evaluate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        batch['input_ids'].to(device), 
                        batch['attention_mask'].to(device), 
                        batch['source_ids'].to(device), 
                        batch['day_of_week'].to(device), 
                        batch['month'].to(device)
                    )
                    l_s = sentiment_criterion(outputs['sentiment'], batch['sentiment_label'].to(device))
                    l_v = event_criterion(outputs['events'], batch['event_label'].to(device))
                    l_t = topic_criterion(outputs['topics'], batch['topic_label'].to(device))
                    l_e = entity_criterion(outputs['entities'], batch['entity_label'].to(device))
                    v_loss = (w_sent * l_s) + (w_evt * l_v) + (w_top * l_t) + (w_ent * l_e)
                    val_loss += v_loss.item()
                    
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        mlflow.log_metric("val_loss", best_val_loss)
        
    return best_val_loss
