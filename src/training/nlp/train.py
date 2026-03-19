"""
Training script for the Multi-Task NLP model.

Pipeline:
    1. Load unified text data from dataset_loader
    2. Tokenize with DeBERTa-v3 tokenizer
    3. Generate weak supervision labels (sentiment, events, topics)
    4. Train multi-task model with combined loss
    5. Extract and save NLP features:
       - features/nlp_signals.parquet  (daily aggregated signals)
       - features/nlp_embeddings.npy   (768-D dense embeddings per day)
"""

import os
import sys
import logging
import json
from datetime import datetime
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.seed import set_seed

try:
    from src.utils.mlflow_utils import setup_experiment, start_run, log_epoch_metrics, log_final_metrics, end_run  # type: ignore[import, no-redef]
    MLFLOW_TRACKING = True
except ImportError:
    MLFLOW_TRACKING = False
    # Provide dummy functions with correct signatures to avoid type errors
    def setup_experiment(*args, **kwargs): return None
    def start_run(*args, **kwargs): return None
    def log_epoch_metrics(*args, **kwargs): pass
    def log_final_metrics(*args, **kwargs): pass
    def end_run(*args, **kwargs): pass

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.training.nlp.dataset import load_all_sources
from src.models.nlp.tokenizer import NLPTokenizer
from src.models.nlp.model import (
    MultiTaskNLPModel,
    generate_weak_sentiment_labels,
    generate_weak_event_labels,
    generate_weak_topic_labels,
    generate_weak_entity_labels,
    NUM_TOPICS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config():
    """Load training configuration from YAML file."""
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'training_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('nlp_model', {})
    # Defaults
    return {
        'batch_size': 16,
        'epochs': 3,
        'learning_rate': 2e-5,
        'max_length': 256,
        'freeze_layers': 6,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }


# ─── Dataset ─────────────────────────────────────────────────────────────────

class NLPDataset(Dataset):
    """PyTorch Dataset wrapping the unified NLP text data."""

    def __init__(self, df: pd.DataFrame, tokenizer: NLPTokenizer):
        self.texts = df['text'].tolist()
        self.source_ids = torch.tensor(df['source_id'].values, dtype=torch.long)
        self.days = torch.tensor(df['day_of_week'].values, dtype=torch.long)
        self.months = torch.tensor(df['month'].values, dtype=torch.long)
        self.dates = df['date'].values

        # Generate weak labels
        logging.info("Generating weak supervision labels...")
        self.sentiment_labels = generate_weak_sentiment_labels(self.texts)
        self.event_labels = generate_weak_event_labels(self.texts)
        self.topic_labels = generate_weak_topic_labels(self.texts)
        self.entity_labels = generate_weak_entity_labels(self.texts)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Streaming tokenization (on-the-fly instead of keeping all in RAM)
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode(text, return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'source_ids': self.source_ids[idx],
            'day_of_week': self.days[idx],
            'month': self.months[idx],
            'sentiment_label': self.sentiment_labels[idx],
            'event_label': self.event_labels[idx],
            'topic_label': self.topic_labels[idx],
            'entity_label': self.entity_labels[idx],
        }


# ─── Training Loop ───────────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, device, scaler):
    """Train for one epoch with multi-task loss and AMP."""
    model.train()
    total_loss = 0
    sentiment_criterion = nn.CrossEntropyLoss()
    event_criterion = nn.CrossEntropyLoss()
    topic_criterion = nn.BCEWithLogitsLoss(reduction='none')  # multi-label with per-element reduction for NaN detection
    entity_criterion = nn.BCEWithLogitsLoss(reduction='none') # multi-label with per-element reduction for NaN detection

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        source_ids = batch['source_ids'].to(device)
        day_of_week = batch['day_of_week'].to(device)
        month = batch['month'].to(device)

        sentiment_labels = batch['sentiment_label'].to(device)
        event_labels = batch['event_label'].to(device)
        topic_labels = batch['topic_label'].to(device)
        entity_labels = batch['entity_label'].to(device)

        # Validate labels for NaN
        if torch.isnan(sentiment_labels).any() or torch.isnan(event_labels).any() or \
           torch.isnan(topic_labels).any() or torch.isnan(entity_labels).any():
            logging.warning(f"Batch {batch_idx} contains NaN labels, skipping...")
            continue

        optimizer.zero_grad()

        # Use Mixed Precision only if CUDA is available
        use_amp = device.type == 'cuda'
        try:
            from torch.cuda.amp import autocast
        except ImportError:
            autocast = None

        if use_amp and autocast:
            with autocast():
                outputs = model(input_ids, attention_mask, source_ids, day_of_week, month)
                loss_sentiment = sentiment_criterion(outputs['sentiment'], sentiment_labels)
                loss_events = event_criterion(outputs['events'], event_labels)
                # Reduce properly for BCE losses
                loss_topics = topic_criterion(outputs['topics'], topic_labels.float()).mean()
                loss_entities = entity_criterion(outputs['entities'], entity_labels.float()).mean()
                loss = 1.0 * loss_sentiment + 0.8 * loss_events + 0.5 * loss_topics + 0.5 * loss_entities
        else:
            outputs = model(input_ids, attention_mask, source_ids, day_of_week, month)
            loss_sentiment = sentiment_criterion(outputs['sentiment'], sentiment_labels)
            loss_events = event_criterion(outputs['events'], event_labels)
            # Reduce properly for BCE losses
            loss_topics = topic_criterion(outputs['topics'], topic_labels.float()).mean()
            loss_entities = entity_criterion(outputs['entities'], entity_labels.float()).mean()
            loss = 1.0 * loss_sentiment + 0.8 * loss_events + 0.5 * loss_topics + 0.5 * loss_entities

        # Skip step if loss is NaN
        if torch.isnan(loss):
            logging.warning(f"NaN loss at batch {batch_idx}, skipping step")
            continue

        if use_amp and autocast:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            logging.info(f"  Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f} "
                         f"(sent: {loss_sentiment.item():.4f}, evt: {loss_events.item():.4f}, "
                         f"topic: {loss_topics.item():.4f})")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ─── Feature Extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, dataloader, df, device, output_dir):
    """
    Run the trained model in inference mode to extract:
      1. nlp_signals.parquet — daily aggregated signal scores
      2. nlp_embeddings.npy   — 768-D dense embeddings per day
    """
    model.eval()

    all_embeddings = []
    all_sentiments = []
    all_events = []
    all_topics = []
    all_dates = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        source_ids = batch['source_ids'].to(device)
        day_of_week = batch['day_of_week'].to(device)
        month = batch['month'].to(device)

        outputs = model(input_ids, attention_mask, source_ids, day_of_week, month)

        all_embeddings.append(outputs['embedding'].cpu().numpy())
        all_sentiments.append(torch.softmax(outputs['sentiment'], dim=-1).cpu().numpy())
        all_events.append(torch.softmax(outputs['events'], dim=-1).cpu().numpy())
        all_topics.append(torch.sigmoid(outputs['topics']).cpu().numpy())

    # Concatenate everything
    embeddings = np.concatenate(all_embeddings, axis=0)
    sentiments = np.concatenate(all_sentiments, axis=0)
    events = np.concatenate(all_events, axis=0)
    topics = np.concatenate(all_topics, axis=0)

    # Build per-sample signals DataFrame
    signals_df = pd.DataFrame({
        'date': df['date'].values[:len(embeddings)],
        'ticker': df['ticker'].values[:len(embeddings)],
        'sentiment_positive': sentiments[:, 0],
        'sentiment_negative': sentiments[:, 1],
        'sentiment_neutral': sentiments[:, 2],
        'event_policy': events[:, 0],
        'event_tech': events[:, 1],
        'event_supply': events[:, 2],
        'event_crash': events[:, 3],
        'event_launch': events[:, 4],
        'event_regulation': events[:, 5],
        'event_economic': events[:, 6],
        'event_none': events[:, 7],
    })

    # Add topic columns
    topic_names = ['AI', 'EV', 'Semiconductors', 'Crypto', 'Climate', 'Energy', 'Healthcare', 'Finance', 'Geopolitics', 'Other']
    for i, name in enumerate(topic_names):
        signals_df[f'topic_{name}'] = topics[:, i]

    # ── Aggregate to daily level ──
    daily_signals = signals_df.groupby(['date', 'ticker']).mean().reset_index()

    # Save signals
    signals_path = os.path.join(output_dir, 'nlp_signals.parquet')
    daily_signals.to_parquet(signals_path, index=False)
    logging.info(f"Saved NLP signals: {daily_signals.shape} to {signals_path}")

    # ── Aggregate embeddings to daily level ──
    emb_df = pd.DataFrame({
        'date': df['date'].values[:len(embeddings)],
        'ticker': df['ticker'].values[:len(embeddings)]
    })
    emb_groups = emb_df.groupby(['date', 'ticker']).groups

    daily_embeddings = []
    daily_dates = []
    daily_tickers = []
    for group_key, indices in sorted(emb_groups.items()):
        date, ticker = group_key  # type: ignore[misc]  # Proper tuple unpacking
        daily_embeddings.append(embeddings[indices].mean(axis=0))
        daily_dates.append(date)
        daily_tickers.append(ticker)

    daily_embeddings = np.array(daily_embeddings)
    emb_path = os.path.join(output_dir, 'nlp_embeddings.npy')
    np.save(emb_path, daily_embeddings)
    logging.info(f"Saved NLP embeddings: {daily_embeddings.shape} to {emb_path}")

    # Save date index for alignment
    pd.DataFrame({'date': daily_dates, 'ticker': daily_tickers}).to_csv(
        os.path.join(output_dir, 'nlp_embedding_dates.csv'), index=False
    )

    return daily_signals, daily_embeddings


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    config = load_config()
    device = torch.device(config.get('device', 'cpu'))
    logging.info(f"Using device: {device}")

    # Defensive: Clean up any leaked MLflow runs before starting
    if MLFLOW_TRACKING:
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass
        setup_experiment('nlp_training')

    model_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    model_path = os.path.join(model_dir, 'nlp_multitask_model.pt')
    
    # ── Check if already trained ──
    if os.path.exists(model_path):
        logging.info("NLP multi-task model already trained and saved. Skipping training.")
        return

    if MLFLOW_TRACKING:
        start_run('nlp_multitask_training', params=config)

    # 1. Load data
    logging.info("Loading all text sources...")
    df = load_all_sources(PROJECT_ROOT)
    if df.empty:
        logging.error("No data loaded. Exiting.")
        if MLFLOW_TRACKING:
            end_run()
        return

    # 2. Tokenize
    tokenizer = NLPTokenizer(max_length=config.get('max_length', 256))

    # For memory efficiency, process in chunks if dataset is very large
    max_samples = 50000  # Limit for initial training
    if len(df) > max_samples:
        logging.info(f"Dataset too large ({len(df)}), sampling {max_samples} rows for training")
        df_sampled = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    else:
        df_sampled = df

    # ── Train/Val Split (80/20) to detect overfitting ──
    val_ratio = 0.2
    n_val = int(len(df_sampled) * val_ratio)
    n_train = len(df_sampled) - n_val
    df_train = df_sampled.iloc[:n_train].reset_index(drop=True)
    df_val = df_sampled.iloc[n_train:].reset_index(drop=True)
    logging.info(f"NLP split: {n_train} train, {n_val} val")

    train_dataset = NLPDataset(df_train, tokenizer)
    val_dataset = NLPDataset(df_val, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 16), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 16), shuffle=False)

    # ── Validate Weak Labels ──
    from src.models.nlp.model import validate_weak_labels
    # Generate weak labels for each task
    texts = df_train['text'].tolist()
    weak_labels_dict = {
        'sentiment': generate_weak_sentiment_labels(texts),
        'event': generate_weak_event_labels(texts),
        'topic': generate_weak_topic_labels(texts),
        'entity': generate_weak_entity_labels(texts)
    }
    label_quality = validate_weak_labels(weak_labels_dict)
    logging.info(f"Weak label quality report:")
    for task_name, quality in label_quality.items():
        logging.info(f"  {task_name}: {quality}")
    # Save label quality report
    quality_path = os.path.join(model_dir, 'nlp_label_quality.json')
    with open(quality_path, 'w') as f:
        json.dump({k: str(v) for k, v in label_quality.items()}, f, indent=2)
    logging.info(f"Label quality saved to {quality_path}")

    # 3. Build model
    model = MultiTaskNLPModel(
        model_name=tokenizer.model_name,
        freeze_encoder_layers=config.get('freeze_layers', 6)
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    # 4. Optimizer & Scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get('learning_rate', 2e-5),
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.get('epochs', 3))

    # ── Checkpoint setup ──
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'nlp_checkpoint.pt')

    start_epoch = 0
    try:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    except ImportError:
        scaler = None
    
    if os.path.exists(checkpoint_path):
        logging.info("Found NLP checkpoint. Resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resuming NLP from epoch {start_epoch+1}")

    # 5. Training loop with validation
    epochs = config.get('epochs', 3)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 5)
    
    for epoch in range(start_epoch, epochs):
        logging.info(f"\n{'='*50}")
        logging.info(f"Epoch {epoch+1}/{epochs}")
        logging.info(f"{'='*50}")

        avg_train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        scheduler.step()

        # ── Validation ──
        model.eval()
        val_loss = 0
        sentiment_criterion = nn.CrossEntropyLoss()
        event_criterion = nn.CrossEntropyLoss()
        topic_criterion = nn.BCEWithLogitsLoss()
        entity_criterion = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                source_ids = batch['source_ids'].to(device)
                day_of_week = batch['day_of_week'].to(device)
                month = batch['month'].to(device)
                
                outputs = model(input_ids, attention_mask, source_ids, day_of_week, month)
                loss = (
                    1.0 * sentiment_criterion(outputs['sentiment'], batch['sentiment_label'].to(device))
                    + 0.8 * event_criterion(outputs['events'], batch['event_label'].to(device))
                    + 0.5 * topic_criterion(outputs['topics'], batch['topic_label'].to(device).float())
                    + 0.5 * entity_criterion(outputs['entities'], batch['entity_label'].to(device).float())
                )
                val_loss += loss.item()
        
        avg_val_loss = val_loss / max(len(val_loader), 1)
        logging.info(f"Epoch {epoch+1} — Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_path = os.path.join(model_dir, 'nlp_multitask_model_best.pt')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"  ✅ New best val loss: {best_val_loss:.4f} — saved")
        else:
            patience_counter += 1
            logging.info(f"  ⚠️ No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                logging.info(f"  🛑 Early stopping triggered at epoch {epoch+1}")
                break

        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, checkpoint_path)

        if MLFLOW_TRACKING:
            log_epoch_metrics(epoch, {'nlp_train_loss': avg_train_loss, 'nlp_val_loss': avg_val_loss})

    # 6. Save trained model
    model_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'nlp_multitask_model.pt')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")

    # 7. Extract features on FULL dataset
    logging.info("\nExtracting NLP features on full dataset...")
    full_dataset = NLPDataset(df, tokenizer)
    full_dataloader = DataLoader(full_dataset, batch_size=config.get('batch_size', 16), shuffle=False)

    features_dir = os.path.join(PROJECT_ROOT, 'data', 'features')
    os.makedirs(features_dir, exist_ok=True)

    extract_features(model, full_dataloader, df, device, features_dir)

    if MLFLOW_TRACKING:
        end_run()

    logging.info("\nNLP TRAINING PIPELINE COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
