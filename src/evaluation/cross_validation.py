"""
Time-Series Cross-Validation (Expanding Window)
================================================
Implements proper temporal cross-validation to avoid data leakage.
Uses expanding (anchored) training windows with fixed-size validation windows.

Example with 5 folds:
  Fold 1: Train [0..200]   Val [200..260]
  Fold 2: Train [0..260]   Val [260..320]
  Fold 3: Train [0..320]   Val [320..380]
  Fold 4: Train [0..380]   Val [380..440]
  Fold 5: Train [0..440]   Val [440..500]
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TimeSeriesCrossValidator:
    """
    Expanding-window cross-validation for time-series models.
    Prevents look-ahead bias by only expanding the training window forward.
    """

    def __init__(self, n_folds: int = 5, min_train_ratio: float = 0.4, val_size_ratio: float = 0.1):
        """
        Args:
            n_folds: Number of CV folds
            min_train_ratio: Minimum fraction of data used for training in the first fold
            val_size_ratio: Fraction of total data used for each validation window
        """
        self.n_folds = n_folds
        self.min_train_ratio = min_train_ratio
        self.val_size_ratio = val_size_ratio

    def get_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/val index splits for expanding window CV."""
        val_size = max(int(n_samples * self.val_size_ratio), 1)
        min_train = max(int(n_samples * self.min_train_ratio), val_size)

        splits = []
        for fold in range(self.n_folds):
            train_end = min_train + fold * val_size
            val_end = train_end + val_size

            if val_end > n_samples:
                break

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(train_end, val_end)
            splits.append((train_idx, val_idx))

        return splits

    def cross_validate(
        self,
        model_class,
        model_kwargs: dict,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = 'cpu'
    ) -> Dict:
        """
        Run expanding-window CV on a time-series model.

        Args:
            model_class: PyTorch model class (e.g., LSTMForecaster)
            model_kwargs: Dict of kwargs to instantiate the model
            X: Full feature array [n_samples, seq_len, n_features]
            y: Full target array [n_samples]
            epochs: Training epochs per fold
            batch_size: Batch size
            lr: Learning rate
            device: 'cpu', 'cuda', or 'mps'

        Returns:
            Dict with per-fold metrics and aggregated results
        """
        splits = self.get_splits(len(X))
        device_obj = torch.device(device)

        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logging.info(f"\n--- CV Fold {fold_idx + 1}/{len(splits)} ---")
            logging.info(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

            # Prepare data
            X_train = torch.tensor(X[train_idx], dtype=torch.float32)
            y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(-1)
            X_val = torch.tensor(X[val_idx], dtype=torch.float32)
            y_val = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(-1)

            train_ds = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)  # No shuffle for TS!

            # Fresh model per fold
            model = model_class(**model_kwargs).to(device_obj)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            # Train
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device_obj), yb.to(device_obj)
                    optimizer.zero_grad()
                    pred, _ = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

            # Evaluate
            model.eval()
            with torch.no_grad():
                X_val_dev = X_val.to(device_obj)
                y_val_dev = y_val.to(device_obj)
                preds, _ = model(X_val_dev)
                val_mse = nn.MSELoss()(preds, y_val_dev).item()
                val_mae = nn.L1Loss()(preds, y_val_dev).item()

                # Directional accuracy
                if len(preds) > 1:
                    pred_dir = (preds[1:] - preds[:-1]) > 0
                    actual_dir = (y_val_dev[1:] - y_val_dev[:-1]) > 0
                    dir_acc = (pred_dir == actual_dir).float().mean().item()
                else:
                    dir_acc = 0.5

            fold_result = {
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'val_mse': val_mse,
                'val_rmse': np.sqrt(val_mse),
                'val_mae': val_mae,
                'directional_accuracy': dir_acc,
            }
            fold_results.append(fold_result)
            logging.info(f"  MSE: {val_mse:.6f} | RMSE: {np.sqrt(val_mse):.6f} | "
                         f"MAE: {val_mae:.6f} | Dir Acc: {dir_acc:.2%}")

        # Aggregate
        metrics = ['val_mse', 'val_rmse', 'val_mae', 'directional_accuracy']
        summary = {}
        for m in metrics:
            values = [r[m] for r in fold_results]
            summary[f'{m}_mean'] = np.mean(values)
            summary[f'{m}_std'] = np.std(values)

        logging.info(f"\n--- CV Summary ({len(splits)} folds) ---")
        logging.info(f"  RMSE: {summary['val_rmse_mean']:.6f} ± {summary['val_rmse_std']:.6f}")
        logging.info(f"  MAE:  {summary['val_mae_mean']:.6f} ± {summary['val_mae_std']:.6f}")
        logging.info(f"  Dir Acc: {summary['directional_accuracy_mean']:.2%} ± {summary['directional_accuracy_std']:.2%}")

        return {
            'folds': fold_results,
            'summary': summary,
            'n_folds': len(splits),
            'timestamp': datetime.now().isoformat(),
        }


def run_ts_cross_validation(device: str = 'cpu', n_folds: int = 5) -> Dict:
    """
    Convenience function: runs expanding-window CV on all 4 TS model types
    using saved model_inputs data.
    """
    from src.models.timeseries.lstm import LSTMForecaster
    from src.models.timeseries.gru import GRUForecaster
    from src.models.timeseries.transformer import TransformerForecaster
    from src.models.timeseries.tft import TFTForecaster

    model_inputs = os.path.join(PROJECT_ROOT, 'data', 'processed', 'model_inputs')
    X = np.load(os.path.join(model_inputs, 'X_train.npy'), mmap_mode='r')
    y = np.load(os.path.join(model_inputs, 'y_train.npy'), mmap_mode='r')

    # Sub-sample for speed (max 5000)
    n = min(len(X), 5000)
    indices = np.random.RandomState(42).choice(len(X), n, replace=False)
    indices.sort()  # Keep temporal order!
    X_sub = np.array(X[indices])
    y_sub = np.array(y[indices])

    input_dim = X_sub.shape[-1]
    cv = TimeSeriesCrossValidator(n_folds=n_folds)

    models = {
        'lstm': (LSTMForecaster, {'input_dim': input_dim, 'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.1}),
        'gru': (GRUForecaster, {'input_dim': input_dim, 'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.1}),
        'transformer': (TransformerForecaster, {'input_dim': input_dim, 'd_model': 128, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1}),
        'tft': (TFTForecaster, {'input_dim': input_dim, 'hidden_dim': 128, 'num_heads': 4, 'num_layers': 2, 'dropout': 0.1}),
    }

    all_results = {}
    for name, (cls, kwargs) in models.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"Cross-validating: {name.upper()}")
        logging.info(f"{'='*60}")
        result = cv.cross_validate(cls, kwargs, X_sub, y_sub, epochs=5, device=device)
        all_results[name] = result

    return all_results


if __name__ == '__main__':
    import json
    results = run_ts_cross_validation()
    # Save results
    out_path = os.path.join(PROJECT_ROOT, 'saved_models', 'cv_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")
