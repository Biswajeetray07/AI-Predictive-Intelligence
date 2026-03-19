"""
Ensemble Ablation Study
=======================
Evaluates how much each model contributes to ensemble performance.
Tests: full ensemble, leave-one-out, and individual model performance.
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EnsembleAblation:
    """Runs ablation studies across all TS models to measure each model's marginal contribution."""

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.models = {}

    def load_models(self):
        """Load all trained TS models."""
        from src.models.timeseries.lstm import LSTMForecaster
        from src.models.timeseries.gru import GRUForecaster
        from src.models.timeseries.transformer import TransformerForecaster
        from src.models.timeseries.tft import TFTForecaster

        model_dir = os.path.join(PROJECT_ROOT, 'saved_models')

        # Need to know input_dim from data
        model_inputs = os.path.join(PROJECT_ROOT, 'data', 'processed', 'model_inputs')
        X = np.load(os.path.join(model_inputs, 'X_train.npy'), mmap_mode='r')
        input_dim = X.shape[-1]

        model_configs = {
            'lstm': (LSTMForecaster, {'input_dim': input_dim, 'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.0}),
            'gru': (GRUForecaster, {'input_dim': input_dim, 'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.0}),
            'transformer': (TransformerForecaster, {'input_dim': input_dim, 'd_model': 128, 'nhead': 4, 'num_layers': 2, 'dropout': 0.0}),
            'tft': (TFTForecaster, {'input_dim': input_dim, 'hidden_dim': 128, 'num_heads': 4, 'num_layers': 2, 'dropout': 0.0}),
        }

        for name, (cls, kwargs) in model_configs.items():
            path = os.path.join(model_dir, f'{name}_model.pt')
            if os.path.exists(path):
                model = cls(**kwargs)
                model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
                model.to(self.device).eval()
                self.models[name] = model
                logging.info(f"Loaded: {name}")
            else:
                logging.warning(f"Model not found: {path}")

        logging.info(f"Loaded {len(self.models)}/{len(model_configs)} models")

    def _evaluate_subset(self, model_names: List[str], X_test: np.ndarray, y_test: np.ndarray,
                          weights: Optional[Dict[str, float]] = None) -> Dict:
        """Evaluate a subset of ensemble models."""
        if not model_names:
            return {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf'), 'dir_acc': 0.0}

        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_true = y_test.flatten()

        predictions = {}
        with torch.no_grad():
            for name in model_names:
                if name in self.models:
                    pred, _ = self.models[name](X_tensor)
                    predictions[name] = pred.cpu().numpy().flatten()

        if not predictions:
            return {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf'), 'dir_acc': 0.0}

        # Weighted average
        if weights is None:
            weights = {k: 1.0 / len(predictions) for k in predictions}
        
        w_sum = sum(weights.get(k, 0) for k in predictions)
        if w_sum == 0:
            w_sum = 1.0

        ensemble_pred = np.zeros(len(y_true))
        for name, pred in predictions.items():
            ensemble_pred += pred * weights.get(name, 1.0 / len(predictions)) / w_sum

        # Metrics
        mse = np.mean((ensemble_pred - y_true) ** 2)
        mae = np.mean(np.abs(ensemble_pred - y_true))
        rmse = np.sqrt(mse)

        # Directional accuracy
        if len(y_true) > 1:
            pred_dir = np.diff(ensemble_pred) > 0
            actual_dir = np.diff(y_true) > 0
            dir_acc = np.mean(pred_dir == actual_dir)
        else:
            dir_acc = 0.5

        return {'mse': float(mse), 'mae': float(mae), 'rmse': float(rmse), 'dir_acc': float(dir_acc)}

    def run_ablation(self, X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None) -> Dict:
        """
        Run full ablation study:
        1. Full ensemble performance
        2. Each model individually
        3. Leave-one-out (drop each model)
        4. Calculate marginal contribution of each model
        """
        if not self.models:
            self.load_models()

        # Load test data if not provided
        if X_test is None or y_test is None:
            model_inputs = os.path.join(PROJECT_ROOT, 'data', 'processed', 'model_inputs')
            X_test = np.array(np.load(os.path.join(model_inputs, 'X_test.npy'), mmap_mode='r')[:1000])
            y_test = np.array(np.load(os.path.join(model_inputs, 'y_test.npy'), mmap_mode='r')[:1000])

        all_names = list(self.models.keys())
        results: Dict[str, Any] = {'models_available': all_names}

        # 1. Full ensemble
        full_result = self._evaluate_subset(all_names, X_test, y_test)
        results['full_ensemble'] = full_result
        logging.info(f"\nFull Ensemble: RMSE={full_result['rmse']:.6f}, Dir Acc={full_result['dir_acc']:.2%}")

        # 2. Individual models
        results['individual'] = {}
        for name in all_names:
            individual = self._evaluate_subset([name], X_test, y_test)
            results['individual'][name] = individual
            logging.info(f"  {name}: RMSE={individual['rmse']:.6f}, Dir Acc={individual['dir_acc']:.2%}")

        # 3. Leave-one-out
        results['leave_one_out'] = {}
        results['marginal_contribution'] = {}
        for name in all_names:
            subset = [n for n in all_names if n != name]
            loo_result = self._evaluate_subset(subset, X_test, y_test)
            results['leave_one_out'][name] = loo_result

            # Marginal contribution = how much RMSE increases when we remove this model
            contribution = loo_result['rmse'] - full_result['rmse']
            results['marginal_contribution'][name] = float(contribution)
            logging.info(f"  Without {name}: RMSE={loo_result['rmse']:.6f} "
                         f"(contribution: +{contribution:.6f})")

        # 4. Rank by contribution
        ranked = sorted(results['marginal_contribution'].items(), key=lambda x: x[1], reverse=True)
        results['contribution_ranking'] = [{'model': k, 'rmse_increase_without': v} for k, v in ranked]

        logging.info(f"\n--- Contribution Ranking ---")
        for r in results['contribution_ranking']:
            logging.info(f"  {r['model']}: RMSE increases by {r['rmse_increase_without']:.6f} when removed")

        results['timestamp'] = datetime.now().isoformat()
        return results


def run_ensemble_ablation(device: str = 'cpu') -> Dict:
    """Convenience function to run ensemble ablation study."""
    ablation = EnsembleAblation(device=device)
    return ablation.run_ablation()


if __name__ == '__main__':
    import json
    results = run_ensemble_ablation()
    out_path = os.path.join(PROJECT_ROOT, 'saved_models', 'ablation_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Ablation results saved to {out_path}")
