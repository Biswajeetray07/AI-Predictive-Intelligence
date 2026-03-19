"""
Prediction Tracker & Logger
============================
Logs all predictions with metadata for:
- Post-hoc accuracy analysis
- Confidence calibration assessment
- Model performance monitoring over time
- Drift detection on prediction distributions
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PredictionTracker:
    """Logs and analyzes prediction history for monitoring."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or os.path.join(PROJECT_ROOT, 'logs', 'predictions')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, 'prediction_log.csv')
        self._ensure_log_exists()

    def _ensure_log_exists(self):
        """Create the prediction log file if it doesn't exist."""
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(columns=[
                'timestamp', 'ticker', 'prediction_1d', 'prediction_5d', 'prediction_30d',
                'confidence', 'regime', 'ensemble_weights', 'model_version',
                'actual_1d', 'actual_5d', 'actual_30d', 'error_1d', 'error_5d', 'error_30d'
            ])
            df.to_csv(self.log_file, index=False)

    def log_prediction(self, ticker: str, predictions: Dict, confidence: float,
                        regime: int = -1, ensemble_weights: Optional[Dict] = None,
                        model_version: str = 'v1') -> None:
        """Log a single prediction."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'prediction_1d': predictions.get('1d', 0.0),
            'prediction_5d': predictions.get('5d', 0.0),
            'prediction_30d': predictions.get('30d', 0.0),
            'confidence': confidence,
            'regime': regime,
            'ensemble_weights': json.dumps(ensemble_weights or {}),
            'model_version': model_version,
            'actual_1d': None,
            'actual_5d': None,
            'actual_30d': None,
            'error_1d': None,
            'error_5d': None,
            'error_30d': None,
        }

        df = pd.DataFrame([record])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

    def update_actuals(self, ticker: str, timestamp: str, actuals: Dict) -> None:
        """Update a prediction record with actual values once known."""
        df = pd.read_csv(self.log_file)

        mask = (df['ticker'] == ticker) & (df['timestamp'] == timestamp)
        if mask.sum() == 0:
            logging.warning(f"No prediction found for {ticker} at {timestamp}")
            return

        for horizon in ['1d', '5d', '30d']:
            if horizon in actuals:
                df.loc[mask, f'actual_{horizon}'] = actuals[horizon]
                pred_col = f'prediction_{horizon}'
                if pd.notna(df.loc[mask, pred_col].values[0]):
                    df.loc[mask, f'error_{horizon}'] = (
                        actuals[horizon] - df.loc[mask, pred_col].values[0]
                    )

        df.to_csv(self.log_file, index=False)

    def get_accuracy_report(self) -> Dict:
        """Generate accuracy report from logged predictions that have actuals."""
        df = pd.read_csv(self.log_file)

        report = {
            'total_predictions': len(df),
            'predictions_with_actuals': int(df['actual_1d'].notna().sum()),
            'timestamp': datetime.now().isoformat(),
        }

        for horizon in ['1d', '5d', '30d']:
            pred_col = f'prediction_{horizon}'
            actual_col = f'actual_{horizon}'
            error_col = f'error_{horizon}'

            valid = df[df[actual_col].notna()]
            if len(valid) == 0:
                report[f'{horizon}_metrics'] = {'status': 'no_actuals_yet'}
                continue

            errors = valid[error_col].astype(float)
            report[f'{horizon}_metrics'] = {
                'n_predictions': len(valid),
                'mae': float(errors.abs().mean()),
                'rmse': float(np.sqrt((errors ** 2).mean())),
                'mean_error': float(errors.mean()),
                'directional_accuracy': float(
                    ((valid[pred_col] > 0) == (valid[actual_col] > 0)).mean()
                ) if len(valid) > 0 else 0.0,
            }

        # Confidence calibration
        valid_conf = df[(df['actual_1d'].notna()) & (df['confidence'].notna())]
        if len(valid_conf) > 10:
            # Bin predictions by confidence
            valid_conf = valid_conf.copy()
            valid_conf['conf_bin'] = pd.cut(valid_conf['confidence'], bins=5)
            valid_conf['correct'] = (
                (valid_conf['prediction_1d'] > 0) == (valid_conf['actual_1d'] > 0)
            )
            calibration = valid_conf.groupby('conf_bin', observed=True)['correct'].mean()
            report['confidence_calibration'] = {
                str(k): float(v) for k, v in calibration.items()
            }

        return report

    def get_recent_predictions(self, n: int = 50) -> pd.DataFrame:
        """Get most recent predictions."""
        df = pd.read_csv(self.log_file)
        return df.tail(n)


def get_tracker() -> PredictionTracker:
    """Get or create a global PredictionTracker instance."""
    return PredictionTracker()


if __name__ == '__main__':
    tracker = get_tracker()
    report = tracker.get_accuracy_report()
    print(json.dumps(report, indent=2))
