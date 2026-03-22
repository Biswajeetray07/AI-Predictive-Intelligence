import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import yaml
import pickle
import joblib
from sklearn.preprocessing import StandardScaler

from dotenv import load_dotenv
load_dotenv()

from src.models.fusion.multi_horizon_fusion import MultiHorizonFusionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logging.getLogger().setLevel(logging.INFO)

class BacktestEngine:
    def __init__(self, device: str = 'cpu', model_path: Optional[str] = None, dataset_path: Optional[str] = None, metadata_path: Optional[str] = None, config_path: Optional[str] = None):
        self.device = torch.device(device)
        self.model_path = model_path or os.path.join(PROJECT_ROOT, 'saved_models', 'fusion_model.pt')
        self.dataset_path = dataset_path or os.path.join(PROJECT_ROOT, 'data', 'processed', 'merged', 'all_merged_dataset.csv')
        self.metadata_path = metadata_path or os.path.join(PROJECT_ROOT, 'data', 'processed', 'model_inputs', 'metadata_test.csv')
        
        self.scaler_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'model_inputs', 'target_scaler.pkl')
        self.logger = logging.getLogger('BacktestEngine')
        
        # Load config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        self.embeddings = self._load_embeddings()
        ts_dim = self.embeddings['ts'].shape[1] if 'ts' in self.embeddings else 64
        self.model = self._load_model(ts_dim=ts_dim)
        
        self.df = pd.read_csv(self.dataset_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.sort_values(['ticker', 'date'], inplace=True)
        
        # Extract unique tickers for backtesting
        self.tickers = self.df['ticker'].unique().tolist()
        if not self.tickers:
            self.logger.warning("No tickers found in dataset.")
            
        self.scaler = self._load_scaler()

    def run(self) -> Dict:
        """Standard high-level entry point used by run_full_pipeline."""
        if self.embeddings and 'meta' in self.embeddings:
            min_dt = self.embeddings['meta']['date'].min().strftime('%Y-%m-%d')
            max_dt = self.embeddings['meta']['date'].max().strftime('%Y-%m-%d')
        else:
            min_dt, max_dt = "2025-01-01", "2025-12-31"
            
        res_df = self.run_backtest(start_date=min_dt, end_date=max_dt)
        return self.compute_metrics(res_df)

    def _load_model(self, ts_dim: int) -> nn.Module:
        """Loads the fusion model weights."""
        nlp_dim = 768
        
        model = MultiHorizonFusionModel(nlp_dim=nlp_dim, ts_dim=ts_dim)
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            model.to(self.device).eval()
            self.logger.info(f"Successfully loaded model from {self.model_path}")
        else:
            self.logger.warning(f"Model path {self.model_path} not found. Using randomly initialized model.")
        return model

    def _load_scaler(self):
        """Loads the target value scaler."""
        if os.path.exists(self.scaler_path):
            try:
                # Scaler is saved with joblib.dump in build_sequences.py
                return joblib.load(self.scaler_path)
            except Exception:
                # Fallback to standard pickle for legacy files
                try:
                    with open(self.scaler_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load scaler from {self.scaler_path}: {e}")
                    return None
        self.logger.warning(f"Scaler not found at {self.scaler_path}. Results will be in normalized units.")
        return None

    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """Loads NLP and TS embeddings along with their date mapping."""
        features_dir = os.path.join(PROJECT_ROOT, 'data', 'features')
        nlp_emb_path = os.path.join(features_dir, 'nlp_embeddings.npy')
        ts_emb_path = os.path.join(features_dir, 'ts_embeddings.npy')
        nlp_meta_path = os.path.join(features_dir, 'nlp_embedding_dates.csv')

        embs = {}
        if os.path.exists(nlp_emb_path) and os.path.exists(ts_emb_path) and os.path.exists(nlp_meta_path):
            embs['nlp'] = np.load(nlp_emb_path)
            embs['ts'] = np.load(ts_emb_path)
            embs['meta'] = pd.read_csv(nlp_meta_path)
            embs['meta']['date'] = pd.to_datetime(embs['meta']['date'])
            self.logger.info("Loaded embeddings and temporal metadata.")
        else:
            self.logger.error("❌ CRITICAL: Embeddings or metadata missing.")
            self.logger.error("Backtest requires REAL embeddings from Phase 6. Cannot proceed.")
            raise FileNotFoundError("Missing real embeddings for Backtesting. Mock fallback disabled.")
        return embs

    def run_backtest(self, start_date: str, end_date: str, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Runs a walk-forward backtest.
        For each date in range, generate forecasts for 1d, 5d, 30d.
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        test_df = self.df[(self.df['date'] >= start_dt) & (self.df['date'] <= end_dt)]
        if tickers:
            test_df = test_df[test_df['ticker'].isin(tickers)]
            
        results = []
        
        self.logger.info(f"Starting backtest from {start_date} to {end_date} for {len(test_df['ticker'].unique())} tickers.")
        
        # Mocking inference for now since we need embeddings and features
        # In actual implementation, we'd call the inference pipeline or model.predict()

        for ticker in tqdm(self.tickers, desc="Backtesting tickers"):
            ticker_data = test_df[test_df['ticker'] == ticker].sort_values('date')
            if ticker_data.empty:
                continue

            dates = ticker_data['date'].values
            # indices = ticker_data.index.values # Not used in the new logic

            # Collect all valid prediction points for this ticker at once
            batch_nlp = []
            batch_ts = []
            batch_dates = []
            # batch_indices = [] # Not used in the new logic

            for dt in dates:
                if not self.embeddings:
                    break
                m = self.embeddings['meta']
                match = m[(m['date'] == dt) & (m['ticker'] == ticker)]
                if len(match) == 0:
                    continue
                midx = match.index[0]  # type: ignore[attr-defined]
                batch_nlp.append(self.embeddings['nlp'][midx])
                batch_ts.append(self.embeddings['ts'][midx])
                batch_dates.append(dt)
                # batch_indices.append(idx) # Not used in the new logic

            if not batch_nlp:
                continue

            # Vectorized batch prediction
            nlp_batch = torch.zeros((len(batch_nlp), 60, 768), device=self.device)
            for i, nlp_vec in enumerate(batch_nlp):
                nlp_batch[i, :, :] = torch.from_numpy(nlp_vec).to(self.device)

            ts_batch = torch.from_numpy(np.array(batch_ts)).to(self.device)

            with torch.no_grad():
                preds = self.model(nlp_batch, ts_batch)
                p1d = preds['1d'].cpu().numpy().flatten()
                p5d = preds['5d'].cpu().numpy().flatten()
                p30d = preds['30d'].cpu().numpy().flatten()

            # Unscale all at once
            if self.scaler:
                p1d = self.scaler.inverse_transform(p1d.reshape(-1, 1)).flatten()
                p5d = self.scaler.inverse_transform(p5d.reshape(-1, 1)).flatten()
                p30d = self.scaler.inverse_transform(p30d.reshape(-1, 1)).flatten()

            # Collect results
            for i, dt in enumerate(batch_dates):
                actual_1d = self._get_future_price(ticker, dt, 1)
                actual_5d = self._get_future_price(ticker, dt, 5)
                actual_30d = self._get_future_price(ticker, dt, 30)

                results.append({
                    'date': dt,
                    'ticker': ticker,
                    'pred_1d': float(p1d[i]),
                    'actual_1d': actual_1d,
                    'pred_5d': float(p5d[i]),
                    'actual_5d': actual_5d,
                    'pred_30d': float(p30d[i]),
                    'actual_30d': actual_30d
                })

        return pd.DataFrame(results)

    def _get_future_price(self, ticker: str, current_date: pd.Timestamp, horizon_days: int) -> float:
        """Helper to get actual future price for metrics calculation."""
        # Find index of current date
        sub = self.df[self.df['ticker'] == ticker]
        current_idx = sub[sub['date'] == current_date].index
        if len(current_idx) == 0:
            return np.nan
        
        # Get the actual row for the current date within the filtered sub-dataframe
        current_row_in_sub = sub[sub['date'] == current_date]
        if current_row_in_sub.empty:
            return np.nan
        
        # Get the position of the current date within the sorted ticker_data
        current_pos = sub.index.get_loc(current_row_in_sub.index[0])
        
        target_pos = current_pos + horizon_days
        
        if target_pos < len(sub):
            return sub.iloc[target_pos]['Close']
        return np.nan

    def compute_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Enhanced performance metrics with RMSE, directional accuracy, and R²."""
        if results_df.empty or 'actual_1d' not in results_df.columns:
            self.logger.warning("No valid predictions generated in this date range. (DataFrame is empty)")
            return {}
            
        metrics = {}
        for h in ['1d', '5d', '30d']:
            valid = results_df.dropna(subset=[f'actual_{h}'])
            if len(valid) == 0:
                continue
            
            pred = valid[f'pred_{h}'].values
            actual = valid[f'actual_{h}'].values
            errors = pred - actual
            
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors ** 2))
            
            # R²
            ss_res = np.sum(errors ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Directional accuracy
            if len(pred) > 1:
                # Calculate direction based on change from previous prediction/actual
                # This assumes a time series of predictions/actuals.
                # For a single prediction point, we can't determine direction.
                # A more robust directional accuracy would compare pred_t+h vs actual_t,
                # and actual_t+h vs actual_t.
                # For simplicity, let's compare the direction of predicted change vs actual change.
                # This requires having the previous actual price.
                # For now, let's use a simplified version:
                # If pred > actual, is actual > previous actual?
                # This is tricky without the previous actual in the results_df.
                # A common approach is to compare the sign of (pred - actual) with the sign of (actual_future - actual_current).
                # Since actual_current is not in results_df, we'll use a simpler definition for now:
                # If pred_t+h > actual_t, and actual_t+h > actual_t.
                # For now, let's use a placeholder that compares the sign of the prediction with the sign of the actual.
                # This is not ideal for price prediction, but for return prediction it makes more sense.
                # Let's assume the predictions are for price levels, and we want to know if the predicted direction of change
                # from the current price matches the actual direction of change from the current price.
                # This would require the current price in the results_df, which is not there.
                # For now, let's use a very basic directional accuracy:
                # If pred > actual, and actual > 0 (or some baseline). This is not standard.
                # Let's use the provided diff's directional accuracy which compares diff(pred) and diff(actual).
                # This implies the predictions and actuals are ordered by time.
                pred_dir = np.diff(pred) > 0
                actual_dir = np.diff(actual) > 0
                dir_acc = np.mean(pred_dir == actual_dir)
            else:
                dir_acc = 0.5 # Default for insufficient data
                
            metrics[f'{h}'] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'directional_accuracy': float(dir_acc),
                'n_predictions': int(len(valid)),
            }
            
        return metrics

if __name__ == "__main__":
    # Test stub
    engine = BacktestEngine(
        model_path="saved_models/fusion_model.pt",
        dataset_path="data/processed/merged/all_merged_dataset.csv",
        metadata_path="data/processed/model_inputs/metadata_test.csv"
    )
    
    if engine.embeddings and 'meta' in engine.embeddings:
        min_dt = engine.embeddings['meta']['date'].min().strftime('%Y-%m-%d')
        max_dt = engine.embeddings['meta']['date'].max().strftime('%Y-%m-%d')
    else:
        min_dt, max_dt = "2025-07-01", "2025-07-10"
        
    res = engine.run_backtest(start_date=min_dt, end_date=max_dt)
    print(res.head())
    metrics = engine.compute_metrics(res)
    print(metrics)
    
    import json
    import os
    os.makedirs("saved_models", exist_ok=True)
    with open("saved_models/backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
