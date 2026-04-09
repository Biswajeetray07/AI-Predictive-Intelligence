"""
Real-Time Feature Builder for AI Predictive Intelligence.

Constructs model-ready [seq_len, n_features] tensors from processed stock data
and technical indicators — completely independent of training split artifacts
(X_test.npy, metadata_test.csv).

This enables TRUE real-time inference: given a ticker (and optionally a date),
build full feature sequences on the fly using the same feature engineering
pipeline and scaler that was used during training.

Usage:
    builder = RealTimeFeatureBuilder()
    sequence = builder.build_sequence("AAPL")           # latest 60 days
    sequence = builder.build_sequence("AAPL", "2025-12-01")  # as of date
"""

import os
import sys
import logging
import joblib
import pickle
import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# Default sequence length (must match training pipeline)
SEQ_LEN = 60

# Columns known to cause data leakage — must be excluded from features
# (mirrors build_sequences.py exclusions exactly)
LEAKY_COLS = ['BB_Upper', 'BB_Lower', 'SMA_10', 'SMA_50', 'EMA_12', 'EMA_26']
EXCLUDE_COLS = ['date', 'ticker', 'Close'] + LEAKY_COLS


class RealTimeFeatureBuilder:
    """
    Constructs model-ready feature tensors from processed/merged data.

    The builder:
    1. Loads the merged dataset (all_merged_dataset.csv) or individual stock parquet files
    2. Applies the same feature column selection as the training pipeline
    3. Applies the saved StandardScaler (feature_scaler.pkl) from training
    4. Clips Z-scores at [-10, 10] for stability
    5. Constructs a [seq_len, n_features] sliding window tensor
    """

    def __init__(self, project_root: str = None, cache_ttl: int = 300):
        """
        Args:
            project_root: Path to the project root directory.
            cache_ttl: Time-to-live in seconds for cached feature sequences (default: 5 min).
                       Set to 0 to disable caching.
        """
        self.project_root = project_root or PROJECT_ROOT
        self._feature_scaler = None
        self._feature_cols = None
        self._merged_df = None
        self._cache_ttl = cache_ttl
        self._sequence_cache: Dict[str, Tuple[float, np.ndarray]] = {}  # key -> (timestamp, array)

    @property
    def feature_scaler(self):
        """Lazy-load the feature scaler saved during training."""
        if self._feature_scaler is None:
            scaler_path = os.path.join(
                self.project_root, 'data', 'processed', 'model_inputs', 'feature_scaler.pkl'
            )
            if not os.path.exists(scaler_path):
                # Try S3 fallback
                scaler = self._load_scaler_from_s3()
                if scaler is not None:
                    self._feature_scaler = scaler
                    return self._feature_scaler
                raise FileNotFoundError(
                    f"Feature scaler not found at {scaler_path}. "
                    f"Run the training pipeline (build_sequences.py) first."
                )
            try:
                self._feature_scaler = joblib.load(scaler_path)
            except Exception:
                with open(scaler_path, 'rb') as f:
                    self._feature_scaler = pickle.load(f)
            logger.info(f"Loaded feature scaler with {self._feature_scaler.n_features_in_} features")
        return self._feature_scaler

    def _load_scaler_from_s3(self):
        """Try loading the feature scaler from S3."""
        try:
            from src.cloud_storage.aws_storage import SimpleStorageService
            use_s3 = os.getenv("USE_S3", "False").lower() in ("true", "1", "yes")
            if not use_s3:
                return None
            s3_bucket = os.getenv("MODEL_BUCKET_NAME", "my-model-mlopsproj012")
            s3 = SimpleStorageService()
            model_obj = s3.load_model(
                'feature_scaler.pkl',
                s3_bucket,
                model_dir='data/processed/model_inputs'
            )
            return model_obj
        except Exception as e:
            logger.warning(f"S3 scaler load failed: {e}")
            return None

    @property
    def feature_cols(self):
        """
        Determine feature columns from the scaler's feature names.
        Falls back to inferring from merged data if the scaler lacks feature_names_in_.
        """
        if self._feature_cols is None:
            scaler = self.feature_scaler
            if hasattr(scaler, 'feature_names_in_'):
                self._feature_cols = list(scaler.feature_names_in_)
            else:
                # Fallback: infer from a sample of merged data
                df = self._load_merged_data()
                if df is not None:
                    self._feature_cols = [
                        c for c in df.columns
                        if c not in EXCLUDE_COLS
                        and df[c].dtype != 'object'
                    ]
                else:
                    raise RuntimeError("Cannot determine feature columns: no scaler names and no merged data.")
        return self._feature_cols

    def _load_merged_data(self, ticker: str = None) -> Optional[pd.DataFrame]:
        """
        Load source data for feature construction.

        Priority:
        1. Individual processed stock parquet (faster, per-ticker)
        2. Merged dataset CSV (all tickers, may be large)
        """
        # Priority 1: Individual processed parquet per ticker (local)
        if ticker:
            parquet_path = os.path.join(
                self.project_root, 'data', 'processed', 'financial', 'stocks', f'{ticker}.parquet'
            )
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                if 'ticker' not in df.columns:
                    df['ticker'] = ticker
                return df

        # Priority 2: Full merged dataset (local)
        merged_path = os.path.join(
            self.project_root, 'data', 'processed', 'merged', 'all_merged_dataset.csv'
        )
        if os.path.exists(merged_path):
            if self._merged_df is None:
                logger.info("Loading merged dataset for feature construction...")
                try:
                    self._merged_df = pd.read_csv(
                        merged_path, on_bad_lines='skip', engine='python'
                    )
                    self._merged_df['date'] = pd.to_datetime(self._merged_df['date'], errors='coerce')
                    self._merged_df.sort_values(by=['ticker', 'date'], inplace=True)
                except Exception as e:
                    logger.error(f"Failed to load merged dataset: {e}")
                    return None
            if ticker:
                return self._merged_df[self._merged_df['ticker'] == ticker].copy()
            return self._merged_df

        # Priority 3: S3 Parquet fetch (Cloud / Streamlit App mode)
        use_s3_env = os.getenv("USE_S3", "False").lower() in ("true", "1", "yes")
        try:
            import streamlit as st
            use_s3_st = st.secrets.get("USE_S3", use_s3_env)
            use_s3 = str(use_s3_st).lower() in ("true", "1", "yes")
            bucket = st.secrets.get("MODEL_BUCKET_NAME", os.getenv("MODEL_BUCKET_NAME", "my-model-mlopsproj012"))
        except Exception:
            use_s3 = use_s3_env
            bucket = os.getenv("MODEL_BUCKET_NAME", "my-model-mlopsproj012")

        if use_s3 and ticker:
            try:
                from src.cloud_storage.aws_storage import SimpleStorageService
                s3 = SimpleStorageService()
                s3_key = f'data/processed/financial/stocks/{ticker}.parquet'
                logger.info(f"Loading {ticker} features from S3...")
                df = s3.read_parquet(s3_key, bucket)
                if df is not None:
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    if 'ticker' not in df.columns:
                        df['ticker'] = ticker
                    return df
            except Exception as e:
                logger.warning(f"Failed to load S3 parquet for {ticker}: {e}")

        return None

    def build_sequence(
        self,
        ticker: str,
        as_of_date: str = None,
        seq_len: int = SEQ_LEN
    ) -> Optional[np.ndarray]:
        """
        Build a model-ready [seq_len, n_features] tensor for a given ticker.

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL')
            as_of_date: Optional target date (YYYY-MM-DD). Uses latest data if None.
            seq_len: Sequence length (default 60, must match training config).

        Returns:
            np.ndarray of shape [seq_len, n_features], or None if insufficient data.
        """
        t0 = time.perf_counter()

        # Check TTL cache
        cache_key = f"{ticker}:{as_of_date or 'latest'}:{seq_len}"
        if self._cache_ttl > 0 and cache_key in self._sequence_cache:
            cached_ts, cached_arr = self._sequence_cache[cache_key]
            if (time.time() - cached_ts) < self._cache_ttl:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.debug(f"Cache hit for {cache_key} ({elapsed:.1f}ms)")
                return cached_arr
            else:
                del self._sequence_cache[cache_key]  # expired

        try:
            df = self._load_merged_data(ticker)
            if df is None or df.empty:
                logger.warning(f"No data available for ticker {ticker}")
                return None

            # Filter to ticker if merged data
            if 'ticker' in df.columns:
                df = df[df['ticker'] == ticker].copy()
                if df.empty:
                    logger.warning(f"Ticker {ticker} not found in data")
                    return None

            # Sort by date
            if 'date' in df.columns:
                df = df.sort_values('date')
                if as_of_date:
                    as_of = pd.to_datetime(as_of_date)
                    df = df[df['date'] <= as_of]

            if len(df) < seq_len:
                logger.warning(
                    f"Insufficient data for {ticker}: need {seq_len} rows, have {len(df)}"
                )
                return None

            # Select features — only columns that exist in the data AND in the scaler
            available_cols = [c for c in self.feature_cols if c in df.columns]
            if not available_cols:
                logger.error(f"No feature columns found for {ticker}. "
                             f"Expected columns like: {self.feature_cols[:5]}")
                return None

            # Forward-fill NaN (mirrors build_sequences.py preprocessing)
            df[available_cols] = df[available_cols].ffill().fillna(0)

            # Extract the last seq_len rows
            feature_values = df[available_cols].iloc[-seq_len:].values.astype(np.float32)

            # If some training columns are missing, pad with zeros
            if len(available_cols) < len(self.feature_cols):
                padded = np.zeros((seq_len, len(self.feature_cols)), dtype=np.float32)
                col_indices = [self.feature_cols.index(c) for c in available_cols]
                padded[:, col_indices] = feature_values
                feature_values = padded

            # Apply the saved StandardScaler
            original_shape = feature_values.shape
            feature_values = self.feature_scaler.transform(feature_values)

            # Clip Z-scores to [-10, 10] for stability (mirrors build_sequences.py)
            feature_values = np.clip(feature_values, -10, 10)

            result = feature_values.astype(np.float32)

            # Store in cache
            if self._cache_ttl > 0:
                self._sequence_cache[cache_key] = (time.time(), result)
                # Evict expired entries periodically (keep cache bounded)
                if len(self._sequence_cache) > 100:
                    now = time.time()
                    self._sequence_cache = {
                        k: v for k, v in self._sequence_cache.items()
                        if (now - v[0]) < self._cache_ttl
                    }

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"Built feature sequence for {ticker}: {result.shape} in {elapsed:.0f}ms")
            return result

        except FileNotFoundError:
            # Scaler not found — can't build features
            logger.warning(f"Feature scaler not available. Cannot build live features for {ticker}.")
            return None
        except Exception as e:
            logger.error(f"Failed to build feature sequence for {ticker}: {e}", exc_info=True)
            return None

    def clear_cache(self):
        """Clear the sequence cache."""
        self._sequence_cache.clear()
        logger.info("Feature sequence cache cleared")


def main():
    """CLI test for the feature builder."""
    import argparse
    parser = argparse.ArgumentParser(description="Build feature sequences for inference")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker")
    parser.add_argument("--date", default=None, help="As-of date (YYYY-MM-DD)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    builder = RealTimeFeatureBuilder()
    seq = builder.build_sequence(args.ticker, as_of_date=args.date)
    if seq is not None:
        print(f"\n✅ Built feature sequence for {args.ticker}")
        print(f"   Shape: {seq.shape}")
        print(f"   Dtype: {seq.dtype}")
        print(f"   Value range: [{seq.min():.4f}, {seq.max():.4f}]")
        print(f"   Mean: {seq.mean():.4f}, Std: {seq.std():.4f}")
    else:
        print(f"\n❌ Failed to build feature sequence for {args.ticker}")


if __name__ == "__main__":
    main()
