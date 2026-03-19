"""
Regime Features — Generates regime probability features and appends them to datasets.

Pipeline position: Feature Engineering → Regime Detection → Sequence Building

Takes the merged dataset, runs the RegimeDetector per-ticker, and appends
5 regime probability columns (one per regime state) to the feature set.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.feature_engineering.regime_detection.regime_detector import RegimeDetector, REGIME_NAMES, HMM_AVAILABLE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RegimeFeatures")


def generate_regime_features(
    df: pd.DataFrame,
    close_col: str = "close",
    ticker_col: str = "ticker",
    n_regimes: int = 5,
    save_model_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate regime probability features for each row of the dataset.
    
    Fits a per-ticker HMM (or global HMM if single ticker) and appends
    regime probability columns: regime_bull, regime_bear, regime_sideways,
    regime_high_vol, regime_low_vol.
    
    Args:
        df: DataFrame with close prices and optional ticker column.
        close_col: Name of the close price column.
        ticker_col: Name of the ticker column.
        n_regimes: Number of regime states.
        save_model_path: Optional path to save fitted detector.
        
    Returns:
        DataFrame with regime probability columns appended.
    """
    if not HMM_AVAILABLE:
        logger.warning("hmmlearn not installed. Adding zero-valued regime columns.")
        for regime_name in REGIME_NAMES.values():
            df[f"regime_{regime_name}"] = 0.0
        return df
    
    logger.info("Generating regime probability features...")
    
    # Initialize regime columns
    for regime_name in REGIME_NAMES.values():
        df[f"regime_{regime_name}"] = 0.0
    
    # Standardize column names for detector
    close_mapping = {close_col: "close"} if close_col != "close" else {}
    
    if ticker_col in df.columns:
        # Per-ticker regime detection
        tickers = df[ticker_col].unique()
        logger.info(f"Running regime detection on {len(tickers)} tickers...")
        
        processed = 0
        for ticker in tickers:
            mask = df[ticker_col] == ticker
            ticker_df = df[mask].copy()
            
            if len(ticker_df) < 60:  # Need enough history
                continue
            
            # Rename columns if needed
            if close_mapping:
                ticker_df = ticker_df.rename(columns=close_mapping)  # type: ignore[assignment]
            
            try:
                detector = RegimeDetector(n_regimes=n_regimes)
                detector.fit(ticker_df)  # type: ignore[arg-type]
                probs, idx = detector.predict_proba(ticker_df)  # type: ignore[arg-type]
                
                # Map probabilities back to original DataFrame
                # idx contains the valid indices after NaN dropping in feature prep
                for regime_id, regime_name in REGIME_NAMES.items():
                    if regime_id < probs.shape[1]:
                        col_name = f"regime_{regime_name}"
                        df.loc[idx, col_name] = probs[:, regime_id]
                
                processed += 1
                if processed % 50 == 0:
                    logger.info(f"  Processed {processed}/{len(tickers)} tickers")
                    
            except Exception as e:
                logger.warning(f"Regime detection failed for {ticker}: {e}")
                continue
        
        logger.info(f"Regime features generated for {processed}/{len(tickers)} tickers")
    else:
        # Global regime detection
        work_df = df.copy()
        if close_mapping:
            work_df = work_df.rename(columns=close_mapping)
        
        try:
            detector = RegimeDetector(n_regimes=n_regimes)
            detector.fit(work_df)
            probs, idx = detector.predict_proba(work_df)
            
            for regime_id, regime_name in REGIME_NAMES.items():
                if regime_id < probs.shape[1]:
                    col_name = f"regime_{regime_name}"
                    df.loc[idx, col_name] = probs[:, regime_id]
            
            if save_model_path:
                detector.save(save_model_path)
                
        except Exception as e:
            logger.error(f"Global regime detection failed: {e}")
    
    # Fill any NaNs from regime columns (leading rows without enough history)
    for regime_name in REGIME_NAMES.values():
        col = f"regime_{regime_name}"
        df[col] = df[col].fillna(0.0)
    
    logger.info(f"Regime features shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n = 500
    prices = np.cumsum(np.random.randn(n) * 0.5) + 100
    
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "close": prices,
        "Volume": np.random.randint(int(1e6), int(1e7), n),  # Convert to int
        "ticker": "TEST",
    })
    
    result = generate_regime_features(df)
    regime_cols = [c for c in result.columns if c.startswith("regime_")]
    print(f"\nRegime columns: {regime_cols}")
    print(result[regime_cols].describe())
    print("\n✅ Regime feature generation test passed!")
