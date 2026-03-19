"""
Market Regime Detector — HMM-Based Market State Classification.

Classifies market conditions into discrete regimes:
    0: Bull Market (strong uptrend, low volatility)
    1: Bear Market (strong downtrend, elevated volatility)
    2: Sideways (range-bound, moderate volatility)
    3: High Volatility (extreme uncertainty, large swings)
    4: Low Volatility (quiet market, small moves)

Uses Gaussian HMM on engineered features: returns, volatility, volume changes.
Outputs regime probabilities as additional features for downstream models.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Tuple, List, Any

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RegimeDetector")

REGIME_NAMES = {
    0: "bull_market",
    1: "bear_market",
    2: "sideways",
    3: "high_volatility",
    4: "low_volatility",
}

N_REGIMES = 5


class RegimeDetector:
    """
    HMM-based market regime detector.
    
    Fits a Gaussian HMM on market features (returns, volatility, volume changes)
    and outputs regime probabilities for each time step.
    
    Usage:
        detector = RegimeDetector()
        detector.fit(market_df)
        probs = detector.predict_proba(market_df)  # [N, 5] array of regime probs
    """

    def __init__(
        self,
        n_regimes: int = N_REGIMES,
        n_iter: int = 100,
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.feature_columns: List[str] = []

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
        """
        Extract regime-relevant features from market data.
        
        Features used:
            - log returns (momentum signal)
            - rolling 20-day volatility (risk signal)
            - rolling 20-day volume change ratio (liquidity signal)
            - rolling 5-day return (short-term momentum)
        """
        features = pd.DataFrame(index=df.index)
        
        # Close price returns
        if 'close' in df.columns:
            close_col = 'close'
        elif 'Close' in df.columns:
            close_col = 'Close'
        else:
            raise ValueError("DataFrame must contain 'close' or 'Close' column")

        close = df[close_col].astype(float)
        
        # Log returns
        features['log_return'] = np.log(close / close.shift(1))
        
        # Rolling volatility (20-day std of log returns)
        features['volatility_20d'] = features['log_return'].rolling(20).std()
        
        # Short-term momentum (5-day return)
        features['momentum_5d'] = close.pct_change(5)
        
        # Volume change ratio
        if 'Volume' in df.columns or 'volume' in df.columns:
            vol_col = 'Volume' if 'Volume' in df.columns else 'volume'
            volume = df[vol_col].astype(float)
            features['volume_ratio'] = volume / volume.rolling(20).mean()
        
        # Drop NaN rows
        features = features.dropna()
        
        self.feature_columns = list(features.columns)
        return features.values, features.index

    def fit(self, df: pd.DataFrame) -> 'RegimeDetector':
        """
        Fit the HMM on historical market data.
        
        Args:
            df: DataFrame with at least 'close'/'Close' column.
            
        Returns:
            Self for chaining.
        """
        if not HMM_AVAILABLE:
            logger.error("hmmlearn not installed. Install with: pip install hmmlearn")
            return self

        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn is required. Install with: pip install hmmlearn")

        from hmmlearn.hmm import GaussianHMM

        X, _ = self._prepare_features(df)
        
        logger.info(f"Fitting HMM with {self.n_regimes} regimes on {len(X)} samples...")
        
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(X)
        
        # Label regimes by their characteristics
        self._label_regimes(X)
        
        logger.info(f"HMM fit complete. Log-likelihood: {self.model.score(X):.2f}")
        return self

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
        """
        Predict regime labels for each time step.
        
        Returns:
            Tuple of (regime_labels [N], index).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X, idx = self._prepare_features(df)
        labels = self.model.predict(X)
        
        # Remap to canonical ordering
        if hasattr(self, 'regime_mapping'):
            labels = np.array([self.regime_mapping.get(l, l) for l in labels])
        
        return labels, idx

    def predict_proba(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
        """
        Get regime probability distribution for each time step.
        
        Returns:
            Tuple of (probabilities [N, n_regimes], index).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X, idx = self._prepare_features(df)
        proba = self.model.predict_proba(X)
        
        # Reorder columns to match canonical regime ordering
        if hasattr(self, 'regime_mapping'):
            new_proba = np.zeros_like(proba)
            for old_idx, new_idx in self.regime_mapping.items():
                new_proba[:, new_idx] = proba[:, old_idx]
            proba = new_proba
        
        return proba, idx

    def _label_regimes(self, X: np.ndarray):
        """
        Auto-label regimes based on their mean characteristics.
        Maps HMM states to canonical regime names by analyzing:
        - Mean return (bull vs bear)
        - Mean volatility (high vol vs low vol)
        """
        if self.model is None:
            return
            
        labels = self.model.predict(X)  # type: ignore
        means = self.model.means_  # type: ignore  # [n_regimes, n_features]
        
        # Feature 0: log_return, Feature 1: volatility_20d
        return_means = means[:, 0]
        vol_means = means[:, 1]
        
        # Sort regimes by return and volatility characteristics
        regime_chars = []
        for i in range(self.n_regimes):
            regime_chars.append({
                'hmm_state': i,
                'mean_return': return_means[i],
                'mean_vol': vol_means[i],
                'count': np.sum(labels == i),
            })
        
        # Assign canonical labels based on characteristics
        sorted_by_return = sorted(regime_chars, key=lambda x: x['mean_return'])
        sorted_by_vol = sorted(regime_chars, key=lambda x: x['mean_vol'])
        
        self.regime_mapping = {}
        assigned = set()
        
        # Highest return → bull (0)
        bull_state = sorted_by_return[-1]['hmm_state']
        self.regime_mapping[bull_state] = 0
        assigned.add(bull_state)
        
        # Lowest return → bear (1)
        bear_state = sorted_by_return[0]['hmm_state']
        if bear_state not in assigned:
            self.regime_mapping[bear_state] = 1
            assigned.add(bear_state)
        
        # Highest volatility → high_vol (3)
        for rc in sorted_by_vol[::-1]:
            if rc['hmm_state'] not in assigned:
                self.regime_mapping[rc['hmm_state']] = 3
                assigned.add(rc['hmm_state'])
                break
        
        # Lowest volatility → low_vol (4)
        for rc in sorted_by_vol:
            if rc['hmm_state'] not in assigned:
                self.regime_mapping[rc['hmm_state']] = 4
                assigned.add(rc['hmm_state'])
                break
        
        # Remaining → sideways (2)
        for i in range(self.n_regimes):
            if i not in assigned:
                self.regime_mapping[i] = 2
                assigned.add(i)
        
        logger.info(f"Regime mapping: {self.regime_mapping}")
        for i, rc in enumerate(regime_chars):
            mapped = self.regime_mapping[rc['hmm_state']]
            logger.info(
                f"  HMM State {rc['hmm_state']} → {REGIME_NAMES[mapped]} "
                f"(return={rc['mean_return']:.4f}, vol={rc['mean_vol']:.4f}, "
                f"count={rc['count']})"
            )

    def save(self, path: str):
        """Save the fitted model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'regime_mapping': getattr(self, 'regime_mapping', {}),
            'feature_columns': self.feature_columns,
            'n_regimes': self.n_regimes,
        }, path)
        logger.info(f"Regime detector saved to {path}")

    def load(self, path: str) -> 'RegimeDetector':
        """Load a fitted model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.regime_mapping = data.get('regime_mapping', {})
        self.feature_columns = data.get('feature_columns', [])
        self.n_regimes = data.get('n_regimes', N_REGIMES)
        logger.info(f"Regime detector loaded from {path}")
        return self


if __name__ == "__main__":
    # Quick verification with synthetic data
    np.random.seed(42)
    n = 1000
    
    # Simulate multi-regime price series
    prices = [100.0]
    for i in range(n - 1):
        if i < 300:
            ret = np.random.normal(0.001, 0.01)  # Bull
        elif i < 500:
            ret = np.random.normal(-0.002, 0.025)  # Bear
        elif i < 700:
            ret = np.random.normal(0.0, 0.005)  # Sideways
        else:
            ret = np.random.normal(0.0005, 0.03)  # High vol
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'close': prices,
        'Volume': np.random.randint(int(1e6), int(1e7), n + 1).astype(float) if len(prices) == n + 1 else np.random.randint(int(1e6), int(1e7), len(prices)).astype(float),
    })
    
    if HMM_AVAILABLE:
        detector = RegimeDetector(n_regimes=5)
        detector.fit(df)
        
        probs, idx = detector.predict_proba(df)
        labels, _ = detector.predict(df)
        
        print(f"\nRegime probabilities shape: {probs.shape}")
        print(f"Regime label distribution:")
        for regime_id, regime_name in REGIME_NAMES.items():
            count = np.sum(labels == regime_id)
            print(f"  {regime_name}: {count} ({count/len(labels):.1%})")
        
        print("\n✅ Regime detection smoke test passed!")
    else:
        print("⚠️ hmmlearn not installed. Skipping smoke test.")
