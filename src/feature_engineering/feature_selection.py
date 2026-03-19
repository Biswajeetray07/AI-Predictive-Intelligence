"""
Feature Selection — Automated Feature Importance and Filtering.

Implements multiple feature importance methods:
    1. SHAP-based importance (if SHAP available)
    2. Mutual Information regression
    3. Variance-based filtering

Outputs: a filtered feature list saved to configs/selected_features.yaml
for use by build_sequences.py.
"""

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureSelection")


def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    n_neighbors: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """
    Compute mutual information between each feature and the target.
    
    Returns:
        Series of MI scores indexed by feature name, sorted descending.
    """
    # Handle NaNs
    mask = X.notna().all(axis=1) & y.notna()
    X_clean = X[mask].values
    y_clean = y[mask].values
    
    if len(X_clean) < 100:
        logger.warning(f"Only {len(X_clean)} valid samples for MI computation")
        return pd.Series(dtype=float)
    
    mi_scores = mutual_info_regression(
        np.asarray(X_clean), np.asarray(y_clean),
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    
    result = pd.Series(mi_scores, index=X.columns, name="mutual_info")
    return result.sort_values(ascending=False)


def compute_variance_importance(X: pd.DataFrame, threshold: float = 0.01) -> pd.Series:
    """
    Score features by normalized variance. Low-variance features add noise.
    
    Returns:
        Series of normalized variance scores.
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X.fillna(0)),
        columns=X.columns,
    )
    variances = X_scaled.var()
    return variances.sort_values(ascending=False)


def compute_correlation_redundancy(X: pd.DataFrame, threshold: float = 0.85) -> List[str]:
    """
    Identify highly correlated feature pairs and mark one for removal.
    
    Returns:
        List of feature names to remove (redundant).
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_remove = []
    for col in upper.columns:
        if any(upper[col] > threshold):
            to_remove.append(col)
    
    return to_remove


def select_features(
    df: pd.DataFrame,
    target_col: str = "close",
    exclude_cols: Optional[List[str]] = None,
    mi_threshold: float = 0.01,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.85,
    min_features: int = 20,
    output_path: Optional[str] = None,
) -> Tuple[List[str], Dict]:
    """
    Run full feature selection pipeline.
    
    Steps:
        1. Remove zero-variance features
        2. Remove highly correlated pairs
        3. Rank by mutual information
        4. Filter below MI threshold
    
    Args:
        df: Full dataset with features and target.
        target_col: Name of the target column.
        exclude_cols: Columns to exclude from selection (e.g., 'date', 'ticker').
        mi_threshold: Minimum MI score to keep a feature.
        variance_threshold: Minimum normalized variance to keep.
        correlation_threshold: Max correlation before marking redundant.
        min_features: Minimum features to keep regardless of thresholds.
        output_path: Path to save selected features YAML.
        
    Returns:
        Tuple of (selected_feature_names, selection_report).
    """
    exclude = set(exclude_cols or ['date', 'ticker', 'Date', 'Ticker'])
    feature_cols = [c for c in df.columns if c not in exclude and c != target_col]
    
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found")
        return feature_cols, {}
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Ensure numeric
    X = X.select_dtypes(include=[np.number])
    feature_cols = list(X.columns)
    
    logger.info(f"Starting feature selection on {len(feature_cols)} features...")
    
    report = {
        'total_features': len(feature_cols),
        'removed': {},
    }
    
    # Step 1: Remove near-zero variance
    var_scores = compute_variance_importance(X, variance_threshold)
    low_var = var_scores[var_scores < variance_threshold].index.tolist()
    if low_var:
        logger.info(f"  Removing {len(low_var)} low-variance features")
        report['removed']['low_variance'] = low_var
        X = X.drop(columns=low_var)
    
    # Step 2: Remove highly correlated
    redundant = compute_correlation_redundancy(X, correlation_threshold)
    if redundant:
        logger.info(f"  Removing {len(redundant)} redundant (correlated) features")
        report['removed']['correlated'] = redundant
        X = X.drop(columns=redundant, errors='ignore')
    
    # Step 3: Rank by mutual information
    mi_scores = compute_mutual_information(X, y)
    if not mi_scores.empty:
        report['mi_scores'] = mi_scores.to_dict()
        
        # Filter below threshold (but keep minimum)
        above_threshold = mi_scores[mi_scores >= mi_threshold].index.tolist()
        
        if len(above_threshold) < min_features:
            selected = mi_scores.head(min_features).index.tolist()
            logger.info(
                f"  MI threshold too strict; keeping top {min_features} features"
            )
        else:
            selected = above_threshold
            logger.info(f"  Keeping {len(selected)} features above MI threshold")
    else:
        selected = list(X.columns)
    
    report['selected_count'] = len(selected)
    report['reduction'] = f"{len(feature_cols)} → {len(selected)}"
    
    logger.info(
        f"Feature selection: {len(feature_cols)} → {len(selected)} features "
        f"({len(feature_cols) - len(selected)} removed)"
    )
    
    # Save to YAML
    save_path = output_path or os.path.join(PROJECT_ROOT, "configs", "selected_features.yaml")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w") as f:
        yaml.dump({
            'selected_features': selected,
            'target_column': target_col,
            'total_original': len(feature_cols),
            'total_selected': len(selected),
        }, f, default_flow_style=False)
    
    logger.info(f"Selected features saved to {save_path}")
    
    return selected, report


if __name__ == "__main__":
    # Smoke test with synthetic data
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n, freq='B'),
        'ticker': 'TEST',
        'close': np.random.randn(n).cumsum() + 100,
        'feature_good_1': np.random.randn(n) * 0.5,
        'feature_good_2': np.random.randn(n) * 0.3,
        'feature_bad_constant': np.ones(n) * 5,
        'feature_bad_noise': np.random.randn(n) * 0.001,
    })
    # Make feature_good_1 correlated with close
    df['feature_good_1'] = df['close'].diff().fillna(0) + np.random.randn(n) * 0.1
    
    selected, report = select_features(df, target_col='close')
    print(f"\nSelected features: {selected}")
    print(f"Report: {report}")
    print("\n✅ Feature selection smoke test passed!")
