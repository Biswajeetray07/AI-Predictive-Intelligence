"""
Feature Analysis & Redundancy Detection
========================================
- Feature correlation matrix analysis
- Highly correlated feature detection (>0.95 correlation)
- Feature importance via permutation importance
- VIF (Variance Inflation Factor) for multicollinearity
- Feature distribution summary statistics
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FeatureAnalyzer:
    """Comprehensive feature quality and redundancy analysis."""

    def __init__(self, correlation_threshold: float = 0.95, vif_threshold: float = 10.0):
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold

    def analyze_correlations(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Dict:
        """Find highly correlated feature pairs."""
        if feature_cols is None:
            feature_cols = df.select_dtypes(include='number').columns.tolist()
        # Ensure feature_cols is always List[str]
        feature_cols = list(feature_cols)  # type: ignore

        # Limit to prevent memory issues
        if len(feature_cols) > 200:
            logging.warning(f"Too many features ({len(feature_cols)}), using first 200")
            feature_cols = feature_cols[:200]

        corr_matrix = df[feature_cols].corr()

        # Find pairs above threshold
        high_corr_pairs = []
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val >= self.correlation_threshold:
                    high_corr_pairs.append({
                        'feature_1': feature_cols[i],
                        'feature_2': feature_cols[j],
                        'correlation': float(corr_val)
                    })

        # Sort by correlation (highest first)
        high_corr_pairs.sort(key=lambda x: x['correlation'], reverse=True)

        # Features to consider removing (one from each pair)
        redundant_features = set()
        for pair in high_corr_pairs:
            # Keep the first feature, mark the second as redundant
            redundant_features.add(pair['feature_2'])

        logging.info(f"Found {len(high_corr_pairs)} highly correlated pairs (>={self.correlation_threshold})")
        logging.info(f"Redundant features that could be removed: {len(redundant_features)}")

        return {
            'total_features': len(feature_cols),
            'high_correlation_pairs': high_corr_pairs[:50],  # Top 50
            'redundant_features': list(redundant_features),
            'n_redundant': len(redundant_features),
            'correlation_threshold': self.correlation_threshold,
        }

    def analyze_distributions(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Dict:
        """Analyze feature distributions for anomalies."""
        if feature_cols is None:
            feature_cols = df.select_dtypes(include='number').columns.tolist()
        # Ensure feature_cols is always List[str]
        feature_cols = list(feature_cols)  # type: ignore

        analysis = []
        problematic = []

        for col in feature_cols:
            series = df[col].dropna()
            if len(series) == 0:
                problematic.append({'feature': col, 'issue': 'all_missing'})
                continue

            stats = {
                'feature': col,
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'missing_pct': float(df[col].isna().mean() * 100),
                'zero_pct': float((series == 0).mean() * 100),
                'unique_ratio': float(series.nunique() / len(series)),
            }

            # Detect issues
            issues = []
            if stats['std'] < 1e-10:
                issues.append('constant_feature')
            if stats['missing_pct'] > 50:
                issues.append('high_missing')
            if stats['zero_pct'] > 95:
                issues.append('mostly_zeros')
            if abs(stats['mean']) > 1e6:
                issues.append('large_values')

            if issues:
                stats['issues'] = issues
                problematic.append(stats)

            analysis.append(stats)

        logging.info(f"Analyzed {len(feature_cols)} features, {len(problematic)} have issues")

        return {
            'total_features': len(feature_cols),
            'problematic_count': len(problematic),
            'problematic_features': problematic,
            'summary': {
                'avg_missing_pct': np.mean([a['missing_pct'] for a in analysis]) if analysis else 0,
                'features_with_issues': len(problematic),
                'constant_features': sum(1 for p in problematic if 'constant_feature' in p.get('issues', [])),
            }
        }

    def permutation_importance(self, model, X: np.ndarray, y: np.ndarray,
                                feature_names: Optional[List[str]] = None, n_repeats: int = 5,
                                device: str = 'cpu') -> Dict:
        """
        Estimate feature importance by shuffling each feature and measuring performance drop.
        Works with any PyTorch model that accepts [batch, seq_len, features] input.
        """
        import torch

        device_obj = torch.device(device)

        # Baseline performance
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device_obj)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(device_obj)
            preds, _ = model(X_tensor)
            baseline_mse = torch.nn.MSELoss()(preds.squeeze(), y_tensor.squeeze()).item()

        n_features = X.shape[-1]
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]

        importances = []
        for feat_idx in range(n_features):
            drops = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                # Shuffle this feature across all samples (along axis 0)
                perm_idx = np.random.permutation(X_perm.shape[0])
                X_perm[:, :, feat_idx] = X_perm[perm_idx, :, feat_idx]

                with torch.no_grad():
                    X_perm_tensor = torch.tensor(X_perm, dtype=torch.float32).to(device_obj)
                    preds_perm, _ = model(X_perm_tensor)
                    perm_mse = torch.nn.MSELoss()(preds_perm.squeeze(), y_tensor.squeeze()).item()

                drops.append(perm_mse - baseline_mse)

            importances.append({
                'feature': feature_names[feat_idx] if feat_idx < len(feature_names) else f'feature_{feat_idx}',
                'importance_mean': float(np.mean(drops)),
                'importance_std': float(np.std(drops)),
            })

        # Sort by importance (higher = more important)
        importances.sort(key=lambda x: x['importance_mean'], reverse=True)

        logging.info(f"\nTop 10 Most Important Features:")
        for imp in importances[:10]:
            logging.info(f"  {imp['feature']}: {imp['importance_mean']:.6f} ± {imp['importance_std']:.6f}")

        return {
            'baseline_mse': baseline_mse,
            'feature_importances': importances,
            'top_10': importances[:10],
            'bottom_10': importances[-10:],
            'n_features': n_features,
        }


def run_feature_analysis() -> Dict:
    """Run full feature analysis on the merged dataset."""
    merged_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'merged', 'all_merged_dataset.csv')
    if not os.path.exists(merged_path):
        logging.error("Merged dataset not found")
        return {}

    df = pd.read_csv(merged_path)
    exclude = ['date', 'ticker', 'Close', 'close']
    feature_cols = [c for c in df.select_dtypes(include='number').columns if c not in exclude]

    analyzer = FeatureAnalyzer()

    logging.info(f"{'='*60}")
    logging.info(f"FEATURE ANALYSIS ({len(feature_cols)} features)")
    logging.info(f"{'='*60}")

    results = {
        'correlations': analyzer.analyze_correlations(df, feature_cols),
        'distributions': analyzer.analyze_distributions(df, feature_cols),
        'timestamp': datetime.now().isoformat(),
    }

    return results


if __name__ == '__main__':
    import json
    results = run_feature_analysis()
    out_path = os.path.join(PROJECT_ROOT, 'saved_models', 'feature_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Feature analysis saved to {out_path}")
