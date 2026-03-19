"""
Model Drift Detection Module for AI-Predictive-Intelligence.

Implements Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests
to monitor feature distribution shifts between training and inference data.
Alerts when drift exceeds configurable thresholds.
"""

import numpy as np
import pandas as pd
from typing import cast
import logging
from scipy import stats
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ── PSI (Population Stability Index) ─────────────────────────────────────────

def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index.
    PSI < 0.1: No significant shift
    PSI 0.1-0.25: Moderate shift (investigate)
    PSI > 0.25: Significant shift (retrain)
    """
    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)
    
    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates
    
    # Calculate proportions in each bin
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    expected_pct = (expected_counts / len(expected)) + epsilon
    actual_pct = (actual_counts / len(actual)) + epsilon
    
    # PSI formula
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


# ── KS Test (Kolmogorov-Smirnov) ─────────────────────────────────────────────

def ks_drift_test(reference: np.ndarray, current: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Two-sample KS test to detect distribution shift.
    Returns test statistic, p-value, and whether drift is detected.
    """
    reference = np.array(reference, dtype=float)
    current = np.array(current, dtype=float)
    
    ks_stat, p_value = stats.ks_2samp(reference, current)
    
    return {
        'ks_statistic': float(cast(tuple, (ks_stat, p_value))[0]),
        'p_value': float(cast(tuple, (ks_stat, p_value))[1]),
        'drift_detected': bool(float(cast(tuple, (ks_stat, p_value))[1]) < alpha)
    }


# ── Full Feature Distribution Monitor ────────────────────────────────────────

class DriftMonitor:
    """
    Monitors feature distributions for concept drift.
    Compares training-time distributions against new incoming data.
    """
    
    def __init__(self, psi_threshold: float = 0.25, ks_alpha: float = 0.05):
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.reference_stats = {}
        
    def fit(self, train_data: pd.DataFrame, feature_columns: List[str]):
        """Store reference distributions from training data."""
        for col in feature_columns:
            if col in train_data.columns:
                values = train_data[col].dropna().values
                self.reference_stats[col] = {
                    'values': values,
                    'mean': float(np.mean(np.asarray(values))),
                    'std': float(np.std(np.asarray(values))),
                    'min': float(np.min(np.asarray(values))),
                    'max': float(np.max(np.asarray(values)))
                }
        logging.info(f"DriftMonitor fitted on {len(self.reference_stats)} features")
    
    def check(self, new_data: pd.DataFrame) -> Dict:
        """
        Run PSI and KS tests on each monitored feature.
        Returns per-feature drift report and overall alert status.
        """
        report = {}
        alerts = []
        
        for col, ref in self.reference_stats.items():
            if col not in new_data.columns:
                continue
                
            current_values = new_data[col].dropna().values
            # Ensure numpy array for compatibility
            current_values = np.asarray(current_values)
            ref_values = np.asarray(ref['values'])
            if len(current_values) < 20:
                continue

            # PSI
            psi_score = calculate_psi(ref_values, current_values)

            # KS Test
            ks_result = ks_drift_test(ref_values, current_values, self.ks_alpha)

            # Mean shift
            current_mean = float(np.mean(current_values))
            mean_shift = abs(current_mean - ref['mean']) / (ref['std'] + 1e-8)
            
            drifted = psi_score > self.psi_threshold or ks_result['drift_detected']
            
            report[col] = {
                'psi': psi_score,
                'ks_stat': ks_result['ks_statistic'],
                'ks_p_value': ks_result['p_value'],
                'mean_shift_zscore': float(mean_shift),
                'drift_detected': drifted
            }
            
            if drifted:
                alerts.append(col)
        
        n_drifted = len(alerts)
        n_total = len(report)
        
        overall_status = 'STABLE'
        if n_drifted > n_total * 0.3:
            overall_status = 'CRITICAL_DRIFT'
        elif n_drifted > 0:
            overall_status = 'WARNING'
        
        if alerts:
            logging.warning(f"🚨 DRIFT ALERT: {n_drifted}/{n_total} features drifted: {alerts[:5]}...")
        else:
            logging.info(f"✅ No drift detected across {n_total} monitored features")
        
        return {
            'status': overall_status,
            'features_monitored': n_total,
            'features_drifted': n_drifted,
            'drifted_features': alerts,
            'per_feature_report': report
        }
