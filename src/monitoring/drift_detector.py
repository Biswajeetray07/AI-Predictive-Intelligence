"""
Drift Detection Module
Detects data and model drift using PSI (Population Stability Index) and KS-test
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    PSI measures how much a variable's distribution has shifted.
    
    PSI > 0.25: High drift (major concern)
    PSI > 0.10: Medium drift (worth investigating)
    PSI < 0.10: Low drift (normal variation)
    """
    # Binning
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)
    
    expected_counts = np.histogram(expected, bins=breakpoints)[0] + 1e-6
    actual_counts = np.histogram(actual, bins=breakpoints)[0] + 1e-6
    
    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def calculate_ks_statistic(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov test for drift detection.
    Returns (statistic, p_value)
    
    p_value < 0.05: Significant drift detected
    """
    statistic, p_value = stats.ks_2samp(expected, actual)
    return float(statistic), float(p_value)


class DriftDetector:
    """
    Detects feature and prediction drift in production.
    """
    
    def __init__(self, baseline_df: pd.DataFrame, psi_threshold: float = 0.10, 
                 ks_threshold: float = 0.05):
        """
        Args:
            baseline_df: Baseline data (e.g., training/validation set)
            psi_threshold: Trigger alert if PSI > threshold
            ks_threshold: Trigger alert if KS p-value < threshold
        """
        self.baseline = baseline_df
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.drift_history: Dict[str, List[float]] = {}
        
        logger.info(f"DriftDetector initialized with {len(baseline_df)} baseline samples")
    
    def check_feature_drift(self, new_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Check for drift in features.
        
        Returns:
            {
                'feature_name': {
                    'psi': float,
                    'ks_statistic': float,
                    'ks_pvalue': float,
                    'drift_detected': bool,
                    'drift_reason': str
                }
            }
        """
        drift_report = {}
        
        for column in new_df.columns:
            if column not in self.baseline.columns:
                logger.warning(f"Column {column} not in baseline; skipping drift check")
                continue
            
            if new_df[column].dtype in ['object', 'bool']:
                logger.debug(f"Skipping non-numeric column {column}")
                continue
            
            baseline_values = self.baseline[column].dropna().values
            new_values = new_df[column].dropna().values
            
            if len(baseline_values) < 10 or len(new_values) < 10:
                logger.debug(f"Insufficient samples for {column}; skipping")
                continue
            
            # Calculate PSI
            psi = calculate_psi(baseline_values, new_values)
            
            # Calculate KS test
            ks_stat, ks_pvalue = calculate_ks_statistic(baseline_values, new_values)
            
            # Determine drift
            drift_detected = (psi > self.psi_threshold) or (ks_pvalue < self.ks_threshold)
            drift_reason = ""
            
            if psi > self.psi_threshold:
                drift_reason += f"PSI={psi:.4f} > threshold"
            if ks_pvalue < self.ks_threshold:
                if drift_reason:
                    drift_reason += "; "
                drift_reason += f"KS p-value={ks_pvalue:.4f} < threshold"
            
            drift_report[column] = {
                'psi': float(psi),
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'drift_detected': drift_detected,
                'drift_reason': drift_reason,
                'baseline_mean': float(np.mean(baseline_values)),
                'baseline_std': float(np.std(baseline_values)),
                'new_mean': float(np.mean(new_values)),
                'new_std': float(np.std(new_values)),
            }
            
            if drift_detected:
                logger.warning(f"🚨 Drift detected in {column}: {drift_reason}")
        
        return drift_report
    
    def check_prediction_drift(self, baseline_predictions: np.ndarray, 
                              new_predictions: np.ndarray) -> Dict:
        """
        Check for drift in model predictions.
        """
        ks_stat, ks_pvalue = calculate_ks_statistic(
            baseline_predictions.flatten(), 
            new_predictions.flatten()
        )
        psi = calculate_psi(
            baseline_predictions.flatten(), 
            new_predictions.flatten()
        )
        
        drift_detected = (psi > self.psi_threshold) or (ks_pvalue < self.ks_threshold)
        
        report = {
            'psi': float(psi),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'drift_detected': drift_detected,
            'baseline_mean': float(np.mean(baseline_predictions)),
            'baseline_std': float(np.std(baseline_predictions)),
            'new_mean': float(np.mean(new_predictions)),
            'new_std': float(np.std(new_predictions)),
        }
        
        if drift_detected:
            logger.warning(f"🚨 Prediction drift detected: PSI={psi:.4f}, KS p-value={ks_pvalue:.4f}")
        
        return report
    
    def get_drift_summary(self, drift_report: Dict) -> Dict:
        """
        Summarize drift detection results.
        """
        drifted_features = [f for f, v in drift_report.items() if v['drift_detected']]
        
        return {
            'total_features_checked': len(drift_report),
            'drifted_features_count': len(drifted_features),
            'drifted_features': drifted_features,
            'drift_rate': len(drifted_features) / max(1, len(drift_report)),
            'severity': 'HIGH' if len(drifted_features) > len(drift_report) * 0.3 
                        else 'MEDIUM' if len(drifted_features) > 0 
                        else 'LOW'
        }
