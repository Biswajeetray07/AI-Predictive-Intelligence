"""
Automated Retraining Trigger — Monitors drift reports and orchestrates retraining.

Integrates with DriftMonitor to detect when model performance degrades,
then triggers the training pipeline to retrain models on fresh data.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.evaluation.monitoring.drift_detection import DriftMonitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RetrainingTrigger")


class RetrainingTrigger:
    """
    Automated retraining orchestrator based on drift detection results.
    
    Workflow:
        1. Load reference distributions from training data
        2. Run drift check on new inference data
        3. If critical drift detected, trigger retraining
        4. Log decision to MLflow and save report
    """

    def __init__(
        self,
        psi_threshold: float = 0.25,
        ks_alpha: float = 0.05,
        drift_ratio_threshold: float = 0.3,
        reports_dir: Optional[str] = None,
    ):
        self.monitor = DriftMonitor(
            psi_threshold=psi_threshold,
            ks_alpha=ks_alpha,
        )
        self.drift_ratio_threshold = drift_ratio_threshold
        self.reports_dir = reports_dir or os.path.join(
            PROJECT_ROOT, "logs", "drift_reports"
        )
        os.makedirs(self.reports_dir, exist_ok=True)
        self.last_report: Optional[Dict] = None

    def fit_reference(self, train_data, feature_columns):
        """Store reference distributions from training data."""
        self.monitor.fit(train_data, feature_columns)
        logger.info(f"Fitted reference distributions on {len(feature_columns)} features")

    def check_and_decide(self, new_data) -> Dict:
        """
        Run drift detection and decide whether to retrain.
        
        Returns:
            Dict with 'should_retrain', 'drift_report', and 'decision_reason'.
        """
        report = self.monitor.check(new_data)
        
        should_retrain = report['status'] == 'CRITICAL_DRIFT'
        
        decision = {
            'timestamp': datetime.now().isoformat(),
            'should_retrain': should_retrain,
            'drift_status': report['status'],
            'features_monitored': report['features_monitored'],
            'features_drifted': report['features_drifted'],
            'drifted_features': report['drifted_features'],
            'decision_reason': self._get_decision_reason(report),
        }
        
        self.last_report = decision
        self._save_report(decision)
        self._log_to_mlflow(decision)
        
        if should_retrain:
            logger.warning(
                f"🚨 RETRAINING TRIGGERED: {report['features_drifted']}/{report['features_monitored']} "
                f"features drifted ({report['drifted_features'][:5]}...)"
            )
        else:
            logger.info(
                f"✅ No retraining needed. Status: {report['status']} "
                f"({report['features_drifted']}/{report['features_monitored']} drifted)"
            )
        
        return decision

    def trigger_retraining(self) -> bool:
        """
        Execute the retraining pipeline.
        
        Returns:
            True if retraining completed successfully.
        """
        logger.info("Starting automated retraining pipeline...")
        
        try:
            from src.utils.pipeline_utils import run_script
            
            # Run the core training scripts in order
            scripts = [
                os.path.join(PROJECT_ROOT, "src", "training", "timeseries", "train.py"),
                os.path.join(PROJECT_ROOT, "src", "training", "nlp", "train.py"),
                os.path.join(PROJECT_ROOT, "src", "training", "fusion", "train.py"),
            ]
            
            for script in scripts:
                if os.path.exists(script):
                    logger.info(f"Running: {script}")
                    success = run_script(script)
                    if not success:
                        logger.error(f"Retraining failed at: {script}")
                        return False
            
            logger.info("✅ Automated retraining completed successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Retraining pipeline error: {e}", exc_info=True)
            return False

    def _get_decision_reason(self, report: Dict) -> str:
        """Generate human-readable decision reason."""
        if report['status'] == 'CRITICAL_DRIFT':
            ratio = report['features_drifted'] / max(report['features_monitored'], 1)
            return (
                f"Critical drift: {report['features_drifted']}/{report['features_monitored']} "
                f"features ({ratio:.0%}) exceeded thresholds. "
                f"Top drifted: {report['drifted_features'][:3]}"
            )
        elif report['status'] == 'WARNING':
            return (
                f"Moderate drift: {report['features_drifted']} features drifted "
                f"but below critical threshold ({self.drift_ratio_threshold:.0%})"
            )
        return "No significant drift detected."

    def _save_report(self, decision: Dict):
        """Save drift report to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.reports_dir, f"drift_report_{timestamp}.json")
        with open(report_path, "w") as f:
            json.dump(decision, f, indent=2, default=str)
        logger.info(f"Drift report saved: {report_path}")

    def _log_to_mlflow(self, decision: Dict):
        """Log drift decision to MLflow if available."""
        try:
            from src.utils.mlflow_utils import MLFLOW_AVAILABLE
            if MLFLOW_AVAILABLE:
                import mlflow
                mlflow.log_metrics({
                    "drift_features_monitored": decision['features_monitored'],
                    "drift_features_drifted": decision['features_drifted'],
                    "drift_should_retrain": int(decision['should_retrain']),
                })
        except ImportError:
            pass


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Quick smoke test
    trigger = RetrainingTrigger()
    
    # Simulate training data
    np.random.seed(42)
    train_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(5, 2, 1000),
        'feature_3': np.random.uniform(0, 1, 1000),
    })
    
    trigger.fit_reference(train_data, ['feature_1', 'feature_2', 'feature_3'])
    
    # Simulate new data with drift in feature_2
    new_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 500),
        'feature_2': np.random.normal(10, 3, 500),  # Shifted!
        'feature_3': np.random.uniform(0, 1, 500),
    })
    
    decision = trigger.check_and_decide(new_data)
    print(f"\nDecision: {json.dumps(decision, indent=2, default=str)}")
