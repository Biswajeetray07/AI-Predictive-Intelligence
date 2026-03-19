"""
MLflow Utilities — Shared experiment tracking helpers for all training scripts.

Provides standardized experiment setup, parameter logging, metric logging,
and model artifact registration across TS, NLP, and Fusion pipelines.
"""

import os
import logging
from typing import Dict, Any, Optional

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MLflowUtils")

# Default MLflow tracking URI (local sqlite)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_TRACKING_URI = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"


def setup_experiment(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Create or get an MLflow experiment.

    Args:
        experiment_name: Name of the experiment (e.g., 'timeseries_training').
        tracking_uri: MLflow tracking server URI. Defaults to local sqlite.
        tags: Optional tags for the experiment.

    Returns:
        Experiment ID or None if MLflow is not available.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
        return None

    mlflow = _require_mlflow()
    uri = tracking_uri or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name, tags=tags or {}
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment '{experiment_name}' (id={experiment_id}) at {uri}")
    return experiment_id


def start_run(
    run_name: str,
    params: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
):
    """
    Start an MLflow run and log initial parameters.

    Args:
        run_name: Human-readable run name.
        params: Hyperparameters to log.
        tags: Optional tags.

    Returns:
        MLflow run object or None.
    """
    if not MLFLOW_AVAILABLE:
        return None
    mlflow = _require_mlflow()
    run = mlflow.start_run(run_name=run_name, tags=tags or {})
    if params:
        # Flatten nested dicts for MLflow
        flat_params = _flatten_dict(params)
        mlflow.log_params(flat_params)
    return run


def log_epoch_metrics(epoch: int, metrics: Dict[str, float]):
    """Log metrics for a specific epoch."""
    if not MLFLOW_AVAILABLE:
        return
    mlflow = _require_mlflow()
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=epoch)


def log_final_metrics(metrics: Dict[str, float]):
    """Log final evaluation metrics."""
    if not MLFLOW_AVAILABLE:
        return
    mlflow = _require_mlflow()
    for key, value in metrics.items():
        mlflow.log_metric(f"final_{key}", value)


def log_model_artifact(model_path: str, artifact_name: str = "model"):
    """Log a model file as an artifact."""
    if not MLFLOW_AVAILABLE:
        return
    mlflow = _require_mlflow()
    if os.path.exists(model_path):
        mlflow.log_artifact(model_path, artifact_name)
        logger.info(f"Logged artifact: {model_path}")


def end_run():
    """End the current MLflow run."""
    if MLFLOW_AVAILABLE:
        mlflow = _require_mlflow()
        mlflow.end_run()


def log_model_card(
    model_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    dataset_info: Optional[Dict[str, Any]] = None,
):
    """
    Log a comprehensive model card with params, metrics, and dataset info.
    
    Args:
        model_name: Name of the model architecture.
        params: Training hyperparameters.
        metrics: Final evaluation metrics.
        dataset_info: Optional dataset metadata (sizes, features, etc.).
    """
    if not MLFLOW_AVAILABLE:
        return
    mlflow = _require_mlflow()
    mlflow.set_tag("model_name", model_name)
    flat_params = _flatten_dict(params)
    mlflow.log_params(flat_params)
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
    if dataset_info:
        for key, value in dataset_info.items():
            mlflow.set_tag(f"dataset.{key}", str(value))


def _require_mlflow():
    if not MLFLOW_AVAILABLE:
        raise ImportError("mlflow is required. Install with: pip install mlflow")
    import mlflow
    return mlflow


def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary for MLflow param logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)
