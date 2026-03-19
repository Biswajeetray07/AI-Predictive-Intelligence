"""
Logging Utilities Module
========================
Structured logging setup for the data collection system.
Logs to both console and `logs/data_collection.log`.
"""

import os
import time
import logging
from typing import Optional
from functools import wraps
from typing import Callable, Any

# Resolve paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "data_collection.log")

os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a structured logger that writes to console and log file.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_execution_metrics(
    logger: logging.Logger,
    collector_name: str,
    records_collected: int,
    start_time: float,
    errors: int = 0,
) -> None:
    """
    Log standardized execution metrics for a collector run.

    Args:
        logger: Logger instance.
        collector_name: Name of the collector.
        records_collected: Number of records successfully collected.
        start_time: Epoch timestamp when collection started (from time.time()).
        errors: Number of errors encountered.
    """
    elapsed = time.time() - start_time
    logger.info(
        f"[{collector_name}] Completed | "
        f"Records: {records_collected:,} | "
        f"Errors: {errors} | "
        f"Duration: {elapsed:.2f}s"
    )


def log_dataset_size(
    logger: logging.Logger,
    dataset_name: str,
    file_path: str,
) -> None:
    """
    Log the size of a saved dataset file.

    Args:
        logger: Logger instance.
        dataset_name: Human-readable name for the dataset.
        file_path: Absolute path to the dataset file.
    """
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"[{dataset_name}] Saved: {file_path} ({size_mb:.2f} MB)")
    else:
        logger.warning(f"[{dataset_name}] File not found: {file_path}")


def log_collector(collector_name: str) -> Callable[..., Any]:
    """
    Decorator that wraps a collector function with execution logging.

    Logs start, completion, record count, and timing automatically.

    Args:
        collector_name: Name of the collector for log messages.

    Returns:
        Decorated function.

    Example:
        @log_collector("WorldBank")
        def collect():
            ...
            return df
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = get_logger(collector_name)
            _logger.info(f"[{collector_name}] Starting collection...")
            start = time.time()
            try:
                result = func(*args, **kwargs)
                record_count = len(result) if hasattr(result, "__len__") else 0
                log_execution_metrics(_logger, collector_name, record_count, start)
                return result
            except Exception as e:
                elapsed = time.time() - start
                _logger.error(
                    f"[{collector_name}] Failed after {elapsed:.2f}s: {e}",
                    exc_info=True,
                )
                raise
        return wrapper  # type: ignore

    return decorator
