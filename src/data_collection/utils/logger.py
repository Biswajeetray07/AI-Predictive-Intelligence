"""
Structured Logging Utility
===========================
Provides consistent, structured logging across all data collectors.
"""

import os
import logging
from datetime import datetime
from typing import Optional


def get_collector_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Create a structured logger for a data collector.

    Args:
        name: Collector name (e.g., 'youtube_collector').
        log_dir: Directory for log files. Defaults to PROJECT_ROOT/logs/.

    Returns:
        Configured logging.Logger instance.
    """
    if log_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
        log_dir = os.path.join(project_root, "logs")

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler
    log_file = os.path.join(log_dir, f"data_collection.log")
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger


def log_collection_summary(logger: logging.Logger, name: str, record_count: int, output_path: str):
    """Log a standardized collection summary."""
    logger.info(f"{'=' * 50}")
    logger.info(f"Collection Summary: {name}")
    logger.info(f"Records collected: {record_count:,}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"{'=' * 50}")
