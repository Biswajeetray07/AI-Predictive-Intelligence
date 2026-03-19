"""
Structured Logging Configuration
Provides dual output: console + file logging with proper formatting
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pythonjsonlogger import jsonlogger


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    use_json: bool = False
) -> logging.Logger:
    """
    Setup dual logging: console + file with optional JSON formatting
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses ./logs/app.log)
        use_json: If True, use JSON formatting for structured logs
    
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    if log_file is None:
        log_file = os.path.join(logs_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create logger
    logger = logging.getLogger("ai_predictive_intelligence")
    logger.setLevel(getattr(logging, log_level))
    
    # Prevent duplicate handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Define formatters
    if use_json:
        console_formatter = jsonlogger.JsonFormatter(
            fmt='%(timestamp)s %(level)s %(name)s %(message)s',
            rename_fields={'timestamp': 'ts', 'level': 'lvl'}
        )
        file_formatter = jsonlogger.JsonFormatter(
            fmt='%(timestamp)s %(level)s %(name)s %(message)s %(pathname)s:%(lineno)d',
            rename_fields={'timestamp': 'ts', 'level': 'lvl'}
        )
    else:
        console_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console Handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File Handler (DEBUG and above)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized | Level: {log_level} | File: {log_file}")
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get logger instance by name"""
    if name is None:
        name = "ai_predictive_intelligence"
    return logging.getLogger(name)


# Initialize default logger on import
default_logger = setup_logging(log_level="INFO")
