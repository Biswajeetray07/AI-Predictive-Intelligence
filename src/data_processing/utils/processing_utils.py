import os
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
from typing import List, Optional
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def setup_processing_logger(name: str) -> logging.Logger:
    """
    Sets up a logger with a rotating file handler to prevent unbounded log growth.
    Logs are saved to the 'logs/' directory at the root of the project.
    """
    # Ensure logs directory exists at the root level
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(root_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'data_processing.log')
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid adding multiple handlers if the logger is already configured
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 10 MB per file, keep 5 backups
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

def safe_read_csv(filepath: str, logger: logging.Logger, **kwargs) -> Optional[pd.DataFrame]:
    """
    Safely reads a CSV file using pandas, catching and logging common errors.
    Returns None if the file could not be read.
    """
    try:
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Successfully loaded '{filepath}' with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.warning(f"File not found: '{filepath}'. Skipping.")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"File is empty: '{filepath}'. Skipping.")
        return None
    except Exception as e:
        logger.error(f"Error reading file '{filepath}': {e}")
        return None

def safe_read_data(filepath: str, logger: logging.Logger, **kwargs) -> Optional[pd.DataFrame]:
    """
    Safely reads a CSV or Parquet file using pandas.
    """
    if filepath.endswith('.parquet'):
        try:
            df = pd.read_parquet(filepath, **kwargs)
            logger.info(f"Successfully loaded parquet '{filepath}' with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error reading parquet file '{filepath}': {e}")
            return None
    else:
        return safe_read_csv(filepath, logger, **kwargs)

def validate_processed_schema(df: pd.DataFrame, expected_columns: List[str], logger: logging.Logger) -> bool:
    """
    Validates that a DataFrame contains all expected columns and has no completely empty expected columns.
    """
    if df is None or df.empty:
        logger.error("Validation failed: DataFrame is None or empty.")
        return False
        
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Validation failed: Missing expected columns {missing_cols}")
        return False
        
    # Optional: check if expected columns are entirely null
    entirely_null_cols = [col for col in expected_columns if df[col].isnull().all()]
    if entirely_null_cols:
        logger.warning(f"Validation warning: These columns are entirely null: {entirely_null_cols}")
        
    logger.info("Schema validation passed.")
    return True
