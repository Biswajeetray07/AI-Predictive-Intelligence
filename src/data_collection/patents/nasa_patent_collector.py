"""
NASA Patent Collector
=====================
Collects patent data from NASA TechTransfer API.
This is a free alternative to USPTO/PatentsView.

API Reference: https://api.nasa.gov/api.html#techtransfer
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# Setup absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

from src.utils.api_client import APIClient
from src.utils.rate_limiter import RateLimiter
from src.utils.retry_handler import retry
from src.utils.data_validator import (
    remove_duplicates,
    handle_missing_fields,
    standardize_timestamps,
)
from src.utils.logging_utils import get_logger, log_execution_metrics, log_dataset_size

logger = get_logger("NASA_Patents")

# Configuration constants
# Direct endpoint that doesn't require keys for basic search
BASE_URL = "https://technology.nasa.gov/api/api"

# Categories/Keywords to query to capture a broad tech portfolio
KEYWORDS = ["AI", "robotics", "computer vision", "machine learning", "energy", "batteries", "sensors", "data"]

# Global rate limiter - moved initialization into collect() to avoid import side-effects
_rate_limiter = None

def get_rate_limiter():
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_calls=2, period=1.0, name="NASA_Patents")
    return _rate_limiter

@retry(max_retries=3, backoff_factor=2.0)
def fetch_nasa_patents_by_keyword(keyword: str) -> List[List]:
    """
    Fetch patents from NASA TechTransfer API for a specific keyword.
    """
    limiter = get_rate_limiter()
    limiter.acquire()
    
    client = APIClient(
        base_url=BASE_URL,
        timeout=60,
    )
    
    try:
        logger.info(f"Searching NASA patents for keyword: {keyword}...")
        data = client.get_json(f"patent/{keyword}")
        
        if not data or not isinstance(data, dict):
            return []
            
        results = data.get("results", [])
        return results
        
    except Exception as e:
        logger.error(f"Error searching NASA patents for '{keyword}': {e}")
        return []
    finally:
        client.close()

def normalize_nasa_record(record: List, keyword: str) -> Dict:
    """
    Normalize NASA record.
    """
    try:
        return {
            "patent_id": record[0] if len(record) > 0 else "",
            "case_id": record[1] if len(record) > 1 else "",
            "title": record[2] if len(record) > 2 else "",
            "description": record[3] if len(record) > 3 else "",
            "search_keyword": keyword,
            "category": record[5] if len(record) > 5 else "",
            "origin_center": record[9] if len(record) > 9 else "",
            "source": "NASA TechTransfer"
        }
    except Exception:
        return {}

def collect() -> pd.DataFrame:
    """Main collection function."""
    # Setup paths relative to project root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "patents")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = time.time()
    errors = 0
    all_normalized = []
    
    for kw in KEYWORDS:
        raw_results = fetch_nasa_patents_by_keyword(kw)
        if not raw_results:
            continue
            
        logger.info(f"  {kw}: Got {len(raw_results)} raw records")
        for record in raw_results:
            if isinstance(record, list):
                norm = normalize_nasa_record(record, kw)
                if norm:
                    all_normalized.append(norm)
                
    if not all_normalized:
        logger.warning("No NASA patents collected across all keywords.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_normalized)
    df = remove_duplicates(df, subset=["patent_id"])
    
    # Save as parquet
    today = datetime.now().strftime("%Y_%m_%d")
    output_path = os.path.join(OUTPUT_DIR, f"nasa_patents_{today}.parquet")
    df.to_parquet(output_path, index=False)
    
    log_dataset_size(logger, "NASA Patents", output_path)
    log_execution_metrics(logger, "NASA Patents", len(df), start_time, errors)
    
    return df

if __name__ == "__main__":
    df = collect()
    if not df.empty:
        print(f"\nCollected {len(df)} NASA patents")
        print(df[["patent_id", "title", "category"]].head(10))
    else:
        print("No patents collected.")
