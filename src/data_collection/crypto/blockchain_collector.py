"""
Blockchain.com Data Collector
==============================
Collects blockchain metrics: market price, transaction count,
hash rate, difficulty, and block size.

API Reference: https://www.blockchain.com/explorer/api/charts
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
# No sys.path.insert here - run with PYTHONPATH=. from root

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.utils.api_client import APIClient
from src.utils.rate_limiter import RateLimiter
from src.utils.retry_handler import retry
from src.utils.data_validator import (
    remove_duplicates,
    handle_missing_fields,
    standardize_timestamps,
)
from src.utils.logging_utils import get_logger, log_execution_metrics, log_dataset_size

logger = get_logger("Blockchain")

# Configuration
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "blockchain")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "https://api.blockchain.info"

# Blockchain.com chart endpoints
METRICS = {
    "market_price": {
        "chart": "market-price",
        "description": "Bitcoin Market Price (USD)",
    },
    "transaction_count": {
        "chart": "n-transactions",
        "description": "Number of Transactions Per Day",
    },
    "hash_rate": {
        "chart": "hash-rate",
        "description": "Hash Rate (TH/s)",
    },
    "difficulty": {
        "chart": "difficulty",
        "description": "Mining Difficulty",
    },
    "block_size": {
        "chart": "avg-block-size",
        "description": "Average Block Size (MB)",
    },
    "mempool_size": {
        "chart": "mempool-size",
        "description": "Mempool Size (bytes)",
    },
    "total_bitcoins": {
        "chart": "total-bitcoins",
        "description": "Total Bitcoins in Circulation",
    },
    "trade_volume": {
        "chart": "trade-volume",
        "description": "Exchange Trade Volume (USD)",
    },
}

TIME_SPAN = "2years"

rate_limiter = RateLimiter(max_calls=10, period=60.0, name="Blockchain")


@retry(max_retries=3, backoff_factor=2.0)
def fetch_chart_data(chart_name: str, timespan: str = TIME_SPAN) -> List[Dict]:
    """
    Fetch chart data from Blockchain.com API.

    Args:
        chart_name: Chart endpoint name.
        timespan: Time span to fetch (e.g., '2years', '1year').

    Returns:
        List of data points with 'x' (timestamp) and 'y' (value).
    """
    rate_limiter.acquire()

    client = APIClient(
        base_url=BASE_URL,
        timeout=30,
        max_retries=3,
    )

    params = {
        "timespan": timespan,
        "format": "json",
        "cors": "true",
    }

    try:
        data = client.get_json(f"charts/{chart_name}", params=params)
        return data.get("values", []) if isinstance(data, dict) else []
    except Exception as e:
        logger.error(f"Error fetching blockchain {chart_name}: {e}")
        return []
    finally:
        client.close()


def normalize_data_point(point: Dict, metric_name: str, description: str) -> Dict:
    """
    Normalize a blockchain data point.

    Args:
        point: Raw data point with 'x' and 'y' keys.
        metric_name: Metric identifier.
        description: Human-readable description.

    Returns:
        Normalized record dict.
    """
    # Convert Unix timestamp to datetime string
    timestamp = point.get("x", 0)
    date_str = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d") if timestamp else ""

    return {
        "date": date_str,
        "metric": metric_name,
        "description": description,
        "value": point.get("y", 0.0),
    }


def collect() -> pd.DataFrame:
    """
    Main collection function.

    Returns:
        Combined DataFrame of all blockchain metrics.
    """
    start_time = time.time()
    all_records: List[Dict] = []
    errors = 0

    logger.info("Starting Blockchain.com data collection...")

    for metric_name, config in METRICS.items():
        try:
            data_points = fetch_chart_data(config["chart"])
            normalized = [
                normalize_data_point(p, metric_name, config["description"])
                for p in data_points
            ]
            all_records.extend(normalized)
            logger.info(f"  {metric_name}: {len(data_points)} data points")
        except Exception as e:
            errors += 1
            logger.error(f"  {metric_name}: {e}")

    if not all_records:
        logger.warning("No blockchain data collected.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = remove_duplicates(df, subset=["date", "metric"])
    df = handle_missing_fields(df)
    df = standardize_timestamps(df, "date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    today = datetime.now().strftime("%Y_%m_%d")
    output_path = os.path.join(OUTPUT_DIR, f"blockchain_{today}.parquet")
    df.to_parquet(output_path, index=False, engine="pyarrow")
    log_dataset_size(logger, "Blockchain", output_path)
    log_execution_metrics(logger, "Blockchain", len(df), start_time, errors)

    return df


if __name__ == "__main__":
    df = collect()
    if not df.empty:
        print(f"\nCollected {len(df)} blockchain data points")
        print(df.head(10))
    else:
        print("No data collected.")
