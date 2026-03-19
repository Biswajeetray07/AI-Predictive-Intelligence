"""
OECD Data Collector
===================
Collects economic indicators (industrial production, trade indices)
from the OECD SDMX-JSON API.

API Reference: https://data-explorer.oecd.org/
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

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

logger = get_logger("OECD")

# Configuration constants
BASE_URL = "https://stats.oecd.org/SDMX-JSON/data"

# OECD datasets to collect
DATASETS = {
    "MEI": {
        "name": "Main Economic Indicators",
        "description": "Industrial production, total",
        "filter": "PRINTO01.IXOBSA.M",
        "dataflow": "MEI"
    },
    "QNA": {
        "name": "Quarterly National Accounts",
        "description": "GDP components, quarterly",
        "filter": "B1_GE.VOBARSA.Q",
        "dataflow": "QNA"
    },
}

# Global rate limiter - moved initialization into collect() to avoid import side-effects
_rate_limiter = None

def get_rate_limiter():
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_calls=5, period=60.0, name="OECD")
    return _rate_limiter


@retry(max_retries=3, backoff_factor=3.0)
def fetch_oecd_dataset(dataset_id: str, config: Dict) -> List[Dict]:
    """
    Fetch data from the OECD SDMX-JSON REST API.
    """
    limiter = get_rate_limiter()
    limiter.acquire()

    client = APIClient(
        base_url=BASE_URL,
        headers={"Accept": "application/vnd.sdmx.data+json;version=1.0.0-wd"},
        timeout=90,
    )

    dataflow = config["dataflow"]
    filter_expr = config["filter"]
    endpoint = f"{dataflow}/{filter_expr}"
    params = {
        "startPeriod": "2015-01",
        "dimensionAtObservation": "AllDimensions",
    }

    try:
        data = client.get_json(endpoint, params=params)
        return _parse_sdmx_json(data, dataset_id)
    except Exception as e:
        logger.error(f"Error fetching OECD {dataset_id}: {e}")
        return []
    finally:
        client.close()


def _parse_sdmx_json(data: Dict, dataset_id: str) -> List[Dict]:
    """
    Parse SDMX-JSON response into flat records.

    Args:
        data: Raw SDMX-JSON response.
        dataset_id: Dataset identifier for labeling.

    Returns:
        List of flat dicts with country, indicator, date, value.
    """
    records = []

    try:
        datasets = data.get("data", {}).get("dataSets", [])
        structure = data.get("data", {}).get("structure", {}).get("dimensions", {})
        observations_dims = structure.get("observation", [])

        if not datasets:
            return records

        observations = datasets[0].get("observations", {})

        # Build dimension value lookups
        dim_values = {}
        for dim in observations_dims:
            dim_id = dim.get("id", "")
            values = dim.get("values", [])
            dim_values[dim_id] = {i: v.get("name", v.get("id", "")) for i, v in enumerate(values)}

        for key, obs_value in observations.items():
            indices = [int(k) for k in key.split(":")]
            record = {"dataset": dataset_id, "value": obs_value[0] if obs_value else None}

            for i, dim in enumerate(observations_dims):
                dim_id = dim.get("id", "")
                if i < len(indices):
                    record[dim_id] = dim_values.get(dim_id, {}).get(indices[i], str(indices[i]))

            records.append(record)

    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Error parsing SDMX-JSON for {dataset_id}: {e}")

    return records


def normalize_oecd_record(record: Dict) -> Dict:
    """
    Normalize an OECD record to a standardized schema.

    Args:
        record: Parsed SDMX record.

    Returns:
        Normalized record dict.
    """
    return {
        "date": record.get("TIME_PERIOD", ""),
        "country": record.get("REF_AREA", ""),
        "indicator": record.get("MEASURE", record.get("dataset", "")),
        "dataset": record.get("dataset", ""),
        "value": record.get("value"),
        "frequency": record.get("FREQ", ""),
        "unit": record.get("UNIT_MEASURE", ""),
    }


def collect() -> pd.DataFrame:
    """Main collection function."""
    # Setup paths relative to project root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "trade")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = time.time()
    all_records: List[Dict] = []
    errors = 0

    logger.info("Starting OECD data collection...")
    logger.info(f"Datasets: {list(DATASETS.keys())}")

    for dataset_id, config in DATASETS.items():
        try:
            logger.info(f"  Fetching {config['name']}...")
            records = fetch_oecd_dataset(dataset_id, config)
            normalized = [normalize_oecd_record(r) for r in records]
            all_records.extend(normalized)
            logger.info(f"  {dataset_id}: {len(records)} records")
        except Exception as e:
            errors += 1
            logger.error(f"  {dataset_id}: {e}")

    if not all_records:
        logger.warning("No OECD records collected.")
        return pd.DataFrame()

    # Build DataFrame
    df = pd.DataFrame(all_records)
    df = remove_duplicates(df)
    df = handle_missing_fields(df)
    df = standardize_timestamps(df, "date")

    # Convert value to numeric
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Save as parquet
    today = datetime.now().strftime("%Y_%m_%d")
    output_path = os.path.join(OUTPUT_DIR, f"oecd_{today}.parquet")
    df.to_parquet(output_path, index=False, engine="pyarrow")
    log_dataset_size(logger, "OECD", output_path)
    log_execution_metrics(logger, "OECD", len(df), start_time, errors)

    return df


if __name__ == "__main__":
    df = collect()
    if not df.empty:
        print(f"\nCollected {len(df)} OECD records")
        print(df.head(10))
    else:
        print("No data collected.")
