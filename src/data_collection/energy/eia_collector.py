"""
EIA (Energy Information Administration) Data Collector
======================================================
Collects US energy data including electricity demand,
oil/gas production, and renewable energy generation.

API Reference: https://www.eia.gov/opendata/
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

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

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
logger = get_logger("EIA")

# Configuration
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "energy")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "https://api.eia.gov/v2"
API_KEY = os.getenv("EIA_API_KEY", "")

# EIA series to collect
SERIES = {
    "electricity_demand": {
        "route": "electricity/retail-sales",
        "description": "Electricity retail sales (demand)",
    },
    "crude_oil_production": {
        "route": "petroleum/crd/crpdn",
        "description": "Crude oil production",
    },
    "natural_gas_production": {
        "route": "natural-gas/prod/sum",
        "description": "Natural gas production",
    },
    "renewable_generation": {
        "route": "electricity/electric-power-operational-data",
        "description": "Renewable energy generation",
    },
    "coal_production": {
        "route": "coal/mine-production",
        "description": "Coal mine production",
    },
}

rate_limiter = RateLimiter(max_calls=10, period=60.0, name="EIA")


@retry(max_retries=3, backoff_factor=2.0)
def fetch_eia_series(route: str, params: Optional[Dict] = None) -> List[Dict]:
    """
    Fetch data from EIA API v2.

    Args:
        route: API route path.
        params: Additional query parameters.

    Returns:
        List of data records.
    """
    rate_limiter.acquire()

    if not API_KEY:
        logger.warning("EIA_API_KEY not set. Skipping.")
        return []

    client = APIClient(
        base_url=BASE_URL,
        timeout=60,
        max_retries=3,
    )

    request_params = {
        "api_key": API_KEY,
        "frequency": params.get("frequency", "monthly") if params else "monthly",
        "start": "2015-01",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 5000,
    }
    # Add route-specific data columns
    if "retail-sales" in route:
        request_params["data[0]"] = "sales"
        request_params["data[1]"] = "revenue"
    elif "coal" in route:
        request_params["data[0]"] = "production"
    else:
        request_params["data[0]"] = "value"

    if params:
        request_params.update(params)

    try:
        data = client.get_json(route, params=request_params)
        if isinstance(data, str):
            logger.error(f"EIA API returned string instead of JSON for {route}: {data[:200]}")
            return []
        if not isinstance(data, dict):
            logger.error(f"EIA API returned unexpected type {type(data)} for {route}")
            return []
            
        if "error" in data:
            logger.error(f"EIA API error for {route}: {data['error']}")
            return []

        response = data.get("response", {})
        if not isinstance(response, dict):
            logger.error(f"EIA 'response' field is not a dict: {type(response)}")
            return []
            
        data_records = response.get("data", [])
        if isinstance(data_records, dict):
            # Check if it's a descriptor or a single row
            if any(k in data_records for k in ["period", "value", "sales", "revenue"]):
                return [data_records]
            return []
            
        return data_records if isinstance(data_records, list) else []
    except Exception as e:
        logger.error(f"Error fetching EIA {route}: {e}")
        return []
    finally:
        client.close()


def normalize_eia_record(record: Dict, series_name: str) -> Dict:
    """
    Normalize an EIA record.

    Args:
        record: Raw EIA record.
        series_name: Series identifier name.

    Returns:
        Normalized record dict.
    """
    return {
        "date": record.get("period", ""),
        "series": series_name,
        "value": record.get("value"),
        "unit": record.get("units", record.get("unit", "")),
        "description": record.get("seriesDescription", record.get("seriesdescription", "")),
        "state": record.get("stateDescription", record.get("location", "")),
        "sector": record.get("sectorDescription", record.get("sector", "")),
    }


def collect() -> pd.DataFrame:
    """
    Main collection function.

    Returns:
        Combined DataFrame of all EIA energy data.
    """
    start_time = time.time()
    all_records: List[Dict] = []
    errors = 0

    if not API_KEY:
        logger.warning("EIA_API_KEY not set in .env. Skipping EIA collection.")
        return pd.DataFrame()

    logger.info("Starting EIA energy data collection...")

    for series_name, config in SERIES.items():
        try:
            params = {}
            if "mine-production" in config["route"]:
                params["frequency"] = "annual"
                
            records = fetch_eia_series(config["route"] + "/data", params=params)
                
            if isinstance(records, dict):
                logger.error(f"  {series_name}: Received dict instead of records list. Content keys: {list(records.keys())}")
                continue
                
            if not isinstance(records, list):
                logger.error(f"  {series_name}: Received unexpected type {type(records)}")
                continue

            normalized = [normalize_eia_record(r, series_name) for r in records if isinstance(r, dict)]
            all_records.extend(normalized)
            logger.info(f"  {series_name}: {len(normalized)} records")
        except Exception as e:
            errors += 1
            logger.error(f"  {series_name}: {e}")

    if not all_records:
        logger.warning("No EIA records collected.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = remove_duplicates(df)
    df = handle_missing_fields(df)
    df = standardize_timestamps(df, "date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    today = datetime.now().strftime("%Y_%m_%d")
    output_path = os.path.join(OUTPUT_DIR, f"eia_{today}.parquet")
    df.to_parquet(output_path, index=False, engine="pyarrow")
    log_dataset_size(logger, "EIA", output_path)
    log_execution_metrics(logger, "EIA", len(df), start_time, errors)

    return df


if __name__ == "__main__":
    df = collect()
    if not df.empty:
        print(f"\nCollected {len(df)} EIA records")
        print(df.head(10))
    else:
        print("No data collected. Ensure EIA_API_KEY is set in .env")
