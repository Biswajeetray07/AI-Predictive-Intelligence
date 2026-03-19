"""
World Bank Data Collector
=========================
Fetches macroeconomic indicators from the World Bank API
and converts them into time-series datasets.

API Reference: https://api.worldbank.org
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
    validate_schema,
    handle_missing_fields,
    standardize_timestamps,
)
from src.utils.logging_utils import get_logger, log_execution_metrics, log_dataset_size

logger = get_logger("WorldBank")

# Configuration constants
BASE_URL = "https://api.worldbank.org/v2"

# Indicators to collect
INDICATORS = {
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "SL.UEM.TOTL.ZS": "Unemployment, total (% of total labor force)",
    "NE.EXP.GNFS.ZS": "Exports of goods and services (% of GDP)",
    "NE.IMP.GNFS.ZS": "Imports of goods and services (% of GDP)",
    "BX.KLT.DINV.WD.GD.ZS": "Foreign direct investment, net inflows (% of GDP)",
    "GC.DOD.TOTL.GD.ZS": "Central government debt, total (% of GDP)",
}

# Countries to collect data for
COUNTRIES = [
    "USA", "CHN", "DEU", "JPN", "GBR", "FRA", "IND",
    "BRA", "CAN", "AUS", "KOR", "RUS", "MEX", "IDN",
]

# Date range
START_YEAR = 2000
END_YEAR = 2025

# Global rate limiter - moved initialization into collect() to avoid import side-effects
_rate_limiter = None

def get_rate_limiter():
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_calls=30, period=60.0, name="WorldBank")
    return _rate_limiter


@retry(max_retries=3, backoff_factor=2.0)
def fetch_indicator(
    indicator_code: str,
    country: str = "all",
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> List[Dict]:
    """
    Fetch a single indicator from the World Bank API.
    """
    limiter = get_rate_limiter()
    limiter.acquire()

    client = APIClient(
        base_url=BASE_URL,
        timeout=60,
        max_retries=3,
    )

    endpoint = f"country/{country}/indicator/{indicator_code}"
    params = {
        "format": "json",
        "date": f"{start_year}:{end_year}",
        "per_page": 1000,
        "page": 1,
    }

    try:
        data = client.get_json(endpoint, params=params)
        # World Bank API returns [metadata, records]
        if isinstance(data, list) and len(data) >= 2:
            return data[1] if data[1] else []
        return []
    except Exception as e:
        logger.error(f"Error fetching {indicator_code} for {country}: {e}")
        return []
    finally:
        client.close()


def normalize_wb_record(record: Dict, indicator_code: str, indicator_name: str) -> Dict:
    """
    Normalize a World Bank API record.

    Args:
        record: Raw API record.
        indicator_code: Indicator code.
        indicator_name: Human-readable indicator name.

    Returns:
        Normalized record dict.
    """
    return {
        "date": record.get("date", ""),
        "country": record.get("country", {}).get("value", ""),
        "country_code": record.get("countryiso3code", record.get("country", {}).get("id", "")),
        "indicator": indicator_name,
        "indicator_code": indicator_code,
        "value": record.get("value"),
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

    logger.info("Starting World Bank data collection...")
    logger.info(f"Indicators: {len(INDICATORS)}")
    logger.info(f"Countries: {len(COUNTRIES)}")

    for indicator_code, indicator_name in INDICATORS.items():
        for country in COUNTRIES:
            try:
                records = fetch_indicator(indicator_code, country)
                normalized = [
                    normalize_wb_record(r, indicator_code, indicator_name)
                    for r in records
                ]
                all_records.extend(normalized)
                logger.info(
                    f"  {country} | {indicator_code}: {len(records)} records"
                )
            except Exception as e:
                errors += 1
                logger.error(f"  {country} | {indicator_code}: {e}")

    if not all_records:
        logger.warning("No World Bank records collected.")
        return pd.DataFrame()

    # Build DataFrame
    df = pd.DataFrame(all_records)

    # Validate and clean
    required_cols = ["date", "country", "indicator", "value"]
    try:
        validate_schema(df, required_cols)
    except ValueError as e:
        logger.error(f"Schema validation failed: {e}")

    df = remove_duplicates(df, subset=["date", "country", "indicator_code"])
    df = handle_missing_fields(df)
    df = standardize_timestamps(df, "date")

    # Convert value to numeric
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Save as parquet
    today = datetime.now().strftime("%Y_%m_%d")
    output_path = os.path.join(OUTPUT_DIR, f"worldbank_{today}.parquet")
    df.to_parquet(output_path, index=False, engine="pyarrow")
    log_dataset_size(logger, "WorldBank", output_path)
    log_execution_metrics(logger, "WorldBank", len(df), start_time, errors)

    return df


if __name__ == "__main__":
    df = collect()
    if not df.empty:
        print(f"\nCollected {len(df)} World Bank records")
        print(df.head(10))
    else:
        print("No data collected.")
