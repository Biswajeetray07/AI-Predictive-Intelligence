"""
UN Population Data Collector
=============================
Collects population, urban population, and growth data
from the UN Population API (World Bank indicators).

API Reference: https://population.un.org/dataportal/about/dataapi
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

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
logger = get_logger("UNPopulation")

# Configuration
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "population")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Using World Bank API for population indicators (free, no key needed)
BASE_URL = "https://api.worldbank.org/v2"

# Population indicators
INDICATORS = {
    "SP.POP.TOTL": "Total Population",
    "SP.URB.TOTL": "Urban Population",
    "SP.POP.GROW": "Population Growth (annual %)",
    "SP.URB.GROW": "Urban Population Growth (annual %)",
    "SP.DYN.CBRT.IN": "Birth Rate (per 1,000 people)",
    "SP.DYN.CDRT.IN": "Death Rate (per 1,000 people)",
    "SP.DYN.LE00.IN": "Life Expectancy at Birth",
    "SP.POP.DPND": "Age Dependency Ratio",
    "EN.URB.MCTY.TL.ZS": "Population in largest city (%)",
}

# Countries to collect data for
COUNTRIES = [
    "USA", "CHN", "IND", "IDN", "PAK", "BRA", "NGA",
    "BGD", "RUS", "MEX", "JPN", "ETH", "PHL", "EGY",
    "DEU", "GBR", "FRA", "THA", "ZAF", "KOR",
]

START_YEAR = 1990
END_YEAR = 2025

rate_limiter = RateLimiter(max_calls=30, period=60.0, name="UNPopulation")


@retry(max_retries=3, backoff_factor=2.0)
def fetch_population_indicator(
    indicator: str,
    country: str = "all",
) -> List[Dict]:
    """
    Fetch population indicator from World Bank API.

    Args:
        indicator: World Bank indicator code.
        country: Country ISO3 code.

    Returns:
        List of data records.
    """
    rate_limiter.acquire()

    client = APIClient(
        base_url=BASE_URL,
        timeout=60,
        max_retries=3,
    )

    endpoint = f"country/{country}/indicator/{indicator}"
    params = {
        "format": "json",
        "date": f"{START_YEAR}:{END_YEAR}",
        "per_page": 1000,
    }

    try:
        data = client.get_json(endpoint, params=params)
        if isinstance(data, list) and len(data) >= 2:
            return data[1] if data[1] else []
        return []
    except Exception as e:
        logger.error(f"Error fetching {indicator} for {country}: {e}")
        return []
    finally:
        client.close()


def normalize_record(record: Dict, indicator_code: str, indicator_name: str) -> Dict:
    """
    Normalize a population data record.

    Args:
        record: Raw API record.
        indicator_code: Indicator code.
        indicator_name: Human-readable name.

    Returns:
        Normalized record dict.
    """
    return {
        "date": record.get("date", ""),
        "country": record.get("country", {}).get("value", ""),
        "country_code": record.get("countryiso3code", ""),
        "indicator": indicator_name,
        "indicator_code": indicator_code,
        "value": record.get("value"),
    }


def collect() -> pd.DataFrame:
    """
    Main collection function.

    Returns:
        Combined DataFrame of population data.
    """
    start_time = time.time()
    all_records: List[Dict] = []
    errors = 0

    logger.info("Starting UN Population data collection...")
    logger.info(f"Indicators: {len(INDICATORS)}")
    logger.info(f"Countries: {len(COUNTRIES)}")

    for indicator_code, indicator_name in INDICATORS.items():
        for country in COUNTRIES:
            try:
                records = fetch_population_indicator(indicator_code, country)
                normalized = [
                    normalize_record(r, indicator_code, indicator_name)
                    for r in records
                ]
                all_records.extend(normalized)
            except Exception as e:
                errors += 1
                logger.error(f"  {country}/{indicator_code}: {e}")

        logger.info(f"  {indicator_code}: collected for {len(COUNTRIES)} countries")

    if not all_records:
        logger.warning("No population records collected.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = remove_duplicates(df, subset=["date", "country_code", "indicator_code"])
    df = handle_missing_fields(df)
    df = standardize_timestamps(df, "date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    today = datetime.now().strftime("%Y_%m_%d")
    output_path = os.path.join(OUTPUT_DIR, f"un_population_{today}.parquet")
    df.to_parquet(output_path, index=False, engine="pyarrow")
    log_dataset_size(logger, "UNPopulation", output_path)
    log_execution_metrics(logger, "UNPopulation", len(df), start_time, errors)

    return df


if __name__ == "__main__":
    df = collect()
    if not df.empty:
        print(f"\nCollected {len(df)} population records")
        print(df.head(10))
    else:
        print("No data collected.")
