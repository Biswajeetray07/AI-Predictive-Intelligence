"""
OpenSky Network Aviation Data Collector
========================================
Collects live flight state data from the OpenSky Network API
and aggregates flight counts per airport.

API Reference: https://openskynetwork.github.io/opensky-api/
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

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
)
from src.utils.logging_utils import get_logger, log_execution_metrics, log_dataset_size

logger = get_logger("OpenSky")

# Configuration constants
BASE_URL = "https://opensky-network.org/api"

# Bounding boxes for major regions [lat_min, lat_max, lon_min, lon_max]
REGIONS = {
    "North_America": {"lamin": 20.0, "lamax": 55.0, "lomin": -130.0, "lomax": -60.0},
    "Europe": {"lamin": 35.0, "lamax": 72.0, "lomin": -25.0, "lomax": 45.0},
    "East_Asia": {"lamin": 10.0, "lamax": 55.0, "lomin": 100.0, "lomax": 150.0},
}

# State vector indices
STATE_KEYS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
    "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
    "spi", "position_source",
]

# Global rate limiter - moved initialization into collect() to avoid import side-effects
_rate_limiter = None

def get_rate_limiter():
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_calls=5, period=60.0, name="OpenSky")
    return _rate_limiter


@retry(max_retries=3, backoff_factor=5.0)
def fetch_flight_states(
    lamin: Optional[float] = None,
    lamax: Optional[float] = None,
    lomin: Optional[float] = None,
    lomax: Optional[float] = None,
) -> List[List]:
    """
    Fetch current flight states from OpenSky Network.
    """
    limiter = get_rate_limiter()
    limiter.acquire()

    client = APIClient(
        base_url=BASE_URL,
        timeout=30,
        max_retries=3,
    )

    params = {}
    if all(v is not None for v in [lamin, lamax, lomin, lomax]):
        params = {
            "lamin": lamin,
            "lamax": lamax,
            "lomin": lomin,
            "lomax": lomax,
        }

    try:
        data = client.get_json("states/all", params=params)
        return data.get("states", []) if isinstance(data, dict) else []
    except Exception as e:
        logger.error(f"Error fetching OpenSky states: {e}")
        return []
    finally:
        client.close()


def normalize_state(state: List) -> Dict:
    """
    Normalize a flight state vector.

    Args:
        state: Raw state vector list (17 elements).

    Returns:
        Dict with named fields.
    """
    record = {}
    for i, key in enumerate(STATE_KEYS):
        record[key] = state[i] if i < len(state) else None

    # Clean callsign
    if record.get("callsign"):
        record["callsign"] = str(record["callsign"]).strip()

    return {
        "icao24": record.get("icao24", ""),
        "callsign": record.get("callsign", ""),
        "origin_country": record.get("origin_country", ""),
        "latitude": record.get("latitude"),
        "longitude": record.get("longitude"),
        "altitude": record.get("baro_altitude"),
        "velocity": record.get("velocity"),
        "on_ground": record.get("on_ground", False),
        "vertical_rate": record.get("vertical_rate"),
        "timestamp": datetime.utcnow().isoformat(),
    }


def aggregate_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate flight counts per origin country.

    Args:
        df: DataFrame of flight states.

    Returns:
        Aggregated DataFrame with flight counts.
    """
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby("origin_country").agg(
        flight_count=("icao24", "nunique"),
        avg_altitude=("altitude", "mean"),
        avg_velocity=("velocity", "mean"),
        grounded_count=("on_ground", "sum"),
    ).reset_index()

    agg["timestamp"] = datetime.utcnow().isoformat()
    return agg


def collect() -> pd.DataFrame:
    """Main collection function."""
    # Setup paths relative to project root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "aviation")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = time.time()
    all_states: List[Dict] = []
    errors = 0

    logger.info("Starting OpenSky aviation data collection...")

    for region_name, bbox in REGIONS.items():
        try:
            states = fetch_flight_states(**bbox)
            normalized = [normalize_state(s) for s in states]
            all_states.extend(normalized)
            logger.info(f"  {region_name}: {len(states)} flight states")
        except Exception as e:
            errors += 1
            logger.error(f"  {region_name}: {e}")

    if not all_states:
        logger.warning("No flight states collected.")
        return pd.DataFrame()

    # Build raw flight states DataFrame
    df = pd.DataFrame(all_states)
    df = remove_duplicates(df, subset=["icao24", "callsign"])
    df = handle_missing_fields(df)

    # Aggregate by country
    df_agg = aggregate_by_country(df)

    # Save raw states
    today = datetime.now().strftime("%Y_%m_%d")
    raw_path = os.path.join(OUTPUT_DIR, f"opensky_states_{today}.parquet")
    df.to_parquet(raw_path, index=False, engine="pyarrow")
    log_dataset_size(logger, "OpenSky_States", raw_path)

    # Save aggregated data
    agg_path = os.path.join(OUTPUT_DIR, f"opensky_aggregated_{today}.parquet")
    df_agg.to_parquet(agg_path, index=False, engine="pyarrow")
    log_dataset_size(logger, "OpenSky_Aggregated", agg_path)

    log_execution_metrics(logger, "OpenSky", len(df), start_time, errors)

    return df


if __name__ == "__main__":
    df = collect()
    if not df.empty:
        print(f"\nCollected {len(df)} flight states")
        print(df.head(10))
    else:
        print("No data collected.")
