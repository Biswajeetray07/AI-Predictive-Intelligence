"""
Adzuna Job Market Data Collector
=================================
Collects job listings from the Adzuna API.

API Reference: https://developer.adzuna.com/
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

logger = get_logger("Adzuna")

# Configuration constants
BASE_URL = "https://api.adzuna.com/v1/api/jobs"
COUNTRIES = ["us", "gb", "de", "fr", "ca", "au"]
CATEGORIES = [
    "it-jobs",
    "engineering-jobs",
    "scientific-qa-jobs",
    "healthcare-nursing-jobs",
    "finance-jobs",
]
MAX_RESULTS_PER_CATEGORY = 200

# Global rate limiter - moved initialization into collect() to avoid import side-effects
_rate_limiter = None

def get_rate_limiter():
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_calls=10, period=60.0, name="Adzuna")
    return _rate_limiter

@retry(max_retries=3, backoff_factor=2.0)
def search_jobs(
    country: str,
    category: str,
    page: int = 1,
    results_per_page: int = 50,
) -> Dict:
    """
    Search job listings on Adzuna.
    """
    limiter = get_rate_limiter()
    limiter.acquire()

    APP_ID = os.getenv("ADZUNA_APP_ID", "")
    APP_KEY = os.getenv("ADZUNA_APP_KEY", "")

    if not APP_ID or not APP_KEY:
        return {"results": []}

    client = APIClient(
        base_url=BASE_URL,
        timeout=30,
        max_retries=3,
    )

    endpoint = f"{country}/search/{page}"
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": results_per_page,
        "category": category,
        "content-type": "application/json",
    }

    try:
        return client.get_json(endpoint, params=params)
    except Exception as e:
        logger.error(f"Error searching Adzuna {country}/{category}: {e}")
        return {"results": []}
    finally:
        client.close()


def normalize_job(job: Dict, country: str) -> Dict:
    """
    Normalize an Adzuna job record.

    Args:
        job: Raw API job record.
        country: Country code.

    Returns:
        Normalized record dict.
    """
    salary_min = job.get("salary_min")
    salary_max = job.get("salary_max")
    salary_range = ""
    if salary_min and salary_max:
        salary_range = f"{salary_min:.0f}-{salary_max:.0f}"
    elif salary_min:
        salary_range = f"{salary_min:.0f}+"

    location = job.get("location", {})
    location_parts = location.get("display_name", "") if isinstance(location, dict) else ""

    return {
        "job_title": job.get("title", ""),
        "company": job.get("company", {}).get("display_name", "") if isinstance(job.get("company"), dict) else "",
        "salary_range": salary_range,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "location": location_parts,
        "country": country.upper(),
        "category": job.get("category", {}).get("label", "") if isinstance(job.get("category"), dict) else "",
        "description": (job.get("description") or "")[:500],
        "created_date": job.get("created", ""),
        "contract_type": job.get("contract_type", ""),
        "contract_time": job.get("contract_time", ""),
    }


def collect() -> pd.DataFrame:
    """Main collection function."""
    # Setup paths relative to project root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "jobs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    APP_ID = os.getenv("ADZUNA_APP_ID", "")
    APP_KEY = os.getenv("ADZUNA_APP_KEY", "")

    start_time = time.time()
    all_jobs: List[Dict] = []
    errors = 0

    if not APP_ID or not APP_KEY:
        logger.warning("ADZUNA_APP_ID/ADZUNA_APP_KEY not set in .env. Skipping.")
        return pd.DataFrame()

    logger.info("Starting Adzuna job data collection...")

    for country in COUNTRIES:
        for category in CATEGORIES:
            try:
                category_jobs = []
                for page in range(1, (MAX_RESULTS_PER_CATEGORY // 50) + 1):
                    result = search_jobs(country, category, page=page)
                    jobs = result.get("results", [])
                    if not jobs:
                        break
                    normalized = [normalize_job(j, country) for j in jobs]
                    category_jobs.extend(normalized)

                all_jobs.extend(category_jobs)
                logger.info(f"  {country}/{category}: {len(category_jobs)} jobs")
            except Exception as e:
                errors += 1
                logger.error(f"  {country}/{category}: {e}")

    if not all_jobs:
        logger.warning("No Adzuna jobs collected.")
        return pd.DataFrame()

    df = pd.DataFrame(all_jobs)
    df = remove_duplicates(df, subset=["job_title", "company", "location"])
    df = handle_missing_fields(df, fill_values={"salary_range": "Not specified"})
    df = standardize_timestamps(df, "created_date")

    today = datetime.now().strftime("%Y_%m_%d")
    output_path = os.path.join(OUTPUT_DIR, f"adzuna_{today}.parquet")
    df.to_parquet(output_path, index=False, engine="pyarrow")
    log_dataset_size(logger, "Adzuna", output_path)
    log_execution_metrics(logger, "Adzuna", len(df), start_time, errors)

    return df


if __name__ == "__main__":
    df = collect()
    if not df.empty:
        print(f"\nCollected {len(df)} job listings")
        print(df[["job_title", "company", "location", "salary_range"]].head(10))
    else:
        print("No data collected. Ensure ADZUNA_APP_ID/ADZUNA_APP_KEY are set in .env")
