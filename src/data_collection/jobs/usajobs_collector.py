"""
USAJobs Data Collector
======================
Collects federal job postings from the USAJobs API.

API Reference: https://developer.usajobs.gov/API-Reference
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any

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

logger = get_logger("USAJobs")

# Configuration constants
BASE_URL = "https://data.usajobs.gov/api"
JOB_CATEGORIES = [
    "2210",  # IT Management
    "1550",  # Computer Science
    "0801",  # General Engineering
    "1301",  # Physical Sciences
    "0110",  # Economist
    "1530",  # Statistics
    "1560",  # Data Science
]
MAX_RESULTS = 500

# Global rate limiter - moved initialization into collect() to avoid import side-effects
_rate_limiter = None

def get_rate_limiter():
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_calls=15, period=60.0, name="USAJobs")
    return _rate_limiter

@retry(max_retries=3, backoff_factor=2.0)
def search_jobs(
    keyword: str = "",
    job_category: str = "",
    page: int = 1,
    results_per_page: int = 100,
) -> Dict:
    """
    Search federal job postings on USAJobs.
    """
    limiter = get_rate_limiter()
    limiter.acquire()

    API_KEY = os.getenv("USAJOBS_API_KEY", "")
    USER_EMAIL = os.getenv("USAJOBS_EMAIL", "")

    if not API_KEY or not USER_EMAIL:
        return {"SearchResult": {"SearchResultItems": []}}

    client = APIClient(
        base_url=BASE_URL,
        headers={
            "Authorization-Key": API_KEY,
            "User-Agent": USER_EMAIL,
            "Host": "data.usajobs.gov",
        },
        timeout=30,
        max_retries=3,
    )

    params: Dict[str, Any] = {
        "Page": page,
        "ResultsPerPage": min(results_per_page, 500),
    }

    if keyword:
        params["Keyword"] = keyword
    if job_category:
        params["JobCategoryCode"] = job_category

    try:
        return client.get_json("Search", params=params)
    except Exception as e:
        logger.error(f"Error searching USAJobs: {e}")
        return {"SearchResult": {"SearchResultItems": []}}
    finally:
        client.close()


def normalize_job(item: Dict) -> Dict:
    """
    Normalize a USAJobs result item.

    Args:
        item: Raw API result item.

    Returns:
        Normalized record dict.
    """
    match_data = item.get("MatchedObjectDescriptor", {})

    # Extract salary
    remuneration = match_data.get("PositionRemuneration", [])
    salary_min = ""
    salary_max = ""
    if remuneration and isinstance(remuneration, list):
        r = remuneration[0]
        salary_min = r.get("MinimumRange", "")
        salary_max = r.get("MaximumRange", "")

    # Extract location
    location_data = match_data.get("PositionLocation", [])
    location = ""
    if location_data and isinstance(location_data, list):
        loc = location_data[0]
        location = loc.get("LocationName", "")

    # Extract category
    categories = match_data.get("JobCategory", [])
    category = ""
    if categories and isinstance(categories, list):
        category = categories[0].get("Name", "")

    return {
        "job_title": match_data.get("PositionTitle", ""),
        "job_category": category,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_range": f"{salary_min}-{salary_max}" if salary_min and salary_max else "",
        "location": location,
        "organization": match_data.get("OrganizationName", ""),
        "department": match_data.get("DepartmentName", ""),
        "posting_date": match_data.get("PublicationStartDate", ""),
        "closing_date": match_data.get("ApplicationCloseDate", ""),
        "position_type": match_data.get("PositionSchedule", [{}])[0].get("Name", "") if match_data.get("PositionSchedule") else "",
        "url": match_data.get("PositionURI", ""),
    }


def collect() -> pd.DataFrame:
    """Main collection function."""
    # Setup paths relative to project root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "jobs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    API_KEY = os.getenv("USAJOBS_API_KEY", "")
    USER_EMAIL = os.getenv("USAJOBS_EMAIL", "")

    start_time = time.time()
    all_jobs: List[Dict] = []
    errors = 0

    if not API_KEY or not USER_EMAIL:
        logger.warning("USAJOBS_API_KEY/USAJOBS_EMAIL not set in .env. Skipping.")
        return pd.DataFrame()

    logger.info("Starting USAJobs data collection...")

    for category_code in JOB_CATEGORIES:
        try:
            result = search_jobs(job_category=category_code, results_per_page=100)
            items = result.get("SearchResult", {}).get("SearchResultItems", [])
            normalized = [normalize_job(item) for item in items]
            all_jobs.extend(normalized)
            logger.info(f"  Category {category_code}: {len(items)} jobs")
        except Exception as e:
            errors += 1
            logger.error(f"  Category {category_code}: {e}")

    if not all_jobs:
        logger.warning("No USAJobs postings collected.")
        return pd.DataFrame()

    df = pd.DataFrame(all_jobs)
    df = remove_duplicates(df, subset=["job_title", "organization", "posting_date"])
    df = handle_missing_fields(df, fill_values={"salary_range": "Not specified"})
    df = standardize_timestamps(df, "posting_date")

    today = datetime.now().strftime("%Y_%m_%d")
    output_path = os.path.join(OUTPUT_DIR, f"usajobs_{today}.parquet")
    df.to_parquet(output_path, index=False, engine="pyarrow")
    log_dataset_size(logger, "USAJobs", output_path)
    log_execution_metrics(logger, "USAJobs", len(df), start_time, errors)

    return df


if __name__ == "__main__":
    df = collect()
    if not df.empty:
        print(f"\nCollected {len(df)} federal job postings")
        print(df[["job_title", "job_category", "salary_range", "location"]].head(10))
    else:
        print("No data collected. Ensure USAJOBS_API_KEY/USAJOBS_EMAIL are set in .env")
