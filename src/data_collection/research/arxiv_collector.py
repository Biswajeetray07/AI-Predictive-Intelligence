"""
arXiv Research Paper Collector
===============================
Collects research papers from arXiv API across AI/ML categories.

API Reference: https://info.arxiv.org/help/api/index.html
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List
import xml.etree.ElementTree as ET

import pandas as pd
from dotenv import load_dotenv

# Setup paths
from src.utils.api_client import APIClient
from src.utils.rate_limiter import RateLimiter
from src.utils.retry_handler import retry
from src.utils.data_validator import (
    remove_duplicates,
    handle_missing_fields,
    standardize_timestamps,
)
from src.utils.logging_utils import get_logger, log_execution_metrics, log_dataset_size

logger = get_logger("ArXiv")

# Configuration constants
BASE_URL = "http://export.arxiv.org/api"
CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"]
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
MAX_RESULTS_PER_QUERY = 500
MAX_TOTAL_RESULTS = 2000

# Global rate limiter - moved initialization into collect() to avoid import side-effects
_rate_limiter = None

def get_rate_limiter():
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_calls=3, period=60.0, name="ArXiv")
    return _rate_limiter

@retry(max_retries=3, backoff_factor=5.0)
def fetch_arxiv_papers(
    category: str,
    start: int = 0,
    max_results: int = MAX_RESULTS_PER_QUERY,
) -> List[Dict]:
    """
    Fetch papers from arXiv API for a given category.
    """
    limiter = get_rate_limiter()
    limiter.acquire()

    client = APIClient(
        base_url=BASE_URL,
        timeout=60,
        max_retries=3,
        rate_limit_calls=3,
        rate_limit_period=60.0,
    )

    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    try:
        response = client.get("query", params=params)
        return _parse_arxiv_xml(response.text, category)
    except Exception as e:
        logger.error(f"Error fetching arXiv {category}: {e}")
        return []
    finally:
        client.close()


def _parse_arxiv_xml(xml_text: str, category: str) -> List[Dict]:
    """
    Parse arXiv Atom XML response into flat records.

    Args:
        xml_text: Raw XML response.
        category: Category being queried.

    Returns:
        List of paper dicts.
    """
    papers = []

    try:
        root = ET.fromstring(xml_text)
        entries = root.findall(f"{ATOM_NS}entry")

        for entry in entries:
            title_elem = entry.find(f"{ATOM_NS}title")
            summary_elem = entry.find(f"{ATOM_NS}summary")
            published_elem = entry.find(f"{ATOM_NS}published")
            id_elem = entry.find(f"{ATOM_NS}id")

            # Extract authors
            authors = []
            for author in entry.findall(f"{ATOM_NS}author"):
                name_elem = author.find(f"{ATOM_NS}name")
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())

            # Extract categories
            categories = []
            for cat in entry.findall(f"{ATOM_NS}category"):
                term = cat.get("term", "")
                if term:
                    categories.append(term)

            paper = {
                "arxiv_id": id_elem.text.strip().split("/abs/")[-1] if id_elem is not None and id_elem.text else "",
                "title": title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else "",
                "authors": "; ".join(authors),
                "abstract": summary_elem.text.strip().replace("\n", " ")[:1000] if summary_elem is not None and summary_elem.text else "",
                "category": category,
                "all_categories": ", ".join(categories),
                "published_date": published_elem.text.strip() if published_elem is not None and published_elem.text else "",
            }
            papers.append(paper)

    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")

    return papers


def collect() -> pd.DataFrame:
    """Main collection function."""
    # Setup paths relative to project root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "research")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = time.time()
    all_papers: List[Dict] = []
    errors = 0

    logger.info("Starting arXiv data collection...")
    logger.info(f"Categories: {CATEGORIES}")

    for category in CATEGORIES:
        try:
            papers_in_category = []
            for start_idx in range(0, MAX_TOTAL_RESULTS, MAX_RESULTS_PER_QUERY):
                papers = fetch_arxiv_papers(
                    category=category,
                    start=start_idx,
                    max_results=MAX_RESULTS_PER_QUERY,
                )
                papers_in_category.extend(papers)
                logger.info(
                    f"  {category}: fetched {len(papers)} papers "
                    f"(total: {len(papers_in_category)})"
                )
                if len(papers) < MAX_RESULTS_PER_QUERY:
                    break  # No more pages

            all_papers.extend(papers_in_category)
        except Exception as e:
            errors += 1
            logger.error(f"  {category}: {e}")

    if not all_papers:
        logger.warning("No arXiv papers collected.")
        return pd.DataFrame()

    # Build DataFrame
    df = pd.DataFrame(all_papers)
    df = remove_duplicates(df, subset=["arxiv_id"])
    df = handle_missing_fields(df, fill_values={"abstract": "", "authors": ""})
    df = standardize_timestamps(df, "published_date")

    # Save as parquet
    today = datetime.now().strftime("%Y_%m_%d")
    output_path = os.path.join(OUTPUT_DIR, f"arxiv_{today}.parquet")
    df.to_parquet(output_path, index=False, engine="pyarrow")
    log_dataset_size(logger, "ArXiv", output_path)
    log_execution_metrics(logger, "ArXiv", len(df), start_time, errors)

    return df


if __name__ == "__main__":
    df = collect()
    if not df.empty:
        print(f"\nCollected {len(df)} arXiv papers")
        print(df[["title", "category", "published_date"]].head(10))
    else:
        print("No data collected.")
