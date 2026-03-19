"""
Multi-Domain Data Collection Pipeline
======================================
Orchestrates all domain collectors, validates datasets,
generates features, and stores results as parquet.

Steps:
1. collect_all_data() — run all domain collectors
2. validate_datasets() — validate collected data integrity
3. generate_features() — compute derived feature indices
4. store_parquet() — final storage verification
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.utils.logging_utils import get_logger, log_execution_metrics

logger = get_logger("DataPipeline")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Import all collectors
COLLECTORS = {
    # ─── Legacy Collectors (script-based, run via subprocess) ─────────────
    # These collectors run as standalone scripts and save CSVs to data/raw/
}

LEGACY_SCRIPT_COLLECTORS = [
    # Social Media
    "src/data_collection/social_media/hackernews_collector.py",
    "src/data_collection/social_media/stackexchange_collector.py",
    "src/data_collection/social_media/github_collector.py",
    "src/data_collection/social_media/mastodon_collector.py",
    "src/data_collection/social_media/youtube_collector.py",
    "src/data_collection/social_media/serpapi_trends_collector.py",
    "src/data_collection/social_media/google_trends_collector.py",
    # News
    "src/data_collection/news/newsapi_collector.py",
    "src/data_collection/news/gdelt_collector.py",
    # Finance
    "src/data_collection/finance/yahoo_finance_collector.py",
    "src/data_collection/finance/alpha_vantage_collector.py",
    "src/data_collection/finance/coingecko_collector.py",
    # Economy
    "src/data_collection/economy/fred_collector.py",
    # Weather
    "src/data_collection/weather/openweather_collector.py",
]

# ─── New Domain Collectors (module-based, run via importlib) ─────────
NEW_DOMAIN_COLLECTORS = {
    # Trade
    "UN Comtrade": "src.data_collection.trade.un_comtrade_collector",
    "World Bank": "src.data_collection.trade.worldbank_collector",
    "OECD": "src.data_collection.trade.oecd_collector",
    # Research
    "ArXiv": "src.data_collection.research.arxiv_collector",
    # Patents
    "NASA": "src.data_collection.patents.nasa_patent_collector",
    # Energy
    "EIA": "src.data_collection.energy.eia_collector",
    # Aviation
    "OpenSky": "src.data_collection.aviation.opensky_collector",
    # Jobs
    "Adzuna": "src.data_collection.jobs.adzuna_collector",
    "USAJobs": "src.data_collection.jobs.usajobs_collector",
    # Population
    "UN Population": "src.data_collection.population.un_population_collector",
    # Crypto
    "Blockchain": "src.data_collection.crypto.blockchain_collector",
}


def is_data_already_collected(collector_name: str) -> bool:
    """
    Check if data for the current date and collector already exists in data/raw/.
    Handles both legacy script names and new domain names.
    """
    import glob
    today_str = datetime.now().strftime("%Y-%m-%d")
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")

    # Search pattern for files with collector name and today's date
    # Format usually: {name}_{date}.csv or {name}_{date}.parquet
    # We use a broad glob to match various structures within data/raw/
    patterns = [
        os.path.join(raw_dir, "**", f"*{collector_name}*{today_str}*"),
        os.path.join(raw_dir, "**", f"*{today_str}*{collector_name}*")
    ]

    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            # Verify it's a file and not empty (optional, but safer)
            for f in files:
                if os.path.isfile(f) and os.path.getsize(f) > 100: # Ignore tiny files (headers only)
                    logger.info(f"⏭️ Skipping {collector_name}: data already exists ({os.path.basename(f)})")
                    return True
    return False


def collect_all_data() -> Dict[str, Tuple[bool, int]]:
    """
    Run all data collectors — both legacy script-based and new module-based.

    Returns:
        Dict mapping collector name → (success: bool, record_count: int).
    """
    import importlib
    import subprocess

    from src.utils.pipeline_utils import run_script

    results: Dict[str, Tuple[bool, int]] = {}
    logger.info("=" * 60)
    logger.info("PHASE 1A: LEGACY SCRIPT-BASED DATA COLLECTION")
    logger.info("=" * 60)

    for script_path in LEGACY_SCRIPT_COLLECTORS:
        name = os.path.basename(script_path).replace('.py', '')
        search_name = name.replace('_collector', '')

        # Smart Skipping Check
        if is_data_already_collected(search_name):
            results[name] = (True, 0)
            continue

        abs_path = os.path.join(PROJECT_ROOT, script_path)
        logger.info(f"\n--- Running: {name} ---")
        try:
            if os.path.exists(abs_path):
                success = run_script(abs_path)
                results[name] = (bool(success), 0)
                status = "✅" if success else "⚠️"
                logger.info(f"{status} {name}: completed")
            else:
                results[name] = (False, 0)
                logger.warning(f"⚠️ {name}: script not found at {abs_path}")
        except Exception as e:
            results[name] = (False, 0)
            logger.error(f"❌ {name}: {e}", exc_info=True)

    legacy_ok = sum(1 for s, _ in results.values() if s)
    logger.info(f"\nLegacy Collection: {legacy_ok}/{len(LEGACY_SCRIPT_COLLECTORS)} succeeded")

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1B: NEW DOMAIN MODULE-BASED DATA COLLECTION")
    logger.info("=" * 60)

    for name, module_path in NEW_DOMAIN_COLLECTORS.items():
        # Smart Skipping Check (using a lowercase/slug version of the name)
        name_slug = name.lower().replace(" ", "_")
        if is_data_already_collected(name_slug):
            results[name] = (True, 0)
            continue

        logger.info(f"\n--- Collecting: {name} ---")
        try:
            module = importlib.import_module(module_path)
            df = module.collect()
            record_count = len(df) if df is not None and not df.empty else 0
            results[name] = (True, record_count)
            logger.info(f"✅ {name}: {record_count:,} records collected")
        except Exception as e:
            results[name] = (False, 0)
            logger.error(f"❌ {name}: {e}", exc_info=True)

    # Summary
    successful = sum(1 for s, _ in results.values() if s)
    total_records = sum(c for _, c in results.values())
    total_collectors = len(LEGACY_SCRIPT_COLLECTORS) + len(NEW_DOMAIN_COLLECTORS)
    logger.info(f"\nCollection Summary: {successful}/{total_collectors} collectors succeeded")
    logger.info(f"Total records collected: {total_records:,}")

    return results


def validate_datasets() -> Dict[str, bool]:
    """
    Validate all collected datasets in data/raw/.

    Returns:
        Dict mapping domain → validation result.
    """
    import glob

    logger.info("=" * 60)
    logger.info("PHASE 2: DATA VALIDATION")
    logger.info("=" * 60)

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    domains = [
        "trade", "research", "patents", "energy",
        "aviation", "jobs", "population", "blockchain", "satellite",
    ]

    results: Dict[str, bool] = {}

    for domain in domains:
        domain_dir = os.path.join(raw_dir, domain)
        parquet_files = glob.glob(os.path.join(domain_dir, "*.parquet"))

        if not parquet_files:
            logger.warning(f"  {domain}: No parquet files found")
            results[domain] = False
            continue

        domain_valid = True
        total_records = 0

        for f in parquet_files:
            try:
                df = pd.read_parquet(f)
                total_records += len(df)

                # Basic validation
                if df.empty:
                    logger.warning(f"  {domain}/{os.path.basename(f)}: empty file")
                    domain_valid = False
                elif len(df.columns) < 2:
                    logger.warning(f"  {domain}/{os.path.basename(f)}: insufficient columns")
                    domain_valid = False
            except Exception as e:
                logger.error(f"  {domain}/{os.path.basename(f)}: {e}")
                domain_valid = False

        results[domain] = domain_valid
        status = "✅" if domain_valid else "⚠️"
        logger.info(f"  {status} {domain}: {len(parquet_files)} files, {total_records:,} records")

    return results


def generate_features() -> bool:
    """
    Run feature generation pipeline.

    Returns:
        True if feature generation succeeded.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: FEATURE GENERATION")
    logger.info("=" * 60)

    try:
        from src.feature_engineering.feature_generator import generate_all_features
        results = generate_all_features()
        successful = sum(1 for v in results.values() if v is not None)
        logger.info(f"Feature generation: {successful}/{len(results)} features computed")
        return successful > 0
    except Exception as e:
        logger.error(f"Feature generation failed: {e}", exc_info=True)
        return False


def store_parquet() -> None:
    """
    Verify and log final parquet storage status.
    """
    import glob

    logger.info("=" * 60)
    logger.info("PHASE 4: STORAGE VERIFICATION")
    logger.info("=" * 60)

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    features_dir = os.path.join(PROJECT_ROOT, "data", "features")

    # Count raw data files
    raw_files = glob.glob(os.path.join(raw_dir, "**", "*.parquet"), recursive=True)
    total_raw_size = sum(os.path.getsize(f) for f in raw_files) / (1024 * 1024)

    # Count feature files
    feature_files = glob.glob(os.path.join(features_dir, "*.parquet"))
    total_feature_size = sum(os.path.getsize(f) for f in feature_files) / (1024 * 1024)

    logger.info(f"Raw data: {len(raw_files)} files ({total_raw_size:.2f} MB)")
    logger.info(f"Features: {len(feature_files)} files ({total_feature_size:.2f} MB)")
    logger.info(f"Total storage: {total_raw_size + total_feature_size:.2f} MB")


def main() -> None:
    """
    Execute the full data pipeline.
    """
    pipeline_start = time.time()
    logger.info("=" * 60)
    logger.info("MULTI-DOMAIN DATA COLLECTION PIPELINE")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Phase 1: Collect data
    collection_results = collect_all_data()

    # Phase 2: Validate datasets
    validation_results = validate_datasets()

    # Phase 3: Generate features
    features_ok = generate_features()

    # Phase 4: Storage verification
    store_parquet()

    # Final summary
    elapsed = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total duration: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("DATA PIPELINE COMPLETED")
    print(f"Duration: {elapsed/60:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
