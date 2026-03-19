"""
Data Collection Scheduler
==========================
APScheduler-based scheduler for automated data collection.

Schedule:
- Social/Real-time data (Aviation, Blockchain) → every 6 hours
- Economic data (Trade, Energy, Jobs, Population) → daily
- Research + Patents → weekly
"""

import os
import sys
import logging
from datetime import datetime

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

from src.utils.logging_utils import get_logger

logger = get_logger("Scheduler")


def run_social_collectors() -> None:
    """Run social/real-time data collectors (every 6 hours)."""
    logger.info("=== Scheduled: Social/Real-time Data Collection ===")
    collectors = [
        ("OpenSky", "src.data_collection.aviation.opensky_collector"),
        ("Blockchain", "src.data_collection.crypto.blockchain_collector"),
        ("HackerNews", "src.data_collection.social_media.hackernews_collector"),
        ("YouTube", "src.data_collection.social_media.youtube_collector"),
        ("NewsAPI", "src.data_collection.news.newsapi_collector"),
        ("GDELT", "src.data_collection.news.gdelt_collector"),
    ]
    _run_collectors(collectors)


def run_economic_collectors() -> None:
    """Run economic data collectors (daily)."""
    logger.info("=== Scheduled: Economic Data Collection ===")
    collectors = [
        ("World Bank", "src.data_collection.trade.worldbank_collector"),
        ("OECD", "src.data_collection.trade.oecd_collector"),
        ("EIA", "src.data_collection.energy.eia_collector"),
        ("Adzuna", "src.data_collection.jobs.adzuna_collector"),
        ("USAJobs", "src.data_collection.jobs.usajobs_collector"),
        ("UN Population", "src.data_collection.population.un_population_collector"),
    ]
    _run_collectors(collectors)


def run_financial_collectors(limit=None) -> None:
    """Run financial data collectors (daily)."""
    logger.info("=== Scheduled: Financial Data Collection ===")
    import importlib
    try:
        name = "Yahoo Finance"
        module_path = "src.data_collection.finance.yahoo_finance_collector"
        logger.info(f"Running {name}...")
        module = importlib.import_module(module_path)
        # Check if collect takes limit
        df = module.collect(limit=limit)
        count = len(df) if df is not None and not df.empty else 0
        logger.info(f"✅ {name}: {count:,} records")
    except Exception as e:
        logger.error(f"❌ Financial collection failed: {e}")


def run_research_collectors() -> None:
    """Run research and patent collectors (weekly)."""
    logger.info("=== Scheduled: Research & Patent Data Collection ===")
    collectors = [
        ("ArXiv", "src.data_collection.research.arxiv_collector"),
        ("USPTO", "src.data_collection.patents.uspto_collector"),
    ]
    _run_collectors(collectors)


def run_feature_generation() -> None:
    """Run feature generation after data collection."""
    logger.info("=== Scheduled: Feature Generation ===")
    try:
        from src.feature_engineering.feature_generator import generate_all_features
        generate_all_features()
        logger.info("Feature generation completed successfully")
    except Exception as e:
        logger.error(f"Feature generation failed: {e}", exc_info=True)


def _run_collectors(collectors: list) -> None:
    """
    Run a list of collectors sequentially.

    Args:
        collectors: List of (name, module_path) tuples.
    """
    import importlib

    for name, module_path in collectors:
        try:
            logger.info(f"Running {name}...")
            module = importlib.import_module(module_path)
            df = module.collect()
            count = len(df) if df is not None and not df.empty else 0
            logger.info(f"✅ {name}: {count:,} records")
        except Exception as e:
            logger.error(f"❌ {name}: {e}")


def start_scheduler() -> None:
    """
    Initialize and start the APScheduler with all job schedules.
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error(
            "APScheduler not installed. Install with: pip install apscheduler"
        )
        print("ERROR: APScheduler not installed. Run: pip install apscheduler")
        return

    scheduler = BlockingScheduler()

    # Social data → every 6 hours
    scheduler.add_job(
        run_social_collectors,
        trigger=IntervalTrigger(hours=6),
        id="social_data",
        name="Social/Real-time Data Collection",
        replace_existing=True,
    )

    # Economic data → daily at 02:00 UTC
    scheduler.add_job(
        run_economic_collectors,
        trigger=CronTrigger(hour=2, minute=0),
        id="economic_data",
        name="Economic Data Collection",
        replace_existing=True,
    )

    # Research + Patents → weekly on Sunday at 04:00 UTC
    scheduler.add_job(
        run_research_collectors,
        trigger=CronTrigger(day_of_week="sun", hour=4, minute=0),
        id="research_data",
        name="Research & Patent Data Collection",
        replace_existing=True,
    )

    # Feature generation → daily at 06:00 UTC (after economic data)
    scheduler.add_job(
        run_feature_generation,
        trigger=CronTrigger(hour=6, minute=0),
        id="feature_generation",
        name="Feature Generation",
        replace_existing=True,
    )

    logger.info("=" * 60)
    logger.info("DATA COLLECTION SCHEDULER STARTED")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Schedule:")
    logger.info("  • Social/Real-time: every 6 hours")
    logger.info("  • Economic data: daily at 02:00 UTC")
    logger.info("  • Research + Patents: weekly (Sunday 04:00 UTC)")
    logger.info("  • Feature generation: daily at 06:00 UTC")
    logger.info("=" * 60)

    print("\nScheduler started. Press Ctrl+C to exit.\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user.")
        print("\nScheduler stopped.")


if __name__ == "__main__":
    start_scheduler()
