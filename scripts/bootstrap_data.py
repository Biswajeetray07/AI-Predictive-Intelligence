"""
Bootstrap Data Collection Script
================================
This script bypasses the 24/7 scheduler and immediately forces
every configured data collector to run right now.

Use this to initialize your database and fill `data/raw/` 
with your first batch of historical data.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logging_utils import get_logger

logger = get_logger("Bootstrap")

def run_all_collectors() -> None:
    logger.info("=" * 60)
    logger.info("STARTING: ONE-TIME BOOTSTRAP DATA COLLECTION")
    logger.info("=" * 60)

    # 1. Social & Real-time Collectors
    logger.info("\n[Phase 1/4] Running Social & Real-time Collectors...")
    try:
        from scripts.advanced_scheduler import run_social_collectors
        run_social_collectors()
    except Exception as e:
        logger.error(f"Social collection failed: {e}")

    # 2. Economic Collectors
    logger.info("\n[Phase 2/4] Running Economic Collectors...")
    try:
        from scripts.advanced_scheduler import run_economic_collectors
        run_economic_collectors()
    except Exception as e:
        logger.error(f"Economic collection failed: {e}")

    # 3. Research & Patent Collectors
    logger.info("\n[Phase 3/4] Running Research & Patent Collectors...")
    try:
        from scripts.advanced_scheduler import run_research_collectors
        run_research_collectors()
    except Exception as e:
        logger.error(f"Research collection failed: {e}")

    # 4. Feature Generation
    logger.info("\n[Phase 4/4] Running Automated Feature Generation...")
    try:
        from scripts.advanced_scheduler import run_feature_generation
        run_feature_generation()
    except Exception as e:
        logger.error(f"Feature generation failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("BOOTSTRAP DATA COLLECTION COMPLETE!")
    logger.info("Check data/raw/ and data/features/ for your new datasets.")
    logger.info("=" * 60)


if __name__ == "__main__":
    print("\nWarning: This will immediately trigger ALL configured API requests.")
    print("Depending on your internet speed and API rate limits, this may take several minutes.")
    
    # Prompt the user just in case
    response = input("\nDo you want to run all collectors right now? (y/n): ")
    if response.lower() == 'y':
        run_all_collectors()
    else:
        print("Initialization cancelled.")
