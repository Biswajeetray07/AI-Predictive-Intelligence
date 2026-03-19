#!/usr/bin/env python3
"""
Run Phase 7 (Evaluation & Backtesting) Only
=============================================
Evaluates all trained models and generates performance metrics.
"""

import os
import sys
import time
import torch
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_full_pipeline import get_device, phase_7_evaluation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("TrainingRunner")

def main():
    phase_start = time.time()
    
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  PHASE 7: EVALUATION & BACKTESTING                              ║")
    logger.info(f"║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                   ║")
    logger.info("╚" + "═" * 68 + "╝")
    
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Phase 7: Evaluation
    try:
        phase_7_evaluation(device)
        logger.info("\n✅ PHASE 7 COMPLETED SUCCESSFULLY!")
    except Exception as e:
        logger.error(f"❌ Phase 7 failed with error: {e}")
        raise
    
    elapsed = time.time() - phase_start
    logger.info("\n" + "═" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Total duration: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info("═" * 70)

if __name__ == '__main__':
    main()
