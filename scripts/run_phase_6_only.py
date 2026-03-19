#!/usr/bin/env python3
"""
Run Phase 6 (Full Training) Only
==================================
Runs TS Ensemble, NLP, and Fusion model training directly.
Assumes data is already prepared from earlier phases.
"""

import os
import sys
import time
import argparse
import torch
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_full_pipeline import get_device, phase_6_full_training

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("TrainingRunner")

def main():
    parser = argparse.ArgumentParser(description="AI Predictive Intelligence — Phase 6 Only (Full Training)")
    parser.add_argument('--hpo', action='store_true', help="Enable Fusion HPO (Phase 6B.5)")
    parser.add_argument('--hpo-trials', type=int, default=5, help="Number of HPO trials for Fusion (default: 5)")
    args = parser.parse_args()

    pipeline_start = time.time()
    
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  PHASE 6: FULL MODEL TRAINING ONLY                               ║")
    logger.info(f"║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                   ║")
    logger.info(f"║  Fusion HPO: {'ENABLED' if args.hpo else 'DISABLED'}                                         ║")
    logger.info("╚" + "═" * 68 + "╝")
    
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Phase 6: Full Training (TS + NLP + Fusion)
    try:
        phase_6_full_training(device, run_fusion_hpo=args.hpo, n_trials=args.hpo_trials)
        logger.info("\n✅ PHASE 6 COMPLETED SUCCESSFULLY!")
    except Exception as e:
        logger.error(f"❌ Phase 6 failed with error: {e}")
        raise
    
    elapsed = time.time() - pipeline_start
    logger.info("\n" + "═" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Total duration: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info("═" * 70)

if __name__ == '__main__':
    main()
