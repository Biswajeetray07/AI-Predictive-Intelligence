#!/usr/bin/env python3
"""
Custom Training Orchestrator
=================================
Runs the pipeline from Phase 2 (Processing) onwards.
- By default: skips HPO entirely (fastest path to real embeddings)
- With --hpo: runs TS HPO (Phase 5) + Fusion HPO (Phase 6B.5, on real data)
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

from scripts.run_full_pipeline import (
    get_device,
    phase_2_data_processing,
    phase_3_feature_engineering,
    phase_4_sequence_generation,
    phase_5_hyperparameter_optimization,
    phase_6_full_training,
    phase_7_evaluation,
    phase_8_regime_detection
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("TrainingRunner")

def main():
    parser = argparse.ArgumentParser(description="AI Predictive Intelligence — Training-Only Pipeline")
    parser.add_argument('--hpo', action='store_true', help="Enable Hyperparameter Optimization (TS in Phase 5, Fusion in Phase 6B.5)")
    parser.add_argument('--hpo-trials', type=int, default=5, help="Number of HPO trials per model (default: 5)")
    args = parser.parse_args()

    pipeline_start = time.time()
    
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  AI PREDICTIVE INTELLIGENCE — TRAINING-ONLY PIPELINE                ║")
    logger.info(f"║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                   ║")
    logger.info(f"║  HPO: {'ENABLED' if args.hpo else 'DISABLED'}                                                ║")
    logger.info("╚" + "═" * 68 + "╝")
    
    device = get_device()
    
    # Phase 2: Data Processing
    phase_2_data_processing()
    
    # Phase 3: Feature Engineering
    phase_3_feature_engineering()
    
    # Phase 4: Sequence Generation
    phase_4_sequence_generation()
    
    # Phase 5: TS Hyperparameter Optimization (opt-in)
    if args.hpo:
        logger.info(f"Running Phase 5: TS HPO with {args.hpo_trials} trials...")
        phase_5_hyperparameter_optimization(device, n_trials=args.hpo_trials)
    else:
        logger.info("⏭️  Skipping Phase 5 (TS HPO) — using existing config. Pass --hpo to enable.")
    
    # Phase 6: Full Training
    # Fusion HPO runs inside Phase 6 (after 6A/6B create real embeddings)
    phase_6_full_training(device, run_fusion_hpo=args.hpo, n_trials=args.hpo_trials)
    
    # Phase 7: Evaluation
    phase_7_evaluation(device)
    
    # Phase 8: Regime Detection
    phase_8_regime_detection()
    
    elapsed = time.time() - pipeline_start
    logger.info("\n" + "═" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info(f"Total duration: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info("═" * 70)

if __name__ == '__main__':
    main()
