import os
import sys
import shutil
import logging

from dotenv import load_dotenv
load_dotenv()

# Setup project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.pipeline_utils import run_script

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_optimized_pipeline():
    logging.info("Starting Optimized Final Training Pipeline")
    
    config_dir = os.path.join(PROJECT_ROOT, 'configs')
    active_config = os.path.join(config_dir, 'training_config.yaml')
    best_config = os.path.join(config_dir, 'best_training_config.yaml')
    backup_config = os.path.join(config_dir, 'training_config.yaml.bak')
    
    if not os.path.exists(best_config):
        logging.error(f"Optimal configuration not found at {best_config}")
        logging.error("Please run `python optimization/run_hyperopt.py` first.")
        sys.exit(1)
        
    # ─── 1. Inject Optimized Config ───────────────────────────────────────────
    # We backup the original config and replace it with our optimized one
    # so the existing robust training scripts automatically inherit the best params.
    if os.path.exists(active_config):
        shutil.copy(active_config, backup_config)
        logging.info("Backed up default configuration.")
        
    shutil.copy(best_config, active_config)
    logging.info("Injected best hyperparameters into active configuration.")
    
    try:
        # ─── 2. Execute Full Training Pipeline ────────────────────────────────
        pipeline_steps = [
            "src/training/timeseries/train.py", 
            "src/training/nlp/train.py",
            "src/training/fusion/train.py"
        ]

        for step in pipeline_steps:
            abs_path = os.path.join(PROJECT_ROOT, step)
            if os.path.exists(abs_path):
                logging.info(f"\n========== RUNNING OPTIMIZED {step} ==========")
                success = run_script(abs_path)
                if not success:
                    logging.error(f"Optimized Pipeline failed at step: {step}. Exiting.")
                    sys.exit(1)
            else:
                logging.error(f"Required script not found at: {abs_path}")
                sys.exit(1)

        logging.info("\n=============================================")
        logging.info("OPTIMIZED END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("=============================================\n")

    finally:
        # ─── 3. Cleanup & Restore ─────────────────────────────────────────────
        if os.path.exists(backup_config):
            shutil.copy(backup_config, active_config)
            os.remove(backup_config)
            logging.info("Restored default training configuration.")
            
if __name__ == "__main__":
    run_optimized_pipeline()
