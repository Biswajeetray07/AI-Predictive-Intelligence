import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PROJECT_ROOT is the container of 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.pipeline_utils import run_script

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting AI-Predictive-Intelligence End-to-End Pipeline")

    pipeline_steps = {
        "Legacy Data Collection": [
            "src/data_collection/finance/yahoo_finance_collector.py",
            "src/data_collection/news/newsapi_collector.py",
            "src/data_collection/news/gdelt_collector.py",
            "src/data_collection/social_media/hackernews_collector.py",
            "src/data_collection/social_media/stackexchange_collector.py",
            "src/data_collection/social_media/github_collector.py",
            "src/data_collection/social_media/mastodon_collector.py",
            "src/data_collection/social_media/youtube_collector.py",
            "src/data_collection/social_media/serpapi_trends_collector.py",
            "src/data_collection/finance/alpha_vantage_collector.py",
            "src/data_collection/finance/coingecko_collector.py",
            "src/data_collection/economy/fred_collector.py",
            "src/data_collection/weather/openweather_collector.py",
        ],
        "Data Pre-Processing": [
            "src/data_processing/financial_processing.py",
            "src/data_processing/news_processing.py",
            "src/data_processing/social_processing.py",
            "src/data_processing/weather_processing.py",
            "src/data_processing/macro_processing.py",
            "src/data_processing/external_processing.py",
        ],
        "New Domain Feature Generation": [
            "src/feature_engineering/feature_generator.py",
        ],
        "Dataset Merging": [
            "src/data_processing/merge_datasets.py",
        ],
        "Feature Engineering": [
            "src/feature_engineering/build_features.py",
        ],
        "Market Regime Detection": [
            "src/feature_engineering/regime_detection/regime_features.py",
        ],
        "Feature Selection": [
            "src/feature_engineering/feature_selection.py",
        ],
        "Sequence Building": [
            "src/data_processing/build_sequences.py",
        ],
        "Model Training": [
            "src/training/timeseries/train.py",
            "src/training/nlp/train.py",
            "src/training/fusion/train.py"
        ],
        "Walk-Forward Validation": [
            "src/training/optimization/walk_forward.py"
        ]
    }

    use_walk_forward = "--walk-forward" in sys.argv
    if use_walk_forward:
        pipeline_steps.pop("Model Training")
        logging.info("Walk-Forward Validation mode enabled. Standard Model Training will be skipped.")
    else:
        pipeline_steps.pop("Walk-Forward Validation")

    for phase_name, steps in pipeline_steps.items():
        logging.info(f"\nPhase: {phase_name} ------------------------")
        for step in steps:
            abs_path = os.path.join(PROJECT_ROOT, step)
            if os.path.exists(abs_path):
                logging.info(f"========== RUNNING {step} ==========")
                success = run_script(abs_path)
                if not success:
                    # Non-critical phases can be skipped
                    if phase_name in ("Market Regime Detection", "Feature Selection"):
                        logging.warning(f"Optional step failed: {step}. Continuing pipeline.")
                        continue
                    logging.error(f"Pipeline failed at step: {step}. Exiting.")
                    sys.exit(1)
            else:
                if phase_name in ("Market Regime Detection", "Feature Selection"):
                    logging.warning(f"Optional script not found: {abs_path}. Skipping.")
                else:
                    logging.error(f"Required script not found at: {abs_path}")
                    sys.exit(1)

    logging.info("\n=============================================")
    logging.info("END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
    logging.info("=============================================\n")



if __name__ == "__main__":
    main()
