"""Phase 1: Import Validation - Tests every module under src/ can be imported."""
import importlib
import sys
import os
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# All modules to test
MODULES = [
    # Top-level packages
    "src",
    "src.data_collection",
    "src.data_processing",
    "src.feature_engineering",
    "src.models",
    "src.training",
    "src.pipelines",
    "src.evaluation",
    "src.utils",
    
    # Data collection
    "src.data_collection.finance.yahoo_finance_collector",
    "src.data_collection.finance.alpha_vantage_collector",
    "src.data_collection.finance.coingecko_collector",
    "src.data_collection.news.newsapi_collector",
    "src.data_collection.news.gdelt_collector",
    "src.data_collection.economy.fred_collector",
    "src.data_collection.weather.openweather_collector",
    "src.data_collection.social_media.github_collector",
    "src.data_collection.social_media.hackernews_collector",
    "src.data_collection.social_media.mastodon_collector",
    "src.data_collection.social_media.serpapi_trends_collector",
    "src.data_collection.social_media.stackexchange_collector",
    "src.data_collection.social_media.youtube_collector",
    "src.data_collection.trade.worldbank_collector",
    "src.data_collection.trade.oecd_collector",
    "src.data_collection.trade.un_comtrade_collector",
    "src.data_collection.satellite.nasa_satellite_collector",
    "src.data_collection.research.arxiv_collector",
    "src.data_collection.research.semantic_scholar_collector",
    "src.data_collection.patents.lens_collector",
    "src.data_collection.patents.uspto_collector",
    "src.data_collection.population.un_population_collector",
    "src.data_collection.aviation.opensky_collector",
    "src.data_collection.jobs.adzuna_collector",
    "src.data_collection.jobs.usajobs_collector",
    "src.data_collection.crypto.blockchain_collector",
    "src.data_collection.energy.eia_collector",
    "src.data_collection.energy.entsoe_collector",

    # Data processing
    "src.data_processing.financial_processing",
    "src.data_processing.news_processing",
    "src.data_processing.social_processing",
    "src.data_processing.weather_processing",
    "src.data_processing.macro_processing",
    "src.data_processing.external_processing",
    "src.data_processing.merge_datasets",
    "src.data_processing.build_sequences",
    "src.data_processing.process_kaggle_data",

    # Feature engineering
    "src.feature_engineering.feature_generator",
    "src.feature_engineering.feature_selection",
    "src.feature_engineering.build_features",
    "src.feature_engineering.regime_detection.regime_features",
    "src.feature_engineering.regime_detection.regime_detector",

    # Models
    "src.models.timeseries.lstm",
    "src.models.timeseries.gru",
    "src.models.timeseries.transformer",
    "src.models.timeseries.tft",
    "src.models.nlp.model",
    "src.models.nlp.tokenizer",
    "src.models.fusion.fusion",

    # Training
    "src.training.timeseries.train",
    "src.training.nlp.train",
    "src.training.fusion.train",
    "src.training.optimization.optuna_ts",
    "src.training.optimization.optuna_nlp",
    "src.training.optimization.optuna_fusion",
    "src.training.optimization.run_hyperopt",
    "src.training.optimization.walk_forward",

    # Evaluation
    "src.evaluation.backtest",
    "src.evaluation.generate_report",
    "src.evaluation.monitoring.retraining_trigger",

    # Pipelines
    "src.pipelines.training_pipeline",
    "src.pipelines.train_optimized_pipeline",
    "src.pipelines.inference_pipeline",
    "src.pipelines.run_data_pipeline",

    # Utils
    "src.utils.pipeline_utils",
    "src.utils.logging_utils",
]

passed = 0
failed = 0
errors = []

print("=" * 60)
print("PHASE 1: IMPORT VALIDATION")
print("=" * 60)

for mod in MODULES:
    try:
        importlib.import_module(mod)
        print(f"  OK  {mod}")
        passed += 1
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        print(f"  FAIL  {mod} -> {err_msg}")
        errors.append((mod, err_msg))
        failed += 1

print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed out of {len(MODULES)}")
print(f"{'='*60}")

if errors:
    print("\nFAILED IMPORTS:")
    for mod, err in errors:
        print(f"  {mod}: {err}")
