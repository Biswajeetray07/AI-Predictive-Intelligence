"""
Kaggle External Dataset Collector
=================================
Downloads a wide variety of high-quality, real-world datasets from Kaggle
(finance, weather, retail, crypto) and formats them for the AI Predictive 
Intelligence models.

Make sure your kaggle.json is placed in your system's .kaggle folder!
(~/.kaggle/kaggle.json on Linux/Mac or C:\\Users\\<User>\\.kaggle\\kaggle.json on Windows)
"""

import os
import sys
import glob
import logging
import pandas as pd

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

from src.utils.logging_utils import get_logger

logger = get_logger("KaggleCollector")

KAGGLE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "custom_kaggle")
os.makedirs(KAGGLE_DIR, exist_ok=True)

# Define the diverse datasets we want to train our models on
# Format: {"Kaggle Dataset Path": ["File inside to process", "Date Column Name"]}
DATASETS = {
    # 1. Broad Economics & Retail (Store Sales)
    "crawford/80-cereals": ["cereal.csv", None], # Static data example
    "rohitsahoo/sales-forecasting": ["train.csv", "Order Date"],
    
    # 2. Daily Crypto/Finance
    "mczielinski/bitcoin-historical-data": ["bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv", "Timestamp"],
    
    # 3. Weather / Climate
    "rohanrao/nifty50-stock-market-data": ["HDFCBANK.csv", "Date"], 
}


def download_and_format() -> None:
    try:
        import kaggle
    except OSError as e:
        logger.error(f"Kaggle API Credentials Missing! {e}")
        print("\n❌ Error: You must create a Kaggle API token.")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New Token'")
        print("3. Place the downloaded 'kaggle.json' file in your C:\\Users\\rajus\\.kaggle\\ folder.")
        return
    except ImportError:
        logger.error("Kaggle package not installed. Run: pip install kaggle")
        return

    logger.info("=" * 60)
    logger.info("STARTING DIVERSE KAGGLE DATA DOWNLOADS")
    logger.info("=" * 60)

    for dataset_path, meta in DATASETS.items():
        file_name, date_col = meta
        dataset_slug = dataset_path.split("/")[-1]
        save_path = os.path.join(KAGGLE_DIR, dataset_slug)
        
        logger.info(f"\nDownloading {dataset_path}...")
        try:
            # Download and unzip the specific dataset
            kaggle.api.dataset_download_files(dataset_path, path=save_path, unzip=True)
            
            # Find the target CSV
            target_csv = os.path.join(save_path, file_name)
            if not os.path.exists(target_csv):
                # Sometimes file names inside zip change, grab the first CSV
                csvs = glob.glob(os.path.join(save_path, "*.csv"))
                if csvs:
                    target_csv = csvs[0]
                else:
                    logger.warning(f"No CSV found for {dataset_slug}")
                    continue

            # Load and standardize the date column
            df = pd.read_csv(target_csv)
            logger.info(f"Loaded {len(df):,} rows from {os.path.basename(target_csv)}")
            
            # If it has a date column, standardize it for the AI pipeline
            if date_col and date_col in df.columns:
                if date_col == "Timestamp": # Handle unix timestamps
                    df["date"] = pd.to_datetime(df[date_col], unit="s").dt.strftime('%Y-%m-%d')
                else:
                    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime('%Y-%m-%d')
                    
                # Drop rows where date failed to parse
                df = df.dropna(subset=["date"])
                
                # Sort chronologically for Time-Series models
                df = df.sort_values("date")
                
            # Save it as a highly compressed Parquet file for the models
            out_parquet = os.path.join(KAGGLE_DIR, f"{dataset_slug}_formatted.parquet")
            df.to_parquet(out_parquet, engine="pyarrow", index=False)
            logger.info(f"✅ Saved clean time-series data to {out_parquet}")
            
            # Clean up the raw massive CSV to save space
            os.remove(target_csv)
            
        except Exception as e:
            logger.error(f"Failed to process {dataset_path}: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("KAGGLE COLLECTION COMPLETE!")
    logger.info(f"Datasets ready for train.py in: {KAGGLE_DIR}")
    logger.info("=" * 60)

if __name__ == "__main__":
    download_and_format()
