import os
import pandas as pd
import logging
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_kaggle_sp500():
    """
    Transforms the external Kaggle SP500 dataset into the format 
    expected by the data processing pipeline.
    """
    input_path = os.path.join(PROJECT_ROOT, 'data', 'external', 'Kaggle_Datasets', 'sp500_datasets', 'sp500_stocks.csv')
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'financial', 'stocks')
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Loading external Kaggle dataset from {input_path}")
    df = pd.read_csv(input_path)
    
    # Rename columns to match pipeline standard
    df = df.rename(columns={
        'Date': 'date',
        'Symbol': 'ticker',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Drop rows without actual closing prices or empty symbols
    df = df.dropna(subset=['close', 'ticker'])
    
    # Sort for sequence generation
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    
    # For speed (and to keep VRAM low), we'll filter to top 50 highest volume stocks
    # if the laptop has low ram, but since this is offline, we can take all of them.
    # We will output one large CSV or per-ticker CSVs. Our pipeline usually loops through stocks.
    
    output_file = os.path.join(output_dir, 'kaggle_sp500_mapped.csv')
    df.to_csv(output_file, index=False)
    logging.info(f"Successfully mapped {len(df)} rows to {output_file}")
    
    # Trigger feature engineering on this data
    from src.data_processing.merge_datasets import merge_all
    
    # We create a dummy merge by saving it to merged so build_features can read it.
    merged_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'merged')
    os.makedirs(merged_dir, exist_ok=True)
    df.to_csv(os.path.join(merged_dir, 'all_merged_dataset.csv'), index=False)
    logging.info("Saved dummy merge to data/processed/merged/all_merged_dataset.csv")
    
    logging.info("Triggering build_features...")
    try:
        from src.feature_engineering.build_features import build_features
        build_features()
    except (ImportError, ModuleNotFoundError) as e:
        logging.warning(f"Could not import build_features: {e}")
    
    logging.info("Triggering build_sequences...")
    from src.data_processing.build_sequences import main as build_seq_main
    build_seq_main()
    
    logging.info("Kaggle data successfully processed into sequences! Ready for Optuna.")

if __name__ == "__main__":
    process_kaggle_sp500()
