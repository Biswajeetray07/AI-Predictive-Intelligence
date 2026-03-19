import pandas as pd
import numpy as np
import os
import logging
from src.feature_engineering.data_transformations import (
    align_time_series, 
    handle_missing_values, 
    generate_lag_features, 
    generate_rolling_features, 
    calculate_technical_indicators
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_features():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_path = os.path.join(base_dir, 'data', 'processed', 'merged', 'all_merged_dataset.csv')
    output_dir = os.path.join(base_dir, 'data', 'processed', 'model_inputs')
    
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Loading merged dataset from {input_path}")
    if not os.path.exists(input_path):
        logging.error(f"Input dataset not found at {input_path}")
        return
        
    df = pd.read_csv(input_path)
    
    # Pre-cleaning: ensure Ticker and Date are correctly named
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'date'})
    if 'ticker' in df.columns and 'Ticker' not in df.columns:
        df = df.rename(columns={'ticker': 'Ticker'})
        
    logging.info("Step 1: Aligning Time Series & Handling Missing Values")
    df = align_time_series(df, date_col='date', freq='B') # Business day frequency
    df = handle_missing_values(df)
    
    logging.info("Step 2: Note - Target Variable (Next Close) is dynamically generated in build_sequences.py")
    # Target shifting moved to src/data_processing/build_sequences.py to prevent double-shifting
    
    logging.info("Step 3: Calculating Technical Indicators")
    df = calculate_technical_indicators(df, group_col='Ticker', close_col='Close')
    
    logging.info("Step 4: Generating Lag & Rolling Features")
    lag_cols = ['Close', 'Volume', 'RSI_14']
    df = generate_lag_features(df, columns=[c for c in lag_cols if c in df.columns], lags=[1, 3, 7])
    df = generate_rolling_features(df, columns=[c for c in lag_cols if c in df.columns], windows=[7, 14, 30])
    
    
    logging.info("Step 5: Exporting Engineered Features")
    
    # Save the engineered feature dataframe to disk
    output_file = os.path.join(output_dir, "engineered_features.csv")
    df.to_csv(output_file, index=False)
    
    logging.info(f"Features successfully built and saved to {output_file}")

if __name__ == "__main__":
    build_features()
