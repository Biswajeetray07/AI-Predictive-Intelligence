"""
Macro-Economic Data Processor
==============================
Processes raw FRED economic indicator CSVs from data/raw/financial/economic_indicators/
into a single wide-format daily macro signal table.

Output: data/processed/economy/macro_signals.csv
"""

import os
import glob
import pandas as pd
import numpy as np
import logging
# No sys.path hack - run with PYTHONPATH=. from root
from dotenv import load_dotenv
load_dotenv()

from src.data_processing.utils.processing_utils import setup_processing_logger, safe_read_csv, validate_processed_schema

class MacroEconomicProcessor:
    def __init__(self, base_dir: str):
        self.project_root = os.path.abspath(base_dir)
        self.raw_dir = os.path.join(self.project_root, 'data', 'raw', 'financial', 'economic_indicators')
        self.processed_dir = os.path.join(self.project_root, 'data', 'processed', 'economy')
        self.logger = setup_processing_logger('MacroEconomicProcessor')

        os.makedirs(self.processed_dir, exist_ok=True)

    def process_indicators(self) -> pd.DataFrame:
        """Load all FRED indicator CSVs and pivot them into a single wide table."""
        self.logger.info("Processing FRED macro-economic indicators...")

        csv_files = glob.glob(os.path.join(self.raw_dir, '*.csv'))
        pq_files = glob.glob(os.path.join(self.raw_dir, '*.parquet'))
        all_files = csv_files + pq_files
        
        if not all_files:
            self.logger.warning(f"No FRED files found in {self.raw_dir}")
            return pd.DataFrame()

        all_dfs = []
        for filepath in all_files:
            if filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            else:
                df = safe_read_csv(filepath, self.logger)
            
            if df is None or df.empty:
                continue

            indicator_name = os.path.basename(filepath).replace('.csv', '').replace('.parquet', '').lower()

            if 'date' not in df.columns:
                self.logger.warning(f"No 'date' column in {filepath}, skipping.")
                continue

            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])

            if 'value' in df.columns:
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df[['date', 'value']].rename(columns={'value': f'macro_{indicator_name}'})
                all_dfs.append(df)

        if not all_dfs:
            self.logger.warning("No valid FRED indicators could be processed.")
            return pd.DataFrame()

        # Merge all indicators on date
        if not all_dfs:
            return pd.DataFrame()
            
        result = all_dfs[0]
        for df in all_dfs[1:]:
            result = pd.merge(result, df, on='date', how='outer')

        result = result.sort_values('date')
        # Forward-fill only (no backward fill) to avoid leakage
        result = result.ffill()
        result = result.drop_duplicates(subset=['date'], keep='last')

        # Fill remaining NaN with 0
        num_cols = result.select_dtypes(include=['number']).columns
        result[num_cols] = result[num_cols].fillna(0)

        result['date'] = result['date'].dt.strftime('%Y-%m-%d')

        output_path = os.path.join(self.processed_dir, 'macro_signals.csv')
        result.to_csv(output_path, index=False)
        self.logger.info(f"Saved {len(result)} rows with {len(result.columns)-1} macro indicators to {output_path}")

        return result

    def run_all(self):
        self.logger.info("Starting macro-economic data processing pipeline.")
        self.process_indicators()
        self.logger.info("Completed macro-economic data processing pipeline.")

if __name__ == '__main__':
    processor = MacroEconomicProcessor(base_dir='data/processed')
    processor.run_all()
