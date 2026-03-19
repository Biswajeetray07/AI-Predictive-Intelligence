import os
import glob
import pandas as pd
import numpy as np
import logging
from typing import List, Optional

# No sys.path hack - run with PYTHONPATH=. from root
from dotenv import load_dotenv
load_dotenv()

from src.data_processing.utils.processing_utils import setup_processing_logger, safe_read_data, validate_processed_schema

class FinancialDataProcessor:
    def __init__(self, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)
        self.raw_dir = os.path.join(self.base_dir, 'data', 'raw', 'financial')
        self.processed_dir = os.path.join(self.base_dir, 'data', 'processed', 'financial')
        self.logger = setup_processing_logger('financial_processing')

        # Ensure processed directories exist
        os.makedirs(os.path.join(self.processed_dir, 'stocks'), exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, 'crypto'), exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, 'alpha_vantage'), exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, 'economic_indicators'), exist_ok=True)
        # Allow overriding the processed file save format via env var: 'parquet' or 'csv'
        self.save_format = os.getenv('PROCESSED_SAVE_FORMAT', 'parquet').lower()

    def process_stocks_data(self):
        """Processes raw stock data files.
        Supports both individual ticker files and combined files with a 'ticker' column.
        Saves split processed files to ensure compatibility with merge_datasets.py.
        """
        self.logger.info("Starting stock data processing...")
        pattern_csv = os.path.join(self.raw_dir, 'stocks', '*.csv')
        pattern_pq = os.path.join(self.raw_dir, 'stocks', '*.parquet')
        stock_files = glob.glob(pattern_csv) + glob.glob(pattern_pq)
        processed_count = 0
        skipped_count = 0
        
        for file_path in stock_files:
            filename = os.path.basename(file_path)
            df = safe_read_data(file_path, self.logger)
            if df is None or df.empty:
                skipped_count += 1
                continue
                
            try:
                # Standardize columns (handle tuple-like string column names from yfinance MultiIndex)
                new_cols = {}
                for col in df.columns:
                    col_str = str(col)
                    if "('Date'," in col_str or col_str.lower() == 'date':
                        new_cols[col] = 'Date'
                    elif "('Close'," in col_str or col_str.lower() == 'close':
                        new_cols[col] = 'Close'
                    elif "('Open'," in col_str or col_str.lower() == 'open':
                        new_cols[col] = 'Open'
                    elif "('High'," in col_str or col_str.lower() == 'high':
                        new_cols[col] = 'High'
                    elif "('Low'," in col_str or col_str.lower() == 'low':
                        new_cols[col] = 'Low'
                    elif "('Volume'," in col_str or col_str.lower() == 'volume':
                        new_cols[col] = 'Volume'
                    elif "('ticker'," in col_str or col_str.lower() == 'ticker':
                        new_cols[col] = 'ticker'
                
                if new_cols:
                    df.rename(columns=new_cols, inplace=True)
                
                if 'Date' not in df.columns:
                    self.logger.warning(f"No 'Date' column found in {filename}. Columns: {df.columns.tolist()}")
                    skipped_count += 1
                    continue
                
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])
                
                # If ticker column is missing, infer from filename
                if 'ticker' not in df.columns:
                    ticker_val = os.path.splitext(filename)[0].upper()
                    df['ticker'] = ticker_val
                
                # Group by ticker and process each
                for ticker, ticker_df in df.groupby('ticker'):
                    ticker = str(ticker).upper()
                    ticker_df = ticker_df.copy()
                    ticker_df.sort_values('Date', inplace=True)
                    
                    # Ensure numeric
                    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
                    for col in numeric_cols:
                        if col in ticker_df.columns:
                            ticker_df[col] = pd.to_numeric(ticker_df[col], errors='coerce')
                    
                    ticker_df.ffill(inplace=True)
                    ticker_df.fillna(0, inplace=True)
                    
                    # ── Technical Indicator Engineering ──
                    if 'Close' in ticker_df.columns and len(ticker_df) > 50:
                        close = ticker_df['Close']
                        
                        # Moving Averages
                        ticker_df['SMA_10'] = close.rolling(window=10).mean()
                        ticker_df['SMA_50'] = close.rolling(window=50).mean()
                        ticker_df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
                        ticker_df['EMA_26'] = close.ewm(span=26, adjust=False).mean()
                        
                        # MACD
                        ticker_df['MACD'] = ticker_df['EMA_12'] - ticker_df['EMA_26']
                        ticker_df['MACD_Signal'] = ticker_df['MACD'].ewm(span=9, adjust=False).mean()
                        
                        # Volatility (annualized rolling 30-day)
                        ticker_df['Daily_Return'] = close.pct_change()
                        ticker_df['Volatility_30'] = ticker_df['Daily_Return'].rolling(window=30).std() * np.sqrt(252)
                        
                        # RSI (14-day)
                        delta = close.diff()
                        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / (loss + 1e-9)
                        ticker_df['RSI_14'] = 100 - (100 / (1 + rs))
                        
                        # Bollinger Bands
                        sma20 = close.rolling(window=20).mean()
                        std20 = close.rolling(window=20).std()
                        ticker_df['BB_Upper'] = sma20 + 2 * std20
                        ticker_df['BB_Lower'] = sma20 - 2 * std20
                        ticker_df['BB_Width'] = (ticker_df['BB_Upper'] - ticker_df['BB_Lower']) / (sma20 + 1e-9)
                        
                        # Price momentum (5d, 10d, 20d returns)
                        ticker_df['Momentum_5d'] = close.pct_change(periods=5)
                        ticker_df['Momentum_10d'] = close.pct_change(periods=10)
                        ticker_df['Momentum_20d'] = close.pct_change(periods=20)
                        
                        # Volume features
                        if 'Volume' in ticker_df.columns:
                            vol = ticker_df['Volume']
                            ticker_df['Volume_SMA_20'] = vol.rolling(window=20).mean()
                            ticker_df['Volume_Ratio'] = vol / (ticker_df['Volume_SMA_20'] + 1e-9)
                        
                        # Forward-fill rolling window NaNs, then zero-fill remaining
                        ticker_df.ffill(inplace=True)
                        ticker_df.fillna(0, inplace=True)
                    
                    # Standardize column names to lowercase 'date' for downstream merger
                    if 'Date' in ticker_df.columns:
                        ticker_df.rename(columns={'Date': 'date'}, inplace=True)
                    
                    output_base = os.path.join(self.processed_dir, 'stocks', f"{ticker}")
                    try:
                        if self.save_format == 'parquet':
                            output_path_pq = output_base + '.parquet'
                            try:
                                ticker_df.to_parquet(output_path_pq, index=False)
                                self.logger.info(f"Saved parquet: {output_path_pq}")
                            except Exception as pq_e:
                                # Fallback to CSV when parquet write fails
                                self.logger.warning(f"Parquet save failed for {ticker}: {pq_e}. Falling back to CSV.")
                                output_path_csv = output_base + '.csv'
                                ticker_df.to_csv(output_path_csv, index=False)
                                self.logger.info(f"Saved csv fallback: {output_path_csv}")
                        else:
                            output_path_csv = output_base + '.csv'
                            ticker_df.to_csv(output_path_csv, index=False)
                            self.logger.info(f"Saved csv: {output_path_csv}")

                        processed_count += 1

                        # Periodic progress log so long runs are visible and interruptible
                        if processed_count % 100 == 0:
                            self.logger.info(f"Processed {processed_count} tickers so far...")

                    except KeyboardInterrupt:
                        self.logger.info("Received KeyboardInterrupt during save — stopping stock processing gracefully.")
                        raise
                    except Exception as e:
                        self.logger.error(f"Failed to save processed data for {ticker}: {e}", exc_info=True)
                
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)
                skipped_count += 1
        
        self.logger.info(f"Stock processing complete. Processed: {processed_count}, Skipped: {skipped_count}")


    def process_crypto_data(self):
        """Processes raw cryptocurrency data files (e.g., bitcoin.csv)."""
        self.logger.info("Starting crypto data processing...")
        pattern_csv = os.path.join(self.raw_dir, 'crypto', '*.csv')
        pattern_pq = os.path.join(self.raw_dir, 'crypto', '*.parquet')
        crypto_files = glob.glob(pattern_csv) + glob.glob(pattern_pq)
        
        for file_path in crypto_files:
            filename = os.path.basename(file_path)
            self.logger.info(f"Processing crypto file: {filename}")
            
            df = safe_read_data(file_path, self.logger)
            if df is None or df.empty:
                continue
                
            try:
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                elif 'Date' in df.columns:
                     df.rename(columns={'Date': 'date'}, inplace=True)
                     df['date'] = pd.to_datetime(df['date'])
                elif 'timestamp' in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    self.logger.warning(f"No date or timestamp column found in {filename}, skipping.")
                    continue
                    
                df.sort_values('date', inplace=True)
                df.reset_index(drop=True, inplace=True)
                
                if 'price' in df.columns:
                    df['price'] = df['price'].ffill().fillna(0)
                    
                    # Feature Engineering
                    df['Daily_Return'] = df['price'].pct_change()
                    df['SMA_7'] = df['price'].rolling(window=7).mean()
                    df['SMA_30'] = df['price'].rolling(window=30).mean()
                    df['Volatility_7d'] = df['Daily_Return'].rolling(window=7).std()
                    
                    df.ffill(inplace=True)
                    df.fillna(0, inplace=True)
                
                core_expected = ['date', 'price', 'coin']
                missing_cols = [col for col in core_expected if col not in df.columns]
                if missing_cols:
                     self.logger.error(f"Missing core columns {missing_cols} in {filename}. Skipping save.")
                     continue
                     
                expected_schema = list(df.columns)
                if not validate_processed_schema(df, expected_schema, self.logger):
                     continue
                     
                output_path = os.path.join(self.processed_dir, 'crypto', f"{filename}")
                df.to_csv(output_path, index=False)
                self.logger.info(f"Successfully processed and saved: {output_path}")
                
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)


    def process_alpha_vantage_data(self):
        """Processes alpha vantage technical indicator data."""
        self.logger.info("Starting alpha vantage data processing...")
        pattern_csv = os.path.join(self.raw_dir, 'alpha_vantage', '*.csv')
        pattern_pq = os.path.join(self.raw_dir, 'alpha_vantage', '*.parquet')
        av_files = glob.glob(pattern_csv) + glob.glob(pattern_pq)
        
        for file_path in av_files:
            filename = os.path.basename(file_path)
            self.logger.info(f"Processing alpha vantage file: {filename}")
            
            df = safe_read_data(file_path, self.logger)
            if df is None or df.empty:
                continue
                
            try:
                # Standardize Date
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                elif 'Date' in df.columns:
                     df.rename(columns={'Date': 'date'}, inplace=True)
                     df['date'] = pd.to_datetime(df['date'])
                else:
                     self.logger.warning(f"No date column found in {filename}, skipping.")
                     continue
                     
                df.sort_values('date', inplace=True)
                df.reset_index(drop=True, inplace=True)
                
                # Forward fill missing indicator values — no bfill to prevent data leakage
                df.ffill(inplace=True)
                df.fillna(0, inplace=True)
                
                # Standardize Ticker/Symbol
                if 'symbol' in df.columns:
                    ticker_col = 'symbol'
                elif 'ticker' in df.columns:
                    ticker_col = 'ticker'
                    df = df.rename(columns={'ticker': 'symbol'})
                else:
                    self.logger.warning(f"No 'symbol' or 'ticker' column found in {filename}, skipping.")
                    continue
                
                expected_schema = list(df.columns)    
                if not validate_processed_schema(df, expected_schema, self.logger):
                     continue
                     
                output_path = os.path.join(self.processed_dir, 'alpha_vantage', f"{filename}")
                df.to_csv(output_path, index=False)
                self.logger.info(f"Successfully processed and saved: {output_path}")
                
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)

    def process_economic_indicators_data(self):
        """Processes raw economic indicators data files (e.g., GDP.csv, CPI.csv)."""
        self.logger.info("Starting economic indicators data processing...")
        pattern_csv = os.path.join(self.raw_dir, 'economic_indicators', '*.csv')
        pattern_pq = os.path.join(self.raw_dir, 'economic_indicators', '*.parquet')
        econ_files = glob.glob(pattern_csv) + glob.glob(pattern_pq)
        
        for file_path in econ_files:
            filename = os.path.basename(file_path)
            self.logger.info(f"Processing economic indicator file: {filename}")
            
            df = safe_read_data(file_path, self.logger)
            if df is None or df.empty:
                continue
                
            try:
                # Standardize Date
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                elif 'Date' in df.columns:
                     df.rename(columns={'Date': 'date'}, inplace=True)
                     df['date'] = pd.to_datetime(df['date'])
                else:
                     self.logger.warning(f"No date column found in {filename}, skipping.")
                     continue
                     
                df.sort_values('date', inplace=True)
                df.reset_index(drop=True, inplace=True)
                
                # Check for missing values in 'value' column and fill if applicable
                if 'value' in df.columns:
                    # Forward fill only — no bfill to prevent data leakage
                    df['value'] = df['value'].ffill().fillna(0)
                
                expected_schema = list(df.columns)
                if not validate_processed_schema(df, expected_schema, self.logger):
                     continue
                     
                output_path = os.path.join(self.processed_dir, 'economic_indicators', f"{filename}")
                df.to_csv(output_path, index=False)
                self.logger.info(f"Successfully processed and saved: {output_path}")
                
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)


    def run_all(self):
        """Runs all financial data processing pipelines."""
        self.logger.info("Starting full financial data processing pipeline.")
        self.process_stocks_data()
        self.process_crypto_data()
        self.process_alpha_vantage_data()
        self.process_economic_indicators_data()
        self.logger.info("Completed full financial data processing pipeline.")

if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    
    processor = FinancialDataProcessor(base_dir=project_root)
    processor.run_all()
