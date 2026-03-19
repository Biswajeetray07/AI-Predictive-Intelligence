import os
import glob
import pandas as pd
import logging
# No sys.path hack - run with PYTHONPATH=. from root
from dotenv import load_dotenv
load_dotenv()

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_all():
    base_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    news_path = os.path.join(base_dir, 'news', 'news_signals.csv')
    social_path = os.path.join(base_dir, 'social_media', 'social_signals.csv')
    weather_path = os.path.join(base_dir, 'weather', 'weather_processed.csv')
    nlp_path = os.path.join(base_dir, '..', 'features', 'nlp_signals.parquet')
    external_dir = os.path.join(base_dir, 'external')
    stocks_dir = os.path.join(base_dir, 'financial', 'stocks')
    output_dir = os.path.join(base_dir, 'merged')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Primary Signals
    global_dfs = []
    ticker_dfs = []

    def process_df(df, name):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            if 'ticker' in df.columns:
                df['ticker'] = df['ticker'].astype(str).str.upper()
                macro_df = df[df['ticker'] == 'MACRO'].drop(columns=['ticker'], errors='ignore')
                tick_df = df[df['ticker'] != 'MACRO']
                if not macro_df.empty: global_dfs.append(macro_df)
                if not tick_df.empty: ticker_dfs.append(tick_df)
                logging.info(f"Loaded {name}: {len(macro_df)} macro, {len(tick_df)} ticker-specific rows.")
            else:
                global_dfs.append(df)
                logging.info(f"Loaded global {name}: {df.shape}")

    if os.path.exists(news_path):
        process_df(pd.read_csv(news_path), "news_signals")
    if os.path.exists(social_path):
        process_df(pd.read_csv(social_path), "social_signals")
    if os.path.exists(weather_path):
        process_df(pd.read_csv(weather_path), "weather_processed")
    if os.path.exists(nlp_path):
        process_df(pd.read_parquet(nlp_path), "nlp_signals")

    # 1b. Load Macro-Economic Signals (FRED)
    macro_path = os.path.join(base_dir, 'economy', 'macro_signals.csv')
    if os.path.exists(macro_path):
        process_df(pd.read_csv(macro_path), "macro_signals")

    # 1c. Load New Domain Feature Indices (from feature_generator.py)
    features_dir = os.path.join(base_dir, '..', 'features')
    
    # Dynamically find ALL generated feature parquets
    feature_parquets = [os.path.basename(p) for p in glob.glob(os.path.join(features_dir, '*.parquet'))]
    
    if not feature_parquets:
        logging.warning(f"No feature parquets found in {features_dir}")
        
    for pq_file in feature_parquets:
        pq_path = os.path.join(features_dir, pq_file)
        if os.path.exists(pq_path):
            try:
                feat_df = pd.read_parquet(pq_path)
                if 'date' in feat_df.columns:
                    feat_df['date'] = pd.to_datetime(feat_df['date'], errors='coerce')
                    # Drop non-numeric, non-date columns that would cause merge issues
                    drop_cols = [c for c in feat_df.columns if c not in ['date'] and feat_df[c].dtype == 'object']
                    feat_df = feat_df.drop(columns=drop_cols, errors='ignore')
                    process_df(feat_df, pq_file.replace('.parquet', ''))
            except Exception as e:
                logging.error(f"Error loading feature index {pq_file}: {e}")

    # 2. Load Kaggle External Signals
    if os.path.exists(external_dir):
        kaggle_files = glob.glob(os.path.join(external_dir, '*.csv'))
        for f in kaggle_files:
            # Skip duplicates that don't have the full path prefix (heuristic)
            # e.g. skip 'Fake_processed.csv' if 'fake_real_news_dataset_Fake_processed.csv' exists
            fname = os.path.basename(f)
            if not "_" in fname and any(os.path.basename(other).startswith(fname.replace('.csv', '')) for other in kaggle_files if other != f):
                 logging.info(f"Skipping potential duplicate/shallow Kaggle file: {fname}")
                 continue

            try:
                df = pd.read_csv(f)
                if 'date' in df.columns and len(df) > 0:
                    df['date'] = pd.to_datetime(df['date'])
                    process_df(df, fname)
            except Exception as e:
                logging.error(f"Error loading {f}: {e}")

    if not global_dfs and not ticker_dfs:
        logging.warning("No external signals found. Exiting.")
        return
        
    # Merge global signals iteratively
    global_signals = global_dfs[0] if global_dfs else None
    if global_signals is not None:
        for df in global_dfs[1:]:
            global_signals = pd.merge(global_signals, df, on='date', how='outer')
        global_signals = global_signals.sort_values('date')
        # LEAKAGE FIX: Only forward-fill (no backward fill) to prevent future data leaking into past
        global_signals = global_signals.ffill()
        # Deduplicate by date
        global_signals = global_signals.drop_duplicates(subset=['date'], keep='last')
        num_cols = global_signals.select_dtypes(include=['number', 'bool']).columns
        global_signals[num_cols] = global_signals[num_cols].fillna(0)
        obj_cols = global_signals.select_dtypes(include=['object']).columns
        global_signals[obj_cols] = global_signals[obj_cols].fillna('')
        logging.info(f"Merged global signals shape: {global_signals.shape} ({len(global_signals.columns)} columns)")

    # Merge ticker signals iteratively
    ticker_signals = ticker_dfs[0] if ticker_dfs else None
    if ticker_signals is not None:
        for df in ticker_dfs[1:]:
            ticker_signals = pd.merge(ticker_signals, df, on=['date', 'ticker'], how='outer')
        logging.info(f"Merged ticker signals shape: {ticker_signals.shape}")
    
    # 3. Load and Merge Financial Data
    stock_files = glob.glob(os.path.join(stocks_dir, '*.csv'))
    # FALLBACK: Try parquet if no csvs found
    is_parquet = False
    if not stock_files:
        stock_files = glob.glob(os.path.join(stocks_dir, '*.parquet'))
        is_parquet = True
        if stock_files:
            logging.info(f"Found {len(stock_files)} stock parquet files for merging.")

    if not stock_files:
        logging.warning(f"No stock data (.csv or .parquet) found in {stocks_dir}. Exiting.")
        return
    
    output_filepath = os.path.join(output_dir, 'all_merged_dataset.csv')
    first_file = True
    
    for count, file_path in enumerate(stock_files):
        try:
            if is_parquet:
                stock_df = pd.read_parquet(file_path)
            else:
                stock_df = pd.read_csv(file_path)
            
            # Ensure Date column formatting handles standard stock format
            if 'Date' in stock_df.columns:
                stock_df = stock_df.rename(columns={'Date': 'date'})
            
            if 'date' not in stock_df.columns:
                continue
                
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            current_ticker = os.path.basename(file_path).replace('.csv', '').replace('.parquet', '').upper()
            
            # Merge with ALL external signals
            if global_signals is not None:
                merged_df = pd.merge(stock_df, global_signals, on='date', how='left')
            else:
                merged_df = stock_df
                
            if ticker_signals is not None:
                ticker_spec_df = ticker_signals[ticker_signals['ticker'] == current_ticker].drop(columns=['ticker'])
                if not ticker_spec_df.empty:
                    merged_df = pd.merge(merged_df, ticker_spec_df, on='date', how='left')
            
            # LEAKAGE FIX: Only forward fill — do not bfill
            merged_df = merged_df.ffill()
            num_cols = merged_df.select_dtypes(include=['number', 'bool']).columns
            merged_df[num_cols] = merged_df[num_cols].fillna(0)
            obj_cols = merged_df.select_dtypes(include=['object']).columns
            merged_df[obj_cols] = merged_df[obj_cols].fillna('')
            
            # Append to CSV
            mode = 'w' if first_file else 'a'
            header = first_file
            merged_df.to_csv(output_filepath, mode=mode, header=header, index=False)
            first_file = False
            
            if (count + 1) % 50 == 0:
                logging.info(f"Processed and appended {count + 1} stock datasets...")
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            
    if first_file:
        logging.warning("No merged stock dataframes were generated.")
    else:
        logging.info(f"Successfully generated merged dataset in {output_filepath}")

if __name__ == "__main__":
    merge_all()
