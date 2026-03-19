import os
import glob
import pandas as pd
import logging
import re
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv()

from src.data_processing.utils.processing_utils import setup_processing_logger, safe_read_csv
from src.data_processing.news_processing import NewsDataProcessor

class ExternalDataProcessor:
    def __init__(self, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)
        self.raw_dir = os.path.join(self.base_dir, 'data', 'external', 'Kaggle_Datasets')
        self.processed_dir = os.path.join(self.base_dir, 'data', 'processed', 'external')
        self.logger = setup_processing_logger('external_processing')
        
        # Instantiate news processor to reuse text cleaning and sentiment analysis tools
        self.news_processor = NewsDataProcessor(self.base_dir)

        # Ensure processed directories exist
        os.makedirs(self.processed_dir, exist_ok=True)

    def _extract_date_from_filename(self, filename: str) -> Optional[str]:
        """Heuristic to extract YYYY-MM or YYYY-MM-DD from filename if columns are missing"""
        match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if match:
            return match.group(1)
        match = re.search(r'(\d{4}-\d{2})', filename)
        if match:
            return match.group(1) + "-01"  # Default to first of month
        return None

    def _standardize_date_column(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """Finds any column that looks like a date and standardizes it to 'date'"""
        date_candidates = ['date', 'time', 'timestamp', 'created_at', 'day', 'datetime', 'period', 'year-month', 'order date', 'published_at', 'publication_date']
        
        # Lowercase columns for matching
        df.columns = [str(c).lower() for c in df.columns]

        found_col = None
        for col in date_candidates:
            if col in df.columns:
                try:
                    df['date'] = pd.to_datetime(df[col], errors='coerce').dt.date
                    # Drop rows where date couldn't be parsed
                    df = df.dropna(subset=['date'])
                    if not df.empty:
                        found_col = col
                        break
                except Exception:
                    pass
        
        # If no date column found, try filename heuristic
        if found_col is None:
            filename = os.path.basename(file_path)
            extracted_date = self._extract_date_from_filename(filename)
            if extracted_date:
                self.logger.info(f"Extracted date {extracted_date} from filename: {filename}")
                df['date'] = pd.to_datetime(extracted_date).date()
        
        return df

    def process_generic_csv(self, file_path: str, dataset_name: str) -> Optional[pd.DataFrame]:
        self.logger.info(f"Processing generic external dataset: {dataset_name} ({file_path})")
        df = safe_read_csv(file_path, self.logger)
        if df is None or df.empty:
            return None

        # Try to find and standardize a date column
        df = self._standardize_date_column(df, file_path)
        if 'date' not in df.columns:
            self.logger.warning(f"Could not find a valid date column in {file_path}. Skipping.")
            return None

        # Expanded text columns (heuristically look for common names)
        text_candidates = ['text', 'title', 'content', 'body', 'tweet', 'headline', 'headlines', 'comment', 'description', 'summary']
        text_cols = [c for c in df.columns if c in text_candidates]
        
        if text_cols:
             target_text_col = text_cols[0]
             self.logger.info(f"Using text column '{target_text_col}' for {dataset_name} count grouping.")
             
             # Group by date for text datasets (count only, NLP is too slow for millions of Kaggle rows)
             daily_df = df.groupby('date').agg(
                 post_count=pd.NamedAgg(column='date', aggfunc='count')
             ).rename(columns={'post_count': f'{dataset_name}_count'}).reset_index()
             
             return daily_df
        else:
             # For numerical datasets, group by date and mean all numeric columns
             numeric_df = df.select_dtypes(include=['number'])
             if numeric_df.empty:
                 self.logger.warning(f"No numeric or text columns found in {file_path}.")
                 return None
                  
             numeric_df['date'] = df['date']
             daily_df = numeric_df.groupby('date').mean().reset_index()
             # Prefix columns to avoid collisions
             rename_dict = {c: f"ext_{dataset_name}_{c}" for c in daily_df.columns if c != 'date'}
             daily_df = daily_df.rename(columns=rename_dict)
             
             return daily_df

    def process_all_datasets(self):
        """Discovers and processes all datasets in the external folder recursively"""
        if not os.path.exists(self.raw_dir):
            self.logger.warning(f"External Kaggle datasets directory not found at {self.raw_dir}")
            return
            
        all_processed_dfs = []
        skip_dirs = ['stock_market_dataset', 'm5_forcasting_dataset']

        # Walk through all directories recursively
        for root, dirs, files in os.walk(self.raw_dir):
            # Prune directories we want to skip
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    # Use a unique dataset name based on path to avoid collisions
                    rel_path = os.path.relpath(file_path, self.raw_dir)
                    dataset_name = rel_path.replace(os.sep, '_').replace('.csv', '')
                    
                    processed_df = self.process_generic_csv(file_path, dataset_name)
                    
                    if processed_df is not None and not processed_df.empty:
                        # Save individual processed output
                        output_path = os.path.join(self.processed_dir, f"{dataset_name}_processed.csv")
                        
                        processed_df['date'] = pd.to_datetime(processed_df['date']).dt.strftime('%Y-%m-%d')
                        processed_df.to_csv(output_path, index=False)
                        all_processed_dfs.append(processed_df)
                        self.logger.info(f"Saved processed external dataset: {output_path}")

        self.logger.info(f"Completed processing {len(all_processed_dfs)} external datasets.")

    def run_all(self):
        self.logger.info("Starting full external data processing pipeline.")
        self.process_all_datasets()
        self.logger.info("Completed full external data processing pipeline.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    processor = ExternalDataProcessor(base_dir=project_root)
    processor.run_all()
