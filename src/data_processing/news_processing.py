import os
import glob
import pandas as pd
import numpy as np
import logging
import re
from typing import List, Optional
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# No sys.path hack - run with PYTHONPATH=. from root
from dotenv import load_dotenv
load_dotenv()

from src.data_processing.utils.processing_utils import setup_processing_logger, safe_read_data, validate_processed_schema

def setup_nltk(logger):
    """Ensures necessary NLTK corpora are downloaded."""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        logger.info("Downloading NLTK VADER lexicon...")
        # Needs to be able to download in environments, so we don't catch all here, just log if fails visually or let it error to be handled properly.
        nltk.download('vader_lexicon', quiet=True)

class NewsDataProcessor:
    def __init__(self, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)
        self.raw_dir = os.path.join(self.base_dir, 'data', 'raw', 'news')
        self.processed_dir = os.path.join(self.base_dir, 'data', 'processed', 'news')
        self.logger = setup_processing_logger('news_processing')
        
        # Ensure processed directories exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Setup NLP
        setup_nltk(self.logger)
        self.sia = SentimentIntensityAnalyzer()

    def clean_text(self, text: str) -> str:
        """Basic text cleaning for NLP."""
        if not isinstance(text, str):
            pd.isna(text)
            return ""
        # Remove HTML tags (simple regex)
        text = re.sub(r'<[^>]+>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove multiple spaces/newlines
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def get_sentiment(self, text: str) -> float:
        """Returns the compound sentiment score from VADER (-1 to 1)."""
        if not text or not isinstance(text, str):
            return 0.0
        return self.sia.polarity_scores(text)['compound']

    def process_gdelt_data(self) -> pd.DataFrame:
        """Processes raw GDELT data."""
        self.logger.info("Starting GDELT data processing...")
        pattern_csv = os.path.join(self.raw_dir, 'gdelt', '*.csv')
        pattern_pq = os.path.join(self.raw_dir, 'gdelt', '*.parquet')
        gdelt_files = glob.glob(pattern_csv) + glob.glob(pattern_pq)
        
        all_daily_stats = []
        
        for file_path in gdelt_files:
            filename = os.path.basename(file_path)
            self.logger.info(f"Processing GDELT file: {filename}")
            
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = safe_read_data(file_path, self.logger)
            if df is None or df.empty:
                continue
                
            try:
                # GDELT typical columns: query, title, url, domain, language, source_country, published_at
                if 'published_at' in df.columns:
                    df['date'] = pd.to_datetime(df['published_at']).dt.date
                else:
                    self.logger.warning(f"No published_at column found in {filename}, skipping.")
                    continue
                
                # Combine title and possibly content for sentiment
                if 'title' in df.columns:
                    df['clean_text'] = df['title'].apply(self.clean_text)
                    df['sentiment'] = df['clean_text'].apply(self.get_sentiment)
                else:
                    self.logger.warning(f"No text column found in {filename}, skipping.")
                    continue
                
                # Aggregate daily signals
                daily = df.groupby('date').agg(
                    gdelt_article_count=('sentiment', 'count'),
                    gdelt_avg_sentiment=('sentiment', 'mean'),
                    gdelt_sentiment_std=('sentiment', 'std')
                ).reset_index()
                
                # Fill NaNs in std (when sequence length is 1)
                daily['gdelt_sentiment_std'] = daily['gdelt_sentiment_std'].fillna(0)
                
                all_daily_stats.append(daily)
                
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)
                
        if all_daily_stats:
            combined = pd.concat(all_daily_stats, ignore_index=True)
            # Re-aggregate over all files
            final_gdelt = combined.groupby('date').agg({
                'gdelt_article_count': 'sum',
                'gdelt_avg_sentiment': 'mean', 
                'gdelt_sentiment_std': 'mean' 
            }).reset_index()
            final_gdelt.sort_values('date', inplace=True)
            self.logger.info("Successfully completed GDELT processing.")
            return final_gdelt
        else:
            self.logger.warning("No GDELT data processed.")
            return pd.DataFrame()

    def process_newsapi_data(self) -> pd.DataFrame:
        """Processes raw NewsAPI data."""
        self.logger.info("Starting NewsAPI data processing...")
        pattern_csv = os.path.join(self.raw_dir, 'newsapi', '*.csv')
        pattern_pq = os.path.join(self.raw_dir, 'newsapi', '*.parquet')
        newsapi_files = glob.glob(pattern_csv) + glob.glob(pattern_pq)
        
        all_daily_stats = []
        
        for file_path in newsapi_files:
            filename = os.path.basename(file_path)
            self.logger.info(f"Processing NewsAPI file: {filename}")
            
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = safe_read_data(file_path, self.logger)
            if df is None or df.empty:
                continue
                
            try:
                # NewsAPI typically has: source, author, title, description, content, url, published_at
                if 'published_at' in df.columns:
                    df['date'] = pd.to_datetime(df['published_at']).dt.date
                else:
                    self.logger.warning(f"No published_at column found in {filename}, skipping.")
                    continue
                
                # Combine title and description for richer text
                df['text_combine'] = ""
                if 'title' in df.columns:
                    df['text_combine'] += df['title'].fillna('') + " "
                if 'description' in df.columns:
                    df['text_combine'] += df['description'].fillna('')
                    
                df['clean_text'] = df['text_combine'].apply(self.clean_text)
                df['sentiment'] = df['clean_text'].apply(self.get_sentiment)
                
                # Aggregate daily signals
                daily = df.groupby('date').agg(
                    newsapi_article_count=('sentiment', 'count'),
                    newsapi_avg_sentiment=('sentiment', 'mean'),
                    newsapi_sentiment_std=('sentiment', 'std')
                ).reset_index()
                
                # Fill NaNs in std
                daily['newsapi_sentiment_std'] = daily['newsapi_sentiment_std'].fillna(0)
                
                all_daily_stats.append(daily)
                
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)
                
        if all_daily_stats:
            combined = pd.concat(all_daily_stats, ignore_index=True)
            # Re-aggregate
            final_news = combined.groupby('date').agg({
                'newsapi_article_count': 'sum',
                'newsapi_avg_sentiment': 'mean', 
                'newsapi_sentiment_std': 'mean' 
            }).reset_index()
            final_news.sort_values('date', inplace=True)
            self.logger.info("Successfully completed NewsAPI processing.")
            return final_news
        else:
            self.logger.warning("No NewsAPI data processed.")
            return pd.DataFrame()

    def run_all(self):
        """Runs all news processing and merges them into a single daily signals file."""
        self.logger.info("Starting full news data processing pipeline.")
        gdelt_signals = self.process_gdelt_data()
        newsapi_signals = self.process_newsapi_data()
        
        # Merge if both exist, otherwise use one
        final_signals = None
        if not gdelt_signals.empty and not newsapi_signals.empty:
            # Merge on date using outer join to capture all dates
            final_signals = pd.merge(gdelt_signals, newsapi_signals, on='date', how='outer')
        elif not gdelt_signals.empty:
            final_signals = gdelt_signals
        elif not newsapi_signals.empty:
            final_signals = newsapi_signals
            
        if final_signals is not None and not final_signals.empty:
            # Sort by date
            final_signals.sort_values('date', inplace=True)
            
            # Convert date to string to standardize CSV saving if preferred, or keep as YYYY-MM-DD
            # final_signals['date'] = pd.to_datetime(final_signals['date']) # Standardize
            
            # Fill missing aggregates with 0 where appropriate
            count_cols = [c for c in final_signals.columns if 'count' in c]
            final_signals[count_cols] = final_signals[count_cols].fillna(0)
            
            # Save final unified news signals
            output_path = os.path.join(self.processed_dir, 'news_signals.csv')
            
            # Validate schema basically
            expected_schema = list(final_signals.columns)
            if validate_processed_schema(final_signals, expected_schema, self.logger):
                final_signals.to_csv(output_path, index=False)
                self.logger.info(f"Successfully processed and saved integrated news signals: {output_path}")
        else:
            self.logger.warning("No news signals generated. Check if raw data exists.")
            
        self.logger.info("Completed full news data processing pipeline.")

if __name__ == "__main__":
    # Correctly identify project root (repo root)
    # File is in src/data_processing/news_processing.py
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    
    processor = NewsDataProcessor(base_dir=project_root)
    processor.run_all()
