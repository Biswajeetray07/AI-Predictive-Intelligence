import os
import glob
import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime

import nltk
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

# No sys.path hack - run with PYTHONPATH=. from root
from src.data_processing.utils.processing_utils import setup_processing_logger, safe_read_data, validate_processed_schema
from dotenv import load_dotenv
load_dotenv()

class SocialMediaProcessor:
    def __init__(self, base_dir, raw_dir='data/raw/social_media', processed_dir='data/processed/social_media', log_dir='logs'):
        # Force absolute paths relative to project root if possible, or assume running from project root
        self.base_dir = base_dir
        self.raw_dir = os.path.join(self.base_dir, raw_dir)
        self.processed_dir = os.path.join(self.base_dir, processed_dir)
        self.log_dir = os.path.join(self.base_dir, log_dir)
        
        # Ensure directories exist
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = setup_processing_logger('SocialMediaProcessor')
        
        try:
            self.sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
            
    def _clean_text(self, text):
        """Clean html, URLs, etc. from text before sentiment analysis"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _get_sentiment(self, text):
        if pd.isna(text) or not isinstance(text, str) or not text:
            return 0.0
        return self.sia.polarity_scores(text)['compound']

    def process_github(self):
        self.logger.info("Processing GitHub data...")
        files = glob.glob(os.path.join(self.raw_dir, 'github_*.csv'))
        if not files:
            self.logger.warning("No GitHub CSV files found.")
            return pd.DataFrame()
        
        df_list = []
        for file in files:
            df = safe_read_data(file, self.logger)
            if df is not None and not df.empty:
                df_list.append(df)
                
        if not df_list:
            return pd.DataFrame()
            
        final_df = pd.concat(df_list, ignore_index=True)
        
        # Drop duplicates based on repo_url if available
        if 'repo_url' in final_df.columns:
            final_df = final_df.drop_duplicates(subset=['repo_url'])
            
        # Time conversion
        if 'created_at' in final_df.columns:
            final_df['date'] = pd.to_datetime(final_df['created_at'], errors='coerce').dt.date
        else:
            self.logger.warning("No 'created_at' column in GitHub data.")
            return pd.DataFrame()
            
        # Ensure numeric columns
        for col in ['stars', 'forks']:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
                
        # Aggregate
        daily_github = final_df.groupby('date').agg(
            github_repo_count=('name', 'count'),
            github_stars_total=('stars', 'sum'),
            github_forks_total=('forks', 'sum')
        ).reset_index()
        
        # Convert date back to string for consistent merging
        daily_github['date'] = pd.to_datetime(daily_github['date']).dt.strftime('%Y-%m-%d')
        return daily_github

    def process_hackernews(self):
        self.logger.info("Processing HackerNews data...")
        pattern_csv = os.path.join(self.raw_dir, 'hackernews_*.csv')
        pattern_pq = os.path.join(self.raw_dir, 'hackernews_*.parquet')
        files = glob.glob(pattern_csv) + glob.glob(pattern_pq)
        if not files:
            self.logger.warning("No HackerNews CSV files found.")
            return pd.DataFrame()
        
        df_list = []
        for file in files:
            if file.endswith('.parquet'):
                df = pd.read_parquet(file)
            else:
                df = safe_read_data(file, self.logger)
            if df is not None and not df.empty:
                df_list.append(df)
                
        if not df_list:
            return pd.DataFrame()
            
        final_df = pd.concat(df_list, ignore_index=True)
        if 'id' in final_df.columns:
            final_df = final_df.drop_duplicates(subset=['id'])
            
        if 'time' in final_df.columns:
            # Assuming unix timestamp
            final_df['date'] = pd.to_datetime(final_df['time'], unit='s', errors='coerce').dt.date
        else:
            return pd.DataFrame()
            
        for col in ['score', 'comments']:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
                
        # Sentiment on Title
        if 'title' in final_df.columns:
            final_df['title_clean'] = final_df['title'].apply(self._clean_text)
            final_df['hn_sentiment'] = final_df['title_clean'].apply(self._get_sentiment)
        else:
            final_df['hn_sentiment'] = 0.0
            
        daily_hn = final_df.groupby('date').agg(
            hn_post_count=('id', 'count'),
            hn_score_total=('score', 'sum'),
            hn_comments_total=('comments', 'sum'),
            hn_sentiment_avg=('hn_sentiment', 'mean')
        ).reset_index()
        
        daily_hn['date'] = pd.to_datetime(daily_hn['date']).dt.strftime('%Y-%m-%d')
        return daily_hn

    def process_youtube(self):
        self.logger.info("Processing YouTube data...")
        pattern_csv = os.path.join(self.raw_dir, 'youtube_*.csv')
        pattern_pq = os.path.join(self.raw_dir, 'youtube_*.parquet')
        files = glob.glob(pattern_csv) + glob.glob(pattern_pq)
        if not files:
            self.logger.warning("No YouTube CSV files found.")
            return pd.DataFrame()
            
        df_list = []
        for file in files:
            if file.endswith('.parquet'):
                df = pd.read_parquet(file)
            else:
                df = safe_read_data(file, self.logger)
            if df is not None and not df.empty:
                df_list.append(df)
                
        if not df_list:
            return pd.DataFrame()
            
        final_df = pd.concat(df_list, ignore_index=True)
        if 'video_id' in final_df.columns:
            final_df = final_df.drop_duplicates(subset=['video_id'])
            
        if 'published_at' in final_df.columns:
            final_df['date'] = pd.to_datetime(final_df['published_at'], errors='coerce').dt.date
        else:
            return pd.DataFrame()
            
        for col in ['views', 'likes', 'comments', 'engagement_score']:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
                
        if 'title' in final_df.columns and 'description' in final_df.columns:
            final_df['text_clean'] = (final_df['title'].fillna('') + ' ' + final_df['description'].fillna('')).apply(self._clean_text)
            final_df['yt_sentiment'] = final_df['text_clean'].apply(self._get_sentiment)
        else:
            final_df['yt_sentiment'] = 0.0
            
        daily_yt = final_df.groupby('date').agg(
            yt_video_count=('video_id', 'count'),
            yt_views_total=('views', 'sum'),
            yt_likes_total=('likes', 'sum'),
            yt_comments_total=('comments', 'sum'),
            yt_engagement_total=('engagement_score', 'sum'),
            yt_sentiment_avg=('yt_sentiment', 'mean')
        ).reset_index()
        
        daily_yt['date'] = pd.to_datetime(daily_yt['date']).dt.strftime('%Y-%m-%d')
        return daily_yt

    def process_mastodon(self):
        self.logger.info("Processing Mastodon data...")
        files = glob.glob(os.path.join(self.raw_dir, 'mastodon_*.csv'))
        if not files:
            self.logger.warning("No Mastodon CSV files found.")
            return pd.DataFrame()
            
        df_list = []
        for file in files:
            df = safe_read_data(file, self.logger)
            if df is not None and not df.empty:
                df_list.append(df)
                
        if not df_list:
            return pd.DataFrame()
            
        final_df = pd.concat(df_list, ignore_index=True)
        if 'post_id' in final_df.columns:
            final_df = final_df.drop_duplicates(subset=['post_id'])
            
        if 'created_at' in final_df.columns:
            final_df['date'] = pd.to_datetime(final_df['created_at'], errors='coerce').dt.date
        else:
            return pd.DataFrame()
            
        for col in ['replies', 'reblogs', 'likes', 'engagement_score']:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
                
        if 'content_clean' in final_df.columns:
            final_df['mastodon_sentiment'] = final_df['content_clean'].apply(self._get_sentiment)
        elif 'content_raw' in final_df.columns:
            final_df['text_clean'] = final_df['content_raw'].apply(self._clean_text)
            final_df['mastodon_sentiment'] = final_df['text_clean'].apply(self._get_sentiment)
        else:
            final_df['mastodon_sentiment'] = 0.0
            
        # Build aggregation dict dynamically based on available columns
        named_aggs = {}
        col_mapping = {
            'mastodon_post_count': ('post_id', 'count'),
            'mastodon_replies_total': ('replies', 'sum'),
            'mastodon_reblogs_total': ('reblogs', 'sum'),
            'mastodon_likes_total': ('likes', 'sum'),
            'mastodon_engagement_total': ('engagement_score', 'sum'),
            'mastodon_sentiment_avg': ('mastodon_sentiment', 'mean')
        }
        for agg_name, (col, func) in col_mapping.items():
            if col in final_df.columns:
                named_aggs[agg_name] = pd.NamedAgg(column=col, aggfunc=func)
        
        if not named_aggs:
            self.logger.warning("No aggregatable columns found for Mastodon data.")
            return pd.DataFrame()
            
        daily_mastodon = final_df.groupby('date').agg(**named_aggs).reset_index()
        
        daily_mastodon['date'] = pd.to_datetime(daily_mastodon['date']).dt.strftime('%Y-%m-%d')
        return daily_mastodon

    def process_stackexchange(self):
        self.logger.info("Processing StackExchange data...")
        files = glob.glob(os.path.join(self.raw_dir, 'stackexchange_*.csv'))
        if not files:
            self.logger.warning("No StackExchange CSV files found.")
            return pd.DataFrame()
            
        df_list = []
        for file in files:
            df = safe_read_data(file, self.logger)
            if df is not None and not df.empty:
                df_list.append(df)
                
        if not df_list:
            return pd.DataFrame()
            
        final_df = pd.concat(df_list, ignore_index=True)
        # Link usually unique in stackexchange dump
        if 'link' in final_df.columns:
            final_df = final_df.drop_duplicates(subset=['link'])
            
        if 'creation_date' in final_df.columns:
            final_df['date'] = pd.to_datetime(final_df['creation_date'], unit='s', errors='coerce').dt.date
        else:
            return pd.DataFrame()
            
        for col in ['score', 'answers']:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
                
        if 'title' in final_df.columns:
            final_df['text_clean'] = final_df['title'].apply(self._clean_text)
            final_df['se_sentiment'] = final_df['text_clean'].apply(self._get_sentiment)
        else:
            final_df['se_sentiment'] = 0.0
            
        daily_se = final_df.groupby('date').agg(
            se_post_count=('link', 'count' if 'link' in final_df.columns else 'size'),
            se_score_total=('score', 'sum' if 'score' in final_df.columns else lambda x: 0),
            se_answers_total=('answers', 'sum' if 'answers' in final_df.columns else lambda x: 0),
            se_sentiment_avg=('se_sentiment', 'mean')
        ).reset_index()
        
        daily_se['date'] = pd.to_datetime(daily_se['date']).dt.strftime('%Y-%m-%d')
        return daily_se

    def process_global_trends(self):
        self.logger.info("Processing Global Trends (SerpApi) data...")
        files = glob.glob(os.path.join(self.raw_dir, 'global_trends_*.csv'))
        if not files:
            self.logger.warning("No Global Trends CSV files found.")
            return pd.DataFrame()
            
        df_list = []
        for file in files:
            df = safe_read_data(file, self.logger)
            if df is not None and not df.empty:
                df_list.append(df)
                
        if not df_list:
            return pd.DataFrame()
            
        final_df = pd.concat(df_list, ignore_index=True)
        if 'date' in final_df.columns:
            final_df['date_dt'] = pd.to_datetime(final_df['date'], errors='coerce').dt.date
        else:
            return pd.DataFrame()
            
        # Parse traffic if possible. Usually looks like '200K+' or 'unknown'
        def parse_traffic(val):
            if pd.isna(val) or val == 'unknown':
                return 0
            val = str(val).lower().replace('+', '').replace(',', '')
            if 'm' in val:
                try: return float(val.replace('m', '')) * 1000000
                except: return 0
            if 'k' in val:
                try: return float(val.replace('k', '')) * 1000
                except: return 0
            try: return float(val)
            except: return 0
            
        if 'traffic' in final_df.columns:
            final_df['traffic_num'] = final_df['traffic'].apply(parse_traffic)
        else:
            final_df['traffic_num'] = 0
            
        daily_gt = final_df.groupby('date_dt').agg(
            gt_trend_count=('keyword', 'count' if 'keyword' in final_df.columns else 'size'),
            gt_traffic_total=('traffic_num', 'sum')
        ).reset_index().rename(columns={'date_dt': 'date'})
        
        daily_gt['date'] = pd.to_datetime(daily_gt['date']).dt.strftime('%Y-%m-%d')
        return daily_gt

    def run_all(self):
        self.logger.info("Starting Social Media Data Processing...")
        
        dfs = []
        
        # GitHub
        df_github = self.process_github()
        if not df_github.empty:
            dfs.append(df_github)
            
        # HackerNews
        df_hn = self.process_hackernews()
        if not df_hn.empty:
            dfs.append(df_hn)
            
        # YouTube
        df_yt = self.process_youtube()
        if not df_yt.empty:
            dfs.append(df_yt)
            
        # Mastodon
        df_mastodon = self.process_mastodon()
        if not df_mastodon.empty:
            dfs.append(df_mastodon)
            
        # StackExchange
        df_se = self.process_stackexchange()
        if not df_se.empty:
            dfs.append(df_se)
            
        # Global Trends
        df_gt = self.process_global_trends()
        if not df_gt.empty:
            dfs.append(df_gt)
            
        if not dfs:
            self.logger.error("No data processed for any social media platform.")
            return None
            
        # Merge all dataframes on 'date'
        self.logger.info("Merging platform signals...")
        final_df = dfs[0]
        for idx in range(1, len(dfs)):
            final_df = pd.merge(final_df, dfs[idx], on='date', how='outer')
            
        # Sort by date
        final_df = final_df.sort_values('date').reset_index(drop=True)
        
        # Fill missing values: volume metrics with 0, sentiments with 0.0
        final_df = final_df.fillna(0)
        
        # Save output
        output_file = os.path.join(self.processed_dir, 'social_signals.csv')
        final_df.to_csv(output_file, index=False)
        self.logger.info(f"Successfully saved merged social media signals to {output_file}")
        self.logger.info(f"Final shape: {final_df.shape}")
        
        # Validate Schema
        validate_processed_schema(final_df, ['date'], self.logger)
        
        return final_df

if __name__ == '__main__':
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    
    processor = SocialMediaProcessor(base_dir=project_root)
    processor.run_all()
