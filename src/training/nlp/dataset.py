"""
NLP Dataset Loader for AI Predictive Intelligence Platform.

Reads raw text data from all sources (NewsAPI, GDELT, Mastodon, YouTube,
HackerNews, StackExchange, GitHub, Google Trends, and Kaggle external datasets),
applies source-specific cleaning, and unifies them into a single schema:
    [date, source, text, metadata]
"""

import os
import re
import glob
import html
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─── Source Registry ─────────────────────────────────────────────────────────
# Each source defines how to extract the text and date columns from its CSV.
SOURCE_REGISTRY = {
    'newsapi': {
        'text_cols': ['title', 'description'],
        'date_col': 'published_at',
        'glob': 'data/raw/news/newsapi/*.csv',
    },
    'gdelt': {
        'text_cols': ['title'],
        'date_col': 'published_at',
        'glob': 'data/raw/news/gdelt/*.csv',
    },
    'hackernews': {
        'text_cols': ['title'],
        'date_col': None,  # date extracted from filename
        'glob': 'data/raw/social_media/hackernews_*.csv',
    },
    'mastodon': {
        'text_cols': ['content'],
        'date_col': 'created_at',
        'glob': 'data/raw/social_media/mastodon_*.csv',
    },
    'youtube': {
        'text_cols': ['title', 'description'],
        'date_col': 'published_at',
        'glob': 'data/raw/social_media/youtube_*.csv',
    },
    'github': {
        'text_cols': ['name', 'description'],
        'date_col': None,  # date extracted from filename
        'glob': 'data/raw/social_media/github_*.csv',
    },
    'stackexchange': {
        'text_cols': ['title'],
        'date_col': None,  # date extracted from filename
        'glob': 'data/raw/social_media/stackexchange_*.csv',
    },
    'google_trends': {
        'text_cols': ['title'],
        'date_col': None,  # date from filename
        'glob': 'data/raw/social_media/global_trends_*.csv',
    },
    # Kaggle external datasets (processed text files)
    'kaggle_news': {
        'text_cols': ['Headlines', 'headline', 'title'],
        'date_col': 'date',
        'glob': 'data/external/Kaggle_Datasets/financial_news_dataset/*.csv',
    },
    'kaggle_fake_news': {
        'text_cols': ['title', 'text'],
        'date_col': 'date',
        'glob': 'data/external/Kaggle_Datasets/fake_real_news_dataset/*.csv',
    },
}

# Source ID mapping for the source embedding layer
SOURCE_TO_ID = {
    'newsapi': 0,
    'gdelt': 1,
    'hackernews': 2,
    'mastodon': 3,
    'youtube': 4,
    'github': 5,
    'stackexchange': 6,
    'google_trends': 7,
    'kaggle_news': 8,
    'kaggle_fake_news': 9,
}
NUM_SOURCES = len(SOURCE_TO_ID)


# ─── Cleaning Functions ─────────────────────────────────────────────────────

def _clean_html(text: str) -> str:
    """Strip HTML tags and decode HTML entities."""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


def _clean_mastodon(text: str) -> str:
    """Mastodon posts often contain HTML markup and URLs."""
    text = _clean_html(text)
    text = re.sub(r'https?://\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)           # remove mentions
    text = re.sub(r'#\w+', '', text)           # remove hashtags (keep them separately if needed)
    return text.strip()


def _clean_youtube(text: str) -> str:
    """YouTube titles/descriptions may contain emojis and special Unicode."""
    text = re.sub(r'https?://\S+', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _clean_github(text: str) -> str:
    """GitHub repo names use underscores/hyphens; descriptions are usually clean."""
    text = text.replace('_', ' ').replace('-', ' ')
    return text.strip()


def _clean_generic(text: str) -> str:
    """Basic cleaning for news and other sources."""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Map source names to their cleaning function
SOURCE_CLEANERS = {
    'mastodon': _clean_mastodon,
    'youtube': _clean_youtube,
    'github': _clean_github,
}


def _extract_date_from_filename(filename: str) -> str:
    """Extract date from filename patterns like 'hackernews_2026-03-07.csv'."""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    return ""  # Return empty string instead of None


# ─── Main Loader ─────────────────────────────────────────────────────────────

def load_all_sources(project_root: str) -> pd.DataFrame:
    """
    Load all registered data sources and unify them into a single DataFrame
    with columns: [date, source, source_id, text, day_of_week, month]
    """
    all_dfs = []

    for source_name, config in SOURCE_REGISTRY.items():
        pattern = os.path.join(project_root, config['glob'])
        files = glob.glob(pattern)

        if not files:
            logging.warning(f"[{source_name}] No files found for pattern: {pattern}")
            continue

        cleaner = SOURCE_CLEANERS.get(source_name, _clean_generic)
        source_id = SOURCE_TO_ID[source_name]

        for fpath in files:
            try:
                df = pd.read_csv(fpath, low_memory=False)
            except Exception as e:
                logging.error(f"[{source_name}] Failed to read {fpath}: {e}")
                continue

            # ── Extract text ──
            text_col_candidates = config['text_cols']
            matched_cols = [c for c in text_col_candidates if c in df.columns]
            if not matched_cols:
                logging.warning(f"[{source_name}] No text columns found in {os.path.basename(fpath)}. Skipping.")
                continue

            # Combine all matched text columns
            df['_combined_text'] = df[matched_cols].fillna('').astype(str).agg(' '.join, axis=1)
            df['_combined_text'] = df['_combined_text'].apply(cleaner)

            # Drop rows with empty text
            df = df[df['_combined_text'].str.len() > 5].copy()
            if df.empty:
                continue

            # ── Extract date ──
            date_col = config.get('date_col')
            if date_col and date_col in df.columns:
                df['_date'] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
                df['_date'] = df['_date'].dt.date
            else:
                # Fall back to filename
                file_date = _extract_date_from_filename(os.path.basename(fpath))
                if file_date:
                    df['_date'] = pd.to_datetime(file_date).date()
                else:
                    logging.warning(f"[{source_name}] No date column or filename date for {os.path.basename(fpath)}. Skipping.")
                    continue

            df = df.dropna(subset=['_date'])

            # ── Build unified row ──
            ticker_col = 'ticker' if 'ticker' in df.columns else 'symbol' if 'symbol' in df.columns else None
            tickers = df[ticker_col].values if ticker_col else ['MACRO'] * len(df)
            
            unified = pd.DataFrame({
                'date': pd.to_datetime(df['_date']),
                'source': source_name,
                'source_id': source_id,
                'text': df['_combined_text'].values,
                'ticker': tickers
            })

            all_dfs.append(unified)
            logging.info(f"[{source_name}] Loaded {len(unified)} rows from {os.path.basename(fpath)}")

    if not all_dfs:
        logging.error("No data was loaded from any source!")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # ── Temporal features for temporal embeddings ──
    combined['day_of_week'] = combined['date'].dt.dayofweek      # 0=Mon ... 6=Sun
    combined['month'] = combined['date'].dt.month - 1            # 0=Jan ... 11=Dec

    combined = combined.sort_values('date').reset_index(drop=True)
    logging.info(f"Total unified dataset: {combined.shape}")
    logging.info(f"Source distribution:\n{combined['source'].value_counts().to_string()}")

    return combined


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    df = load_all_sources(project_root)
    if not df.empty:
        out_path = os.path.join(project_root, 'data', 'features', 'unified_nlp_dataset.csv')
        df.to_csv(out_path, index=False)
        logging.info(f"Saved unified NLP dataset to {out_path}")
