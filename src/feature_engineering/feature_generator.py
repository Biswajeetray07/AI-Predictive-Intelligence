"""
Feature Generator Module
========================
Generates derived features/indices from raw data collected by
the multi-domain data collection system.

Features Generated:
- trade_growth_rate
- energy_demand_growth
- research_activity_index
- patent_innovation_index
- job_market_index
- air_traffic_index
- blockchain_activity_index
- satellite_economic_activity_index
"""

import os
import sys
import glob
import time
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import numpy as np

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

from src.utils.logging_utils import get_logger, log_execution_metrics, log_dataset_size

logger = get_logger("FeatureGenerator")

# Paths
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
os.makedirs(FEATURES_DIR, exist_ok=True)


def _load_latest_parquet(domain: str, prefix: str) -> Optional[pd.DataFrame]:
    """
    Load the most recent parquet file for a domain.

    Args:
        domain: Subdirectory under data/raw/.
        prefix: File prefix to match.

    Returns:
        DataFrame or None if not found.
    """
    pattern = os.path.join(RAW_DIR, domain, f"{prefix}*.parquet")
    files = sorted(glob.glob(pattern), reverse=True)
    if files:
        logger.info(f"Loading {files[0]}")
        return pd.read_parquet(files[0])
    logger.warning(f"No parquet files found for {domain}/{prefix}")
    return None


def compute_trade_growth_rate() -> Optional[pd.DataFrame]:
    """
    Compute trade growth rate from World Bank trade data.

    Returns:
        DataFrame with country-level trade growth rates.
    """
    df = _load_latest_parquet("trade", "worldbank")
    if df is None or df.empty:
        return None

    # Filter for GDP indicator
    gdp = df[df["indicator_code"] == "NY.GDP.MKTP.CD"].copy()
    if gdp.empty:
        return None

    gdp = gdp.sort_values(["country_code", "date"])
    gdp["value"] = pd.to_numeric(gdp["value"], errors="coerce")
    gdp["trade_growth_rate"] = gdp.groupby("country_code")["value"].pct_change()

    result = gdp[["date", "country_code", "country", "trade_growth_rate"]].dropna()
    logger.info(f"trade_growth_rate: {len(result)} records")
    return result


def compute_energy_demand_growth() -> Optional[pd.DataFrame]:
    """
    Compute energy demand growth from EIA data.

    Returns:
        DataFrame with energy demand growth rates.
    """
    df = _load_latest_parquet("energy", "eia")
    if df is None or df.empty:
        return None

    demand = df[df["series"] == "electricity_demand"].copy()
    if demand.empty:
        # Fall back to any available series
        demand = df.copy()

    demand = demand.sort_values("date")
    demand["value"] = pd.to_numeric(demand["value"], errors="coerce")
    demand["energy_demand_growth"] = demand.groupby("series")["value"].pct_change()

    result = demand[["date", "series", "energy_demand_growth"]].dropna()
    logger.info(f"energy_demand_growth: {len(result)} records")
    return result


def compute_research_activity_index() -> Optional[pd.DataFrame]:
    """
    Compute research activity index from arXiv and Semantic Scholar data.
    Index = normalized paper count × average citation count.

    Returns:
        DataFrame with monthly research activity index.
    """
    arxiv_df = _load_latest_parquet("research", "arxiv")
    scholar_df = _load_latest_parquet("research", "semantic_scholar")

    dfs = []
    if arxiv_df is not None and not arxiv_df.empty:
        arxiv_df["published_date"] = pd.to_datetime(arxiv_df["published_date"], errors="coerce")
        arxiv_df["month"] = arxiv_df["published_date"].dt.to_period("M")
        arxiv_agg = arxiv_df.groupby("month").agg(
            paper_count=("arxiv_id", "count"),
        ).reset_index()
        dfs.append(arxiv_agg)

    if scholar_df is not None and not scholar_df.empty:
        scholar_df["publication_date"] = pd.to_datetime(scholar_df["publication_date"], errors="coerce")
        scholar_df["month"] = scholar_df["publication_date"].dt.to_period("M")
        scholar_agg = scholar_df.groupby("month").agg(
            avg_citations=("citation_count", "mean"),
        ).reset_index()
        dfs.append(scholar_agg)

    if not dfs:
        return None

    if len(dfs) == 2:
        merged = pd.merge(dfs[0], dfs[1], on="month", how="outer")
    else:
        merged = dfs[0]

    # Normalize paper count and compute index
    for col in ["paper_count", "avg_citations"]:
        if col in merged.columns:
            max_val = merged[col].max()
            if max_val > 0:
                merged[f"{col}_norm"] = merged[col] / max_val

    if "paper_count_norm" in merged.columns and "avg_citations_norm" in merged.columns:
        merged["research_activity_index"] = (
            merged["paper_count_norm"] * 0.6 + merged["avg_citations_norm"] * 0.4
        )
    elif "paper_count_norm" in merged.columns:
        merged["research_activity_index"] = merged["paper_count_norm"]
    else:
        merged["research_activity_index"] = 0.0

    merged["date"] = merged["month"].astype(str)
    result = merged[["date", "research_activity_index"]].dropna()
    logger.info(f"research_activity_index: {len(result)} records")
    return result


def compute_patent_innovation_index() -> Optional[pd.DataFrame]:
    """
    Compute patent innovation index from USPTO data.
    Index = normalized patent count × average citations.

    Returns:
        DataFrame with monthly patent innovation index.
    """
    df = _load_latest_parquet("patents", "nasa_patents")
    if df is None or df.empty:
        return None

    if "publication_date" not in df.columns:
        df["publication_date"] = pd.Timestamp.now()
        
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["month"] = df["publication_date"].dt.to_period("M")

    agg = df.groupby("month").agg(
        patent_count=("patent_id", "count"),
        # Use a dummy or calculated citation if available, NASA data usually has category/center
    ).reset_index()
    agg["avg_citations"] = 0.0 # Placeholder

    # Normalize
    for col in ["patent_count", "avg_citations"]:
        max_val = agg[col].max()
        if max_val > 0:
            agg[f"{col}_norm"] = agg[col] / max_val
        else:
            agg[f"{col}_norm"] = 0.0

    agg["patent_innovation_index"] = (
        agg["patent_count_norm"] * 0.5 + agg["avg_citations_norm"] * 0.5
    )

    agg["date"] = agg["month"].astype(str)
    result = agg[["date", "patent_innovation_index"]].dropna()
    logger.info(f"patent_innovation_index: {len(result)} records")
    return result


def compute_job_market_index() -> Optional[pd.DataFrame]:
    """
    Compute job market index from Adzuna data.
    Index based on job count and average salary.

    Returns:
        DataFrame with job market index per country.
    """
    df = _load_latest_parquet("jobs", "adzuna")
    if df is None or df.empty:
        df = _load_latest_parquet("jobs", "usajobs")
    if df is None or df.empty:
        return None

    if "salary_min" in df.columns:
        df["salary_min"] = pd.to_numeric(df["salary_min"], errors="coerce")
        agg = df.groupby("country").agg(
            job_count=("job_title", "count"),
            avg_salary=("salary_min", "mean"),
        ).reset_index()
    else:
        agg = df.groupby("country" if "country" else "location").agg(
            job_count=("job_title", "count"),
        ).reset_index()
        agg["avg_salary"] = 0.0

    for col in ["job_count", "avg_salary"]:
        max_val = agg[col].max()
        if max_val > 0:
            agg[f"{col}_norm"] = agg[col] / max_val
        else:
            agg[f"{col}_norm"] = 0.0

    agg["job_market_index"] = (
        agg.get("job_count_norm", 0) * 0.6 + agg.get("avg_salary_norm", 0) * 0.4
    )
    agg["date"] = datetime.now().strftime("%Y-%m-%d")

    result = agg[["date", "country", "job_market_index"]].dropna()
    logger.info(f"job_market_index: {len(result)} records")
    return result


def compute_air_traffic_index() -> Optional[pd.DataFrame]:
    """
    Compute air traffic index from OpenSky aggregated data.

    Returns:
        DataFrame with air traffic index per country.
    """
    df = _load_latest_parquet("aviation", "opensky_aggregated")
    if df is None or df.empty:
        return None

    max_flights = df["flight_count"].max()
    if max_flights > 0:
        df["air_traffic_index"] = df["flight_count"] / max_flights
    else:
        df["air_traffic_index"] = 0.0

    df["date"] = df.get("timestamp", datetime.now().isoformat())
    result = df[["date", "origin_country", "flight_count", "air_traffic_index"]].dropna()
    logger.info(f"air_traffic_index: {len(result)} records")
    return result


def compute_blockchain_activity_index() -> Optional[pd.DataFrame]:
    """
    Compute blockchain activity index from Blockchain.com data.
    Composite of normalized market price, transaction count, and hash rate.

    Returns:
        DataFrame with daily blockchain activity index.
    """
    df = _load_latest_parquet("blockchain", "blockchain")
    if df is None or df.empty:
        return None

    # Pivot metrics to columns
    pivot = df.pivot_table(index="date", columns="metric", values="value", aggfunc="first")

    index_cols = ["market_price", "transaction_count", "hash_rate"]
    available = [c for c in index_cols if c in pivot.columns]

    if not available:
        return None

    for col in available:
        pivot[col] = pd.to_numeric(pivot[col], errors="coerce")
        max_val = pivot[col].max()
        if max_val > 0:
            pivot[f"{col}_norm"] = pivot[col] / max_val
        else:
            pivot[f"{col}_norm"] = 0.0

    norm_cols = [f"{c}_norm" for c in available]
    pivot["blockchain_activity_index"] = pivot[norm_cols].mean(axis=1)

    result = pivot.reset_index()[["date", "blockchain_activity_index"]].dropna()
    logger.info(f"blockchain_activity_index: {len(result)} records")
    return result


def compute_population_index() -> Optional[pd.DataFrame]:
    """
    Compute population index from UN Population data.
    """
    df = _load_latest_parquet("population", "un_population")
    if df is None or df.empty:
        return None

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    agg = df.groupby("date")["value"].sum().reset_index()
    
    max_val = agg["value"].max()
    if max_val > 0:
        agg["population_index"] = agg["value"] / max_val
    else:
        agg["population_index"] = 0.0

    result = agg[["date", "population_index"]].dropna()
    logger.info(f"population_index: {len(result)} records")
    return result


def compute_technical_indicators() -> Optional[pd.DataFrame]:
    """
    Compute advanced technical indicators (Volatility, RSI, MACD, Moving Averages)
    on the underlying stock data to provide high-quality ML features.
    """
    # Try to load the raw stock pricing data (assumes Alpha Vantage or Yahoo Finance format)
    # The merger script drops raw 'stocks' directory, so we look in data/raw/financial/stocks
    stock_dir = os.path.join(RAW_DIR, "financial", "stocks")
    stock_files = glob.glob(os.path.join(stock_dir, "*.csv"))
    
    if not stock_files:
        logger.warning(f"No stock data found in {stock_dir} to compute technical indicators.")
        return None
        
    all_tickers = []
    
    for count, fpath in enumerate(stock_files):
        try:
            # Skip the second header-like row often found in some scraper outputs
            df = pd.read_csv(fpath, skiprows=[1]) 
            
            # Case-insensitive column detection
            cols_map = {c.lower(): c for c in df.columns}
            
            close_col = cols_map.get('close')
            date_col = cols_map.get('date')
            ticker_col = cols_map.get('ticker')
            
            if not all([close_col, date_col, ticker_col]):
                continue
                
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col]).sort_values(date_col)
            
            # 1. Moving Averages
            df['SMA_10'] = df[close_col].rolling(window=10).mean()
            df['SMA_50'] = df[close_col].rolling(window=50).mean()
            df['EMA_12'] = df[close_col].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df[close_col].ewm(span=26, adjust=False).mean()

            # 2. MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            # 3. Volatility
            df['Returns'] = df[close_col].pct_change()
            df['Volatility_30'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)

            # 4. RSI
            delta = df[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            df['RSI_14'] = 100 - (100 / (1 + rs))

            tech_cols = ['SMA_10', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'Volatility_30', 'RSI_14']
            df[tech_cols] = df[tech_cols].ffill().fillna(0)

            # Standardize column naming for the merger (lowercase keys)
            df_out = df[[date_col, ticker_col] + tech_cols].copy()
            df_out.columns = ['date', 'ticker'] + tech_cols
            all_tickers.append(df_out)
            
        except Exception as e:
            logger.warning(f"Error computing indicators for {fpath}: {e}")
            
    if not all_tickers:
        return None
        
    final_df = pd.concat(all_tickers, ignore_index=True)
    logger.info(f"technical_indicators: {len(final_df)} records across {len(all_tickers)} tickers")
    return final_df


def generate_all_features() -> Dict[str, Optional[pd.DataFrame]]:
    """
    Generate all feature indices and save to parquet.

    Returns:
        Dictionary mapping feature name to its DataFrame (or None).
    """
    start_time = time.time()
    logger.info("Starting feature generation pipeline...")

    features = {
        "trade_growth_rate": compute_trade_growth_rate,
        "energy_demand_growth": compute_energy_demand_growth,
        "research_activity_index": compute_research_activity_index,
        "patent_innovation_index": compute_patent_innovation_index,
        "job_market_index": compute_job_market_index,
        "air_traffic_index": compute_air_traffic_index,
        "blockchain_activity_index": compute_blockchain_activity_index,
        "population_index": compute_population_index,
        "technical_indicators": compute_technical_indicators,
    }

    results = {}
    for name, func in features.items():
        try:
            df = func()
            results[name] = df
            if df is not None and not df.empty:
                output_path = os.path.join(FEATURES_DIR, f"{name}.parquet")
                df.to_parquet(output_path, index=False, engine="pyarrow")
                log_dataset_size(logger, name, output_path)
            else:
                logger.warning(f"No data for feature: {name}")
        except Exception as e:
            logger.error(f"Error generating {name}: {e}", exc_info=True)
            results[name] = None

    # Log summary
    successful = sum(1 for v in results.values() if v is not None and not v.empty)
    total_records = sum(len(v) for v in results.values() if v is not None and not v.empty)
    log_execution_metrics(logger, "FeatureGenerator", total_records, start_time)
    logger.info(f"Feature generation complete: {successful}/{len(features)} features generated")

    # ── Save to Feature Store ──
    try:
        from src.feature_engineering.feature_store import FeatureStore
        store = FeatureStore(PROJECT_ROOT)
        for name, df in results.items():
            if df is not None and not df.empty:
                version = store.save_features(
                    name, df, description=f"Auto-generated by FeatureGenerator",
                    tags=["auto", "feature_engineering"],
                )
                logger.info(f"  → FeatureStore: saved '{name}' v{version}")
        logger.info("All features persisted to FeatureStore")
    except Exception as e:
        logger.warning(f"FeatureStore persistence skipped: {e}")

    return results


if __name__ == "__main__":
    results = generate_all_features()
    print("\n=== Feature Generation Summary ===")
    for name, df in results.items():
        if df is not None and not df.empty:
            print(f"  ✅ {name}: {len(df)} records")
        else:
            print(f"  ⚠️ {name}: no data (raw data may not be collected yet)")
