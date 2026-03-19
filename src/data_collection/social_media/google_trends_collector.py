"""
Google Trends Collector
========================
Collects trending search topics using SerpAPI Google Trends endpoint.
Falls back to pytrends if SerpAPI key is unavailable.

Saves to: data/raw/social_media/google_trends_YYYY-MM-DD.parquet
"""

import os
from typing import cast
from datetime import datetime as dt
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─── Path Setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "social_media")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SERPAPI_KEY = os.getenv("SERPAPI_KEY")


def collect_with_serpapi():
    """Collect trending searches via SerpAPI."""
    if not SERPAPI_KEY:
        print("⚠️ SERPAPI_KEY not set, skipping SerpAPI collection")
        return []

    BASE_URL = "https://serpapi.com/search"
    regions = ["US", "GB", "IN", "DE", "JP", "BR", "FR", "AU", "CA", "KR"]
    all_trends = []

    for region in regions:
        try:
            params = {
                "engine": "google_trends_trending_now",
                "geo": region,
                "api_key": SERPAPI_KEY,
            }
            response = requests.get(BASE_URL, params=params, timeout=20)

            if response.status_code != 200:
                print(f"  API error for {region}: {response.status_code}")
                continue

            data = response.json()
            trends = data.get("trending_searches", data.get("daily_searches", []))

            if isinstance(trends, list):
                for trend in trends:
                    if isinstance(trend, dict):
                        all_trends.append({
                            "keyword": trend.get("query", trend.get("title", "")),
                            "traffic": trend.get("search_volume", trend.get("traffic", "unknown")),
                            "region": region,
                            "date": datetime.today().strftime("%Y-%m-%d"),
                            "source": "serpapi",
                        })
                    elif isinstance(trend, str):
                        all_trends.append({
                            "keyword": trend,
                            "traffic": "unknown",
                            "region": region,
                            "date": datetime.today().strftime("%Y-%m-%d"),
                            "source": "serpapi",
                        })

            print(f"  {region}: {len(trends) if isinstance(trends, list) else 0} trends")
            time.sleep(1)

        except Exception as e:
            print(f"  Error for {region}: {e}")

    return all_trends


def collect_with_pytrends():
    """Fallback: collect using pytrends library."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("⚠️ pytrends not installed. pip install pytrends")
        return []

    pytrends = TrendReq(hl="en-US", tz=360)
    all_trends = []

    keywords = [
        "artificial intelligence", "machine learning", "deep learning",
        "ChatGPT", "generative AI", "robotics", "data science",
        "neural networks", "LLM", "AI regulation",
    ]

    for kw in keywords:
        try:
            pytrends.build_payload([kw], timeframe="now 7-d")
            interest = pytrends.interest_over_time()

            if not interest.empty and kw in interest.columns:
                for idx, row in interest.iterrows():
                    try:
                        date_str = cast(dt, idx).strftime("%Y-%m-%d")
                    except (AttributeError, TypeError):
                        date_str = str(idx)
                    all_trends.append({
                        "keyword": kw,
                        "interest_score": int(row[kw]),
                        "date": date_str,
                        "source": "pytrends",
                    })

            time.sleep(2)  # Respect rate limits

        except Exception as e:
            print(f"  Error for '{kw}': {e}")

    return all_trends


def collect():
    """Main collection entry point."""
    print("Starting Google Trends Collection...\n")

    # Try SerpAPI first
    trends = collect_with_serpapi()

    # Fallback to pytrends
    if not trends:
        print("Falling back to pytrends...")
        trends = collect_with_pytrends()

    if trends:
        df = pd.DataFrame(trends)
        filename = os.path.join(
            OUTPUT_DIR, f"google_trends_{datetime.today().date()}.parquet"
        )
        df.to_parquet(filename, index=False)
        print(f"\nGoogle Trends collected: {len(df)} records")
        print(f"Saved to: {filename}")
        return df
    else:
        print("No Google Trends data collected.")
        return pd.DataFrame()


if __name__ == "__main__":
    collect()
