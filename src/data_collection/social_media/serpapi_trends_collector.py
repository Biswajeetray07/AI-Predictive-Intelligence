"""
SerpAPI Trends Collector
=========================
Collects global trending search topics using SerpAPI.

Saves to: data/raw/social_media/global_trends_YYYY-MM-DD.csv
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

REGIONS = ["US", "GB", "IN", "DE", "JP", "BR", "FR", "AU", "CA", "KR"]

def collect():
    """Collect trending searches from SerpAPI."""
    load_dotenv()
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    BASE_URL = "https://serpapi.com/search"
    
    # Setup paths relative to project root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "social_media")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Starting SerpAPI Trends Collection...\n")

    if not SERPAPI_KEY:
        print("⚠️ SERPAPI_KEY not set. Skipping collection.")
        return pd.DataFrame()

    all_trends = []

    for region in REGIONS:
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
                        })
                    elif isinstance(trend, str):
                        all_trends.append({
                            "keyword": trend,
                            "traffic": "unknown",
                            "region": region,
                            "date": datetime.today().strftime("%Y-%m-%d"),
                        })

            print(f"  {region}: {len(trends) if isinstance(trends, list) else 0} trends")
            time.sleep(1.5)

        except Exception as e:
            print(f"  Error for {region}: {e}")

    if all_trends:
        df = pd.DataFrame(all_trends)
        filename = os.path.join(
            OUTPUT_DIR, f"global_trends_{datetime.today().date()}.csv"
        )
        df.to_csv(filename, index=False)
        print(f"\nSerpAPI Trends collected: {len(df)} records")
        print(f"Saved to: {filename}")
        return df
    else:
        print("No SerpAPI data collected.")
        return pd.DataFrame()

if __name__ == "__main__":
    collect()