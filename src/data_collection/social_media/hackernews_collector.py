"""
Hacker News Collector — Async Edition
=======================================
Fetches story IDs from multiple endpoints and then retrieves up to 5000
story details concurrently using aiohttp.

Saves to: data/raw/social_media/hackernews_YYYY-MM-DD.parquet
"""

import asyncio
import pandas as pd
import os
from datetime import datetime

from src.data_collection.async_fetcher import AsyncFetcher

# Setup absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "social_media")

BASE_URL = "https://hacker-news.firebaseio.com/v0"

ENDPOINTS = [
    "topstories",
    "newstories",
    "beststories",
    "askstories",
    "showstories",
    "jobstories",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


async def collect_hackernews():
    fetcher = AsyncFetcher(max_concurrency=50, rate_limit_delay=0.02, timeout=10)

    # --- Phase 1: Fetch all story IDs concurrently ---
    print("Fetching story IDs from all endpoints...")
    endpoint_urls = [f"{BASE_URL}/{ep}.json" for ep in ENDPOINTS]
    id_results = await fetcher.fetch_all(endpoint_urls)

    all_ids = set()
    for url, data in id_results:
        if data and isinstance(data, list):
            all_ids.update(data)

    all_ids = list(all_ids)[:5000]
    print(f"Unique story IDs collected: {len(all_ids)}")

    # --- Phase 2: Fetch all story details concurrently ---
    print(f"Fetching details for {len(all_ids)} stories (async)...")
    story_urls = [f"{BASE_URL}/item/{sid}.json" for sid in all_ids]
    story_results = await fetcher.fetch_all(story_urls)

    data = []
    for url, story in story_results:
        if story is None:
            continue
        data.append({
            "id": story.get("id"),
            "title": story.get("title"),
            "type": story.get("type"),
            "author": story.get("by"),
            "score": story.get("score"),
            "time": story.get("time"),
            "url": story.get("url"),
            "comments": story.get("descendants"),
        })

    df = pd.DataFrame(data)
    filename = os.path.join(OUTPUT_DIR, f"hackernews_{datetime.today().date()}.parquet")
    df.to_parquet(filename, index=False)

    print(f"Total posts collected: {len(df)}")
    print(f"Data saved to: {filename}")
    return df

def collect():
    """Sync wrapper for scheduler."""
    return asyncio.run(collect_hackernews())

if __name__ == "__main__":
    collect()