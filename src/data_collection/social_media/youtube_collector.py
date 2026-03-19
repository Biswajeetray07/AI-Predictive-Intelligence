"""
YouTube Trends Collector — Async Edition
==========================================
Searches for AI/Tech videos across multiple topics and fetches
video statistics concurrently using aiohttp.

Saves to: data/raw/social_media/youtube_trends_YYYY-MM-DD.parquet
"""

import asyncio
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any, cast, Optional
from dotenv import load_dotenv

from src.data_collection.async_fetcher import AsyncFetcher

load_dotenv()

# Setup absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

TOPICS = [
    "Artificial Intelligence", "Machine Learning", "AI News", "Deep Learning",
    "AI Startups", "AI Technology", "Robotics", "Data Science", "AI Research", "AI Future",
]

async def collect_youtube():
    load_dotenv()
    API_KEY = os.getenv("YOUTUBE_API_KEY")
    SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "social_media")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fetcher = AsyncFetcher(max_concurrency=10, rate_limit_delay=0.3, timeout=15)

    # --- Phase 1: Collect video IDs from all topics concurrently ---
    search_urls = [SEARCH_URL] * len(TOPICS)
    search_params = [
        {
            "part": "snippet",
            "q": topic,
            "type": "video",
            "maxResults": 50,
            "key": API_KEY,
        }
        for topic in TOPICS
    ]

    print(f"Searching {len(TOPICS)} topics concurrently...")
    search_results = await fetcher.fetch_all(search_urls, cast(Optional[List[Optional[Dict[str, Any]]]], search_params))

    # Collect video IDs and snippets
    video_ids = []
    snippets = {}
    topic_map = {}

    for i, (url, data) in enumerate(search_results):
        if data is None:
            continue
        for item in data.get("items", []):
            vid_id = item.get("id", {}).get("videoId")
            if vid_id:
                video_ids.append(vid_id)
                snippets[vid_id] = item["snippet"]
                topic_map[vid_id] = TOPICS[i]

    print(f"Collected {len(video_ids)} video IDs. Fetching statistics...")

    # --- Phase 2: Fetch video statistics in batches of 50 (API limit) ---
    batch_size = 50
    stats_urls = []
    stats_params = []
    for start in range(0, len(video_ids), batch_size):
        batch = video_ids[start : start + batch_size]
        stats_urls.append(VIDEO_URL)
        stats_params.append({
            "part": "statistics",
            "id": ",".join(batch),
            "key": API_KEY,
        })

    stats_results = await fetcher.fetch_all(stats_urls, stats_params)

    videos = []
    for url, stats_data in stats_results:
        if stats_data is None:
            continue
        for video in stats_data.get("items", []):
            vid = video["id"]
            stats = video.get("statistics", {})
            snippet = snippets.get(vid, {})

            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))
            comments = int(stats.get("commentCount", 0))

            videos.append({
                "video_id": vid,
                "topic": topic_map.get(vid, "unknown"),
                "title": snippet.get("title"),
                "channel": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"),
                "views": views,
                "likes": likes,
                "comments": comments,
                "engagement_score": likes + comments,
                "description": snippet.get("description"),
            })

    if videos:
        df = pd.DataFrame(videos)
        filename = os.path.join(
            OUTPUT_DIR, f"youtube_trends_{datetime.today().date()}.parquet"
        )
        df.to_parquet(filename, index=False)
        print(f"\nVideos collected: {len(df)}")
        print(f"Saved to: {filename}")
        return df
    else:
        print("No YouTube data collected.")
        return pd.DataFrame()

def collect():
    """Sync wrapper for scheduler."""
    return asyncio.run(collect_youtube())

if __name__ == "__main__":
    collect()