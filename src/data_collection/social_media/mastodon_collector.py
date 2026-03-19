"""
Mastodon Social Media Collector
=================================
Collects public posts from multiple Mastodon instances.
Extracts engagement metrics, hashtags, and content for sentiment analysis.

Saves to: data/raw/social_media/mastodon_trends_YYYY-MM-DD.csv
"""

import requests
import pandas as pd
import os
import time
import random
import re
from datetime import datetime
from dotenv import load_dotenv

# ─── Path Setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

# ─── Mastodon Instances ──────────────────────────────────────────────────────
INSTANCES = [
    "https://mastodon.social",
    "https://mastodon.world",
    "https://fosstodon.org",
    "https://techhub.social",
    "https://hachyderm.io",
    "https://mastodon.online",
    "https://mstdn.social"
]

def get_headers():
    load_dotenv()
    ACCESS_TOKEN = os.getenv("MASTODON_ACCESS_TOKEN")
    return {
        "User-Agent": "TrendCollectorBot/1.0",
        "Accept": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }

# ─── State ──────────────────────────────────────────────────────────────────
seen_ids = set()


def clean_text(text):
    text = re.sub("<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = text.replace("\n", " ").strip()
    return text


def extract_hashtags(text):
    return re.findall(r"#(\w+)", text)


def fetch_posts(instance, headers):
    collected = []
    max_id = None
    pages = 20

    for page in range(pages):
        try:
            url = f"{instance}/api/v1/timelines/public"
            params = {
                "limit": 40,
                "max_id": max_id,
                "local": False
            }

            response = requests.get(
                url, headers=headers, params=params, timeout=20
            )

            if response.status_code != 200:
                print("  API error:", response.status_code)
                break

            data = response.json()
            if not isinstance(data, list) or len(data) == 0:
                break

            for post in data:
                post_id = post.get("id")
                if post_id in seen_ids:
                    continue
                seen_ids.add(post_id)

                raw_text = post.get("content", "")
                clean = clean_text(raw_text)
                hashtags = extract_hashtags(clean)
                user = post.get("account", {})

                engagement = (
                    post.get("replies_count", 0)
                    + post.get("reblogs_count", 0)
                    + post.get("favourites_count", 0)
                )

                collected.append({
                    "post_id": post_id,
                    "instance": instance,
                    "created_at": post.get("created_at"),
                    "language": post.get("language"),
                    "user_id": user.get("id"),
                    "username": user.get("username"),
                    "followers": user.get("followers_count"),
                    "content_raw": raw_text,
                    "content_clean": clean,
                    "hashtags": ",".join(hashtags),
                    "replies": post.get("replies_count"),
                    "reblogs": post.get("reblogs_count"),
                    "likes": post.get("favourites_count"),
                    "engagement_score": engagement
                })

            max_id = data[-1]["id"]
            time.sleep(random.uniform(2, 4))

        except Exception as e:
            print("  Error:", instance, e)
            break

    return collected


def collect():
    """Main collection entry point."""
    print("Starting Mastodon Collector...\n")
    
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "social_media")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    headers = get_headers()
    all_posts = []

    for instance in INSTANCES:
        print("Collecting from:", instance)
        new_posts = fetch_posts(instance, headers)
        if new_posts:
            all_posts.extend(new_posts)
            print("  Collected:", len(new_posts))
        else:
            print("  No posts collected")
        time.sleep(random.uniform(2, 5))

    if all_posts:
        df = pd.DataFrame(all_posts)
        filename = os.path.join(
            OUTPUT_DIR,
            f"mastodon_trends_{datetime.today().date()}.csv"
        )
        df.to_csv(filename, index=False)
        print("\nTotal posts collected:", len(df))
        print("Dataset saved:", filename)
        return df
    else:
        print("\nNo Mastodon posts collected.")
        return pd.DataFrame()


if __name__ == "__main__":
    collect()