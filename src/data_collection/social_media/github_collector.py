"""
GitHub Repository Collector
=============================
Collects trending AI repository data from GitHub Search API.

Saves to: data/raw/social_media/github_YYYY-MM-DD.csv
"""

import requests
import pandas as pd
import os
from datetime import datetime
import time
from tqdm import tqdm

# Setup absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "social_media")

BASE_URL = "https://api.github.com/search/repositories"

os.makedirs(OUTPUT_DIR, exist_ok=True)

repos_data = []

print("Collecting repository data from GitHub...")
for page in tqdm(range(1, 11)):
    try:
        params = {
            "q": "artificial intelligence",
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
            "page": page
        }

        response = requests.get(BASE_URL, params=params, timeout=15)
        
        if response.status_code != 200:
            print(f"\nAPI Error {response.status_code}: {response.text}")
            if response.status_code == 403:
                print("Rate limit reached. GitHub Search API has strict limits for unauthenticated requests.")
            break
            
        data = response.json()
        repos = data.get("items", [])

        if not repos:
            break

        for repo in repos:
            repos_data.append({
                "name": repo.get("name"),
                "stars": repo.get("stargazers_count"),
                "forks": repo.get("forks_count"),
                "language": repo.get("language"),
                "created_at": repo.get("created_at"),
                "updated_at": repo.get("updated_at"),
                "repo_url": repo.get("html_url")
            })

        # GitHub search API has a limit of 10 requests per minute for unauthenticated users
        time.sleep(7) 

    except Exception as e:
        print(f"\nError on page {page}: {e}")
        break

if repos_data:
    df = pd.DataFrame(repos_data)
    filename = os.path.join(OUTPUT_DIR, f"github_{datetime.today().date()}.csv")
    df.to_csv(filename, index=False)
    print(f"GitHub collection finished: {len(df)}")
    print(f"Saved to: {filename}")
else:
    print("No GitHub data collected.")