"""
StackExchange Collector
========================
Collects questions from Stack Overflow via the StackExchange API.

Saves to: data/raw/social_media/stackexchange_YYYY-MM-DD.csv
"""

import requests
import pandas as pd
import os
import time
from datetime import datetime
from tqdm import tqdm

# Setup absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "social_media")

BASE_URL = "https://api.stackexchange.com/2.3/questions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_questions = []
page = 1

print("Collecting questions from Stack Overflow...")
while page <= 10:   # collect 1000 questions
    try:
        params = {
            "order": "desc",
            "sort": "activity",
            "site": "stackoverflow",
            "pagesize": 100,
            "page": page
        }

        response = requests.get(BASE_URL, params=params, timeout=15)
        
        if response.status_code != 200:
            print(f"\nAPI Error {response.status_code}: {response.text}")
            break
            
        data = response.json()
        items = data.get("items", [])
        
        if not items:
            break

        for q in items:
            all_questions.append({
                "title": q.get("title"),
                "score": q.get("score"),
                "answers": q.get("answer_count"),
                "tags": ",".join(q.get("tags", [])),
                "creation_date": q.get("creation_date"),
                "link": q.get("link")
            })

        page += 1
        time.sleep(1)
    except Exception as e:
        print(f"\nError on page {page}: {e}")
        break

if all_questions:
    df = pd.DataFrame(all_questions)
    filename = os.path.join(OUTPUT_DIR, f"stackexchange_{datetime.today().date()}.csv")
    df.to_csv(filename, index=False)
    print(f"StackExchange collection finished: {len(df)}")
    print(f"Saved to: {filename}")
else:
    print("No StackExchange questions collected.")