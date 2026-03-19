import requests
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import time

def collect_gdelt_data():
    """Collect news metadata from GDELT."""
    # Setup absolute paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "news", "gdelt")
    
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    queries = [
        "AI", "technology", "finance", "stock market",
        "cryptocurrency", "economy", "startup", "global markets"
    ]
    headers = {"User-Agent": "AI-Predictive-Intelligence-Collector"}
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_articles = []

    print("Collecting articles from GDELT...")
    for query in tqdm(queries):
        params = {
            "query": query, "mode": "ArtList", "maxrecords": 250,
            "format": "json", "sort": "DateDesc"
        }

        try:
            response = requests.get(BASE_URL, params=params, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"\nRequest failed for {query}: {response.status_code}")
                continue
            if not response.text.strip():
                continue

            data = response.json()
            if "articles" not in data:
                continue

            for article in data["articles"]:
                all_articles.append({
                    "query": query,
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "domain": article.get("domain"),
                    "language": article.get("language"),
                    "source_country": article.get("sourcecountry"),
                    "published_at": article.get("seendate")
                })
            time.sleep(1) # prevent API blocking
        except Exception as e:
            print(f"\nError for query {query}: {e}")
            continue

    if all_articles:
        df = pd.DataFrame(all_articles)
        df = df.drop_duplicates(subset="url")
        filename = os.path.join(OUTPUT_DIR, f"gdelt_{datetime.today().date()}.csv")
        df.to_csv(filename, index=False)
        print(f"Total articles collected: {len(df)}")
        print(f"Data saved to: {filename}")
    else:
        print("No articles collected.")

def main():
    collect_gdelt_data()

if __name__ == "__main__":
    main()