import requests
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def collect_newsapi_data():
    """Collect news articles from NewsAPI."""
    # Setup absolute paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "news", "newsapi")
    
    API_KEY = os.getenv("NEWSAPI_KEY")
    BASE_URL = "https://newsapi.org/v2/everything"
    query = "AI OR technology OR finance OR economy"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_articles = []

    print(f"Fetching news articles for query: {query}")
    for page in tqdm(range(1, 11)):
        try:
            params = {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "page": page,
                "apiKey": API_KEY
            }

            response = requests.get(BASE_URL, params=params, timeout=15)
            data = response.json()

            if data.get("status") != "ok":
                print(f"\nAPI error on page {page}: {data}")
                break

            articles = data.get("articles", [])

            if not articles:
                print(f"\nNo more articles available at page {page}.")
                break

            for article in articles:
                all_articles.append({
                    "source": article.get("source", {}).get("name"),
                    "author": article.get("author"),
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "content": article.get("content"),
                    "url": article.get("url"),
                    "published_at": article.get("publishedAt")
                })
        except Exception as e:
            print(f"\nError fetching page {page}: {e}")
            continue

    if all_articles:
        df = pd.DataFrame(all_articles)
        filename = os.path.join(OUTPUT_DIR, f"newsapi_{datetime.today().date()}.csv")
        df.to_csv(filename, index=False)
        print(f"Total articles collected: {len(df)}")
        print(f"Data saved to: {filename}")
    else:
        print("No articles were successfully collected.")

def main():
    collect_newsapi_data()

if __name__ == "__main__":
    main()