from pycoingecko import CoinGeckoAPI
import pandas as pd
import os
import time
from tqdm import tqdm

def collect_crypto_data():
    """Collect market charts for top 100 coins."""
    # Setup absolute paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "financial", "crypto")
    
    cg = CoinGeckoAPI()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching top 100 coins by market cap...")
    try:
        coins = cg.get_coins_markets(
            vs_currency="usd",
            order="market_cap_desc",
            per_page=100,
            page=1
        )
        coin_ids = [coin["id"] for coin in coins]
    except Exception as e:
        print(f"Failed to fetch coin list: {e}")
        coin_ids = []

    if not coin_ids:
        print("No coins found to collect.")
        return

    print(f"Collecting market charts for {len(coin_ids)} coins...")
    for coin in tqdm(coin_ids):
        filename = os.path.join(OUTPUT_DIR, f"{coin}.csv")
        if os.path.exists(filename) and os.path.getsize(filename) > 500:
            continue

        try:
            data = cg.get_coin_market_chart_by_id(
                id=coin,
                vs_currency="usd",
                days=365
            )

            prices = data.get("prices", [])
            if not prices:
                continue

            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["coin"] = coin

            df.to_csv(filename, index=False)
            time.sleep(1.5)  # Slightly increased delay for stability

        except Exception as e:
            print(f"\nError for {coin}: {e}")
            if "429" in str(e):
                print("Rate limit detected, sleeping for 30s...")
                time.sleep(30)
    print("Crypto data collection completed.")

def main():
    collect_crypto_data()

if __name__ == "__main__":
    main()