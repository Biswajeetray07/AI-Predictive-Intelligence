import yfinance as yf
import pandas as pd
import requests
import os
import time
from datetime import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup

# ... (imports)

def get_tickers():
    """Fetch S&P 500 tickers from Wikipedia."""
    print("Fetching S&P 500 tickers from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    tickers = []
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        
        if table:
            for row in table.find_all('tr')[1:]:
                ticker = row.find_all('td')[0].text.strip()
                tickers.append(ticker)
            
            # fix tickers like BRK.B -> BRK-B
            tickers = [t.replace(".", "-") for t in tickers]
            print(f"Total tickers found: {len(tickers)}")
        else:
            # Fallback to pandas if BS fails to find the table by ID
            print("Table 'constituents' not found by ID, trying fallback...")
            tables = pd.read_html(response.text)
            for t in tables:
                if "Symbol" in t.columns:
                    tickers = t["Symbol"].tolist()
                    tickers = [str(tick).replace(".", "-") for tick in tickers]
                    break
            print(f"Total tickers found (fallback): {len(tickers)}")
    except Exception as e:
        print(f"Error fetching tickers: {e}")
    return tickers

def collect_stock_data(tickers):
    """Collect daily stock data for given tickers."""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "financial", "stocks")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Output directory: {OUTPUT_DIR}")
    print("Starting stock data collection...")
    
    for ticker in tqdm(tickers):
        filename = os.path.join(OUTPUT_DIR, f"{ticker}.csv")
        if os.path.exists(filename) and os.path.getsize(filename) > 1000:
            continue

        try:
            data = yf.download(
                ticker,
                start="2015-01-01",
                end=datetime.today().strftime("%Y-%m-%d"),
                progress=False
            )

            if data is None or data.empty:
                continue

            data = data.reset_index()
            data["ticker"] = ticker
            data.to_csv(filename, index=False)
            
            # prevent rate limits
            time.sleep(1)
        except Exception as e:
            print(f"\nError downloading {ticker}: {e}")

def main():
    tickers = get_tickers()
    if not tickers:
        print("No tickers found. Exiting.")
        return
    collect_stock_data(tickers)
    print("\nStock data collection completed")

if __name__ == "__main__":
    main()