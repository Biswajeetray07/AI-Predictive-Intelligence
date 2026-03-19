"""
UN Comtrade Collector
======================
Collects international trade data (imports/exports) from the UN Comtrade API.

Endpoint: https://comtradeapi.un.org/data/v1/get/C/A/
Saves to: data/raw/trade/un_comtrade_YYYY-MM-DD.parquet
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Configuration constants
BASE_URL = "https://comtradeapi.un.org/data/v1/get/C/A"

# Major reporting countries (ISO codes)
REPORTER_CODES = ["840", "156", "276", "392", "826", "356", "076", "250", "410", "036"]
# Top commodity codes (HS2): Machinery, Electronics, Vehicles, Chemicals, Fuels
COMMODITY_CODES = ["84", "85", "87", "29", "27"]


def collect() -> pd.DataFrame:
    """Collect UN Comtrade trade data."""
    # Setup paths relative to project root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "trade")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Starting UN Comtrade Collection...\n")

    all_records = []
    current_year = datetime.now().year

    for reporter in REPORTER_CODES:
        try:
            params = {
                "reporterCode": reporter,
                "period": str(current_year - 1),
                "partnerCode": "0",  # World
                "cmdCode": ",".join(COMMODITY_CODES),
                "flowCode": "M,X",  # Import and Export
                "maxRecords": 500,
            }

            response = requests.get(BASE_URL, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                records = data.get("data", [])

                for record in records:
                    all_records.append({
                        "reporter": record.get("reporterDesc", ""),
                        "reporter_code": record.get("reporterCode", ""),
                        "partner": record.get("partnerDesc", ""),
                        "partner_code": record.get("partnerCode", ""),
                        "commodity": record.get("cmdDesc", ""),
                        "commodity_code": record.get("cmdCode", ""),
                        "flow": record.get("flowDesc", ""),
                        "trade_value_usd": record.get("primaryValue", 0),
                        "net_weight_kg": record.get("netWgt", 0),
                        "qty": record.get("qty", 0),
                        "period": record.get("period", ""),
                        "year": record.get("refYear", current_year - 1),
                    })

                print(f"  Reporter {reporter}: {len(records)} records")
            elif response.status_code == 403:
                print(f"  Reporter {reporter}: Access forbidden (API key may be required)")
            elif response.status_code == 429:
                print(f"  Rate limited. Waiting 30s...")
                time.sleep(30)
            else:
                print(f"  Reporter {reporter}: HTTP {response.status_code}")

            time.sleep(2)  # Respect rate limits

        except Exception as e:
            print(f"  Error for reporter {reporter}: {e}")

    if all_records:
        df = pd.DataFrame(all_records)
        filename = os.path.join(
            OUTPUT_DIR, f"un_comtrade_{datetime.today().date()}.parquet"
        )
        df.to_parquet(filename, index=False)
        print(f"\nUN Comtrade collected: {len(df)} records")
        print(f"Saved to: {filename}")
        return df
    else:
        print("No UN Comtrade data collected.")
        return pd.DataFrame()


if __name__ == "__main__":
    collect()
