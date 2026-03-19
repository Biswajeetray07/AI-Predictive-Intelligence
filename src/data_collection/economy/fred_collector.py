import os
import sys
import ssl
import pandas as pd
from datetime import datetime

# Workaround for SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

from fredapi import Fred
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Setup absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

# Cleaned up trailing space from API key
API_KEY = os.getenv("FRED_API_KEY")

try:
    if not API_KEY:
        print("FRED_API_KEY not found in .env")
        fred = None
    else:
        fred = Fred(api_key=API_KEY)
except Exception as e:
    print(f"Failed to initialize FRED API: {e}")
    fred = None

# important macroeconomic indicators
series_ids = {
    "GDP": "GDP",
    "REAL_GDP": "GDPC1",
    "CPI": "CPIAUCSL",
    "PPI": "PPIACO",
    "UNEMPLOYMENT_RATE": "UNRATE",
    "PAYROLL_EMPLOYMENT": "PAYEMS",
    "INTEREST_RATE": "FEDFUNDS",
    "10Y_TREASURY": "DGS10",
    "2Y_TREASURY": "DGS2",
    "CONSUMER_SENTIMENT": "UMCSENT",
    "RETAIL_SALES": "RSAFS",
    "INDUSTRIAL_PRODUCTION": "INDPRO",
    "HOUSING_STARTS": "HOUST",
    "BUILDING_PERMITS": "PERMIT",
    "MONEY_SUPPLY_M1": "M1SL",
    "MONEY_SUPPLY_M2": "M2SL",
    "PERSONAL_INCOME": "PI",
    "PERSONAL_CONSUMPTION": "PCE",
    "EXPORTS": "EXPGS",
    "IMPORTS": "IMPGS",
    "BUSINESS_INVENTORIES": "BUSINV",
    "CORPORATE_PROFITS": "CP",
    "DOLLAR_INDEX": "DTWEXBGS",
    "CRUDE_OIL_PRICE": "DCOILWTICO",
    "NATURAL_GAS_PRICE": "DHHNGSP",
    "GOLD_PRICE": "ID7108",
    "SILVER_PRICE": "IP7106",
    "MORTGAGE_RATE": "MORTGAGE30US",
    "CREDIT_CARD_RATE": "TERMCBCCALLNS",
    "BANK_PRIME_RATE": "DPRIME",
    "TOTAL_PUBLIC_DEBT": "GFDEBTN",
    "FED_BALANCE_SHEET": "WALCL",
    "HOUSE_PRICE_INDEX": "CSUSHPINSA",
    "AUTO_SALES": "TOTALSA",
    "TRUCK_SALES": "TRUCKD11",
    "AIR_PASSENGERS": "ENPLANE",
    "MANUFACTURING_OUTPUT": "OUTMS",
    "CAPACITY_UTILIZATION": "TCU",
    "LABOR_PRODUCTIVITY": "OPHNFB",
    "UNIT_LABOR_COST": "ULCNFB",
    "JOB_OPENINGS": "JTSJOL",
    "QUIT_RATE": "JTSQUR",
    "LAYOFF_RATE": "JTSLDR",
    "SMALL_BUSINESS_OPTIMISM": "NFIBINDEX",
    "VIX_INDEX": "VIXCLS",
    "CORPORATE_BOND_YIELD": "BAA10Y",
    "AAA_BOND_YIELD": "AAA10Y",
    "COMMERCIAL_PAPER_RATE": "DCPN3M"
}

# Consolidate all series into one DataFrame
all_data = []

if fred:
    print(f"Collecting {len(series_ids)} economic indicators...")
    for name, series in tqdm(series_ids.items()):
        try:
            data = fred.get_series(series)
            if data is not None and not data.empty:
                df_series = pd.DataFrame(data, columns=["value"])
                df_series["date"] = df_series.index
                df_series["indicator"] = name
                all_data.append(df_series)
        except Exception as e:
            print(f"\nError collecting indicator {name} ({series}): {e}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Ensure 'value' is numeric and no list-like objects exist
        if 'value' in final_df.columns:
            final_df['value'] = pd.to_numeric(final_df['value'], errors='coerce')
        
        # Clean up any column that might have list-like objects (unlikely but causes factorize errors)
        for col in final_df.columns:
            if final_df[col].apply(lambda x: isinstance(x, list)).any():
                final_df[col] = final_df[col].apply(lambda x: str(x) if isinstance(x, list) else x)

        # Update output directory to standard path
        OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "financial", "economic_indicators")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(OUTPUT_DIR, f"fred_collector_{today_str}.parquet")
        final_df.to_parquet(filename, index=False)
        print(f"Economic data collection completed. Saved to {filename}")
    else:
        print("No economic data collected.")
else:
    print("Skipping FRED collection due to initialization error.")