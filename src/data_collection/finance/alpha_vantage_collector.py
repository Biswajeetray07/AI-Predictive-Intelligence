import os
import requests
import pandas as pd
import time
from typing import List, Optional
from dotenv import load_dotenv

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class AlphaVantageCollector:
    """
    Collector for raw OHLCV financial data from Alpha Vantage API.
    Designed as a fallback or alternative to Yahoo Finance.
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize the Alpha Vantage Collector.
        
        Args:
            api_key: Alpha Vantage API key (defaults to ALPHA_VANTAGE_KEY env var)
            output_dir: Directory to save CSVs (defaults to data/raw/financial/stocks)
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_KEY")
        if not self.api_key:
            logger.warning("ALPHA_VANTAGE_KEY environment variable not set. API calls will fail.")
            
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
        
        self.output_dir = output_dir or os.path.join(project_root, "data", "raw", "financial", "alpha_vantage")
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_ohlcv(self, symbol: str, output_size: str = "full") -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV data for a given symbol.
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            output_size: 'compact' (last 100 days) or 'full' (up to 20 years)
            
        Returns:
            DataFrame containing OHLCV data, or None if failed.
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",
            "apikey": self.api_key,
            "datatype": "json"
        }
        
        try:
            logger.info(f"Fetching Alpha Vantage daily data for {symbol}...")
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors or rate limits
            if "Error Message" in data:
                logger.error(f"Alpha Vantage Error for {symbol}: {data['Error Message']}")
                return None
            if "Note" in data: # Standard Free Tier rate limit note
                logger.warning(f"Alpha Vantage Rate Limit reached while fetching {symbol}")
                return None
                
            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                logger.warning(f"No daily time series data found for {symbol}")
                return None
                
            # Convert JSON to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            df.index.name = "date"
            df = df.reset_index()
            
            # Rename columns to standard OHLCV format
            df = df.rename(columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume"
            })
            
            # Convert types
            df["date"] = pd.to_datetime(df["date"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])
                
            # Sort explicitly ascending (oldest to newest)
            df = df.sort_values("date").reset_index(drop=True)
            df["ticker"] = symbol
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
            return None

    def collect_multiple(self, symbols: List[str], delay_seconds: int = 15):
        """
        Collect data for multiple symbols with rate limiting.
        Note: Free tier is 5 requests/minute, hence 12-15s delay.
        """
        collected = 0
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol)
            if df is not None and not df.empty:
                filepath = os.path.join(self.output_dir, f"{symbol}_av.csv")
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {len(df)} rows for {symbol} to {filepath}")
                collected += 1
                
            time.sleep(delay_seconds)
            
        logger.info(f"Successfully collected Alpha Vantage data for {collected}/{len(symbols)} symbols")

if __name__ == "__main__":
    # Test execution
    collector = AlphaVantageCollector()
    collector.collect_multiple(["AAPL", "MSFT"])
