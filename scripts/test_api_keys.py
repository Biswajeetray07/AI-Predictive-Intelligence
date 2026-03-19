import os
import requests
import asyncio
from dotenv import load_dotenv
import pandas as pd
from fredapi import Fred
from pycoingecko import CoinGeckoAPI
from serpapi import GoogleSearch
import ssl
import json

# Load environment variables
load_dotenv()

results_data = []

def record_result(name, success, message=""):
    status = "✅ PASS" if success else "❌ FAIL"
    results_data.append({"Service": name, "Status": status, "Details": message})
    print(f"[{status}] {name}: {message}")

def test_newsapi():
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        record_result("NewsAPI", False, "NEWSAPI_KEY missing in .env")
        return
    url = f"https://newsapi.org/v2/everything?q=AI&apiKey={api_key}&pageSize=1"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            record_result("NewsAPI", True, "Authenticated successfully")
        else:
            try:
                error_msg = resp.json().get('message', resp.text)
            except:
                error_msg = resp.text
            record_result("NewsAPI", False, f"HTTP {resp.status_code}: {error_msg}")
    except Exception as e:
        record_result("NewsAPI", False, str(e))

def test_alpha_vantage():
    api_key = os.getenv("ALPHA_VANTAGE_KEY")
    if not api_key:
        record_result("Alpha Vantage", False, "ALPHA_VANTAGE_KEY missing in .env")
        return
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "Time Series (Daily)" in data:
            record_result("Alpha Vantage", True, "Authenticated successfully")
        elif "Note" in data:
            record_result("Alpha Vantage", True, "Authenticated but Rate Limited (Free Tier)")
        else:
            record_result("Alpha Vantage", False, f"Error: {data.get('Error Message', 'Unknown')}")
    except Exception as e:
        record_result("Alpha Vantage", False, str(e))

def test_fred():
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        record_result("FRED", False, "FRED_API_KEY missing in .env")
        return
    try:
        # Workaround for SSL issues on Mac
        import urllib.request
        orig_context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        
        fred = Fred(api_key=api_key)
        data = fred.get_series("GDP", limit=1)
        
        # Restore SSL context
        ssl._create_default_https_context = orig_context
        
        if not data.empty:
            record_result("FRED", True, "Authenticated successfully")
        else:
            record_result("FRED", False, "No data returned")
    except Exception as e:
        record_result("FRED", False, f"Error: {str(e)}")

def test_openweather():
    api_key = os.getenv("OPENWEATHER_KEY")
    if not api_key:
        record_result("OpenWeather", False, "OPENWEATHER_KEY missing in .env")
        return
    url = f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            record_result("OpenWeather", True, "Authenticated successfully")
        else:
            record_result("OpenWeather", False, f"HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        record_result("OpenWeather", False, str(e))

def test_eia():
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        record_result("EIA", False, "EIA_API_KEY missing in .env")
        return
    # Fixed parameter from 'value' to 'sales' as per error message
    url = f"https://api.eia.gov/v2/electricity/retail-sales/data/?api_key={api_key}&frequency=monthly&data[0]=sales&length=1"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            record_result("EIA", True, "Authenticated successfully")
        else:
            record_result("EIA", False, f"HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        record_result("EIA", False, str(e))

def test_adzuna():
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        record_result("Adzuna", False, "ADZUNA_APP_ID/KEY missing in .env")
        return
    url = f"https://api.adzuna.com/v1/api/jobs/us/search/1?app_id={app_id}&app_key={app_key}&results_per_page=1"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            record_result("Adzuna", True, "Authenticated successfully")
        else:
            record_result("Adzuna", False, f"HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        record_result("Adzuna", False, str(e))

def test_usajobs():
    api_key = os.getenv("USAJOBS_API_KEY")
    email = os.getenv("USAJOBS_EMAIL")
    if not api_key or not email:
        record_result("USAJobs", False, "USAJOBS_API_KEY/EMAIL missing in .env")
        return
    headers = {
        "Authorization-Key": api_key,
        "User-Agent": email,
        "Host": "data.usajobs.gov",
    }
    url = "https://data.usajobs.gov/api/Search?Keyword=IT"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            record_result("USAJobs", True, "Authenticated successfully")
        else:
            record_result("USAJobs", False, f"HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        record_result("USAJobs", False, str(e))

def test_serpapi():
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        record_result("SerpAPI", False, "SERPAPI_KEY missing in .env")
        return
    
    # Debug: Print masked key
    if api_key and len(api_key) > 8:
        masked = f"{api_key[:4]}...{api_key[-4:]}"
    else:
        masked = "****"
    print(f"DEBUG: SerpAPI Key used: {masked}")
    try:
        params = {
            "engine": "google_trends_trending_now",
            "geo": "US",
            "api_key": api_key
        }
        # Direct check instead of using search library to control timeouts better
        url = "https://serpapi.com/search.json"
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if "error" not in data:
            record_result("SerpAPI", True, "Authenticated successfully")
        else:
            record_result("SerpAPI", False, data.get("error"))
    except Exception as e:
        record_result("SerpAPI", False, str(e))

def test_mastodon():
    token = os.getenv("MASTODON_ACCESS_TOKEN")
    if not token:
        record_result("Mastodon", False, "MASTODON_ACCESS_TOKEN missing in .env")
        return
    headers = {"Authorization": f"Bearer {token}"}
    url = "https://mastodon.social/api/v1/timelines/public?limit=1"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            record_result("Mastodon", True, "Authenticated successfully")
        else:
            record_result("Mastodon", False, f"HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        record_result("Mastodon", False, str(e))

def test_youtube():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        record_result("YouTube", False, "YOUTUBE_API_KEY missing in .env")
        return
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q=AI&type=video&maxResults=1&key={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            record_result("YouTube", True, "Authenticated successfully")
        else:
            try:
                error_msg = resp.json().get('error', {}).get('message', resp.text)
            except:
                error_msg = resp.text
            record_result("YouTube", False, f"HTTP {resp.status_code}: {error_msg}")
    except Exception as e:
        record_result("YouTube", False, str(e))

def test_coingecko():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    try:
        # Public API is heavily rate limited.
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            record_result("CoinGecko", True, "Successfully fetched price (Public API)")
        elif resp.status_code == 429:
             record_result("CoinGecko", False, "Rate Limited (HTTP 429). Public API is restricted.")
        else:
            record_result("CoinGecko", False, f"HTTP {resp.status_code}")
    except Exception as e:
        record_result("CoinGecko", False, str(e))

def test_gdelt():
    url = "https://api.gdeltproject.org/api/v2/doc/doc?query=AI&mode=ArtList&maxrecords=1&format=json"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            try:
                data = resp.json()
                if "articles" in data:
                    record_result("GDELT", True, "Successfully fetched articles")
                else:
                    record_result("GDELT", False, "No articles in response (might be query specific)")
            except:
                record_result("GDELT", False, "Failed to parse JSON response")
        else:
            record_result("GDELT", False, f"HTTP {resp.status_code}")
    except Exception as e:
        record_result("GDELT", False, str(e))

def main():
    print("--- API Key Validation Script (Refined) ---")
    
    # Run tests
    test_newsapi()
    test_alpha_vantage()
    test_fred()
    test_openweather()
    test_eia()
    test_adzuna()
    test_usajobs()
    test_serpapi()
    test_mastodon()
    test_youtube()
    test_coingecko()
    test_gdelt()
    
    print("\n--- Summary Report ---")
    df = pd.DataFrame(results_data)
    print(df.to_string(index=False))
    print("-------------------------------------------")

if __name__ == "__main__":
    main()
