"""
OpenWeather Data Collector
==========================
Collects live weather data for major global cities using the OpenWeather API.
"""

import os
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# Setup absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

CITIES = [
    "New York","Los Angeles","Chicago","Houston","San Francisco","Seattle","Boston","Miami","Dallas","Atlanta",
    "London","Paris","Berlin","Madrid","Rome","Amsterdam","Brussels","Vienna","Zurich","Prague",
    "Moscow","Istanbul","Athens","Warsaw","Stockholm","Copenhagen","Helsinki","Oslo","Lisbon","Dublin",
    "Delhi","Mumbai","Bangalore","Chennai","Hyderabad","Kolkata","Pune","Ahmedabad","Jaipur","Lucknow",
    "Tokyo","Osaka","Seoul","Beijing","Shanghai","Hong Kong","Singapore","Bangkok","Jakarta","Kuala Lumpur",
    "Manila","Hanoi","Ho Chi Minh City","Taipei","Dhaka","Karachi","Lahore","Colombo","Kathmandu","Yangon",
    "Dubai","Abu Dhabi","Riyadh","Doha","Kuwait City","Tehran","Baghdad","Tel Aviv","Amman","Muscat",
    "Sydney","Melbourne","Brisbane","Perth","Auckland","Wellington",
    "Cape Town","Johannesburg","Nairobi","Cairo","Lagos","Accra","Addis Ababa","Casablanca","Tunis","Algiers",
    "Toronto","Vancouver","Montreal","Ottawa","Calgary",
    "Mexico City","Bogota","Lima","Santiago","Buenos Aires","Sao Paulo","Rio de Janeiro","Caracas","Quito","Montevideo"
]

def collect() -> pd.DataFrame:
    """Main collection function."""
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    API_KEY = os.getenv("OPENWEATHER_KEY")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "weather")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not API_KEY:
        print("OPENWEATHER_KEY not set in .env. Skipping.")
        return pd.DataFrame()

    print(f"Collecting weather data for {len(CITIES)} cities...")
    all_weather = []
    
    for city in tqdm(CITIES):
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": city,
                "appid": API_KEY,
                "units": "metric"
            }

            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                continue
                
            data = response.json()
            main_data = data.get("main", {})
            wind_data = data.get("wind", {})
            weather_list = data.get("weather", [{}])
            
            weather_data = {
                "city": city,
                "temperature": main_data.get("temp"),
                "humidity": main_data.get("humidity"),
                "pressure": main_data.get("pressure"),
                "wind_speed": wind_data.get("speed"),
                "weather": weather_list[0].get("main") if weather_list else None,
                "timestamp": datetime.now()
            }
            all_weather.append(weather_data)

            # Save individual city data as CSV (backward compatibility)
            city_df = pd.DataFrame([weather_data])
            filename = os.path.join(OUTPUT_DIR, f"{city.replace(' ', '_')}.csv")
            city_df.to_csv(filename, index=False)

        except Exception:
            continue

    if not all_weather:
        return pd.DataFrame()

    return pd.DataFrame(all_weather)

if __name__ == "__main__":
    df = collect()
    if not df.empty:
        print(f"Weather data collection completed. Collected {len(df)} records.")
    else:
        print("No weather data collected.")