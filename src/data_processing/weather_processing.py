import os
import glob
import pandas as pd
import logging
from typing import Optional

# No sys.path hack - run with PYTHONPATH=. from root
from dotenv import load_dotenv
load_dotenv()

from src.data_processing.utils.processing_utils import setup_processing_logger, safe_read_csv, validate_processed_schema

class WeatherDataProcessor:
    def __init__(self, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)
        self.raw_dir = os.path.join(self.base_dir, 'data', 'raw', 'weather')
        self.processed_dir = os.path.join(self.base_dir, 'data', 'processed', 'weather')
        self.logger = setup_processing_logger('weather_processing')

        # Ensure processed directories exist
        os.makedirs(self.processed_dir, exist_ok=True)

    def process_weather_data(self) -> Optional[pd.DataFrame]:
        """Processes raw weather data combining cities into a global daily average."""
        self.logger.info("Starting Weather Data Processing...")
        
        all_files = glob.glob(os.path.join(self.raw_dir, '*.csv'))
        
        if not all_files:
            self.logger.warning("No weather CSV files found in the raw directory.")
            return None

        df_list = []
        for file in all_files:
            df = safe_read_csv(file, self.logger)
            if df is not None and not df.empty:
                df_list.append(df)
                
        if not df_list:
            self.logger.warning("No valid weather data could be loaded.")
            return None

        # Combine all weather data
        weather_df = pd.concat(df_list, ignore_index=True)
        
        required_cols = ['timestamp', 'temperature', 'humidity', 'pressure', 'wind_speed']
        for col in required_cols:
            if col not in weather_df.columns:
                self.logger.error(f"Missing essential column '{col}' in weather data. Available: {weather_df.columns.tolist()}")
                return None
                
        # Parse timestamp to date
        weather_df['date'] = pd.to_datetime(weather_df['timestamp'], errors='coerce').dt.date
        weather_df = weather_df.dropna(subset=['date'])
        
        # Ensure numeric columns
        numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
        for col in numeric_cols:
            weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')
            
        weather_df = weather_df.dropna(subset=numeric_cols)
        
        if 'weather' in weather_df.columns:
            weather_df['is_extreme_weather'] = weather_df['weather'].str.lower().str.contains('storm|hurricane|tornado|extreme', na=False).astype(int)
            weather_df['is_raining'] = weather_df['weather'].str.lower().str.contains('rain|drizzle|shower', na=False).astype(int)
        else:
            weather_df['is_extreme_weather'] = 0
            weather_df['is_raining'] = 0

        # Calculate Global Daily Averages across all cities
        # Since cities differ by row, we just group by date and mean numeric columns
        # To make it robust, dynamically build dict
        
        daily_weather = weather_df.groupby('date').agg(
            global_avg_temp=pd.NamedAgg(column='temperature', aggfunc='mean'),
            global_avg_humidity=pd.NamedAgg(column='humidity', aggfunc='mean'),
            global_avg_pressure=pd.NamedAgg(column='pressure', aggfunc='mean'),
            global_avg_wind_speed=pd.NamedAgg(column='wind_speed', aggfunc='mean'),
            extreme_weather_events=pd.NamedAgg(column='is_extreme_weather', aggfunc='sum'),
            rain_events=pd.NamedAgg(column='is_raining', aggfunc='sum')
        ).reset_index()
        
        if 'city' in weather_df.columns:
             city_count = weather_df.groupby('date')['city'].nunique().reset_index(name='cities_reported')
             daily_weather = pd.merge(daily_weather, city_count, on='date', how='left')
        
        daily_weather['date'] = pd.to_datetime(daily_weather['date']).dt.strftime('%Y-%m-%d')
        daily_weather = daily_weather.sort_values('date')
        
        output_path = os.path.join(self.processed_dir, 'weather_processed.csv')
        
        if validate_processed_schema(daily_weather, ['date', 'global_avg_temp', 'global_avg_humidity'], self.logger):
             daily_weather.to_csv(output_path, index=False)
             self.logger.info(f"Successfully processed {len(daily_weather)} days of weather data. Saved to {output_path}")
             return daily_weather
        else:
             self.logger.error("Weather data failed schema validation.")
             return None

    def run_all(self):
        self.logger.info("Starting full weather data processing pipeline.")
        self.process_weather_data()
        self.logger.info("Completed full weather data processing pipeline.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    processor = WeatherDataProcessor(base_dir=project_root)
    processor.run_all()
