import pandas as pd
import numpy as np
import logging

def align_time_series(df: pd.DataFrame, date_col='date', freq='D') -> pd.DataFrame:
    """
    Ensures the dataframe has a continuous date index per ticker.
    This is critical for time-series forecasting to avoid jumps.
    """
    # Group by ticker if exists, otherwise assume single target
    if 'Ticker' not in df.columns:
        df['Ticker'] = 'UNKNOWN'
        
    df[date_col] = pd.to_datetime(df[date_col])
    
    aligned_dfs = []
    # Using groupby can be slow for 500 stocks, but guarantees contiguous indices
    for ticker, group in df.groupby('Ticker'):
        # Deduplicate to avoid "cannot reindex on an axis with duplicate labels"
        group = group.drop_duplicates(subset=[date_col])
        group = group.set_index(date_col).sort_index()
        # Resample to business or daily frequency
        idx = pd.date_range(start=group.index.min(), end=group.index.max(), freq=freq)
        group = group.reindex(idx)
        group['Ticker'] = ticker
        aligned_dfs.append(group.reset_index().rename(columns={'index': date_col}))
        
    return pd.concat(aligned_dfs, ignore_index=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies forward fill only to handle NaNs introduced during alignment.
    NO backward fill — bfill leaks future information into past rows.
    """
    df = df.ffill()
    # Fill remaining NaNs (leading rows with no history) with 0
    df = df.fillna(0)
    return df


def generate_lag_features(df: pd.DataFrame, columns: list, lags: list, group_col='Ticker') -> pd.DataFrame:
    """
    Generates lag features for time-series models.
    """
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df.groupby(group_col)[col].shift(lag)
    return df


def generate_rolling_features(df: pd.DataFrame, columns: list, windows: list, group_col='Ticker') -> pd.DataFrame:
    """
    Generates rolling mean and standard deviation for time-series models.
    """
    for col in columns:
        if col not in df.columns:
            continue
        for window in windows:
            # Shift by 1 to prevent data leakage (using only past data to calc rolling for current prediction)
            shifted_group = df.groupby(group_col)[col].shift(1)
            df[f'{col}_rollmean_{window}'] = shifted_group.rolling(window).mean()
            df[f'{col}_rollstd_{window}'] = shifted_group.rolling(window).std()
    return df


def calculate_technical_indicators(df: pd.DataFrame, group_col='Ticker', close_col='Close') -> pd.DataFrame:
    """
    Calculates basic technical indicators like SMA, EMA, RSI, MACD.
    Optimized for large multi-ticker datasets.
    """
    if close_col not in df.columns:
        return df
        
    logging.info("Calculating technical indicators (optimized for large datasets)...")
    
    # Pre-calculate grouped column to avoid repeated groupbys
    df = df.sort_values([group_col, 'date'] if 'date' in df.columns else [group_col])
    grouped = df.groupby(group_col)[close_col]
    shifted_close = grouped.shift(1)
    
    # 1. Simple Moving Averages
    df['SMA_10'] = shifted_close.rolling(window=10, min_periods=1).mean().reset_index(0, drop=True)
    df['SMA_50'] = shifted_close.rolling(window=50, min_periods=1).mean().reset_index(0, drop=True)
    
    # 2. Exponential Moving Averages (optimized group-by)
    # ewm within groupby is notoriously slow in pandas for large datasets
    # We use transform to apply ewm per group
    df['EMA_10'] = df.groupby(group_col)[close_col].transform(lambda x: x.shift(1).ewm(span=10, adjust=False).mean())
    df['EMA_50'] = df.groupby(group_col)[close_col].transform(lambda x: x.shift(1).ewm(span=50, adjust=False).mean())
    
    # 3. RSI
    delta = grouped.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = -delta.clip(upper=0).fillna(0)
    
    # Using simple SMA for RSI approximation to speed up computation
    avg_gain = gain.groupby(df[group_col]).transform(lambda x: x.shift(1).rolling(window=14, min_periods=1).mean())
    avg_loss = loss.groupby(df[group_col]).transform(lambda x: x.shift(1).rolling(window=14, min_periods=1).mean())
    
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 4. MACD
    ema_12 = df.groupby(group_col)[close_col].transform(lambda x: x.shift(1).ewm(span=12, adjust=False).mean())
    ema_26 = df.groupby(group_col)[close_col].transform(lambda x: x.shift(1).ewm(span=26, adjust=False).mean())
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df.groupby(group_col)['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    
    # 5. Bollinger Bands (Phase 8 — new)
    bb_period = 20
    bb_sma = shifted_close.rolling(window=bb_period, min_periods=1).mean().reset_index(0, drop=True)
    bb_std = shifted_close.rolling(window=bb_period, min_periods=1).std().reset_index(0, drop=True)
    df['BB_Upper'] = bb_sma + 2 * bb_std
    df['BB_Lower'] = bb_sma - 2 * bb_std
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (bb_sma + 1e-10)
    
    # 6. Average True Range — ATR (Phase 8 — new)
    if 'High' in df.columns and 'Low' in df.columns:
        high = df.groupby(group_col)['High'].shift(1)
        low = df.groupby(group_col)['Low'].shift(1)
        prev_close = shifted_close
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR_14'] = true_range.groupby(df[group_col]).transform(lambda x: x.rolling(14, min_periods=1).mean())
    
    # 7. On-Balance Volume — OBV (Phase 8 — new)
    if 'Volume' in df.columns:
        price_change = grouped.diff()
        obv_direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
        volume_shifted = df.groupby(group_col)['Volume'].shift(1).fillna(0)
        df['OBV'] = (volume_shifted * obv_direction)
        df['OBV'] = df.groupby(group_col)['OBV'].cumsum()
    
    # Fill any NaNs created by rolling windows
    df = df.fillna(0)
    return df


def generate_sentiment_velocity(df: pd.DataFrame, sentiment_col: str = 'sentiment_positive', group_col: str = 'Ticker') -> pd.DataFrame:
    """
    Phase 8 — Sentiment Velocity: rate of change over 3 and 7 days.
    Captures momentum in market sentiment shifts.
    """
    if sentiment_col in df.columns:
        shifted = df.groupby(group_col)[sentiment_col].shift(1)
        df[f'{sentiment_col}_velocity_3d'] = shifted - df.groupby(group_col)[sentiment_col].shift(3)
        df[f'{sentiment_col}_velocity_7d'] = shifted - df.groupby(group_col)[sentiment_col].shift(7)
    return df
