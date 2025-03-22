import pandas as pd
import numpy as np
import requests
import datetime
from datetime import timedelta
import os

# Make sure data directories exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

def fetch_bitcoin_data(start_date, end_date, interval='1h'):
    """
    Fetch Bitcoin price data from a public API.
    Params:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data granularity (1h for 1-hour data)
    Returns:
        DataFrame with OHLCV data
    """
    # Convert dates to timestamps
    start_ts = int(datetime.datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    
    # Binance API endpoint for historical klines (candlestick) data
    url = 'https://api.binance.com/api/v3/klines'
    
    # Parameters for API request
    params = {
        'symbol': 'BTCUSDT',
        'interval': interval,
        'startTime': start_ts,
        'endTime': end_ts,
        'limit': 1000  # Max limit per request
    }
    
    all_data = []
    
    # Fetch data in chunks if needed
    while start_ts < end_ts:
        params['startTime'] = start_ts
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        
        # Update start_ts for next iteration
        start_ts = data[-1][0] + 1
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'quote_asset_volume', 'trades', 
                                         'taker_buy_base', 'taker_buy_quote', 'ignored'])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert price columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    return df[['open', 'high', 'low', 'close', 'volume']]

if __name__ == "__main__":
    print("Fetching Bitcoin price data...")
    # Fetch 4 months of 1-hour Bitcoin data
    start_date = '2024-11-01'
    end_date = '2025-03-01'
    bitcoin_data = fetch_bitcoin_data(start_date, end_date)
    
    # Save the raw data
    bitcoin_data.to_csv('data/raw/bitcoin_raw_data.csv')
    print(f"Raw data saved to data/raw/bitcoin_raw_data.csv ({len(bitcoin_data)} rows)")