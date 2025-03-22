import pandas as pd
import numpy as np
import os

def load_preprocessed_data():
    """
    Load preprocessed Bitcoin price data.
    """
    file_path = 'data/processed/bitcoin_preprocessed.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessed data file not found at {file_path}. Run preprocess.py first.")
    
    return pd.read_csv(file_path, index_col=0, parse_dates=True)

def create_features(df):
    """
    Create technical indicators and features for Bitcoin price prediction.
    """
    # Make a copy of the dataframe
    data = df.copy()
    print(f"Initial data shape: {data.shape}")
    
    # Price-based features
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    
    # Volatility features
    data['volatility_1h'] = data['returns'].rolling(window=1).std() * np.sqrt(24)
    data['volatility_24h'] = data['returns'].rolling(window=24).std() * np.sqrt(24)
    
    # Simple Moving Averages
    data['sma_6h'] = data['close'].rolling(window=6).mean()
    data['sma_12h'] = data['close'].rolling(window=12).mean()
    data['sma_24h'] = data['close'].rolling(window=24).mean()
    
    # Exponential Moving Averages
    data['ema_6h'] = data['close'].ewm(span=6, adjust=False).mean()
    data['ema_12h'] = data['close'].ewm(span=12, adjust=False).mean()
    data['ema_24h'] = data['close'].ewm(span=24, adjust=False).mean()
    
    # MACD
    data['macd'] = data['ema_12h'] - data['ema_24h']
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    # RSI (Relative Strength Index)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    # Check if loss contains any zeros before division
    if (loss == 0).any():
        print("Warning: Division by zero in RSI calculation")
        # Replace zeros with a small value to avoid division by zero
        loss = loss.replace(0, 1e-10)
    
    rs = gain / loss
    data['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['bb_middle'] = data['close'].rolling(window=20).mean()
    data['bb_std'] = data['close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
    data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    
    # Volume features
    data['volume_change'] = data['volume'].pct_change()
    data['volume_ma_6h'] = data['volume'].rolling(window=6).mean()
    data['volume_ma_24h'] = data['volume'].rolling(window=24).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma_24h']
    
    # Time-based features (hour of day, day of week)
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    
    # Count NaN values before dropping
    nan_counts = data.isna().sum()
    print("NaN counts per column:")
    print(nan_counts)
    print(f"Total rows with at least one NaN: {data.isna().any(axis=1).sum()}")
    
    # Change to more selective NaN removal
    # Instead of dropping all rows with any NaN, keep rows with essential data
    essential_columns = ['close', 'volume', 'returns', 'rsi_14', 'macd']
    data_before_dropna = data.shape[0]
    data = data.dropna(subset=essential_columns)
    data_after_dropna = data.shape[0]
    print(f"Rows before dropna: {data_before_dropna}, after dropna: {data_after_dropna}")
    
    # If still losing too many rows, consider filling NaNs instead
    if data_after_dropna < 100:  # Arbitrary threshold
        print("Too many rows dropped, attempting to fill NaNs instead")
        data = df.copy()
        # Apply features again but fill NaNs for rolling calculations
        # This is a simplified example
        data['returns'] = data['close'].pct_change().fillna(0)
        # ... repeat other feature calculations with NaN filling ...
    
    print(f"Final data shape: {data.shape}")
    return data

if __name__ == "__main__":
    print("Creating features...")
    # Load preprocessed data
    bitcoin_data = load_preprocessed_data()
    
    # Add basic data inspection
    print(f"Loaded preprocessed data shape: {bitcoin_data.shape}")
    print(f"Loaded preprocessed data columns: {bitcoin_data.columns.tolist()}")
    print(f"First few rows of preprocessed data:")
    print(bitcoin_data.head())
    
    # Check for NaN values in input data
    print(f"NaN values in preprocessed data: {bitcoin_data.isna().sum().sum()}")
    
    # Create features
    bitcoin_features = create_features(bitcoin_data)
    
    # Save features
    bitcoin_features.to_csv('data/processed/bitcoin_features.csv')
    print(f"Features created and saved to data/processed/bitcoin_features.csv ({len(bitcoin_features)} rows)")