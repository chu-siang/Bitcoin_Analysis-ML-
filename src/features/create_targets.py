import pandas as pd
import numpy as np
import os

def load_feature_data():
    """
    Load Bitcoin data with features.
    """
    file_path = 'data/processed/bitcoin_features.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature data file not found at {file_path}. Run create_features.py first.")
    
    return pd.read_csv(file_path, index_col=0, parse_dates=True)

def create_targets(df, prediction_horizon=24):
    """
    Create target variables for price prediction.
    Params:
        df: DataFrame with features
        prediction_horizon: Number of hours to predict into the future
    """
    print(f"Input data shape: {df.shape}")
    data = df.copy()
    
    # Future price change percentage
    target_col = f'future_return_{prediction_horizon}h'
    data[target_col] = data['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
    
    # Binary target for price direction
    direction_col = f'price_up_{prediction_horizon}h'
    data[direction_col] = (data[target_col] > 0).astype(int)
    
    # Price volatility target
    volatility_col = f'future_volatility_{prediction_horizon}h'
    data[volatility_col] = data['returns'].rolling(window=prediction_horizon).std().shift(-prediction_horizon) * np.sqrt(prediction_horizon)
    
    # Print NaN counts in target columns
    print(f"NaN counts in target columns:")
    print(f"{target_col}: {data[target_col].isna().sum()}")
    print(f"{direction_col}: {data[direction_col].isna().sum()}")
    print(f"{volatility_col}: {data[volatility_col].isna().sum()}")
    
    # Only remove rows with NaN target values
    rows_before = len(data)
    data = data.dropna(subset=[target_col, direction_col, volatility_col])
    rows_after = len(data)
    print(f"Rows before dropping NaN targets: {rows_before}, after: {rows_after}")
    
    return data

if __name__ == "__main__":
    print("Creating target variables...")
    # Load feature data
    bitcoin_features = load_feature_data()
    print(f"Loaded feature data shape: {bitcoin_features.shape}")
    
    # Create targets for 24-hour prediction
    bitcoin_ml_data = create_targets(bitcoin_features, prediction_horizon=24)
    
    # Save ML-ready data
    bitcoin_ml_data.to_csv('data/processed/bitcoin_ml_data.csv')
    print(f"Target variables created and saved to data/processed/bitcoin_ml_data.csv ({len(bitcoin_ml_data)} rows)")