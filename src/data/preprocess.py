import pandas as pd
import os

def load_raw_data():
    """
    Load raw Bitcoin price data.
    """
    file_path = 'data/raw/bitcoin_raw_data.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at {file_path}. Run fetch_data.py first.")
    
    return pd.read_csv(file_path, index_col=0, parse_dates=True)

if __name__ == "__main__":
    print("Loading and preprocessing raw data...")
    # Load raw data
    bitcoin_data = load_raw_data()
    
    # Perform basic preprocessing (if needed)
    # Remove duplicate rows
    bitcoin_data = bitcoin_data.drop_duplicates()
    
    # Sort by timestamp
    bitcoin_data = bitcoin_data.sort_index()
    
    # Save preprocessed data
    bitcoin_data.to_csv('data/processed/bitcoin_preprocessed.csv')
    print(f"Preprocessed data saved to data/processed/bitcoin_preprocessed.csv ({len(bitcoin_data)} rows)")