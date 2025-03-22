import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Make sure directories exist
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

def load_ml_data():
    """
    Load ML-ready Bitcoin data.
    """
    file_path = 'data/processed/bitcoin_ml_data.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ML data file not found at {file_path}. Run create_targets.py first.")
    
    return pd.read_csv(file_path, index_col=0, parse_dates=True)

def experiment_data_size(df, target_col):
    """
    Experiment with different training data sizes.
    """
    # Define features and target
    feature_cols = [col for col in df.columns if col not in [target_col, 'open', 'high', 'low', 'close', 'volume']]
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Define different training sizes
    train_sizes = [0.2, 0.4, 0.6, 0.8]
    results = []
    
    # Fixed test set (last 20% of data)
    test_size = 0.2
    test_idx = int(len(X) * (1 - test_size))
    X_test = X[test_idx:]
    y_test = y[test_idx:]
    
    from sklearn.preprocessing import StandardScaler
    
    for size in train_sizes:
        # Calculate actual number of samples for this training size
        n_samples = int(len(X) * size)
        
        # Use the appropriate portion for training
        X_train = X[:n_samples]
        y_train = y[:n_samples]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train RF model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        rf_pred = rf.predict(X_test_scaled)
        rf_mse = mean_squared_error(y_test, rf_pred)
        
        # Train SVR model
        svr = SVR(kernel='rbf', C=10)
        svr.fit(X_train_scaled, y_train)
        svr_pred = svr.predict(X_test_scaled)
        svr_mse = mean_squared_error(y_test, svr_pred)
        
        # Store results
        results.append({
            'train_size': size,
            'rf_mse': rf_mse,
            'svr_mse': svr_mse,
            'n_samples': n_samples
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['train_size'], results_df['rf_mse'], marker='o', label='Random Forest')
    plt.plot(results_df['train_size'], results_df['svr_mse'], marker='x', label='SVR')
    plt.xlabel('Training Data Size (proportion)')
    plt.ylabel('Mean Squared Error')
    plt.title('Effect of Training Data Size on Model Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/figures/training_size_experiment.png')
    plt.close()
    
    # Save results
    results_df