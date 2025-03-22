import pandas as pd
import numpy as np
import os
import joblib
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Make sure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

def load_ml_data():
    """
    Load ML-ready Bitcoin data.
    """
    file_path = 'data/processed/bitcoin_ml_data.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ML data file not found at {file_path}. Run create_targets.py first.")
    
    return pd.read_csv(file_path, index_col=0, parse_dates=True)

def prepare_data(df, target_col, test_size=0.2):
    """
    Prepare data for machine learning models.
    """
    # Define features and target
    feature_cols = [col for col in df.columns if col not in [target_col, 'open', 'high', 'low', 'close', 'volume']]
    X = df[feature_cols]
    y = df[target_col]
    
    # Check for NaN values
    print(f"NaN values in features before cleaning: {X.isna().sum().sum()}")
    
    # 檢查是否有全為 NaN 的列，並刪除
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        print(f"Dropping columns with all NaN values: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)
    
    # 使用中位數填補剩餘的 NaN 值
    X = X.fillna(X.median())
    
    print(f"NaN values in features after cleaning: {X.isna().sum().sum()}")
    
    # Split data into train and test sets
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()

def train_svr(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Support Vector Regression model.
    """
    # Create and train the model
    svr_model = SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale')
    svr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = svr_model.predict(X_test)
    
    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"SVR MSE: {mse:.6f}")
    print(f"SVR R²: {r2:.6f}")
    
    # Save metrics
    metrics = pd.DataFrame({
        'model': ['SVR'],
        'mse': [mse],
        'r2': [r2]
    })
    metrics.to_csv('results/metrics/svr_performance.csv', index=False)
    
    return svr_model, y_pred

if __name__ == "__main__":
    print("Training SVR model...")
    # Load ML data
    bitcoin_ml_data = load_ml_data()
    
    # Prepare data for price prediction
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(
        bitcoin_ml_data, 'future_return_24h', test_size=0.2
    )
    
    # Train and evaluate SVR model
    svr_model, svr_pred = train_svr(X_train, y_train, X_test, y_test)
    
    # Save model and scaler
    joblib.dump(svr_model, 'models/svr_model.pkl')
    joblib.dump(scaler, 'models/svr_scaler.pkl')
    joblib.dump(feature_cols, 'models/svr_features.pkl')
    
    print("SVR model trained and saved.")