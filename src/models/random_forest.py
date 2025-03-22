import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure directories exist
os.makedirs('models', exist_ok=True)
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

def prepare_data(df, target_col, test_size=0.2):
    """
    Prepare data for machine learning models.
    """
    # Define features and target
    feature_cols = [col for col in df.columns if col not in [target_col, 'open', 'high', 'low', 'close', 'volume']]
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols

def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    """
    Train and evaluate a Random Forest regression model.
    """
    # Create and train the model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest MSE: {mse:.6f}")
    print(f"Random Forest R²: {r2:.6f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importance in Random Forest Model')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('results/figures/rf_feature_importance.png')
    plt.close()
    
    # Save metrics
    metrics = pd.DataFrame({
        'model': ['Random Forest'],
        'mse': [mse],
        'r2': [r2]
    })
    metrics.to_csv('results/metrics/rf_performance.csv', index=False)
    
    return rf_model, y_pred, feature_importance

if __name__ == "__main__":
    print("Training Random Forest model...")
    # Load ML data
    bitcoin_ml_data = load_ml_data()
    
    # Prepare data for price prediction
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(
        bitcoin_ml_data, 'future_return_24h', test_size=0.2
    )
    
    # Train and evaluate Random Forest model
    rf_model, rf_pred, rf_importance = train_random_forest(X_train, y_train, X_test, y_test, feature_cols)
    
    # Save model and scaler
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    joblib.dump(scaler, 'models/rf_scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    
    print("Random Forest model trained and saved.")