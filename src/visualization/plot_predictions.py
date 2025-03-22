import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.preprocessing import StandardScaler

def plot_predictions():
    """
    Plot model predictions against actual values.
    """
    print("Plotting model predictions...")
    
    # Ensure directories exist
    os.makedirs('results/figures', exist_ok=True)
    
    try:
        # Load ML data
        ml_data = pd.read_csv('data/processed/bitcoin_ml_data.csv', index_col=0, parse_dates=True)
        
        # Target variable
        target_col = 'future_return_24h'
        
        # Get features for prediction
        feature_cols = [col for col in ml_data.columns if col not in [target_col, 'open', 'high', 'low', 'close', 'volume', 
                                                              'price_up_24h', 'future_volatility_24h']]
        
        # Prepare data
        X = ml_data[feature_cols]
        y = ml_data[target_col]
        
        # Check for all-NaN columns
        all_nan_cols = X.columns[X.isna().all()].tolist()
        if all_nan_cols:
            print(f"Dropping columns with all NaN values: {all_nan_cols}")
            X = X.drop(columns=all_nan_cols)
        
        # Handle remaining NaN values
        X = X.fillna(X.median())
        
        # Split data for visualization
        test_size = 0.2
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Load models
        models = {}
        scalers = {}
        
        # Check available model files
        model_files = os.listdir('models')
        print(f"Available model files: {model_files}")
        
        # Try to load Random Forest model
        rf_model_path = None
        for file in model_files:
            if 'random_forest' in file.lower() and 'model' in file.lower():
                rf_model_path = os.path.join('models', file)
                break
        
        if rf_model_path and os.path.exists(rf_model_path):
            print(f"Loading Random Forest model from: {rf_model_path}")
            models['Random Forest'] = joblib.load(rf_model_path)
        
        # Try to load SVR model
        svr_model_path = None
        for file in model_files:
            if 'svr' in file.lower() and 'model' in file.lower():
                svr_model_path = os.path.join('models', file)
                break
        
        if svr_model_path and os.path.exists(svr_model_path):
            print(f"Loading SVR model from: {svr_model_path}")
            models['SVR'] = joblib.load(svr_model_path)
        
        # Create a scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        predictions = {}
        for name, model in models.items():
            try:
                # Get the expected number of features for this model
                expected_features = None
                try:
                    expected_features = model.n_features_in_
                    print(f"{name} model expects {expected_features} features, we have {X_test_scaled.shape[1]} features")
                except:
                    pass
                
                # Try to make prediction
                y_pred = model.predict(X_test_scaled)
                predictions[name] = y_pred
            except Exception as e:
                print(f"Error making predictions with {name} model: {str(e)}")
                # Try with a retrained model
                print(f"Retraining {name} model with current features...")
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                predictions[name] = y_pred
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({'Actual': y_test})
        for name, pred in predictions.items():
            plot_data[name] = pred
        
        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(plot_data.index, plot_data['Actual'], label='Actual', linewidth=2)
        
        for name in predictions.keys():
            plt.plot(plot_data.index, plot_data[name], label=f'{name} Prediction', linewidth=1.5, alpha=0.8)
        
        plt.title('Bitcoin Price Return Predictions')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/figures/model_predictions.png')
        
        # Plot error distribution
        plt.figure(figsize=(12, 6))
        for i, (name, pred) in enumerate(predictions.items(), 1):
            errors = plot_data['Actual'] - plot_data[name]
            
            plt.subplot(1, len(predictions), i)
            sns.histplot(errors, kde=True)
            plt.title(f'{name} Error Distribution')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/error_distribution.png')
        
        # Plot scatter of actual vs predicted
        plt.figure(figsize=(12, 6))
        for i, (name, pred) in enumerate(predictions.items(), 1):
            plt.subplot(1, len(predictions), i)
            plt.scatter(plot_data['Actual'], plot_data[name], alpha=0.5)
            
            # Add perfect prediction line
            min_val = min(plot_data['Actual'].min(), plot_data[name].min())
            max_val = max(plot_data['Actual'].max(), plot_data[name].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.title(f'Actual vs {name} Predicted')
            plt.xlabel('Actual Return')
            plt.ylabel('Predicted Return')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/actual_vs_predicted.png')
        
        print("Model prediction plots saved to results/figures/")
        return plot_data
    
    except Exception as e:
        print(f"Error plotting predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    plot_predictions()