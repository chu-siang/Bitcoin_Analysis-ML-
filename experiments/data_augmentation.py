import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def experiment_data_augmentation():
    """
    Experiment: Study the impact of data augmentation on model performance
    """
    print("Running data augmentation experiment...")
    
    # Ensure directories exist
    os.makedirs('results/experiments', exist_ok=True)
    
    try:
        # Check what files are in the models directory
        print("Checking model files in directory...")
        model_files = os.listdir('models')
        print(f"Files in models directory: {model_files}")
        
        # Try to find the correct model files
        rf_model_path = None
        rf_scaler_path = None
        
        for file in model_files:
            if 'random_forest' in file.lower() and 'model' in file.lower():
                rf_model_path = os.path.join('models', file)
            elif 'random_forest' in file.lower() and 'scaler' in file.lower():
                rf_scaler_path = os.path.join('models', file)
        
        # Load models if found, otherwise create new ones
        if rf_model_path and os.path.exists(rf_model_path):
            print(f"Loading Random Forest model from: {rf_model_path}")
            rf_model = joblib.load(rf_model_path)
        else:
            print("Creating new Random Forest model")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Load ML data
        ml_data = pd.read_csv('data/processed/bitcoin_ml_data.csv', index_col=0, parse_dates=True)
        
        # Target variable
        target_col = 'future_return_24h'
        
        # Define feature columns
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
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Create or load scaler
        if rf_scaler_path and os.path.exists(rf_scaler_path):
            print(f"Loading scaler from: {rf_scaler_path}")
            rf_scaler = joblib.load(rf_scaler_path)
            X_train_scaled = rf_scaler.transform(X_train)
            X_test_scaled = rf_scaler.transform(X_test)
        else:
            print("Creating new scaler")
            rf_scaler = StandardScaler()
            X_train_scaled = rf_scaler.fit_transform(X_train)
            X_test_scaled = rf_scaler.transform(X_test)
        
        # Train Random Forest model for baseline
        rf_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_base = rf_model.predict(X_test_scaled)
        
        # Evaluate baseline performance
        base_mse = mean_squared_error(y_test, y_pred_base)
        base_r2 = r2_score(y_test, y_pred_base)
        
        # Data augmentation experiment results
        results = [{
            'method': 'Baseline (No Augmentation)',
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mse': base_mse,
            'r2': base_r2
        }]
        
        # Data augmentation method 1: Adding noise
        noise_levels = [0.01, 0.05, 0.1]
        
        for noise in noise_levels:
            # Copy training data
            X_train_noisy = X_train.copy()
            
            # Add noise
            for col in X_train_noisy.columns:
                noise_factor = noise * X_train_noisy[col].std()
                X_train_noisy[col] = X_train_noisy[col] + np.random.normal(0, noise_factor, size=len(X_train_noisy))
            
            # Scale features
            X_train_noisy_scaled = rf_scaler.transform(X_train_noisy)
            
            # Train model
            rf_model.fit(X_train_noisy_scaled, y_train)
            
            # Predictions
            y_pred_noisy = rf_model.predict(X_test_scaled)
            
            # Evaluate performance
            noisy_mse = mean_squared_error(y_test, y_pred_noisy)
            noisy_r2 = r2_score(y_test, y_pred_noisy)
            
            results.append({
                'method': f'Gaussian Noise (std={noise})',
                'train_samples': len(X_train_noisy),
                'test_samples': len(X_test),
                'mse': noisy_mse,
                'r2': noisy_r2
            })
        
        # Data augmentation method 2: Synthetic sample creation
        X_train_synth = X_train.copy()
        y_train_synth = y_train.copy()
        
        # Simple mixing to generate new samples
        for i in range(int(len(X_train) * 0.2)):  # Add 20% more samples
            idx1, idx2 = np.random.randint(0, len(X_train), 2)
            alpha = np.random.random()
            
            # Create mixed sample
            new_sample = X_train.iloc[idx1] * alpha + X_train.iloc[idx2] * (1 - alpha)
            new_target = y_train.iloc[idx1] * alpha + y_train.iloc[idx2] * (1 - alpha)
            
            # Add to training set
            X_train_synth.loc[len(X_train) + i] = new_sample
            y_train_synth.loc[len(y_train) + i] = new_target
        
        # Scale features
        X_train_synth_scaled = rf_scaler.transform(X_train_synth)
        
        # Train model
        rf_model.fit(X_train_synth_scaled, y_train_synth)
        
        # Predictions
        y_pred_synth = rf_model.predict(X_test_scaled)
        
        # Evaluate performance
        synth_mse = mean_squared_error(y_test, y_pred_synth)
        synth_r2 = r2_score(y_test, y_pred_synth)
        
        results.append({
            'method': 'Synthetic Samples (Mixing)',
            'train_samples': len(X_train_synth),
            'test_samples': len(X_test),
            'mse': synth_mse,
            'r2': synth_r2
        })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv('results/experiments/data_augmentation_results.csv', index=False)
        
        # Plot results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(results_df['method'], results_df['mse'])
        plt.title('MSE by Data Augmentation Method')
        plt.xlabel('Method')
        plt.ylabel('Mean Squared Error')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(results_df['method'], results_df['r2'])
        plt.title('R² by Data Augmentation Method')
        plt.xlabel('Method')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/experiments/data_augmentation_experiment.png')
        
        print("Data augmentation experiment completed. Results saved.")
        return results_df
    
    except Exception as e:
        print(f"Error in experiment: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    experiment_data_augmentation()