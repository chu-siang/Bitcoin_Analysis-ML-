import pandas as pd
import numpy as np
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def experiment_pca():
    """
    Experiment: Study the impact of PCA dimensionality reduction on model performance
    """
    print("Running PCA dimensionality reduction experiment...")
    
    try:
        # Ensure directories exist
        os.makedirs('results/experiments', exist_ok=True)
        
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
            elif ('rf' in file.lower() or 'random_forest' in file.lower()) and 'scaler' in file.lower():
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
            try:
                rf_scaler = joblib.load(rf_scaler_path)
                X_train_scaled = rf_scaler.transform(X_train)
                X_test_scaled = rf_scaler.transform(X_test)
            except ValueError as ve:
                print(f"Error using saved scaler: {ve}")
                print("Creating new scaler instead")
                rf_scaler = StandardScaler()
                X_train_scaled = rf_scaler.fit_transform(X_train)
                X_test_scaled = rf_scaler.transform(X_test)
        else:
            print("Creating new scaler")
            rf_scaler = StandardScaler()
            X_train_scaled = rf_scaler.fit_transform(X_train)
            X_test_scaled = rf_scaler.transform(X_test)
        # Baseline performance (without PCA)
        rf_model.fit(X_train_scaled, y_train)
        y_pred_base = rf_model.predict(X_test_scaled)
        base_mse = mean_squared_error(y_test, y_pred_base)
        base_r2 = r2_score(y_test, y_pred_base)
        
        # Results storage
        results = [{
            'n_components': 'Original',
            'variance_explained': 1.0,
            'mse': base_mse,
            'r2': base_r2
        }]
        
        # Try different numbers of PCA components
        n_features = X_train_scaled.shape[1]
        components_to_try = [2, 3, 5, 7, 10, 15]
        components_to_try = [c for c in components_to_try if c < n_features]
        
        for n_components in components_to_try:
            # Apply PCA
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            
            # Get variance explained
            variance_explained = sum(pca.explained_variance_ratio_)
            
            # Train model
            rf_pca = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_pca.fit(X_train_pca, y_train)
            
            # Predict
            y_pred_pca = rf_pca.predict(X_test_pca)
            
            # Evaluate
            pca_mse = mean_squared_error(y_test, y_pred_pca)
            pca_r2 = r2_score(y_test, y_pred_pca)
            
            # Save results
            results.append({
                'n_components': n_components,
                'variance_explained': variance_explained,
                'mse': pca_mse,
                'r2': pca_r2
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv('results/experiments/pca_results.csv', index=False)
        
        # Plot results
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Components vs MSE
        plt.subplot(2, 2, 1)
        plt.plot([str(x) for x in results_df['n_components']], results_df['mse'], 'o-', linewidth=2)
        plt.title('MSE vs Number of PCA Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Mean Squared Error')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Components vs R²
        plt.subplot(2, 2, 2)
        plt.plot([str(x) for x in results_df['n_components']], results_df['r2'], 'o-', linewidth=2)
        plt.title('R² vs Number of PCA Components')
        plt.xlabel('Number of Components')
        plt.ylabel('R² Score')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Variance Explained vs Performance
        plt.subplot(2, 2, 3)
        plt.scatter(results_df['variance_explained'], results_df['mse'], s=100, alpha=0.7)
        for i, txt in enumerate(results_df['n_components']):
            plt.annotate(str(txt), (results_df['variance_explained'].iloc[i], results_df['mse'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points')
        plt.title('MSE vs Variance Explained')
        plt.xlabel('Cumulative Variance Explained')
        plt.ylabel('Mean Squared Error')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Feature Importance Heatmap (for the first few components)
        if len(components_to_try) > 0:
            pca = PCA(n_components=min(5, n_features))
            pca.fit(X_train_scaled)
            
            # Create a DataFrame for the heatmap
            component_df = pd.DataFrame(
                pca.components_,
                columns=X.columns,
                index=[f'PC{i+1}' for i in range(pca.n_components_)]
            )
            
            plt.subplot(2, 2, 4)
            sns.heatmap(component_df, cmap='viridis', annot=False, linewidths=0.5)
            plt.title('PCA Component Loadings')
            plt.tight_layout()
        
        plt.savefig('results/experiments/pca_experiment.png')
        
        print("PCA dimensionality reduction experiment completed. Results saved.")
        return results_df
    
    except Exception as e:
        print(f"Error in experiment: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    experiment_pca()