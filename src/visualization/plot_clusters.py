import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.decomposition import PCA

def visualize_clusters():
    """
    Visualize K-means clustering results on Bitcoin data.
    """
    print("Visualizing cluster results...")
    
    # Ensure directories exist
    os.makedirs('results/figures', exist_ok=True)
    
    try:
        # Check if the clustered data file exists
        cluster_file = 'data/processed/bitcoin_clusters.csv'
        if not os.path.exists(cluster_file):
            print(f"Cluster data file not found: {cluster_file}")
            
            # Look for alternative file
            ml_data_file = 'data/processed/bitcoin_ml_data.csv'
            if os.path.exists(ml_data_file):
                print(f"Loading ML data from: {ml_data_file}")
                data = pd.read_csv(ml_data_file, index_col=0, parse_dates=True)
                
                # Check if kmeans model exists
                model_files = os.listdir('models')
                kmeans_model_path = None
                
                for file in model_files:
                    if 'kmeans' in file.lower() and 'model' in file.lower():
                        kmeans_model_path = os.path.join('models', file)
                        break
                
                if kmeans_model_path and os.path.exists(kmeans_model_path):
                    print(f"Loading K-means model from: {kmeans_model_path}")
                    kmeans_model = joblib.load(kmeans_model_path)
                    
                    # Prepare features for clustering
                    feature_cols = ['returns', 'volatility_24h', 'rsi_14', 'volume_ratio', 'macd', 'bb_width']
                    valid_features = [col for col in feature_cols if col in data.columns]
                    
                    X = data[valid_features]
                    X = X.fillna(X.median())
                    
                    # Apply clustering
                    clusters = kmeans_model.predict(X)
                    data['cluster'] = clusters
                else:
                    print("K-means model not found. Creating random clusters for visualization.")
                    # Create random clusters for demonstration
                    np.random.seed(42)
                    data['cluster'] = np.random.randint(0, 4, size=len(data))
            else:
                print("No data files found for clustering visualization.")
                return None
        else:
            print(f"Loading cluster data from: {cluster_file}")
            data = pd.read_csv(cluster_file, index_col=0, parse_dates=True)
            
            # If only cluster column is present, load full data
            if len(data.columns) <= 1:
                print("Loading full data for visualization...")
                ml_data = pd.read_csv('data/processed/bitcoin_ml_data.csv', index_col=0, parse_dates=True)
                data = pd.concat([ml_data, data], axis=1)
        
        # 1. Plot time series with cluster colors
        plt.figure(figsize=(14, 7))
        scatter = plt.scatter(data.index, data['close'], c=data['cluster'], 
                              cmap='viridis', alpha=0.7, s=40)
        
        plt.title('Bitcoin Price with Cluster Classifications')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/figures/cluster_time_series.png')
                

        # 2. Perform PCA for visualization
        # Select numerical columns for PCA
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        num_cols = [col for col in num_cols if col != 'cluster']

        if len(num_cols) > 2:
            # Select a subset of columns that have minimal NaN values
            X = data[num_cols].copy()
            
            # Check which columns have NaN values
            nan_count = X.isna().sum()
            good_cols = nan_count[nan_count < len(X) * 0.1].index.tolist()  # Columns with <10% NaNs
            
            if len(good_cols) < 2:
                print("Not enough good columns for PCA. Using the least NaN columns.")
                # Get columns with the least NaNs
                good_cols = nan_count.nsmallest(min(5, len(nan_count))).index.tolist()
            
            print(f"Using {len(good_cols)} columns for PCA: {good_cols}")
            X = X[good_cols]
            
            # Handle any remaining NaN values
            X = X.fillna(X.median())
            
            # Apply PCA
            try:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                
                # Create DataFrame for plotting
                pca_df = pd.DataFrame({
                    'PC1': X_pca[:, 0],
                    'PC2': X_pca[:, 1],
                    'Cluster': data.loc[X.index, 'cluster']
                })
                
                # Plot PCA results
                plt.figure(figsize=(10, 8))
                sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=60)
                plt.title('PCA Visualization of Bitcoin Market States')
                plt.xlabel(f'PC1 (variance)')
                plt.ylabel(f'PC2 (variance)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('results/figures/cluster_pca.png')
            except Exception as e:
                print(f"Error during PCA visualization: {str(e)}")
        # 3. Plot cluster statistics
        # Calculate key statistics by cluster
        cluster_stats = data.groupby('cluster').agg({
            'future_return_24h': ['mean', 'std'],
            'returns': ['mean', 'std'],
            'volume': 'mean',
            'rsi_14': 'mean'
        })
        
        # Plot return by cluster
        plt.figure(figsize=(12, 10))
        
        # Plot average future return by cluster
        plt.subplot(2, 2, 1)
        returns_by_cluster = cluster_stats['future_return_24h']['mean']
        plt.bar(returns_by_cluster.index, returns_by_cluster.values)
        plt.title('Average 24h Future Return by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Avg. Return (%)')
        plt.grid(True, alpha=0.3)
        
        # Plot return volatility by cluster
        plt.subplot(2, 2, 2)
        volatility_by_cluster = cluster_stats['future_return_24h']['std']
        plt.bar(volatility_by_cluster.index, volatility_by_cluster.values)
        plt.title('Return Volatility by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Return Std. Dev.')
        plt.grid(True, alpha=0.3)
        
        # Plot average RSI by cluster
        plt.subplot(2, 2, 3)
        if 'rsi_14' in cluster_stats.columns.get_level_values(0):
            rsi_by_cluster = cluster_stats['rsi_14']['mean']
            plt.bar(rsi_by_cluster.index, rsi_by_cluster.values)
            plt.title('Average RSI by Cluster')
            plt.xlabel('Cluster')
            plt.ylabel('Avg. RSI')
            plt.grid(True, alpha=0.3)
        
        # Plot volume by cluster
        plt.subplot(2, 2, 4)
        if 'volume' in cluster_stats.columns.get_level_values(0):
            volume_by_cluster = cluster_stats['volume']['mean']
            plt.bar(volume_by_cluster.index, volume_by_cluster.values)
            plt.title('Average Volume by Cluster')
            plt.xlabel('Cluster')
            plt.ylabel('Avg. Volume')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/cluster_statistics.png')
        
        print("Cluster visualizations saved to results/figures/")
        return data
    
    except Exception as e:
        print(f"Error visualizing clusters: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    visualize_clusters()