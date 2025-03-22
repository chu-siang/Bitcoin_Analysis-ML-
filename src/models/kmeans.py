import pandas as pd
import numpy as np
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

def load_ml_data():
    """
    Load ML-ready Bitcoin data.
    """
    file_path = 'data/processed/bitcoin_ml_data.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ML data file not found at {file_path}. Run create_targets.py first.")
    
    return pd.read_csv(file_path, index_col=0, parse_dates=True)

def prepare_data_for_clustering(df):
    """
    Prepare data for clustering.
    """
    # Select features for clustering
    cluster_features = ['returns', 'volatility_24h', 'rsi_14', 'volume_ratio', 'macd', 'bb_width']
    X = df[cluster_features]
    
    # check NaN value
    print(f"NaN values before cleaning: {X.isna().sum().sum()}")
    
    # check is there any the row of all NaN values, and deleted.
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        print(f"Dropping columns with all NaN values: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)
        # update cluster_features table
        cluster_features = [col for col in cluster_features if col not in all_nan_cols]
    
    # fill the remain NaN values.
    X = X.fillna(X.median())
    print(f"NaN values after cleaning: {X.isna().sum().sum()}")
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, cluster_features

def perform_kmeans_clustering(X, n_clusters=4):
    """
    Perform K-means clustering on Bitcoin data.
    """
    # Create and train the model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Return cluster centers and labels
    return kmeans, clusters

def visualize_clusters(X, clusters):
    """
    Visualize K-means clusters using PCA for dimensionality reduction.
    """
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create a DataFrame for plotting
    cluster_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': clusters
    })
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=cluster_df, palette='viridis')
    plt.title('K-means Clustering of Bitcoin Market States')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Market State')
    plt.savefig('results/figures/kmeans_clusters.png')
    plt.close()
    
    return cluster_df

if __name__ == "__main__":
    print("Performing K-means clustering...")
    # Load ML data
    bitcoin_ml_data = load_ml_data()
    
    # Prepare data for clustering
    X, scaler, cluster_features = prepare_data_for_clustering(bitcoin_ml_data)
    
    # check the dimension
    print(f"Clustering data shape: {X.shape}")
    print(f"Features used: {cluster_features}")
    
    # Perform clustering
    kmeans_model, cluster_labels = perform_kmeans_clustering(X, n_clusters=4)
    
    # Visualize clusters
    cluster_df = visualize_clusters(X, cluster_labels)
    
    # Add cluster labels to original data
    bitcoin_ml_data['cluster'] = cluster_labels
    bitcoin_ml_data[['cluster']].to_csv('data/processed/bitcoin_clusters.csv')
    
    # Save model and scaler
    joblib.dump(kmeans_model, 'models/kmeans_model.pkl')
    joblib.dump(scaler, 'models/kmeans_scaler.pkl')
    joblib.dump(cluster_features, 'models/kmeans_features.pkl')
    
    print("K-means clustering performed and saved.")