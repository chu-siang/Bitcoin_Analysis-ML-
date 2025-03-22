# Bitcoin Price Analysis and Prediction Using Machine Learning

## Web link to dataset
[GitHub Repository: https://github.com/chu-siang/Bitcoin_Analysis_ML](https://github.com/chu-siang/Bitcoin_Analysis_ML)

## Research Question
This research investigates how effectively machine learning models can predict Bitcoin price movements based on historical price data and technical indicators using 1-hour intervals over a 4-month period (November 2024 to March 2025). Additionally, we explore whether unsupervised learning can identify distinct market states with different behavior patterns.

## Dataset Documentation

### Data Type and External Source
The dataset consists of Bitcoin (BTC/USDT) price data collected from the Binance API, including 1-hour candlestick information from November 1, 2024, to March 1, 2025. The raw data includes time-series records of Bitcoin's price and trading volume, with each record containing a timestamp and OHLCV (Open, High, Low, Close, Volume) values.

### Dataset Preprocessing and Feature Engineering
I conducted several preprocessing steps to ensure data quality:
- Converting timestamps to datetime format
- Ensuring consistent hourly intervals
- Removing missing timestamps and sorting chronologically
- Creating technical indicators as features
- Generating target variables for prediction

From the raw data, I derived over 30 technical indicators and features including:
- Simple Moving Averages (SMA): 12-hour SMA for smoothing short-term fluctuations
- Exponential Moving Averages (EMA): 6h, 12h, 24h EMAs emphasizing recent price trends
- Relative Strength Index (RSI): 14-period RSI signaling overbought (>70) or oversold (<30) conditions
- Moving Average Convergence Divergence (MACD): Trend momentum indicators
- Bollinger Bands: For volatility and extreme price deviation detection
- Volatility measures: 24-hour volatility and volume-related features
- Time-based features: Hour of day (0-23) and weekend flags (1 for Saturday/Sunday, 0 otherwise)

### Target Variable Construction
For supervised learning, we created the following targets:
- 24-hour future return (return_24h): Percentage change from current time (t) to 24 hours later (t+24h)
- Binary price direction (price_up_24h): 1 if future return positive, 0 otherwise
- Future volatility (future_volatility_24h): For cluster analysis only, not prediction

![Bitcoin Price with Cluster Classifications](https://raw.githubusercontent.com/chu-siang/Bitcoin_Analysis_ML/refs/heads/main/results/figures/cluster_time_series.png)
*Figure 1: Bitcoin price chart with identified market state clusters. Each point represents Bitcoin price colored by its assigned cluster, showing how market states evolve over time from November 2024 to March 2025. Purple points (Cluster 0) appear during sideways movement, green points (Cluster 2) during uptrends, blue points (Cluster 1) during corrections, and yellow points (Cluster 3) often after price drops.*

## Description of Supervised and Unsupervised Methods

### Supervised Learning Methods

#### 1. Random Forest Regression
Random Forest is an ensemble learning method that constructs multiple decision trees and outputs the average prediction to improve accuracy and control overfitting.

**Implementation Details:**
- Framework: scikit-learn's RandomForestRegressor
- Features: 30+ technical indicators
- Target: 24-hour future returns
- Hyperparameters: 100 trees, default depth

![Random Forest Feature Importance](https://raw.githubusercontent.com/chu-siang/Bitcoin_Analysis_ML/refs/heads/main/results/figures/rf_feature_importance.png)
*Figure 2: Top 15 feature importances in the Random Forest model. The chart shows which features most influenced predictions, with price_up_24h and future_volatility_24h having the highest importance scores (indicating some target leakage). Among legitimate technical indicators, Bollinger Bands (bb_upper) and moving averages (sma_24h) contributed the most to predictions.*

#### 2. Support Vector Regression (SVR)
SVR finds a function that best predicts the continuous output value for a given input value, while maximizing the margin.

**Implementation Details:**
- Framework: scikit-learn's SVR
- Features: Same technical indicators used for Random Forest
- Target: 24-hour future returns
- Hyperparameters: RBF kernel, C=10, epsilon=0.1

### Unsupervised Learning Method

#### K-means Clustering
K-means clustering was applied to identify distinct market states or regimes in Bitcoin trading patterns by grouping data into clusters that minimize intra-cluster variance.

**Implementation Details:**
- Framework: scikit-learn's KMeans
- Features: Selected subset including returns, volatility, RSI, volume ratio, MACD, Bollinger Band width
- Parameters: 4 clusters (k=4)
- No future/target information was included in clustering

![K-means Clustering Visualization](https://raw.githubusercontent.com/chu-siang/Bitcoin_Analysis_ML/refs/heads/main/results/figures/kmeans_clusters.png)
*Figure 3: K-means clustering of Bitcoin market states visualized in principal component space. Each point represents an hourly observation projected onto two principal components and colored by cluster. The visualization shows clear separation between the four market states: purple points (Cluster 0) forming a dense group at bottom-left, green points (Cluster 2) spreading to the right, blue points (Cluster 1) in the upper-middle region, and yellow points (Cluster 3) appearing in the upper-left.*

## Description of Experiments and Evaluation Results

### Experiment 1: Regression Performance and Cross-Validation
I compared the performance of Random Forest and SVR models in predicting 24-hour Bitcoin price returns, using chronological train-test splitting and 5-fold cross-validation.

**Results:**
The Random Forest significantly outperformed SVR, achieving lower MSE and better R² values. However, both models had negative R² scores on the test set, indicating limited predictive power for the highly volatile Bitcoin returns. The RF partially captured directional movements, while SVR predicted mostly near-zero returns.

![Model Prediction Comparison](https://raw.githubusercontent.com/chu-siang/Bitcoin_Analysis_ML/refs/heads/main/results/figures/actual_vs_predicted.png)
*Figure 4: Scatter plots of actual vs. predicted 24h returns for Random Forest (left) and SVR (right). The red dashed line represents perfect prediction. The Random Forest shows predictions somewhat correlated with actual returns, while SVR predictions cluster horizontally near zero, indicating its failure to capture return variability.*

![Error Distribution](https://raw.githubusercontent.com/chu-siang/Bitcoin_Analysis_ML/refs/heads/main/results/figures/error_distribution.png)
*Figure 5: Histograms showing prediction error distributions for Random Forest (left) and SVR (right). The RF errors are more tightly centered around zero with fewer extreme errors, while SVR shows a broader, skewed distribution, confirming its tendency to underpredict returns.*

![Bitcoin Price Return Predictions](https://raw.githubusercontent.com/chu-siang/Bitcoin_Analysis_ML/refs/heads/main/results/figures/model_predictions.png)
*Figure 6: Time series of actual Bitcoin returns (blue) compared with Random Forest predictions (orange) and SVR predictions (green) during February 2025. The RF partially tracks actual return patterns but underestimates extreme movements, while SVR predictions remain near zero throughout, demonstrating its limited predictive power.*

### Experiment 2: Data Augmentation Effects
I investigated four augmentation techniques beyond the baseline (no augmentation):
- Gaussian noise addition (std=0.01, 0.05, 0.1)
- Synthetic sample mixing (averaging random pairs of training examples)

**Results:**
Moderate Gaussian noise (std=0.05) yielded the best improvement, reducing MSE from 0.00075 to 0.00071 and improving R² from -0.8 to -0.7. This suggests that adding controlled noise can act as a regularizer, helping the model generalize better. Smaller noise levels were ineffective, while synthetic mixing did not help.

![Data Augmentation Results](https://raw.githubusercontent.com/chu-siang/Bitcoin_Analysis_ML/refs/heads/main/results/experiments/data_augmentation_experiment.png)
*Figure 7: Bar charts comparing MSE (left) and R² (right) for different data augmentation methods. The baseline shows moderate performance, tiny noise (std=0.01) slightly worsens results, moderate noise (std=0.05) gives the best performance with lowest MSE and highest R², while higher noise (std=0.1) and synthetic mixing show results similar to baseline.*

### Experiment 3: PCA Dimensionality Reduction
I tested how using principal component analysis (PCA) to reduce feature dimensions affected model performance.

**Results:**
PCA dimensionality reduction degraded model performance. Using fewer components (e.g., 3) significantly increased MSE and worsened R². Performance gradually improved as more components were added, approaching but not exceeding the original feature set performance. This indicates that the full feature set contains valuable information not captured by the first few principal components.

![PCA Experiment Results](https://raw.githubusercontent.com/chu-siang/Bitcoin_Analysis_ML/refs/heads/main/results/experiments/pca_experiment.png)
*Figure 8: Effects of PCA dimensionality reduction on model performance. The top-left chart shows MSE increasing dramatically with fewer components and gradually improving as components increase. Top-right shows R² following the opposite pattern. Bottom-left relates MSE to variance explained, and bottom-right displays the component loadings heatmap showing how original features contribute to principal components.*

### Experiment 4: Market State Clustering
I analyzed the four market states identified by K-means clustering to understand their characteristics and temporal distribution.

**Results:**
The clustering revealed four distinct market regimes with clear differences in future returns, volatility, RSI, and volume:

- **Cluster 0 (Purple)**: "Calm Market" - Low returns (~0.05%), neutral RSI (~50), lowest volatility (~0.022) and volume (~800). Represents sideways, low-activity periods.

- **Cluster 1 (Blue)**: "Correction Phase" - Moderate returns (~0.25%), relatively low RSI (~35-40), high volatility (~0.028), moderate volume (~1500). Represents recovering or dipping markets.

- **Cluster 2 (Green)**: "Bull Market" - High returns (~0.30%), very high RSI (~70), highest volatility (~0.030), high volume (~2500). Represents bullish trending states.

- **Cluster 3 (Yellow)**: "Reversal State" - Highest returns (~0.35%), lowest RSI (~30), high volatility (~0.028), highest volume (~3200). Represents post-crash rebound situations.

![PCA Visualization of Clusters](https://raw.githubusercontent.com/chu-siang/Bitcoin_Analysis_ML/refs/heads/main/results/figures/cluster_pca.png)
*Figure 9: Alternative PCA visualization of the Bitcoin market state clusters. This projection shows the distinctive distribution of cluster points across the principal component space with different variance scaling, highlighting outlier points and revealing the large-scale structure of the dataset.*

![Cluster Characteristics](https://raw.githubusercontent.com/chu-siang/Bitcoin_Analysis_ML/refs/heads/main/results/figures/cluster_statistics.png)
*Figure 10: Bar charts showing key statistical properties of each cluster. Top-left shows average 24h future returns (highest in Cluster 3); top-right shows return volatility (lowest in Cluster 0); bottom-left shows average RSI (highest in Cluster 2); bottom-right shows trading volume (highest in Cluster 3). These metrics reveal the distinct nature of each market state.*

The temporal distribution of clusters aligned with observable market behavior. Green cluster points appeared during strong uptrends, yellow cluster points often followed local price minima, blue cluster points appeared during corrections, and purple cluster points predominated during flat periods.

## Discussion

### Key Findings

1. **Predictive Challenge**: Both supervised models struggled to predict exact 24-hour returns (negative R²), with Random Forest performing better than SVR. This aligns with the efficient market hypothesis that short-term price movements are difficult to predict.

2. **Technical Indicators**: No single technical indicator strongly predicted future returns. The model relied on combinations of features with Bollinger Bands and moving averages showing modest predictive value.

3. **Data Augmentation**: Adding moderate Gaussian noise (5-10% of feature scale) during training improved model performance slightly, suggesting it helps mitigate overfitting.

4. **Feature Dimensionality**: PCA dimensionality reduction decreased performance, indicating the model benefits from the full feature space and complex feature interactions.

5. **Market State Identification**: Unsupervised clustering successfully identified four meaningful market regimes: calm sideways markets, corrections, bullish trends, and post-crash reversals. These clusters showed distinct characteristics in returns, volatility, RSI, and volume.

### Limitations and Future Work

1. **Advanced Models**: Test deep learning approaches like LSTM networks that might better capture temporal dependencies in price data.

2. **Additional Data**: Incorporate sentiment analysis, on-chain metrics, or macroeconomic indicators to provide broader context.

3. **Alternative Approaches**: Reframe the prediction task as classification rather than regression to potentially achieve better results.

4. **Time Sequence Modeling**: Explore sequence models that account for autocorrelation in returns and indicators over time.

5. **Practical Applications**: While direct return prediction remains challenging, the market state clustering offers practical value for risk management and trading strategy adaptation.

## References

1. Random forests. Machine Learning :  https://medium.com/chung-yi/ml%E5%85%A5%E9%96%80-%E5%8D%81%E4%B8%83-%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97-random-forest-6afc24871857
2. Support-vector networks. Machine Learning. https://scikit-learn.org/stable/modules/svm.html 
3. Scikit-learn: Machine Learning in Python. https://scikit-learn.org/stable/
4. pandas: a Foundational Python Library for Data Analysis. https://pandas.pydata.org/ 
5. Binance API Documentation. https://binance-docs.github.io/apidocs/

## Appendix: Project Structure

The repository contains the following key files and directories:

```
bitcoin-analysis/
├── data/
│   ├── raw/
│   │   └── bitcoin_raw_data.csv
│   └── processed/
│       ├── bitcoin_features.csv
│       └── bitcoin_ml_data.csv
├── src/
│   ├── data/
│   │   ├── fetch_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── create_features.py
│   │   └── create_targets.py
│   ├── models/
│   │   ├── random_forest.py
│   │   ├── svr.py
│   │   └── kmeans.py
│   └── visualization/
│       ├── plot_predictions.py
│       ├── plot_clusters.py
│       └── plot_experiments.py
├── results/
│   ├── figures/
│   │   └── [various plot images]
│   └── metrics/
│       └── [performance metrics CSV files]
├── requirements.txt
└── README.md
```