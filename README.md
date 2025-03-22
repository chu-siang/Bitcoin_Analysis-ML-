# Bitcoin Analysis Project

This project analyzes Bitcoin data.

-------
### Folder Structure
```plaintext
bitcoin-analysis/
├── README.md
├── requirements.txt
├── Makefile
├── data/
│   ├── raw/
│   │   └── bitcoin_raw_data.csv
│   └── processed/
│       ├── bitcoin_features.csv
│       └── bitcoin_ml_data.csv
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetch_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── create_features.py
│   │   └── create_targets.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── random_forest.py
│   │   ├── svr.py
│   │   └── kmeans.py
│   └── visualization/
│       ├── __init__.py
│       ├── plot_predictions.py
│       ├── plot_clusters.py
│       └── plot_experiments.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_analysis_and_results.ipynb
├── experiments/
│   ├── __init__.py
│   ├── data_size.py
│   ├── data_augmentation.py
│   └── dimensionality_reduction.py
├── models/
│   ├── random_forest_model.pkl
│   ├── svr_model.pkl
│   ├── kmeans_model.pkl
│   ├── optimized_random_forest_model.pkl
│   └── optimized_svr_model.pkl
├── results/
│   ├── figures/
│   │   ├── prediction_comparison.png
│   │   ├── feature_importance.png
│   │   ├── kmeans_clusters.png
│   │   ├── training_size_experiment.png
│   │   ├── augmentation_experiment.png
│   │   ├── pca_experiment.png
│   │   └── cluster_analysis.png
│   └── metrics/
│       └── model_performance.csv
└── report/
    ├── bitcoin_analysis_report.pdf
    └── figures/
        ├── price_chart.png
        ├── model_comparison.png
        └── cluster_visualization.png
