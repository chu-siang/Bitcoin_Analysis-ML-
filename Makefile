.PHONY: clean data features models experiments visualize test report all

# Default target
all: setup data features models experiments visualize report

# Install dependencies
setup:
	pip install -r requirements.txt

# Fetch raw data
data:
	python3 -m src.data.fetch_data

# Create features
features: data
	python3 -m src.data.preprocess
	python3 -m src.features.create_features
	python3 -m src.features.create_targets

# Train models
models: features
	python3 -m src.models.random_forest
	python3 -m src.models.svr
	python3 -m src.models.kmeans

# Run experiments
experiments: models
	python3 -m experiments.data_size
	python3 -m experiments.data_augmentation
	python3 -m experiments.dimensionality_reduction

# Create visualizations
visualize: models experiments
	python3 -m src.visualization.plot_predictions
	python3 -m src.visualization.plot_clusters

# Run tests
test:
	pytest

# Generate report
report: visualize
	# Try PDF first, fall back to HTML if it fails
	jupyter nbconvert --to pdf report/bitcoin_analysis_report.ipynb || jupyter nbconvert --to html report/bitcoin_analysis_report.ipynb

# Clean all generated files
clean:
	rm -rf data/raw/*.csv
	rm -rf data/processed/*.csv
	rm -rf models/*.pkl
	rm -rf results/figures/*.png
	rm -rf results/metrics/*.csv
	rm -rf report/*.pdf