# Expose experiment functions
from .data_size import experiment_data_size
from .data_augmentation import experiment_data_augmentation
from .dimensionality_reduction import experiment_pca

__all__ = ['experiment_data_size', 'experiment_data_augmentation', 'experiment_pca']