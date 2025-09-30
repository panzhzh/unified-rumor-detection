"""Configuration module for loading and managing experiment configs"""

from .config_loader import (
    ConfigDict,
    load_config,
    load_default_config,
    load_dataset_config,
    get_train_test_datasets,
    is_cross_dataset,
    validate_config,
    save_config,
    merge_configs,
    get_available_configs,
    AVAILABLE_DATASETS
)

__all__ = [
    'ConfigDict',
    'load_config',
    'load_default_config',
    'load_dataset_config',
    'get_train_test_datasets',
    'is_cross_dataset',
    'validate_config',
    'save_config',
    'merge_configs',
    'get_available_configs',
    'AVAILABLE_DATASETS'
]