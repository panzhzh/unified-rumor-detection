"""Configuration loader for YAML config files"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


class ConfigDict(dict):
    """Dictionary that supports dot notation access"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key):
        try:
            value = self[key]
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


def load_config(config_path: str) -> ConfigDict:
    """Load configuration from YAML file

    Args:
        config_path: Path to YAML config file

    Returns:
        ConfigDict object with dot notation access

    Example:
        >>> config = load_config("config/amg.yaml")
        >>> print(config.dataset.name)
        'AMG'
        >>> print(config.training.batch_size)
        32
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    return ConfigDict(config_dict)


def load_default_config(config_dir: str = "config") -> ConfigDict:
    """Load the default master configuration

    Args:
        config_dir: Directory containing config files

    Returns:
        ConfigDict object

    Example:
        >>> config = load_default_config()
        >>> print(config.dataset.name)
        'AMG'
    """
    config_path = Path(config_dir) / "config.yaml"
    return load_config(str(config_path))


def load_dataset_config(
    dataset_name: Optional[str] = None,
    config_dir: str = "config",
    use_default: bool = True
) -> ConfigDict:
    """Load configuration for a specific dataset with inheritance from master config

    Args:
        dataset_name: Name of the dataset (e.g., 'AMG', 'DGM4')
                     If None, loads from master config's dataset.name field
        config_dir: Directory containing config files
        use_default: If True, merge with default config.yaml

    Returns:
        ConfigDict object with merged configuration

    Example:
        >>> # Load default config (from config.yaml)
        >>> config = load_dataset_config()
        >>> print(config.dataset.name)
        'AMG'

        >>> # Load specific dataset config merged with defaults
        >>> config = load_dataset_config("DGM4")
        >>> print(config.model.type)
        'multimodal'

        >>> # Load dataset config without defaults
        >>> config = load_dataset_config("AMG", use_default=False)
    """
    config_dir = Path(config_dir)

    if use_default:
        # Load master config as base
        base_config = load_default_config(config_dir)

        # If no dataset specified, use the one from master config
        if dataset_name is None:
            dataset_name = base_config.dataset.name

        # Check if dataset-specific config exists
        dataset_config_path = config_dir / f"{dataset_name.lower()}.yaml"
        if dataset_config_path.exists():
            # Load dataset-specific config
            dataset_config = load_config(str(dataset_config_path))
            # Merge: dataset-specific overrides default
            merged = merge_configs(dict(base_config), dict(dataset_config))
            return ConfigDict(merged)
        else:
            # No dataset-specific config, just return default with dataset name set
            base_config.dataset.name = dataset_name
            return base_config
    else:
        # Load only dataset-specific config without defaults
        if dataset_name is None:
            raise ValueError("dataset_name must be specified when use_default=False")
        config_path = config_dir / f"{dataset_name.lower()}.yaml"
        return load_config(str(config_path))


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file

    Args:
        config: Configuration dictionary
        save_path: Path to save the config file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(dict(config), f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations, with override_config taking precedence

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def validate_config(config: ConfigDict) -> bool:
    """Validate configuration structure

    Args:
        config: Configuration to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    required_sections = ['dataset', 'model', 'training', 'experiment']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    # Validate dataset section
    required_dataset_fields = ['name', 'data_root', 'modality']
    for field in required_dataset_fields:
        if field not in config.dataset:
            raise ValueError(f"Missing required dataset field: {field}")

    # Validate model section
    required_model_fields = ['type', 'num_classes']
    for field in required_model_fields:
        if field not in config.model:
            raise ValueError(f"Missing required model field: {field}")

    # Validate training section
    required_training_fields = ['optimizer', 'learning_rate', 'num_epochs', 'batch_size']
    for field in required_training_fields:
        if field not in config.training:
            raise ValueError(f"Missing required training field: {field}")

    # Validate experiment section
    required_experiment_fields = ['name', 'seed', 'device']
    for field in required_experiment_fields:
        if field not in config.experiment:
            raise ValueError(f"Missing required experiment field: {field}")

    return True


def get_train_test_datasets(config: ConfigDict) -> tuple:
    """Get training and testing dataset names from config

    Args:
        config: Configuration object

    Returns:
        Tuple of (train_dataset, test_dataset)
        If same_dataset=True, both are the same

    Example:
        >>> config = load_dataset_config()
        >>> train_ds, test_ds = get_train_test_datasets(config)
        >>> print(f"Train: {train_ds}, Test: {test_ds}")
        Train: AMG, Test: AMG
    """
    train_dataset = config.dataset.name

    if config.dataset.get('same_dataset', True):
        # Train and test on same dataset
        test_dataset = train_dataset
    else:
        # Cross-dataset: use specified test dataset
        test_dataset = config.dataset.get('test_dataset', train_dataset)

    return train_dataset, test_dataset


def is_cross_dataset(config: ConfigDict) -> bool:
    """Check if configuration is for cross-dataset evaluation

    Args:
        config: Configuration object

    Returns:
        True if cross-dataset, False if same dataset

    Example:
        >>> config = load_dataset_config()
        >>> if is_cross_dataset(config):
        ...     print("Cross-dataset evaluation")
        ... else:
        ...     print("Same dataset evaluation")
    """
    return not config.dataset.get('same_dataset', True)


# Available datasets
AVAILABLE_DATASETS = ['AMG', 'DGM4', 'FineFake', 'MR2']


def get_available_configs(config_dir: str = "config") -> list:
    """Get list of available dataset configurations

    Args:
        config_dir: Directory containing config files

    Returns:
        List of available dataset names
    """
    config_dir = Path(config_dir)
    if not config_dir.exists():
        return []

    configs = []
    for config_file in config_dir.glob("*.yaml"):
        if config_file.stem not in ['base', 'default', 'config_loader']:
            configs.append(config_file.stem.upper())

    return configs


if __name__ == "__main__":
    # Test config loading
    print("=" * 70)
    print("Testing Config Loader with Inheritance")
    print("=" * 70)

    # Test 1: Load default config
    print("\n[1] Load default config (config.yaml):")
    try:
        config = load_default_config()
        print(f"✓ Default config loaded")
        print(f"  - Dataset: {config.dataset.name}")
        print(f"  - Same dataset: {config.dataset.same_dataset}")
        print(f"  - Epochs: {config.training.num_epochs}")
        print(f"  - Batch size: {config.training.batch_size}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 2: Load dataset-specific configs with inheritance
    print("\n[2] Load dataset configs with inheritance from config.yaml:")
    for dataset in AVAILABLE_DATASETS:
        try:
            config = load_dataset_config(dataset, use_default=True)
            validate_config(config)
            train_ds, test_ds = get_train_test_datasets(config)
            print(f"\n✓ {dataset} config (merged with defaults):")
            print(f"  - Modality: {config.dataset.modality}")
            print(f"  - Model: {config.model.type}")
            print(f"  - Batch size: {config.training.batch_size}")
            print(f"  - Epochs: {config.training.num_epochs}")
            print(f"  - Train on: {train_ds}, Test on: {test_ds}")
        except Exception as e:
            print(f"\n✗ {dataset} error: {e}")

    # Test 3: Load config from master config.yaml's dataset field
    print("\n[3] Load config using dataset.name from config.yaml:")
    try:
        config = load_dataset_config()  # No dataset specified
        print(f"✓ Loaded config for: {config.dataset.name}")
        print(f"  - Batch size: {config.training.batch_size}")
        print(f"  - Model: {config.model.type}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 4: Test cross-dataset configuration
    print("\n[4] Test cross-dataset detection:")
    config = load_dataset_config("AMG")
    print(f"  - Is cross-dataset: {is_cross_dataset(config)}")
    train_ds, test_ds = get_train_test_datasets(config)
    print(f"  - Train dataset: {train_ds}")
    print(f"  - Test dataset: {test_ds}")

    # Test 5: Override check
    print("\n[5] Test config override (AMG overrides default batch_size):")
    default_config = load_default_config()
    amg_config = load_dataset_config("AMG", use_default=True)
    print(f"  - Default batch_size: {default_config.training.batch_size}")
    print(f"  - AMG batch_size: {amg_config.training.batch_size}")
    print(f"  - Override successful: {amg_config.training.batch_size == 32}")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)