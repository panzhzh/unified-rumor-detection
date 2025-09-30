# Configuration System Usage Guide

## Overview

The configuration system uses a **hierarchical inheritance model**:
- `config.yaml` - Master configuration with all default parameters
- `<dataset>.yaml` - Dataset-specific configs that override defaults

## Configuration Files

### 1. Master Config: `config.yaml`

The master configuration file contains:
- Default parameters for all experiments
- Dataset selection (`dataset.name`)
- Cross-dataset testing configuration (`same_dataset`, `test_dataset`)
- All training hyperparameters

**Key fields:**
```yaml
dataset:
  name: "AMG"              # Primary dataset
  same_dataset: true       # Train and test on same dataset
  test_dataset: null       # If same_dataset=false, specify test dataset
```

### 2. Dataset-Specific Configs

Each dataset has its own config file (e.g., `amg.yaml`, `dgm4.yaml`) that **overrides** the master config.

**Inheritance rule:** Dataset config > Master config

## Usage Examples

### Example 1: Load Default Configuration

Load configuration from `config.yaml`:

```python
from src.config import load_dataset_config

# Load the dataset specified in config.yaml (dataset.name = "AMG")
config = load_dataset_config()

print(config.dataset.name)        # "AMG"
print(config.training.batch_size)  # 32
print(config.training.num_epochs)  # 10
```

### Example 2: Load Specific Dataset Config

Load a specific dataset config merged with defaults:

```python
# Load DGM4 config (merges dgm4.yaml with config.yaml)
config = load_dataset_config("DGM4")

print(config.dataset.name)         # "DGM4"
print(config.dataset.modality)     # "multimodal"
print(config.training.batch_size)  # 16 (from dgm4.yaml)
print(config.training.num_epochs)  # 15 (from dgm4.yaml)
print(config.training.optimizer)   # "adamw" (from config.yaml)
```

### Example 3: Same Dataset Training & Testing

Train and test on the same dataset (default behavior):

```yaml
# config.yaml
dataset:
  name: "AMG"
  same_dataset: true  # Train and test on AMG
```

```python
from src.config import load_dataset_config, get_train_test_datasets

config = load_dataset_config("AMG")
train_ds, test_ds = get_train_test_datasets(config)

print(f"Train: {train_ds}, Test: {test_ds}")  # Train: AMG, Test: AMG
```

### Example 4: Cross-Dataset Evaluation

Train on one dataset, test on another:

**Method 1: Modify `config.yaml`**
```yaml
# config.yaml
dataset:
  name: "AMG"              # Train on AMG
  same_dataset: false      # Enable cross-dataset
  test_dataset: "DGM4"     # Test on DGM4
```

**Method 2: Modify config in code**
```python
config = load_dataset_config("AMG")

# Enable cross-dataset testing
config.dataset.same_dataset = false
config.dataset.test_dataset = "DGM4"

train_ds, test_ds = get_train_test_datasets(config)
print(f"Train: {train_ds}, Test: {test_ds}")  # Train: AMG, Test: DGM4
```

### Example 5: Override Configuration Parameters

Override any parameter at runtime:

```python
config = load_dataset_config("AMG")

# Override training parameters
config.training.batch_size = 64
config.training.num_epochs = 20
config.training.learning_rate = 3e-5

# Override experiment settings
config.experiment.device = "cpu"
config.experiment.log_interval = 50

# Save modified config
from src.config import save_config
save_config(config, "config/amg_custom.yaml")
```

### Example 6: Check Cross-Dataset Mode

Check if configuration is for cross-dataset evaluation:

```python
from src.config import is_cross_dataset

config = load_dataset_config()

if is_cross_dataset(config):
    print("Running cross-dataset evaluation")
    train_ds, test_ds = get_train_test_datasets(config)
    print(f"Training on {train_ds}, testing on {test_ds}")
else:
    print("Running same-dataset evaluation")
```

## Configuration Inheritance

### How It Works

1. **Load master config** (`config.yaml`)
2. **Load dataset config** (e.g., `amg.yaml`)
3. **Merge**: Dataset-specific values override master values
4. **Return** merged configuration

### Example: Batch Size Inheritance

**config.yaml:**
```yaml
training:
  batch_size: 32  # Default
  num_epochs: 10
```

**dgm4.yaml:**
```yaml
training:
  batch_size: 16  # Override for DGM4
  # num_epochs not specified, inherits 10 from config.yaml
```

**Result:**
```python
config = load_dataset_config("DGM4")
print(config.training.batch_size)  # 16 (from dgm4.yaml)
print(config.training.num_epochs)  # 10 (from config.yaml)
```

## Common Use Cases

### Use Case 1: Quick Experiment on Different Dataset

```python
# Just change the dataset name
config = load_dataset_config("MMFakeBench")
# All MMFakeBench-specific settings loaded automatically
```

### Use Case 2: Hyperparameter Tuning

```python
config = load_dataset_config("AMG")

# Try different learning rates
for lr in [1e-5, 2e-5, 5e-5]:
    config.training.learning_rate = lr
    config.experiment.name = f"amg_lr_{lr}"
    # Run training with this config...
```

### Use Case 3: Cross-Dataset Generalization Study

```python
# Train on multiple source datasets, test on target
source_datasets = ["AMG", "DGM4", "FineFake"]
target_dataset = "MMFakeBench"

for source in source_datasets:
    config = load_dataset_config(source)
    config.dataset.same_dataset = False
    config.dataset.test_dataset = target_dataset
    config.experiment.name = f"cross_{source}_to_{target}"
    # Run training and evaluation...
```

### Use Case 4: Reproducibility

```python
# Always use the same seed for reproducibility
config = load_dataset_config("AMG")
config.experiment.seed = 42

# Save config for future reference
save_config(config, f"outputs/{config.experiment.name}/config.yaml")
```

## Configuration Validation

Validate configuration before training:

```python
from src.config import validate_config

config = load_dataset_config("AMG")

try:
    validate_config(config)
    print("✓ Configuration is valid")
except ValueError as e:
    print(f"✗ Configuration error: {e}")
```

## Tips

1. **Always use inheritance**: Use `load_dataset_config()` with defaults
2. **Save modified configs**: Save custom configs for reproducibility
3. **Check cross-dataset mode**: Use `is_cross_dataset()` and `get_train_test_datasets()`
4. **Validate before training**: Use `validate_config()` to catch errors early
5. **Use descriptive experiment names**: Set `config.experiment.name` based on your setup

## API Reference

### Functions

- `load_default_config()` - Load master config.yaml
- `load_dataset_config(dataset_name=None, use_default=True)` - Load dataset config with inheritance
- `get_train_test_datasets(config)` - Get train and test dataset names
- `is_cross_dataset(config)` - Check if cross-dataset mode
- `validate_config(config)` - Validate configuration structure
- `save_config(config, path)` - Save configuration to YAML
- `merge_configs(base, override)` - Merge two configurations

### ConfigDict

The configuration object supports dot notation:

```python
config.dataset.name              # Access
config.training.batch_size = 64  # Modify
```