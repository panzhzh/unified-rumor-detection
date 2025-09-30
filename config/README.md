# Configuration Files

This directory contains **YAML configuration files only**. The configuration loading logic is in `src/config/config_loader.py`.

## Available Configs

- `amg.yaml` - AMG dataset (text-only)
- `dgm4.yaml` - DGM4 dataset (multimodal)
- `finefake.yaml` - FineFake dataset (multimodal)
- `mmfakebench.yaml` - MMFakeBench dataset (multimodal)
- `mr2.yaml` - MR2 dataset (multimodal with OCR)

## Configuration Structure

Each config file has 4 main sections:

### 1. Dataset Configuration
```yaml
dataset:
  name: "AMG"
  data_root: "data"
  modality: "text"  # text, image, multimodal

  train_split: "train"
  val_split: "val"
  test_split: "test"

  has_text: true
  has_image: false
  use_ocr: false

  max_text_length: 512
  image_size: 224
```

### 2. Model Configuration
```yaml
model:
  type: "text"  # text, image, multimodal
  num_classes: 2

  text_encoder: "bert-base-uncased"
  text_hidden_size: 768

  image_encoder: "resnet50"
  image_hidden_size: 2048

  fusion_method: "concat"
  dropout: 0.3
```

### 3. Training Configuration
```yaml
training:
  optimizer: "adamw"
  learning_rate: 2.0e-5
  weight_decay: 0.01

  scheduler: "linear_warmup"
  warmup_ratio: 0.1

  num_epochs: 10
  batch_size: 32
  gradient_accumulation_steps: 1

  loss_fn: "cross_entropy"
  early_stopping: true
  patience: 3
```

### 4. Experiment Configuration
```yaml
experiment:
  name: "amg_text_detection"
  seed: 42
  device: "cuda"

  output_dir: "outputs/amg"
  checkpoint_dir: "checkpoints/amg"
  log_dir: "logs/amg"

  log_interval: 100
  eval_interval: 500
  save_interval: 1000
```

## Usage

### Load Configuration

```python
from config.config_loader import load_dataset_config

# Load config for a dataset
config = load_dataset_config("AMG")

# Access config values using dot notation
print(config.dataset.name)           # "AMG"
print(config.model.text_encoder)     # "bert-base-uncased"
print(config.training.batch_size)    # 32
print(config.experiment.device)      # "cuda"
```

### Validate Configuration

```python
from config.config_loader import load_dataset_config, validate_config

config = load_dataset_config("AMG")
validate_config(config)  # Raises ValueError if invalid
```

### Override Configuration

```python
config = load_dataset_config("AMG")

# Override specific values
config.training.batch_size = 64
config.training.num_epochs = 20
config.experiment.device = "cpu"
```

### Save Configuration

```python
from config.config_loader import save_config

config = load_dataset_config("AMG")
config.training.batch_size = 64

# Save modified config
save_config(config, "config/amg_custom.yaml")
```

## Dataset-Specific Notes

### AMG
- **Text-only** dataset
- Use `bert-base-uncased` or similar text encoders
- Binary classification (real/fake)

### DGM4
- **Multimodal** (text + image)
- Images contain manipulated faces
- Use multimodal fusion models

### FineFake
- **Multimodal** with knowledge graph features
- Longer text (news articles)
- Consider using `roberta-base` for better news understanding

### MMFakeBench
- **Small benchmark** dataset (~1K samples)
- Use smaller batch size and more epochs
- Good for quick prototyping

### MR2
- **Chinese + English** multimodal dataset
- Has **OCR text** from images
- Use `bert-base-multilingual-cased`
- **3-class** classification (real/fake/unverified)

## Creating Custom Configs

To create a custom config for experiments:

1. Copy an existing config file
2. Modify the parameters as needed
3. Save with a descriptive name (e.g., `amg_large_batch.yaml`)
4. Load it: `config = load_config("config/amg_large_batch.yaml")`

## Future: Cross-Dataset Configs

For cross-dataset experiments (to be implemented):

```yaml
# config/cross_amg_to_dgm4.yaml
experiment:
  type: "cross_dataset"
  source_datasets: ["AMG", "DGM4"]
  target_dataset: "MMFakeBench"

# Training only on source, testing on target
```