# Unified Rumor Detection Framework

A unified framework for multimodal rumor/fake news detection that supports multiple public datasets, single-dataset training, multi-dataset training, and cross-dataset evaluation.

## Features

- **Unified Data Format**: Standardized data structure across 5+ public datasets
- **Flexible Training Modes**:
  - Single dataset training
  - Multi-dataset joint training
  - Cross-dataset generalization testing
- **Multimodal Support**: Text, Image, and Text+Image models
- **Multiple Datasets**:
  - AMG (Aligned Multimodal Graph)
  - DGM4 (DeepFake Generation Methods)
  - FineFake (Fine-grained Fake News)
  - MMFakeBench (Multimodal Fake Benchmark)
  - MR2 (Multimodal Rumor Recognition)

## Project Structure

```
unified-rumor-detection/
├── data/                       # Raw datasets
│   ├── AMG/
│   ├── DGM4/
│   ├── FineFake/
│   ├── MMFakeBench/
│   └── MR2/
│
├── config/                     # YAML configuration files only
│   ├── config.yaml             # Master configuration
│   ├── amg.yaml                # AMG dataset config
│   ├── dgm4.yaml               # DGM4 dataset config
│   ├── finefake.yaml           # FineFake dataset config
│   ├── mmfakebench.yaml        # MMFakeBench dataset config
│   └── mr2.yaml                # MR2 dataset config
│
├── src/
│   ├── data/                   # Data loading
│   │   ├── data_loader.py      # Unified data loader
│   │   └── dataset.py          # PyTorch datasets
│   │
│   ├── models/                 # Model architectures
│   │   ├── base_model.py       # Base model interface
│   │   ├── text_models.py      # Text-only models (TODO)
│   │   ├── image_models.py     # Image-only models (TODO)
│   │   └── multimodal_models.py # Multimodal fusion models (TODO)
│   │
│   ├── training/               # Training logic
│   │   ├── trainer.py          # Training loop (TODO)
│   │   └── evaluator.py        # Evaluation (TODO)
│   │
│   ├── config/                 # Configuration utilities
│   │   └── config_loader.py    # Config loading logic
│   │
│   └── utils/                  # Utilities
│       └── metrics.py          # Evaluation metrics
│
├── checkpoints/                # Model checkpoints (created at runtime)
├── outputs/                    # Experiment outputs (created at runtime)
└── logs/                       # Training logs (created at runtime)
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd unified-rumor-detection

# Install dependencies
pip install torch torchvision transformers
pip install scikit-learn pandas numpy pillow
pip install tensorboard wandb  # Optional for logging
```

## Quick Start

### 1. Load Data

```python
from src.data import UnifiedDataLoader

# Initialize loader
loader = UnifiedDataLoader(data_root="data")

# Load single dataset
amg_data = loader.load_dataset("AMG", split="train")

# Load multiple datasets
multi_data = loader.load_multiple(["AMG", "DGM4"], split="train")

# Load only multimodal data
multimodal_data = loader.load_by_modality(
    split="train",
    require_text=True,
    require_image=True
)
```

### 2. Create PyTorch Dataset

```python
from src.data.dataset import create_dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create multimodal dataset
dataset = create_dataset(
    data=multi_data,
    mode='multimodal',
    text_tokenizer=tokenizer
)
```

### 3. Load Configuration

```python
from src.config import load_dataset_config

# Load configuration for a dataset
config = load_dataset_config("AMG")

# Access config values using dot notation
print(config.dataset.name)           # "AMG"
print(config.model.text_encoder)     # "bert-base-uncased"
print(config.training.batch_size)    # 32
print(config.experiment.device)      # "cuda"

# Modify config as needed
config.training.num_epochs = 20
config.training.batch_size = 64
```

### 4. Train Model (TODO)

```python
# Training implementation coming soon
# from src.training import Trainer
#
# trainer = Trainer(model, config)
# trainer.train()
```

## Unified Data Format

All datasets are converted to a unified format:

```python
@dataclass
class UnifiedDataItem:
    # Core fields (always present)
    id: str                      # Unique identifier
    label: int                   # 0=real, 1=fake, 2=unverified
    dataset_name: str            # Source dataset
    split: str                   # train/val/test

    # Content fields (may be None)
    text: str                    # Main text content
    image_path: str              # Path to image
    ocr: str                     # OCR text from image
    entities: List[str]          # Named entities
    timestamp: float             # Publication time
    author: str                  # Author/source
    language: str                # Language code

    # Extended fields
    metadata: Dict               # Dataset-specific info
```

## Label Convention

All datasets use unified labels:
- **0**: Real/True/Non-rumor
- **1**: Fake/False/Rumor
- **2**: Unverified (MR2 only)

Original labels are preserved in `metadata` for reference.

## Supported Datasets

| Dataset | Modality | Language | Size | Label Type |
|---------|----------|----------|------|------------|
| AMG | Text | EN | ~50K | Fine-grained (0-5) |
| DGM4 | Text+Image | EN | ~20K | Binary (original/manipulated) |
| FineFake | Text+Image | EN | ~30K | Binary + Fine-grained |
| MMFakeBench | Text+Image | EN | ~1K | Binary |
| MR2 | Text+Image+OCR | ZH/EN | ~10K | Ternary (0/1/2) |

## Experiment Modes

### 1. Single Dataset Training
Train and test on the same dataset:
```python
config = get_single_dataset_config("AMG")
```

### 2. Multi-Dataset Training
Train on multiple datasets jointly:
```python
config = get_multi_dataset_config(["AMG", "DGM4", "FineFake"])
```

### 3. Cross-Dataset Evaluation
Train on source datasets, test on target:
```python
config = get_cross_dataset_config(
    source_datasets=["AMG", "DGM4"],
    target_dataset="MMFakeBench"
)
```

## Evaluation Metrics

- Accuracy
- Precision (macro/weighted/per-class)
- Recall (macro/weighted/per-class)
- F1-score (macro/weighted/per-class)
- AUC-ROC (binary classification)
- Confusion Matrix
- Per-dataset metrics (for multi-dataset evaluation)

## TODO

- [ ] Implement text-only models (BERT, RoBERTa)
- [ ] Implement image-only models (ResNet, ViT)
- [ ] Implement multimodal fusion models (CLIP, ViLT, etc.)
- [ ] Implement trainer and evaluator
- [ ] Add visualization tools
- [ ] Add experiment scripts
- [ ] Add pre-trained model zoo
- [ ] Add detailed tutorials

## Citation

If you use this framework, please cite the original dataset papers:
- AMG: [Paper Link]
- DGM4: [Paper Link]
- FineFake: [Paper Link]
- MMFakeBench: [Paper Link]
- MR2: [Paper Link]

## License

MIT License