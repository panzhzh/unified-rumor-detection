# Unified Rumor Detection Framework

A unified framework for multimodal rumor/fake news detection that supports multiple public datasets, single-dataset training, multi-dataset training, and cross-dataset evaluation.

## Features

- **Unified Data Format**: Standardized data structure across 5+ public datasets
- **Flexible Training Modes**: Single dataset, multi-dataset joint training, cross-dataset evaluation
- **Multimodal Support**: Text, Image, and Text+Image models
- **Modular Architecture**: Reusable base components and isolated experiment models

## Supported Datasets

| Dataset | Modality | Language | Size | Label Type |
|---------|----------|----------|------|------------|
| AMG | Text | EN | ~50K | Fine-grained (0-5) |
| DGM4 | Text+Image | EN | ~20K | Binary |
| FineFake | Text+Image | EN | ~30K | Binary + Fine-grained |
| MMFakeBench | Text+Image | EN | ~1K | Binary |
| MR2 | Text+Image+OCR | ZH/EN | ~10K | Ternary (0/1/2) |

## Installation

```bash
# Install dependencies
pip install torch torchvision transformers
pip install scikit-learn pandas numpy pillow
```

## Project Structure

```
unified-rumor-detection/
├── data/                       # Raw datasets and features
│   ├── AMG/
│   ├── DGM4/
│   ├── FineFake/
│   ├── MMFakeBench/
│   ├── MR2/
│   └── features/               # Pre-computed features
│
├── src/
│   ├── data/                   # Unified data loading
│   ├── preprocessing/          # Text processing utilities
│   └── models/
│       ├── base/               # Reusable components (encoders, attention)
│       └── experiments/        # Paper-specific models
│
├── scripts/
│   ├── ocr/                    # OCR extraction
│   └── features/               # Feature extraction
│
├── configs/                    # YAML configurations
├── results/                    # Experiment results
└── README.md
```

## Quick Start

### 1. Load Data

```python
from src.data import UnifiedDataLoader

loader = UnifiedDataLoader(data_root="data")
amg_data = loader.load_dataset("AMG", split="train")
```

### 2. Extract Features

```bash
# Extract image features
python scripts/features/extract_image_features.py --model clip_large

# Extract OCR text
python scripts/ocr/extract_ocr.py --dataset MR2
```

### 3. Use Base Components

```python
from src.models.base import TextEncoder, ImageEncoder, DeepFusionLayer

text_encoder = TextEncoder()
image_encoder = ImageEncoder()
fusion_layer = DeepFusionLayer()
```

## Data Format

All datasets use a unified format:

```python
@dataclass
class UnifiedDataItem:
    id: str                      # Unique identifier
    label: int                   # 0=real, 1=fake, 2=unverified
    dataset_name: str            # Source dataset
    split: str                   # train/val/test
    text: str                    # Main text content
    image_path: str              # Path to image
    ocr: str                     # OCR text from image
    metadata: Dict               # Dataset-specific info
```

**Label Convention**:
- 0: Real/True/Non-rumor
- 1: Fake/False/Rumor
- 2: Unverified (MR2 only)

## Adding New Experiments

### Step 1: Create Model

```python
# src/models/experiments/my_paper/model.py
from ...base import TextEncoder, ImageEncoder, DeepFusionLayer

class MyPaperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.fusion = DeepFusionLayer()
```

### Step 2: Create Config

```yaml
# configs/experiments/my_paper.yaml
model:
  name: "MyPaperModel"
  num_classes: 2

data:
  datasets: ["AMG", "MR2"]
  batch_size: 32

training:
  epochs: 20
  lr: 1e-4
```

### Step 3: Run

```bash
python scripts/experiments/run_my_paper.py --config configs/experiments/my_paper.yaml
```

## License

MIT License
