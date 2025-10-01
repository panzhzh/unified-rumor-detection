# Image Feature Extraction Scripts

This directory contains scripts for extracting pre-computed image features from the rumor detection datasets.

## Features

- **Multiple Model Support**: ViT (base/large), CLIP (base/large)
- **Animated Image Handling**: Multi-frame extraction with attention pooling
- **Extreme Aspect Ratio Handling**: Padding for very wide/tall images
- **Unified Format**: Works with all 5 datasets (AMG, DGM4, FineFake, MMFakeBench, MR2)
- **GPU Acceleration**: Automatic GPU detection
- **Efficient Caching**: Features saved as `.npy` files

## Prerequisites

Install required packages:
```bash
pip install torch torchvision transformers Pillow numpy tqdm
```

## Usage

### Extract features from all datasets (default)

```bash
python scripts/features/extract_image_features.py
```

This uses CLIP-Large by default and processes all datasets (train/val/test splits).

### Specify model

```bash
# CLIP Large (1024-dim, recommended)
python scripts/features/extract_image_features.py --model clip_large

# CLIP Base (768-dim, faster)
python scripts/features/extract_image_features.py --model clip_base

# ViT Large (1024-dim)
python scripts/features/extract_image_features.py --model vit_large

# ViT Base (768-dim, fastest)
python scripts/features/extract_image_features.py --model vit_base
```

### Extract from specific datasets

```bash
python scripts/features/extract_image_features.py \
    --model clip_large \
    --datasets AMG FineFake MR2
```

### Extract from specific splits

```bash
python scripts/features/extract_image_features.py \
    --model clip_large \
    --splits train val
```

### Force re-extraction (overwrite existing files)

```bash
python scripts/features/extract_image_features.py --force
```

### Combine options

```bash
python scripts/features/extract_image_features.py \
    --model vit_large \
    --datasets AMG MR2 \
    --splits train test \
    --force
```

## Output Format

Features are saved in `data/features/{DATASET}/{MODEL}_{split}.npy` as Python dictionaries:

```python
{
    "item_id_1": np.array([...]),  # shape: (feature_dim,)
    "item_id_2": np.array([...]),
    ...
}
```

**Example:**
- `data/features/AMG/clip_large_train.npy` - CLIP-Large features for AMG training set
- `data/features/MR2/vit_base_test.npy` - ViT-Base features for MR2 test set

**Loading features:**
```python
import numpy as np

# Load features
features = np.load('data/features/AMG/clip_large_train.npy', allow_pickle=True).item()

# Access feature for specific item
item_id = "123"
feature_vector = features[item_id]  # shape: (1024,) for CLIP-Large
```

## Output Locations

### Structure
```
data/features/
├── AMG/
│   ├── clip_large_train.npy
│   ├── clip_large_val.npy
│   └── clip_large_test.npy
├── DGM4/
│   ├── vit_large_train.npy
│   ├── vit_large_val.npy
│   └── vit_large_test.npy
├── FineFake/
├── MMFakeBench/
└── MR2/
```

## Model Information

| Model | HuggingFace Model | Feature Dim | Speed | Recommended |
|-------|------------------|-------------|-------|-------------|
| `clip_large` | `openai/clip-vit-large-patch14` | 1024 | Medium | ✅ Yes |
| `clip_base` | `openai/clip-vit-base-patch32` | 768 | Fast | Good |
| `vit_large` | `google/vit-large-patch16-224` | 1024 | Slow | Good |
| `vit_base` | `google/vit-base-patch16-224` | 768 | Fastest | Budget |

## Special Features

### Animated Image Processing

For animated images (GIFs, animated WebP):
1. Extracts up to 8 keyframes
2. Extracts features for each frame
3. Pools frames using attention mechanism
4. Returns single feature vector

### Extreme Aspect Ratio Handling

For images with extreme aspect ratios (>5:1 or <1:5):
1. Resizes to fit within 224×224 while maintaining aspect ratio
2. Pads with white background to make square
3. Prevents information loss from aggressive cropping

## Notes

- The script automatically skips existing feature files unless `--force` is specified
- Progress bars show extraction progress for each dataset/split
- Statistics are printed after each dataset is processed
- Feature extraction uses GPU if available, falls back to CPU
- All features are saved using item IDs from the unified data loader for consistency

## Comparison with OCR Extraction

Similar to the OCR extraction pipeline (`scripts/ocr/extract_ocr.py`), this script:
- ✅ Works with unified data format
- ✅ Supports all 5 datasets
- ✅ Provides progress tracking
- ✅ Supports force re-extraction
- ✅ Uses consistent output format
- ✅ Integrates with `UnifiedDataLoader`

**Key Difference**:
- OCR extracts text from images → saved as JSON
- This script extracts visual features → saved as .npy (numpy arrays)
