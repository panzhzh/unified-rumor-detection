# OCR Extraction Scripts

This directory contains scripts for extracting OCR text from images in the rumor detection datasets.

## Prerequisites

**GPU Required**: This script requires CUDA-enabled GPU (CUDA 12.x supported).

Install PaddlePaddle and PaddleOCR:
```bash
# Install PaddlePaddle GPU version (CUDA 12.x compatible)
python -m pip install paddlepaddle-gpu==3.0.0

# Install PaddleOCR
python -m pip install paddleocr
```

This script uses **PP-OCRv5_server** models for state-of-the-art OCR performance.

## Usage

### Extract OCR from all datasets (default)

```bash
python scripts/ocr/extract_ocr.py
```

This will process all datasets (AMG, DGM4, FineFake, MMFakeBench, MR2) for train/val/test splits.

### Extract from specific datasets

```bash
python scripts/ocr/extract_ocr.py --datasets AMG DGM4
```

### Extract from specific splits

```bash
python scripts/ocr/extract_ocr.py --splits train val
```

### Force re-extraction (overwrite existing files)

```bash
python scripts/ocr/extract_ocr.py --force
```

### Combine options

```bash
python scripts/ocr/extract_ocr.py \
    --datasets AMG FineFake \
    --splits train test \
    --force
```

## Output Format

### Main OCR Results

OCR results are saved in `data/{DATASET}/ocr_{split}.json` with the following format:

```json
[
  {
    "id": "AMG_12345",
    "ocr": "detected text from image",
    "confidence": 0.95
  }
]
```

**Fields:**
- **id**: Unique identifier matching the original dataset
- **ocr**: Extracted text from the image (empty string if no text detected)
- **confidence**: Average confidence score (0-1) from OCR engine

### MR2 Evidence OCR Results

MR2 evidence OCR results are saved in `data/MR2/evidence_ocr_{split}.json` with the following format:

```json
{
  "0": [
    {
      "image_path": "val/img_html_news/2/78.jpg",
      "ocr": "detected text from evidence image",
      "confidence": 0.92
    },
    {
      "image_path": "val/img_html_news/2/79.jpg",
      "ocr": "another evidence text",
      "confidence": 0.88
    }
  ],
  "1": []
}
```

**Structure:**
- Top-level keys are sample IDs (as strings)
- Each sample has a list of evidence OCR results (one entry per evidence image)
- Empty lists indicate samples with no evidence images

**Fields:**
- **image_path**: Relative path to the evidence image (serves as unique identifier)
- **ocr**: Extracted text from the evidence image
- **confidence**: Average confidence score (0-1) from OCR engine

## Output Locations

### Main OCR Results
- AMG: `data/AMG/ocr_train.json`, `data/AMG/ocr_val.json`, `data/AMG/ocr_test.json`
- DGM4: `data/DGM4/ocr_train.json`, `data/DGM4/ocr_val.json`, `data/DGM4/ocr_test.json`
- FineFake: `data/FineFake/ocr_train.json`, `data/FineFake/ocr_val.json`, `data/FineFake/ocr_test.json`
- MMFakeBench: `data/MMFakeBench/ocr_val.json`, `data/MMFakeBench/ocr_test.json`
- MR2: `data/MR2/ocr_train.json`, `data/MR2/ocr_val.json`, `data/MR2/ocr_test.json`

### MR2 Evidence OCR Results
- MR2 Evidence: `data/MR2/evidence_ocr_train.json`, `data/MR2/evidence_ocr_val.json`, `data/MR2/evidence_ocr_test.json`

The MR2 evidence OCR files contain OCR results for evidence images associated with each claim, organized by sample ID.

## Notes

- The script automatically skips existing output files unless `--force` is specified
- Progress bars show extraction progress for each dataset/split
- Statistics are printed after each dataset is processed
- All datasets (including MR2) will be re-extracted for unified format consistency
- Uses PP-OCRv5 server models which support both Chinese and English text recognition
- GPU is required for optimal performance