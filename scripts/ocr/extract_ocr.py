#!/usr/bin/env python3
"""
OCR Extraction Script for Unified Rumor Detection Datasets

This script extracts OCR text from images in all datasets and saves the results
in a unified format: [{id, ocr, confidence}]

Supports: AMG, DGM4, FineFake, MMFakeBench, MR2

Requires: GPU (CUDA-enabled)
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import UnifiedDataLoader

try:
    from paddleocr import PaddleOCR
    import paddle
except ImportError:
    print("Error: PaddleOCR not installed. Please run: pip install paddleocr")
    sys.exit(1)

# Check GPU availability
if not paddle.is_compiled_with_cuda():
    print("Error: PaddlePaddle is not compiled with CUDA support.")
    print("GPU is required for this script. Please install GPU-enabled PaddlePaddle:")
    print("  python -m pip install paddlepaddle-gpu==3.0.0+")
    sys.exit(1)

if not paddle.device.cuda.device_count() > 0:
    print("Error: No GPU detected. This script requires GPU acceleration.")
    sys.exit(1)


@dataclass
class OCRResult:
    """OCR result for a single item"""
    id: str
    ocr: str
    confidence: float


class OCRExtractor:
    """Extract OCR text from images using PaddleOCR PP-OCRv5 (GPU-accelerated)"""

    def __init__(self):
        """Initialize PaddleOCR with PP-OCRv5 server models (supports Chinese+English)"""
        print("Initializing PaddleOCR PP-OCRv5 with GPU...")
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_textline_orientation=True
        )
        print("PaddleOCR PP-OCRv5 initialized successfully with GPU!")

    @staticmethod
    def _print_statistics(title: str, stats: dict, output_path: Path):
        """Print OCR extraction statistics"""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        for key, value in stats.items():
            print(f"{key}: {value}")
        print(f"Output saved to: {output_path}")
        print(f"{'='*60}\n")

    def extract_from_image(self, image_path: str) -> Optional[OCRResult]:
        """
        Extract OCR text from a single image

        Args:
            image_path: Path to image file

        Returns:
            OCRResult or None if extraction failed
        """
        if not os.path.exists(image_path):
            return None

        try:
            # Run OCR with new API
            result = self.ocr.predict(image_path)

            # Extract text from new API format
            if result and len(result) > 0:
                rec_texts = result[0].get('rec_texts', [])
                rec_scores = result[0].get('rec_scores', [])

                if rec_texts and len(rec_texts) > 0:
                    ocr_text = " ".join(rec_texts)
                    avg_confidence = sum(rec_scores) / len(rec_scores) if rec_scores else 0.0

                    return OCRResult(
                        id="",
                        ocr=ocr_text,
                        confidence=round(avg_confidence, 4)
                    )

            # No text detected
            return OCRResult(
                id="",
                ocr="",
                confidence=0.0
            )

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def extract_mr2_evidence_ocr(
        self,
        split: str,
        data_root: str = "data",
        output_dir: str = "data",
        force: bool = False
    ) -> dict:
        """
        Extract OCR from MR2 evidence images

        Args:
            split: Data split (train/val/test)
            data_root: Root directory of datasets
            output_dir: Directory to save OCR results
            force: Force re-extraction even if output file exists

        Returns:
            Dict mapping sample_id -> list of evidence OCR results
        """
        output_path = Path(output_dir) / "MR2" / f"evidence_ocr_{split}.json"
        if output_path.exists() and not force:
            print(f"Evidence OCR file already exists: {output_path}")
            print("Use --force to re-extract. Skipping...")
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Load evidence JSON
        evidence_path = Path(data_root) / "MR2" / f"evidence_{split}.json"
        print(f"\nLoading MR2 evidence from {evidence_path}...")

        with open(evidence_path, 'r', encoding='utf-8') as f:
            evidence_data = json.load(f)

        print(f"Found evidence for {len(evidence_data)} samples")

        # Extract OCR for each sample's evidence
        all_evidence_ocr = {}
        total_images = 0
        processed_images = 0

        # Count total images first
        for sample_id, evidences in evidence_data.items():
            if evidences:
                total_images += sum(1 for ev in evidences if ev.get('image_path'))

        print(f"Found {total_images} evidence images to process")

        with tqdm(total=total_images, desc=f"MR2 Evidence {split}") as pbar:
            for sample_id, evidences in evidence_data.items():
                if not evidences:
                    all_evidence_ocr[sample_id] = []
                    continue

                sample_evidence_ocr = []

                for evidence in evidences:
                    image_path = evidence.get('image_path')

                    if not image_path:
                        continue

                    # Build full image path
                    full_image_path = Path(data_root) / "MR2" / image_path

                    # Extract OCR
                    result = self.extract_from_image(str(full_image_path))

                    if result:
                        sample_evidence_ocr.append({
                            "image_path": image_path,
                            "ocr": result.ocr,
                            "confidence": result.confidence
                        })
                        processed_images += 1

                    pbar.update(1)

                all_evidence_ocr[sample_id] = sample_evidence_ocr

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_evidence_ocr, f, ensure_ascii=False, indent=2)

        # Calculate statistics
        samples_with_evidence = sum(1 for v in all_evidence_ocr.values() if v)
        total_evidence_ocr = sum(len(v) for v in all_evidence_ocr.values())
        evidence_with_text = sum(
            1 for sample_evs in all_evidence_ocr.values()
            for ev in sample_evs if ev['ocr'].strip()
        )
        avg_conf = 0
        if total_evidence_ocr > 0:
            total_conf = sum(
                ev['confidence'] for sample_evs in all_evidence_ocr.values()
                for ev in sample_evs
            )
            avg_conf = total_conf / total_evidence_ocr

        # Print statistics
        self._print_statistics(
            f"MR2 Evidence OCR Complete: {split}",
            {
                "Total samples": len(evidence_data),
                "Samples with evidence": samples_with_evidence,
                "Total evidence images processed": processed_images,
                "Evidence with text detected": f"{evidence_with_text} ({evidence_with_text/max(processed_images,1)*100:.1f}%)",
                "Average confidence": f"{avg_conf:.4f}"
            },
            output_path
        )

        return all_evidence_ocr

    def extract_from_dataset(
        self,
        dataset_name: str,
        split: str,
        data_root: str = "data",
        output_dir: str = "data",
        force: bool = False
    ) -> List[OCRResult]:
        """
        Extract OCR for all images in a dataset split

        Args:
            dataset_name: Name of dataset (AMG, DGM4, FineFake, MMFakeBench, MR2)
            split: Data split (train/val/test)
            data_root: Root directory of datasets
            output_dir: Directory to save OCR results
            force: Force re-extraction even if output file exists

        Returns:
            List of OCRResult
        """
        # Check if output already exists
        output_path = Path(output_dir) / dataset_name / f"ocr_{split}.json"
        if output_path.exists() and not force:
            print(f"Output file already exists: {output_path}")
            print("Use --force to re-extract. Skipping...")
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            return [OCRResult(**item) for item in existing_data]

        # Load dataset
        print(f"\nLoading {dataset_name} {split} dataset...")
        loader = UnifiedDataLoader(data_root=data_root)

        try:
            data = loader.load_dataset(dataset_name, split)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

        # Count items with images
        items_with_images_count = sum(1 for item in data if item.has_image())
        print(f"Found {len(data)} total items, {items_with_images_count} with images")

        if len(data) == 0:
            print("No items found. Skipping...")
            return []

        # Extract OCR for ALL items (maintain index alignment)
        ocr_results = []
        print(f"\nExtracting OCR from {len(data)} items...")

        for item in tqdm(data, desc=f"{dataset_name} {split}"):
            # If item has no image, create empty OCR result
            if not item.has_image():
                ocr_results.append(OCRResult(
                    id=item.id,
                    ocr="",
                    confidence=0.0
                ))
            else:
                # Extract OCR from image
                result = self.extract_from_image(item.image_path)
                if result:
                    result.id = item.id
                    ocr_results.append(result)
                else:
                    # Image exists but OCR failed
                    ocr_results.append(OCRResult(
                        id=item.id,
                        ocr="",
                        confidence=0.0
                    ))

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(r) for r in ocr_results],
                f,
                ensure_ascii=False,
                indent=2
            )

        # Calculate and print statistics
        items_with_text = sum(1 for r in ocr_results if r.ocr.strip())
        avg_confidence = sum(r.confidence for r in ocr_results) / len(ocr_results) if ocr_results else 0

        self._print_statistics(
            f"OCR Extraction Complete: {dataset_name} {split}",
            {
                "Total items processed": len(ocr_results),
                "Items with text detected": f"{items_with_text} ({items_with_text/len(ocr_results)*100:.1f}%)",
                "Average confidence": f"{avg_confidence:.4f}"
            },
            output_path
        )

        return ocr_results


def main():
    """Main function to run OCR extraction"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract OCR text from images in rumor detection datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["AMG", "DGM4", "FineFake", "MMFakeBench", "MR2"],
        help="Datasets to process (default: all datasets)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Data splits to process (default: train val test)"
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory of datasets (default: data)"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for OCR results (default: data)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if output files exist"
    )

    args = parser.parse_args()

    # Initialize OCR extractor (GPU only)
    extractor = OCRExtractor()

    # Process all datasets and splits
    print("\n" + "="*60)
    print("OCR Extraction Pipeline (GPU-Accelerated)")
    print("="*60)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"Device: GPU")
    print("="*60)

    all_results = {}

    for dataset_name in args.datasets:
        all_results[dataset_name] = {}

        for split in args.splits:
            try:
                results = extractor.extract_from_dataset(
                    dataset_name=dataset_name,
                    split=split,
                    data_root=args.data_root,
                    output_dir=args.output_dir,
                    force=args.force
                )
                all_results[dataset_name][split] = len(results)
            except Exception as e:
                print(f"Error processing {dataset_name} {split}: {e}")
                all_results[dataset_name][split] = 0

        # For MR2, also extract evidence OCR
        if dataset_name == "MR2":
            print(f"\n{'='*60}")
            print(f"Processing MR2 Evidence Images")
            print(f"{'='*60}")

            for split in args.splits:
                try:
                    extractor.extract_mr2_evidence_ocr(
                        split=split,
                        data_root=args.data_root,
                        output_dir=args.output_dir,
                        force=args.force
                    )
                except Exception as e:
                    print(f"Error processing MR2 evidence {split}: {e}")

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for dataset_name, splits in all_results.items():
        print(f"\n{dataset_name}:")
        for split, count in splits.items():
            print(f"  {split}: {count} items")
    print("\n" + "="*60)
    print("All done!")


if __name__ == "__main__":
    main()