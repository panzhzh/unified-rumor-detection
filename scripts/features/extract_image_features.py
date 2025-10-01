#!/usr/bin/env python3
"""
Image Feature Extraction Script for Unified Rumor Detection Datasets

This script extracts image features from all datasets and saves them as .npy files.
Supports: AMG, DGM4, FineFake, MMFakeBench, MR2

Features:
- Multiple model support (ViT, CLIP)
- Animated image handling (multi-frame extraction + pooling)
- Extreme aspect ratio handling
- GPU acceleration
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from transformers import (
    ViTModel, ViTImageProcessor,
    CLIPModel, CLIPProcessor
)

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import UnifiedDataLoader


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    hf_name: str
    feature_dim: int


# Supported models
MODELS = {
    "vit_large": ModelConfig("vit_large", "google/vit-large-patch16-224", 1024),
    "vit_base": ModelConfig("vit_base", "google/vit-base-patch16-224", 768),
    "clip_large": ModelConfig("clip_large", "openai/clip-vit-large-patch14", 1024),
    "clip_base": ModelConfig("clip_base", "openai/clip-vit-base-patch32", 768)
}


class AnimatedImageProcessor:
    """Handle animated image processing with multi-frame pooling"""

    def __init__(self, max_frames: int = 8, pooling_strategy: str = "attention"):
        self.max_frames = max_frames
        self.pooling_strategy = pooling_strategy

    def is_animated(self, image_path: str) -> bool:
        """Check if image is animated"""
        try:
            with Image.open(image_path) as img:
                return getattr(img, 'is_animated', False) and img.n_frames > 1
        except Exception:
            return False

    def extract_frames(self, image_path: str) -> List[Image.Image]:
        """Extract frames from animated image"""
        frames = []
        try:
            with Image.open(image_path) as img:
                if not self.is_animated(image_path):
                    return [img.convert('RGB')]

                # Extract frames from animated image
                total_frames = img.n_frames
                step = max(1, total_frames // self.max_frames)

                for i in range(0, min(total_frames, self.max_frames * step), step):
                    img.seek(i)
                    frame = img.copy().convert('RGB')
                    frames.append(frame)

        except Exception as e:
            print(f"Error extracting frames from {image_path}: {e}")
            try:
                with Image.open(image_path) as img:
                    frames = [img.convert('RGB')]
            except Exception:
                frames = []

        return frames

    def pool_frame_features(self, frame_features: torch.Tensor) -> torch.Tensor:
        """Pool features from multiple frames"""
        if frame_features.shape[0] == 1:
            return frame_features.squeeze(0)

        if self.pooling_strategy == "mean":
            return frame_features.mean(dim=0)
        elif self.pooling_strategy == "max":
            return frame_features.max(dim=0)[0]
        elif self.pooling_strategy == "attention":
            # Simple attention pooling
            weights = F.softmax(frame_features.norm(dim=1), dim=0)
            return (frame_features * weights.unsqueeze(1)).sum(dim=0)
        else:
            return frame_features.mean(dim=0)


class ImageFeatureExtractor:
    """Extract features from images using pre-trained models"""

    def __init__(self, model_config: ModelConfig, device: str = "auto"):
        self.model_config = model_config
        self.device = self._get_device(device)
        self.model = None
        self.processor = None
        self.animated_processor = AnimatedImageProcessor(max_frames=8, pooling_strategy="attention")
        self._setup_transforms()
        self._load_model()

    def _get_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _setup_transforms(self):
        """Setup image transforms"""
        if "clip" in self.model_config.name:
            self.transform = T.Compose([
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        else:
            self.transform = T.Compose([
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def _preprocess_extreme_aspect_ratio_image(self, image: Image.Image) -> Image.Image:
        """Handle images with extreme aspect ratios (very wide or very tall)"""
        width, height = image.size
        aspect_ratio = width / height

        # Define thresholds for extreme aspect ratios
        max_aspect_ratio = 5.0
        min_aspect_ratio = 0.2

        if aspect_ratio > max_aspect_ratio or aspect_ratio < min_aspect_ratio:
            target_size = 224

            if aspect_ratio > 1:
                new_width = target_size
                new_height = int(target_size / aspect_ratio)
            else:
                new_height = target_size
                new_width = int(target_size * aspect_ratio)

            image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

            # Create square canvas and paste resized image in center
            canvas = Image.new('RGB', (target_size, target_size), color=(255, 255, 255))
            paste_x = (target_size - new_width) // 2
            paste_y = (target_size - new_height) // 2
            canvas.paste(image, (paste_x, paste_y))

            return canvas

        return image

    def _load_model(self):
        """Load the specified model and processor"""
        print(f"Loading model: {self.model_config.hf_name}")

        if "clip" in self.model_config.name:
            self.model = CLIPModel.from_pretrained(self.model_config.hf_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_config.hf_name)
        else:
            self.model = ViTModel.from_pretrained(self.model_config.hf_name).to(self.device)
            self.processor = ViTImageProcessor.from_pretrained(self.model_config.hf_name)

        self.model.eval()
        print(f"Model loaded on {self.device}")

    def extract_single_frame_features(self, frame: Image.Image) -> torch.Tensor:
        """Extract features from a single frame"""
        try:
            preprocessed_frame = self._preprocess_extreme_aspect_ratio_image(frame)
            image_tensor = self.transform(preprocessed_frame).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if "clip" in self.model_config.name:
                    vision_outputs = self.model.vision_model(pixel_values=image_tensor)
                    features = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs.last_hidden_state.mean(dim=1)
                else:
                    outputs = self.model(pixel_values=image_tensor)
                    features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)

            return features.squeeze(0).cpu()

        except Exception as e:
            print(f"Error extracting features: {e}")
            return torch.zeros(self.model_config.feature_dim)

    def extract_image_features(self, image_path: str) -> torch.Tensor:
        """Extract features from image (handling animated images)"""
        if not os.path.exists(image_path):
            return torch.zeros(self.model_config.feature_dim)

        # Extract frames
        frames = self.animated_processor.extract_frames(image_path)

        if not frames:
            return torch.zeros(self.model_config.feature_dim)

        # Extract features from each frame
        frame_features = []
        for frame in frames:
            features = self.extract_single_frame_features(frame)
            frame_features.append(features)

        # Stack and pool
        frame_features = torch.stack(frame_features)
        pooled_features = self.animated_processor.pool_frame_features(frame_features)

        return pooled_features


class FeatureExtractor:
    """Main class for extracting and caching image features"""

    def __init__(self, model_name: str, data_root: str = "data", cache_root: str = "data/features"):
        self.model_config = MODELS[model_name]
        self.data_root = Path(data_root)
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.extractor = None

    def _init_extractor(self):
        """Lazy initialization of feature extractor"""
        if self.extractor is None:
            self.extractor = ImageFeatureExtractor(self.model_config)

    def extract_dataset_features(
        self,
        dataset_name: str,
        split: str,
        force: bool = False
    ) -> Dict[str, np.ndarray]:
        """Extract features for a dataset split"""

        # Check if output already exists
        output_path = self.cache_root / dataset_name / f"{self.model_config.name}_{split}.npy"
        if output_path.exists() and not force:
            print(f"Features already exist: {output_path}")
            print("Use --force to re-extract. Skipping...")
            return np.load(output_path, allow_pickle=True).item()

        # Load dataset
        print(f"\nLoading {dataset_name} {split} dataset...")
        loader = UnifiedDataLoader(data_root=str(self.data_root))

        try:
            data = loader.load_dataset(dataset_name, split)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return {}

        # Initialize extractor
        self._init_extractor()

        # Extract features
        feature_dict = {}
        items_with_images = [item for item in data if item.has_image()]

        print(f"Found {len(data)} total items, {len(items_with_images)} with images")
        print(f"Extracting {self.model_config.name} features...")

        for item in tqdm(items_with_images, desc=f"{dataset_name} {split}"):
            features = self.extractor.extract_image_features(item.image_path)
            # Use item.id as key for consistency
            feature_dict[item.id] = features.numpy()

        # Save features
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, feature_dict)

        print(f"\n{'='*60}")
        print(f"Feature Extraction Complete: {dataset_name} {split}")
        print(f"{'='*60}")
        print(f"Total items with features: {len(feature_dict)}")
        print(f"Feature dimension: {self.model_config.feature_dim}")
        print(f"Output saved to: {output_path}")
        print(f"{'='*60}\n")

        return feature_dict


def main():
    """Main function to run feature extraction"""
    parser = argparse.ArgumentParser(
        description="Extract image features from rumor detection datasets"
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="clip_large",
        help="Model to use for feature extraction (default: clip_large)"
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
        "--cache-root",
        default="data/features",
        help="Root directory for feature cache (default: data/features)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if cache exists"
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "="*60)
    print("Image Feature Extraction Pipeline")
    print("="*60)
    print(f"Model: {MODELS[args.model].hf_name}")
    print(f"Feature dimension: {MODELS[args.model].feature_dim}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("="*60)

    # Initialize extractor
    extractor = FeatureExtractor(
        model_name=args.model,
        data_root=args.data_root,
        cache_root=args.cache_root
    )

    # Process all datasets and splits
    all_results = {}

    for dataset_name in args.datasets:
        all_results[dataset_name] = {}

        for split in args.splits:
            try:
                features = extractor.extract_dataset_features(
                    dataset_name=dataset_name,
                    split=split,
                    force=args.force
                )
                all_results[dataset_name][split] = len(features)
            except Exception as e:
                print(f"Error processing {dataset_name} {split}: {e}")
                all_results[dataset_name][split] = 0

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for dataset_name, splits in all_results.items():
        print(f"\n{dataset_name}:")
        for split, count in splits.items():
            print(f"  {split}: {count} features")
    print("\n" + "="*60)
    print("All done!")


if __name__ == "__main__":
    main()
