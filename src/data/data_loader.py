"""
Unified Data Loader for Multiple Rumor Detection Datasets
Supports: AMG, DGM4, FineFake, MR2
"""

import json
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class UnifiedDataItem:
    """Unified data format for all datasets

    Unified Label Convention:
    - 0: Real/True/Non-rumor
    - 1: Fake/False/Rumor
    - 2: Unverified (only for MR2 dataset)

    Unified Field Structure:
    Core fields (always present):
    - id: unique identifier
    - label: unified label (0/1/2)
    - dataset_name: source dataset
    - split: train/val/test

    Content fields (may be None if not available):
    - text: main text content (news body, caption, tweet, etc.)
    - image_path: path to image file
    - ocr: OCR text extracted from image
    - entities: named entities or keywords
    - timestamp: publication time or creation time
    - author: author or source information
    - language: language code (en, zh, etc.)

    Extended fields (dataset-specific, stored in metadata):
    - original_label: original label before conversion
    - fine_grained_label: detailed fake type classification
    - manipulation_type: type of image manipulation (for DGM4)
    - evidence_paths: paths to evidence/verification data (for MR2)
    - topic: news topic or category
    - platform: source platform (Twitter, Facebook, etc.)
    - etc.

    Original labels and dataset-specific fields are preserved in metadata.
    """
    # Core fields
    id: str
    label: int
    dataset_name: str
    split: str

    # Content fields (may be None)
    text: Optional[str] = None
    image_path: Optional[str] = None
    ocr: Optional[str] = None
    entities: Optional[List[str]] = None
    timestamp: Optional[float] = None
    author: Optional[str] = None
    language: Optional[str] = None

    # Extended fields
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def has_image(self) -> bool:
        """Check if item has image"""
        return self.image_path is not None

    def has_text(self) -> bool:
        """Check if item has text"""
        return self.text is not None and len(self.text.strip()) > 0

    def has_ocr(self) -> bool:
        """Check if item has OCR text"""
        return self.ocr is not None and len(self.ocr.strip()) > 0

    def is_multimodal(self) -> bool:
        """Check if item is multimodal (has both text and image)"""
        return self.has_text() and self.has_image()

    def get_all_text(self) -> str:
        """Get concatenated text from all text sources"""
        texts = []
        if self.has_text():
            texts.append(self.text)
        if self.has_ocr():
            texts.append(f"[OCR]: {self.ocr}")
        return " ".join(texts)


class BaseDatasetLoader:
    """Base class for dataset loaders"""

    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)

    def load(self, split: str) -> List[UnifiedDataItem]:
        """Load dataset for a specific split"""
        raise NotImplementedError

    def get_stats(self, split: str) -> Dict:
        """Get dataset statistics"""
        data = self.load(split)
        total = len(data)
        fake_count = sum(1 for item in data if item.label == 1)
        real_count = total - fake_count
        return {
            "total": total,
            "fake": fake_count,
            "real": real_count,
            "fake_ratio": fake_count / total if total > 0 else 0
        }


class AMGLoader(BaseDatasetLoader):
    """AMG (Attributing Multi-granularity Multimodal) Dataset Loader

    Original labels: 0=Real, 1-5=Fake (fine-grained)
    Unified: 0=Real, 1=Fake

    AMG is a multimodal dataset with images in AMG_MEDIA folder
    """

    def __init__(self, data_root: str = "data"):
        super().__init__(data_root)
        self.dataset_dir = self.data_root / "AMG"
        self.media_dir = self.dataset_dir / "AMG_MEDIA"

    def load(self, split: str) -> List[UnifiedDataItem]:
        json_path = self.dataset_dir / f"{split}.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        unified_data = []
        for item in raw_data:
            original_label = int(item['label'])
            # AMG: 0=real, 1-5=fake -> Unified: 0=real, 1=fake
            unified_label = 0 if original_label == 0 else 1

            # Construct image path: AMG_MEDIA/{split}/{id}.jpg
            item_id = str(item['Id'])  # Original ID is string like "1", "2", "3"
            image_path = self.media_dir / split / f"{item_id}.jpg"
            image_path_str = str(image_path) if image_path.exists() else None

            unified_data.append(UnifiedDataItem(
                id=item_id,  # Keep original ID
                label=unified_label,
                dataset_name="AMG",
                split=split,
                text=item['content'],
                image_path=image_path_str,  # AMG has images
                timestamp=item.get('create_time'),
                metadata={
                    "original_label": original_label,
                    "fine_grained_label": original_label  # 0-5 classification
                }
            ))
        return unified_data


class DGM4Loader(BaseDatasetLoader):
    """DGM4 (DeepFake Generation Methods) Dataset Loader"""

    def __init__(self, data_root: str = "data"):
        super().__init__(data_root)
        self.dataset_dir = self.data_root / "DGM4"

    def load(self, split: str) -> List[UnifiedDataItem]:
        json_path = self.dataset_dir / "metadata" / f"{split}.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        unified_data = []
        for idx, item in enumerate(raw_data):
            # DGM4: original images are real (0), manipulated are fake (1)
            label = 0 if item['fake_cls'] == 'orig' else 1

            # Use original ID (IDs are unique across all splits)
            item_id = str(item['id'])

            # Image path in JSON already includes 'DGM4/' prefix, so strip it
            # Path format: DGM4/manipulation/infoswap/995762-043201-infoswap.jpg
            # Actual path: data/DGM4/manipulation/infoswap/infoswap/995762-043201-infoswap.jpg
            # The directory structure has duplicate subfolder names (e.g., infoswap/infoswap/)
            image_rel_path = item['image']
            if image_rel_path.startswith('DGM4/'):
                image_rel_path = image_rel_path[5:]  # Remove 'DGM4/' prefix

            # Insert duplicate subfolder name: manipulation/infoswap/XXX.jpg -> manipulation/infoswap/infoswap/XXX.jpg
            from pathlib import Path as P
            parts = P(image_rel_path).parts
            if len(parts) >= 3:  # e.g., ('manipulation', 'infoswap', 'file.jpg')
                # Insert duplicate of second-level folder
                parts = parts[:2] + (parts[1],) + parts[2:]
                image_rel_path = str(P(*parts))

            unified_data.append(UnifiedDataItem(
                id=item_id,
                label=label,
                dataset_name="DGM4",
                split=split,
                text=item['text'],
                image_path=str(self.dataset_dir / image_rel_path),
                metadata={
                    "original_id": item['id'],  # Preserve original ID
                    "manipulation_type": item['fake_cls'],  # original, face_swap, etc.
                    "fake_image_box": item.get('fake_image_box'),
                    "fake_text_pos": item.get('fake_text_pos'),
                    "mtcnn_boxes": item.get('mtcnn_boxes')
                }
            ))
        return unified_data


class FineFakeLoader(BaseDatasetLoader):
    """FineFake Dataset Loader

    Original labels: 0=Fake, 1=Real (reversed!)
    Unified: 0=Real, 1=Fake

    FineFake is stored as JSON files (train.json, val.json, test.json)
    """

    def __init__(self, data_root: str = "data"):
        super().__init__(data_root)
        self.dataset_dir = self.data_root / "FineFake"

    def load(self, split: str) -> List[UnifiedDataItem]:
        import json

        # Load from pre-split JSON files
        json_path = self.dataset_dir / f"{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"JSON file not found: {json_path}. "
                f"Please run scripts/data_analysis/export_finefake_to_json.py first."
            )

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        unified_data = []
        for item in raw_data:
            # Use original ID from JSON
            item_id = item['id']

            original_label = item['label']
            # FineFake: 0=fake, 1=real -> Unified: 0=real, 1=fake (REVERSE!)
            unified_label = 1 - original_label

            # Extract entities
            entities = None
            if item.get('entity_id') is not None:
                entity_val = item['entity_id']
                if isinstance(entity_val, list):
                    entities = [str(e) for e in entity_val if e is not None]
                else:
                    entities = [str(entity_val)]
                entities = entities if entities else None

            # Image path
            image_path = None
            if item.get('image_path') and str(item['image_path']).strip():
                image_path = str(self.dataset_dir / item['image_path'])

            # Extract text
            text = item.get('text', '')
            if text is None:
                text = ''

            # Extract timestamp
            timestamp = item.get('date')

            # Extract author (can be list or string)
            author = None
            if item.get('author') is not None:
                author_val = item['author']
                if isinstance(author_val, list):
                    author = ', '.join(str(a) for a in author_val if a is not None)
                else:
                    author = str(author_val)

            unified_data.append(UnifiedDataItem(
                id=item_id,
                label=unified_label,
                dataset_name="FineFake",
                split=split,
                text=text,
                image_path=image_path,
                entities=entities,
                timestamp=timestamp,
                author=author,
                metadata={
                    "original_label": original_label,
                    "fine_grained_label": item.get('fine_grained_label'),
                    "topic": item.get('topic'),
                    "platform": item.get('platform'),
                    "description": item.get('description'),
                    "relation": item.get('relation')
                }
            ))
        return unified_data


class MR2Loader(BaseDatasetLoader):
    """MR2 (Multimodal Rumor) Dataset Loader

    Original labels: 0=Non-rumor, 1=Rumor, 2=Unverified
    Unified: 0=Real (non-rumor), 1=Fake (rumor), 2=Unverified (kept as-is)
    """

    def __init__(self, data_root: str = "data"):
        super().__init__(data_root)
        self.dataset_dir = self.data_root / "MR2"

    def load(self, split: str) -> List[UnifiedDataItem]:
        json_path = self.dataset_dir / f"dataset_items_{split}.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        unified_data = []
        for item_id, item in raw_data.items():
            original_label = int(item['label'])
            # MR2: 0=non-rumor, 1=rumor, 2=unverified
            # Keep as-is since it matches our convention (0=real, 1=fake)
            # Note: label=2 (unverified) is kept for compatibility

            # Use original ID (already unique string keys like "0", "1", "2")
            unified_data.append(UnifiedDataItem(
                id=item_id,
                label=original_label,
                dataset_name="MR2",
                split=split,
                text=item['caption'],
                image_path=str(self.dataset_dir / item['image_path']),
                ocr=item.get('ocr'),
                language=item.get('language'),
                metadata={
                    "direct_search_path": item.get('direct_path'),
                    "inverse_search_path": item.get('inv_path')
                }
            ))
        return unified_data


class MMFakeBenchLoader(BaseDatasetLoader):
    """MMFakeBench (Multimodal Fake Benchmark) Dataset Loader

    Original labels: 'True'=Real, 'Fake'=Fake
    Unified: 0=Real, 1=Fake

    Note: MMFakeBench only has val and test splits, no train split
    """

    def __init__(self, data_root: str = "data"):
        super().__init__(data_root)
        self.dataset_dir = self.data_root / "MMFakeBench"

    def load(self, split: str) -> List[UnifiedDataItem]:
        # MMFakeBench only has val and test
        if split not in ["val", "test"]:
            raise ValueError(f"MMFakeBench only has 'val' and 'test' splits, got: {split}")

        json_path = self.dataset_dir / f"MMFakeBench_{split}.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        unified_data = []
        for idx, item in enumerate(raw_data):
            # MMFakeBench: 'True'=real, 'Fake'=fake -> Unified: 0=real, 1=fake
            original_label = item['gt_answers']
            unified_label = 0 if original_label == 'True' else 1

            # Use index as ID (MMFakeBench has no original IDs)
            item_id = str(idx)

            # Image path handling: remove leading '/' and prepend split directory
            image_rel_path = item['image_path']
            if image_rel_path.startswith('/'):
                image_rel_path = image_rel_path[1:]  # Remove leading '/'

            # Path is like: real/bbc_val_50/BBC_val_0.png
            # Full path: data/MMFakeBench/MMFakeBench_val/real/bbc_val_50/BBC_val_0.png
            image_path = self.dataset_dir / f"MMFakeBench_{split}" / image_rel_path

            unified_data.append(UnifiedDataItem(
                id=item_id,
                label=unified_label,
                dataset_name="MMFakeBench",
                split=split,
                text=item['text'],
                image_path=str(image_path) if image_path.exists() else None,
                metadata={
                    "original_label": original_label,
                    "text_source": item.get('text_source'),
                    "image_source": item.get('image_source'),
                    "fake_cls": item.get('fake_cls')  # original or manipulation type
                }
            ))
        return unified_data


class UnifiedDataLoader:
    """Main unified data loader supporting multiple datasets

    Features:
    - Load single or multiple datasets
    - Filter by available fields (e.g., only multimodal data)
    - Get dataset statistics
    - Support cross-dataset training and testing
    """

    DATASET_LOADERS = {
        "AMG": AMGLoader,
        "DGM4": DGM4Loader,
        "FineFake": FineFakeLoader,
        "MMFakeBench": MMFakeBenchLoader,
        "MR2": MR2Loader
    }

    def __init__(self, data_root: str = "data"):
        self.data_root = data_root
        self.loaders = {
            name: loader_cls(data_root)
            for name, loader_cls in self.DATASET_LOADERS.items()
        }

    def load_dataset(self, dataset_name: str, split: str,
                     filter_fn: Optional[callable] = None) -> List[UnifiedDataItem]:
        """Load a single dataset with optional filtering

        Args:
            dataset_name: Name of the dataset
            split: train/val/test
            filter_fn: Optional function to filter items (e.g., lambda x: x.has_image())

        Returns:
            List of UnifiedDataItem
        """
        if dataset_name not in self.loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.loaders.keys())}")

        data = self.loaders[dataset_name].load(split)

        if filter_fn:
            data = [item for item in data if filter_fn(item)]

        return data

    def load_multiple(self, dataset_names: List[str], split: str,
                     filter_fn: Optional[callable] = None) -> List[UnifiedDataItem]:
        """Load and merge multiple datasets

        Args:
            dataset_names: List of dataset names to load
            split: train/val/test
            filter_fn: Optional function to filter items

        Returns:
            List of UnifiedDataItem from all specified datasets
        """
        all_data = []
        for dataset_name in dataset_names:
            data = self.load_dataset(dataset_name, split, filter_fn)
            all_data.extend(data)
        return all_data

    def load_all(self, split: str, filter_fn: Optional[callable] = None) -> List[UnifiedDataItem]:
        """Load all available datasets

        Args:
            split: train/val/test
            filter_fn: Optional function to filter items

        Returns:
            List of UnifiedDataItem from all datasets
        """
        return self.load_multiple(list(self.loaders.keys()), split, filter_fn)

    def load_by_modality(self, dataset_names: Optional[List[str]] = None,
                         split: str = "train",
                         require_text: bool = True,
                         require_image: bool = False,
                         require_ocr: bool = False) -> List[UnifiedDataItem]:
        """Load datasets filtered by required modalities

        Args:
            dataset_names: List of datasets to load (None = all)
            split: train/val/test
            require_text: Require text field
            require_image: Require image field
            require_ocr: Require OCR field

        Returns:
            Filtered list of UnifiedDataItem
        """
        if dataset_names is None:
            dataset_names = list(self.loaders.keys())

        def filter_fn(item: UnifiedDataItem) -> bool:
            if require_text and not item.has_text():
                return False
            if require_image and not item.has_image():
                return False
            if require_ocr and not item.has_ocr():
                return False
            return True

        return self.load_multiple(dataset_names, split, filter_fn)

    def get_stats(self, dataset_names: Optional[List[str]] = None, split: str = "train") -> Dict:
        """Get statistics for specified datasets"""
        if dataset_names is None:
            dataset_names = list(self.loaders.keys())

        stats = {}
        for name in dataset_names:
            if name in self.loaders:
                stats[name] = self.loaders[name].get_stats(split)

        # Calculate total stats
        total_items = sum(s['total'] for s in stats.values())
        total_fake = sum(s['fake'] for s in stats.values())
        total_real = sum(s['real'] for s in stats.values())

        stats['total'] = {
            "total": total_items,
            "fake": total_fake,
            "real": total_real,
            "fake_ratio": total_fake / total_items if total_items > 0 else 0
        }

        return stats


def main():
    """Example usage demonstrating various loading modes"""
    loader = UnifiedDataLoader(data_root="data")

    print("=" * 60)
    print("Unified Rumor Detection Data Loader")
    print("=" * 60)

    # 1. Load single dataset
    print("\n[1] Load single dataset (AMG train):")
    amg_train = loader.load_dataset("AMG", "train")
    print(f"   Total items: {len(amg_train)}")
    if amg_train:
        sample = amg_train[0]
        print(f"   Sample ID: {sample.id}")
        print(f"   Has text: {sample.has_text()}, Has image: {sample.has_image()}")
        print(f"   Label: {sample.label}")

    # 2. Load multiple datasets
    print("\n[2] Load multiple datasets (AMG + DGM4 train):")
    multi_train = loader.load_multiple(["AMG", "DGM4"], "train")
    print(f"   Total items: {len(multi_train)}")
    print(f"   Datasets: {set(item.dataset_name for item in multi_train)}")

    # 3. Load only multimodal data (text + image)
    print("\n[3] Load only multimodal data (require text + image):")
    multimodal_data = loader.load_by_modality(
        split="train",
        require_text=True,
        require_image=True
    )
    print(f"   Total multimodal items: {len(multimodal_data)}")
    print(f"   Datasets: {set(item.dataset_name for item in multimodal_data)}")

    # 4. Load data with OCR
    print("\n[4] Load data with OCR (MR2 only):")
    ocr_data = loader.load_by_modality(
        dataset_names=["MR2"],
        split="train",
        require_ocr=True
    )
    print(f"   Total items with OCR: {len(ocr_data)}")

    # 5. Custom filtering - only fake news
    print("\n[5] Load only fake news (label=1):")
    fake_data = loader.load_multiple(
        ["AMG", "DGM4"],
        "train",
        filter_fn=lambda x: x.label == 1
    )
    print(f"   Total fake news: {len(fake_data)}")

    # 6. Get dataset statistics
    print("\n[6] Dataset statistics:")
    stats = loader.get_stats(split="train")
    for name, stat in stats.items():
        if name == "total":
            print(f"\n   {name.upper()}:")
        else:
            print(f"\n   {name}:")
        for key, value in stat.items():
            if key == "fake_ratio":
                print(f"      {key}: {value:.2%}")
            else:
                print(f"      {key}: {value}")

    # 7. Field availability check
    print("\n[7] Field availability by dataset:")
    for dataset_name in loader.loaders.keys():
        try:
            data = loader.load_dataset(dataset_name, "train")
            if data:
                sample = data[0]
                print(f"\n   {dataset_name}:")
                print(f"      text: {sample.has_text()}")
                print(f"      image: {sample.has_image()}")
                print(f"      ocr: {sample.has_ocr()}")
                print(f"      entities: {sample.entities is not None}")
                print(f"      timestamp: {sample.timestamp is not None}")
                print(f"      language: {sample.language is not None}")
        except Exception as e:
            print(f"\n   {dataset_name}: Error - {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()