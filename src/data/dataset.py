"""PyTorch Dataset wrapper for unified rumor detection data"""

import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any
from PIL import Image
import torchvision.transforms as transforms

from .data_loader import UnifiedDataItem


class RumorDetectionDataset(Dataset):
    """PyTorch Dataset for rumor detection

    Supports:
    - Text-only models
    - Image-only models
    - Multimodal models
    - Custom preprocessing
    """

    def __init__(
        self,
        data: List[UnifiedDataItem],
        text_tokenizer=None,
        image_transform=None,
        max_text_length: int = 512,
        use_ocr: bool = True,
        return_metadata: bool = False
    ):
        """
        Args:
            data: List of UnifiedDataItem
            text_tokenizer: Text tokenizer (e.g., from transformers)
            image_transform: Image transform (e.g., torchvision transforms)
            max_text_length: Maximum text length for tokenization
            use_ocr: Whether to include OCR text
            return_metadata: Whether to return metadata in output
        """
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.image_transform = image_transform or self._default_image_transform()
        self.max_text_length = max_text_length
        self.use_ocr = use_ocr
        self.return_metadata = return_metadata

    def _default_image_transform(self):
        """Default image preprocessing"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        output = {
            'id': item.id,
            'label': item.label,
            'dataset_name': item.dataset_name
        }

        # Process text
        if item.has_text():
            text = item.get_all_text() if self.use_ocr else item.text

            if self.text_tokenizer:
                encoded = self.text_tokenizer(
                    text,
                    max_length=self.max_text_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                output['input_ids'] = encoded['input_ids'].squeeze(0)
                output['attention_mask'] = encoded['attention_mask'].squeeze(0)
            else:
                output['text'] = text
        else:
            output['text'] = ""

        # Process image
        if item.has_image():
            try:
                image = Image.open(item.image_path).convert('RGB')
                output['image'] = self.image_transform(image)
                output['has_image'] = True
            except Exception as e:
                # Fallback: create zero tensor if image loading fails
                output['image'] = torch.zeros(3, 224, 224)
                output['has_image'] = False
        else:
            output['image'] = torch.zeros(3, 224, 224)
            output['has_image'] = False

        # Add metadata if requested
        if self.return_metadata:
            output['metadata'] = item.metadata

        return output


class TextOnlyDataset(Dataset):
    """Dataset for text-only models"""

    def __init__(self, data: List[UnifiedDataItem], text_tokenizer, max_length: int = 512):
        self.data = [item for item in data if item.has_text()]
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        encoded = self.text_tokenizer(
            item.text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': item.label,
            'id': item.id
        }


class ImageOnlyDataset(Dataset):
    """Dataset for image-only models"""

    def __init__(self, data: List[UnifiedDataItem], image_transform=None):
        self.data = [item for item in data if item.has_image()]
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        image = Image.open(item.image_path).convert('RGB')
        return {
            'image': self.image_transform(image),
            'label': item.label,
            'id': item.id
        }


class MultimodalDataset(Dataset):
    """Dataset for multimodal models (text + image)"""

    def __init__(
        self,
        data: List[UnifiedDataItem],
        text_tokenizer,
        image_transform=None,
        max_text_length: int = 512,
        use_ocr: bool = True
    ):
        self.data = [item for item in data if item.is_multimodal()]
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.use_ocr = use_ocr
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # Process text
        text = item.get_all_text() if self.use_ocr else item.text
        encoded = self.text_tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Process image
        image = Image.open(item.image_path).convert('RGB')

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'image': self.image_transform(image),
            'label': item.label,
            'id': item.id,
            'dataset_name': item.dataset_name
        }


def create_dataset(
    data: List[UnifiedDataItem],
    mode: str = 'multimodal',
    text_tokenizer=None,
    image_transform=None,
    **kwargs
) -> Dataset:
    """Factory function to create appropriate dataset

    Args:
        data: List of UnifiedDataItem
        mode: 'text', 'image', 'multimodal', or 'flexible'
        text_tokenizer: Text tokenizer
        image_transform: Image transform
        **kwargs: Additional arguments for dataset

    Returns:
        Dataset instance
    """
    if mode == 'text':
        return TextOnlyDataset(data, text_tokenizer, **kwargs)
    elif mode == 'image':
        return ImageOnlyDataset(data, image_transform, **kwargs)
    elif mode == 'multimodal':
        return MultimodalDataset(data, text_tokenizer, image_transform, **kwargs)
    elif mode == 'flexible':
        return RumorDetectionDataset(data, text_tokenizer, image_transform, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'text', 'image', 'multimodal', 'flexible'")