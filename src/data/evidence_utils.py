"""Evidence filtering and text formatting utilities"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from PIL import Image


def select_top_evidence(
    caption: str,
    evidence_list: List[Dict],
    text_encoder,
    tokenizer,
    device,
    max_evidence: int = 5
) -> List[Dict]:
    """
    Select top-k evidence based on similarity with caption.

    Args:
        caption: Main claim text
        evidence_list: List of evidence items with 'text' field
        text_encoder: XLM-RoBERTa model for encoding
        tokenizer: XLM-RoBERTa tokenizer
        device: torch device
        max_evidence: Maximum number of evidence to select (default: 5)

    Returns:
        List of top-k evidence items sorted by similarity
    """
    if not evidence_list or len(evidence_list) <= max_evidence:
        return evidence_list[:max_evidence] if evidence_list else []

    # Encode caption
    caption_tokens = tokenizer(
        caption,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        caption_output = text_encoder(
            caption_tokens['input_ids'],
            caption_tokens['attention_mask']
        )
        caption_vec = caption_output[1]  # [CLS] pooled output
        caption_vec = F.normalize(caption_vec, p=2, dim=1)

    # Encode all evidence texts
    evidence_texts = [evi.get('text', '') for evi in evidence_list]
    evidence_tokens = tokenizer(
        evidence_texts,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        evidence_output = text_encoder(
            evidence_tokens['input_ids'],
            evidence_tokens['attention_mask']
        )
        evidence_vecs = evidence_output[1]  # [batch, hidden]
        evidence_vecs = F.normalize(evidence_vecs, p=2, dim=1)

    # Calculate cosine similarity
    similarities = (caption_vec @ evidence_vecs.T).squeeze(0)  # [num_evidence]

    # Select top-k
    top_indices = similarities.argsort(descending=True)[:max_evidence].cpu().tolist()

    return [evidence_list[i] for i in top_indices]


def format_evidence_text(
    caption: str,
    ocr: str,
    evidence_list: List[Dict],
    tokenizer,
    cap_budget: int = 64,
    ocr_budget: int = 192,
    evi_budget: int = 256,
    evi_per_item: int = 50
) -> str:
    """
    Format text as: [CAP] ... [OCR] ... [EVI] ...

    Args:
        caption: Main claim text
        ocr: OCR text
        evidence_list: List of selected evidence items
        tokenizer: Tokenizer for counting tokens
        cap_budget: Max tokens for caption
        ocr_budget: Max tokens for OCR
        evi_budget: Max tokens for all evidence
        evi_per_item: Max tokens per evidence item

    Returns:
        Formatted text string
    """
    parts = []

    # [CAP] section
    cap_tokens = tokenizer.encode(caption, add_special_tokens=False)
    if len(cap_tokens) > cap_budget:
        cap_tokens = cap_tokens[:cap_budget]
        caption_truncated = tokenizer.decode(cap_tokens)
    else:
        caption_truncated = caption
    parts.append(f"[CAP] {caption_truncated}")

    # [OCR] section
    if ocr:
        ocr_tokens = tokenizer.encode(ocr, add_special_tokens=False)
        if len(ocr_tokens) > ocr_budget:
            ocr_tokens = ocr_tokens[:ocr_budget]
            ocr_truncated = tokenizer.decode(ocr_tokens)
        else:
            ocr_truncated = ocr
        parts.append(f"[OCR] {ocr_truncated}")

    # [EVI] section
    if evidence_list:
        evi_texts = []
        remaining_budget = evi_budget

        for evi in evidence_list:
            evi_text = evi.get('text', '')
            if not evi_text:
                continue

            evi_tokens = tokenizer.encode(evi_text, add_special_tokens=False)
            max_tokens = min(evi_per_item, remaining_budget)

            if len(evi_tokens) > max_tokens:
                evi_tokens = evi_tokens[:max_tokens]
                evi_text_truncated = tokenizer.decode(evi_tokens)
            else:
                evi_text_truncated = evi_text

            evi_texts.append(evi_text_truncated)
            remaining_budget -= len(evi_tokens)

            if remaining_budget <= 0:
                break

        if evi_texts:
            parts.append(f"[EVI] {' '.join(evi_texts)}")

    return " ".join(parts)


def load_evidence_images(
    evidence_list: List[Dict],
    image_transform,
    max_images: int = 5
) -> torch.Tensor:
    """
    Load and transform evidence images.

    Args:
        evidence_list: List of evidence items with 'image_path' field
        image_transform: torchvision transform
        max_images: Maximum number of images to load

    Returns:
        Tensor of shape [num_images, 3, 224, 224]
    """
    images = []

    for evi in evidence_list[:max_images]:
        img_path = evi.get('image_path')
        if img_path:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = image_transform(img)
                images.append(img_tensor)
            except:
                continue

    if images:
        return torch.stack(images)
    else:
        return torch.zeros(0, 3, 224, 224)
