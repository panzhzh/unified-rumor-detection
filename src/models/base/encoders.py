"""
Base encoder modules for text and image processing.
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaModel, CLIPVisionModel


class TextEncoder(nn.Module):
    """
    Text encoder using XLM-RoBERTa.

    Args:
        model_name: HuggingFace model name (default: xlm-roberta-large)
        freeze: Whether to freeze encoder weights
    """

    def __init__(self, model_name: str = "xlm-roberta-large", freeze: bool = False):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            last_hidden_state: [batch_size, seq_len, hidden_size]
            pooled_output: [batch_size, hidden_size] (CLS token)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, outputs.last_hidden_state[:, 0]


class ImageEncoder(nn.Module):
    """
    Image encoder using CLIP vision model.

    Args:
        model_name: HuggingFace model name (default: openai/clip-vit-large-patch14)
        freeze_backbone: Whether to freeze CLIP backbone
        use_trainable_adapter: Whether to add trainable MLP adapter
        adapter_hidden_dim: Hidden dimension for adapter (if used)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        freeze_backbone: bool = True,
        use_trainable_adapter: bool = True,
        adapter_hidden_dim: int = 1024
    ):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(model_name)
        self.hidden_size = self.clip.config.hidden_size

        # Freeze CLIP backbone
        if freeze_backbone:
            for param in self.clip.parameters():
                param.requires_grad = False

        # Optional trainable adapter
        self.use_adapter = use_trainable_adapter
        if use_trainable_adapter:
            self.adapter = nn.Sequential(
                nn.Linear(self.hidden_size, adapter_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(adapter_hidden_dim, self.hidden_size)
            )

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: [batch_size, 3, 224, 224] or [batch_size, num_images, 3, 224, 224]

        Returns:
            last_hidden_state: [batch_size, num_patches, hidden_size] or
                              [batch_size, num_images, num_patches, hidden_size]
            pooled_output: [batch_size, hidden_size] or [batch_size, num_images, hidden_size]
        """
        original_shape = pixel_values.shape

        # Handle multiple images per batch
        if pixel_values.dim() == 5:
            batch_size, num_images = original_shape[:2]
            pixel_values = pixel_values.view(-1, *original_shape[2:])
        else:
            batch_size, num_images = original_shape[0], 1

        # Extract features
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.clip(pixel_values=pixel_values)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = outputs.pooler_output

        # Apply adapter if enabled
        if self.use_adapter:
            last_hidden_state = self.adapter(last_hidden_state)
            pooled_output = self.adapter(pooled_output)

        # Reshape back if multiple images
        if num_images > 1:
            num_patches = last_hidden_state.shape[1]
            last_hidden_state = last_hidden_state.view(batch_size, num_images, num_patches, self.hidden_size)
            pooled_output = pooled_output.view(batch_size, num_images, self.hidden_size)

        return last_hidden_state, pooled_output
