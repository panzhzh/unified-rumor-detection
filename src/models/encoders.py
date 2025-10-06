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
    Image encoder using CLIP vision model with partial unfreezing.

    Args:
        model_name: HuggingFace model name (default: openai/clip-vit-base-patch16)
        unfreeze_last_n_layers: Number of last layers to unfreeze (default: 3)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        unfreeze_last_n_layers: int = 3
    ):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(model_name)
        self.hidden_size = self.clip.config.hidden_size
        self.unfreeze_last_n_layers = unfreeze_last_n_layers

        # First freeze all parameters
        for param in self.clip.parameters():
            param.requires_grad = False

        # Then unfreeze last N layers if specified
        if unfreeze_last_n_layers > 0:
            num_layers = len(self.clip.vision_model.encoder.layers)
            if unfreeze_last_n_layers > num_layers:
                print(f"Warning: unfreeze_last_n_layers ({unfreeze_last_n_layers}) > total layers ({num_layers})")
                unfreeze_last_n_layers = num_layers

            print(f"Unfreezing last {unfreeze_last_n_layers} layers of CLIP (layers {num_layers - unfreeze_last_n_layers} to {num_layers - 1})")
            for layer in self.clip.vision_model.encoder.layers[-unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

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

        # Extract CLIP features
        outputs = self.clip(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        # Reshape back if multiple images
        if num_images > 1:
            num_patches = last_hidden_state.shape[1]
            last_hidden_state = last_hidden_state.view(batch_size, num_images, num_patches, self.hidden_size)
            pooled_output = pooled_output.view(batch_size, num_images, self.hidden_size)

        return last_hidden_state, pooled_output
