"""
Evidence Fusion Model for Multimodal Rumor Detection.

This model uses:
- Text encoder (XLM-RoBERTa)
- Image encoder (CLIP)
- Multi-layer cross-modal attention fusion
- Evidence image fusion with attention pooling
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer

from ...base import TextEncoder, ImageEncoder, DeepFusionLayer, EvidenceAttentionPooling


class EvidenceFusionModel(nn.Module):
    """
    Multimodal rumor detection model with evidence fusion.

    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        text_model_name: Text encoder model name
        image_model_name: Image encoder model name
        num_fusion_layers: Number of deep fusion layers
        use_evidence: Whether to use evidence images
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_classes: int = 2,
        text_model_name: str = "xlm-roberta-large",
        image_model_name: str = "openai/clip-vit-large-patch14",
        num_fusion_layers: int = 3,
        use_evidence: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_evidence = use_evidence
        self.hidden_dim = 1024  # Assuming large models

        # Encoders
        self.text_encoder = TextEncoder(text_model_name, freeze=False)
        self.image_encoder = ImageEncoder(
            image_model_name,
            freeze_backbone=True,
            use_trainable_adapter=True
        )

        # Multi-layer cross-modal fusion
        self.num_fusion_layers = num_fusion_layers
        self.fusion_layers = nn.ModuleList([
            DeepFusionLayer(
                dim=self.hidden_dim,
                num_heads=8,
                dropout=dropout
            ) for _ in range(num_fusion_layers)
        ])

        # Evidence processing (if enabled)
        if use_evidence:
            # Tokenizer for caption encoding
            self.caption_tokenizer = XLMRobertaTokenizer.from_pretrained(text_model_name)
            self.caption_max_length = 128

            # Evidence attention pooling
            self.evidence_attention = EvidenceAttentionPooling(
                dim=self.hidden_dim,
                num_heads=8,
                dropout=dropout
            )

            # Gate mechanism for evidence contribution
            self.evidence_gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )

            # Pooling layers
            self.text_pooler = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.image_pooler = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.evidence_pooler = nn.Linear(self.hidden_dim, self.hidden_dim)

            # Classifier with evidence
            classifier_input_dim = self.hidden_dim * 3
        else:
            # Pooling layers (no evidence)
            self.text_pooler = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.image_pooler = nn.Linear(self.hidden_dim, self.hidden_dim)

            # Classifier without evidence
            classifier_input_dim = self.hidden_dim * 2

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        caption=None,
        num_evidence_images=None,
        return_attention_weights=False
    ):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            pixel_values: [batch_size, 3, 224, 224] or [batch_size, num_images, 3, 224, 224]
            caption: Optional list of caption strings for evidence querying
            num_evidence_images: Optional list of evidence counts per sample
            return_attention_weights: Whether to return attention weights

        Returns:
            logits: [batch_size, num_classes]
            attention_info: Optional dict with attention weights (if return_attention_weights=True)
        """
        # Encode text - use full sequence
        text_hidden, text_cls = self.text_encoder(input_ids, attention_mask)
        # text_hidden: [batch_size, seq_len, hidden_dim]

        # Encode images
        if pixel_values.dim() == 4:
            # Single image per sample
            image_hidden, image_pooled = self.image_encoder(pixel_values)
            # image_hidden: [batch_size, num_patches, hidden_dim]
            main_image_features = image_hidden
            evidence_image_features = None
        else:
            # Multiple images: first is main, rest are evidence
            image_hidden, image_pooled = self.image_encoder(pixel_values)
            # image_hidden: [batch_size, num_images, num_patches, hidden_dim]
            main_image_features = image_hidden[:, 0, :, :]  # [batch_size, num_patches, hidden_dim]

            if self.use_evidence and image_hidden.shape[1] > 1:
                # Extract evidence features: [batch_size, num_evidence, num_patches, hidden_dim]
                evidence_features = image_hidden[:, 1:, :, :]
                # Use CLS token: [batch_size, num_evidence, hidden_dim]
                evidence_cls = evidence_features[:, :, 0, :]

                # Process evidence with attention
                if return_attention_weights:
                    evidence_image_features, attention_info = self._process_evidence(
                        evidence_cls,
                        caption,
                        num_evidence_images,
                        return_attention_weights
                    )
                else:
                    evidence_image_features = self._process_evidence(
                        evidence_cls,
                        caption,
                        num_evidence_images,
                        return_attention_weights
                    )
            else:
                evidence_image_features = None

        # Multi-layer cross-modal fusion (token-level)
        for fusion_layer in self.fusion_layers:
            text_hidden, main_image_features = fusion_layer(text_hidden, main_image_features)

            if evidence_image_features is not None:
                # Also fuse with evidence
                text_hidden, evidence_image_features = fusion_layer(text_hidden, evidence_image_features)
                main_image_features, evidence_image_features = fusion_layer(main_image_features, evidence_image_features)

        # Pool to single representations
        text_pooled = text_hidden[:, 0, :]  # CLS token
        main_pooled = main_image_features[:, 0, :]  # CLS token

        # Apply pooling layers
        text_pooled = torch.tanh(self.text_pooler(text_pooled))
        main_pooled = torch.tanh(self.image_pooler(main_pooled))

        # Fuse features
        if self.use_evidence and evidence_image_features is not None:
            evidence_pooled = evidence_image_features[:, 0, :]
            evidence_pooled = torch.tanh(self.evidence_pooler(evidence_pooled))
            fused_features = torch.cat([text_pooled, main_pooled, evidence_pooled], dim=1)
        elif self.use_evidence:
            # No evidence images, use zero padding
            evidence_pooled = torch.zeros(text_pooled.shape[0], self.hidden_dim, device=text_pooled.device)
            evidence_pooled = torch.tanh(self.evidence_pooler(evidence_pooled))
            fused_features = torch.cat([text_pooled, main_pooled, evidence_pooled], dim=1)
        else:
            # Evidence disabled
            fused_features = torch.cat([text_pooled, main_pooled], dim=1)

        # Classify
        logits = self.classifier(fused_features)

        if return_attention_weights:
            return logits, attention_info
        else:
            return logits

    def _process_evidence(self, evidence_features, caption, num_evidence_images, return_attention_weights=False):
        """
        Process evidence images with attention pooling.

        Args:
            evidence_features: [batch_size, num_evidence, hidden_dim]
            caption: List of caption strings
            num_evidence_images: List of evidence counts
            return_attention_weights: Whether to return attention weights

        Returns:
            evidence_patch_features: [batch_size, num_patches, hidden_dim]
            attention_info: Optional dict (if return_attention_weights=True)
        """
        batch_size = evidence_features.shape[0]

        # Encode caption as query
        caption_query = self._encode_caption(caption)  # [batch_size, hidden_dim]

        # Handle None caption (use zero vector as fallback)
        if caption_query is None:
            caption_query = torch.zeros(batch_size, self.hidden_dim, device=evidence_features.device)

        # Attention pooling
        pooled_evidence, attention_weights = self.evidence_attention(
            caption_query,
            evidence_features
        )

        # Gate mechanism
        gate_input = torch.cat([
            caption_query,
            evidence_features.mean(dim=1),  # Average evidence
            pooled_evidence
        ], dim=1)

        gate_weight = self.evidence_gate(gate_input)  # [batch_size, 1]

        # Mask samples with no evidence
        if num_evidence_images is not None:
            has_evidence = torch.tensor(
                [n > 0 for n in num_evidence_images],
                device=evidence_features.device,
                dtype=torch.float32
            ).unsqueeze(1)
            gate_weight = gate_weight * has_evidence

        # Create patch-level features (expand pooled evidence to patch format)
        # For simplicity, we replicate the pooled feature across patches
        num_patches = 257  # CLIP ViT-L has 257 patches (16x16 + 1 CLS)
        evidence_patch_features = pooled_evidence.unsqueeze(1).expand(-1, num_patches, -1)

        # Apply gate
        gate_weight_expanded = gate_weight.unsqueeze(1)
        evidence_patch_features = evidence_patch_features * gate_weight_expanded

        if return_attention_weights:
            attention_info = {
                'attention_weights': attention_weights,
                'gate_weights': gate_weight.squeeze(1),
                'num_evidence_images': num_evidence_images,
                'captions': caption
            }
            return evidence_patch_features, attention_info
        else:
            return evidence_patch_features

    def _encode_caption(self, caption):
        """Encode caption strings as query vectors."""
        # Check for None first before trying to get len()
        if caption is None:
            # Return None to signal caller to use fallback
            return None

        if not hasattr(self, 'caption_tokenizer'):
            # Fallback - now safe to use len()
            batch_size = 1 if isinstance(caption, str) else len(caption)
            return torch.zeros(batch_size, self.hidden_dim, device=next(self.parameters()).device)

        device = next(self.parameters()).device

        # Handle string or list
        if isinstance(caption, str):
            captions = [caption]
        else:
            captions = caption

        # Tokenize
        caption_inputs = self.caption_tokenizer(
            captions,
            max_length=self.caption_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        caption_inputs = {k: v.to(device) for k, v in caption_inputs.items()}

        # Encode
        with torch.no_grad():
            caption_hidden, caption_cls = self.text_encoder(
                caption_inputs['input_ids'],
                caption_inputs['attention_mask']
            )

        return caption_cls
