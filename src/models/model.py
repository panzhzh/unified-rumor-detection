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

from .encoders import TextEncoder, ImageEncoder
from .attention import DeepFusionLayer, EvidenceAttentionPooling


class MultimodalModel(nn.Module):
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
        num_classes,
        text_model_name,
        image_model_name,
        num_fusion_layers,
        num_heads,
        use_evidence,
        caption_max_length,
        dropout,
        unfreeze_clip_layers,
        hidden_dim,
        classifier_hidden_dim
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_evidence = use_evidence
        self.hidden_dim = hidden_dim

        # Encoders
        self.text_encoder = TextEncoder(text_model_name, freeze=False)
        self.image_encoder = ImageEncoder(
            image_model_name,
            unfreeze_last_n_layers=unfreeze_clip_layers
        )

        # Projection layers to align dimensions
        text_dim = self.text_encoder.hidden_size  # 1024 for xlm-roberta-large
        image_dim = self.image_encoder.hidden_size  # 768 for clip-vit-base-patch16

        # Project text to common dimension
        if text_dim != self.hidden_dim:
            self.text_projection = nn.Linear(text_dim, self.hidden_dim)
        else:
            self.text_projection = nn.Identity()

        # Project image to common dimension
        if image_dim != self.hidden_dim:
            self.image_projection = nn.Linear(image_dim, self.hidden_dim)
        else:
            self.image_projection = nn.Identity()

        # Multi-layer cross-modal fusion
        self.num_fusion_layers = num_fusion_layers
        self.fusion_layers = nn.ModuleList([
            DeepFusionLayer(
                dim=self.hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_fusion_layers)
        ])

        # Text-guided image attention pooling (for evidence)
        if use_evidence:
            # Attention weights: w^T [v_i; t; v_i ⊙ t]
            self.image_attention_fc = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, classifier_hidden_dim),
                nn.Tanh(),
                nn.Linear(classifier_hidden_dim, 1)
            )

            # Prior weights for main vs evidence images
            self.register_buffer('main_prior_weight', torch.tensor([2.0]))  # Main image prior: 1.5~3.0
            self.register_buffer('evidence_prior_weight', torch.tensor([0.75]))  # Evidence prior: 0.5~1.0

        # Pooling layers
        self.text_pooler = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.image_pooler = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Classifier
        classifier_input_dim = self.hidden_dim * 2

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        evidence_pixel_values=None,
        return_attention_weights=False
    ):
        """
        Args:
            input_ids: [batch_size, seq_len] - Formatted text with [CAP][OCR][EVI]
            attention_mask: [batch_size, seq_len]
            pixel_values: [batch_size, 3, 224, 224] - Main image
            evidence_pixel_values: [batch_size, num_evidence, 3, 224, 224] - Evidence images (optional)

        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size = input_ids.shape[0]

        # Encode text (with evidence text already concatenated)
        text_hidden, _ = self.text_encoder(input_ids, attention_mask)
        text_hidden = self.text_projection(text_hidden)
        text_cls = text_hidden[:, 0, :]  # CLS token after projection

        # Encode main image (keep patch tokens)
        main_image_hidden, _ = self.image_encoder(pixel_values)
        main_image_hidden = self.image_projection(main_image_hidden)
        main_cls = main_image_hidden[:, 0, :]
        main_patches = main_image_hidden[:, 1:, :]

        # Prepare image token sequence (start with CLS-like token)
        image_hidden = torch.cat([main_cls.unsqueeze(1), main_patches], dim=1)

        # Process evidence images if available
        if self.use_evidence and evidence_pixel_values is not None and evidence_pixel_values.shape[1] > 0:
            num_evidence = evidence_pixel_values.shape[1]

            # Encode evidence images
            evi_flat = evidence_pixel_values.view(batch_size * num_evidence, 3, 224, 224)
            evi_hidden, _ = self.image_encoder(evi_flat)
            evi_hidden = self.image_projection(evi_hidden)
            seq_len = evi_hidden.shape[1]
            evi_hidden = evi_hidden.contiguous().view(batch_size, num_evidence, seq_len, self.hidden_dim)
            evi_cls = evi_hidden[:, :, 0, :]

            # Text-guided attention pooling over global CLS tokens
            image_cls_all = torch.cat([main_cls.unsqueeze(1), evi_cls], dim=1)
            text_cls_expanded = text_cls.unsqueeze(1).expand(-1, 1 + num_evidence, -1)

            attention_input = torch.cat([
                image_cls_all,
                text_cls_expanded,
                image_cls_all * text_cls_expanded
            ], dim=-1)

            attention_logits = self.image_attention_fc(attention_input).squeeze(-1)

            prior_weights = torch.cat([
                self.main_prior_weight.expand(batch_size, 1),
                self.evidence_prior_weight.expand(batch_size, num_evidence)
            ], dim=1)

            attention_logits = attention_logits + torch.log(prior_weights + 1e-8)
            attention_weights = torch.softmax(attention_logits, dim=1)

            # Weighted CLS pooling for downstream classifier
            image_cls_pooled = (attention_weights.unsqueeze(-1) * image_cls_all).sum(dim=1)

            # Rebuild image token sequence: pooled token + weighted patches
            main_weight = attention_weights[:, 0].unsqueeze(-1).unsqueeze(-1)
            weighted_main = main_patches * main_weight

            evi_weights = attention_weights[:, 1:].unsqueeze(-1).unsqueeze(-1)
            weighted_evi = (evi_hidden[:, :, 1:, :] * evi_weights).contiguous().view(batch_size, -1, self.hidden_dim)

            image_hidden = torch.cat([
                image_cls_pooled.unsqueeze(1),
                weighted_main,
                weighted_evi
            ], dim=1)
        else:
            # No evidence: use main CLS as pooled representation
            image_cls_pooled = main_cls

        # Cross-modal fusion
        for fusion_layer in self.fusion_layers:
            text_hidden, image_hidden = fusion_layer(text_hidden, image_hidden)

        # Pool representations (use updated CLS tokens)
        text_pooled = torch.tanh(self.text_pooler(text_hidden[:, 0, :]))
        image_pooled = torch.tanh(self.image_pooler(image_hidden[:, 0, :]))

        # Concatenate and classify
        fused_features = torch.cat([text_pooled, image_pooled], dim=1)
        logits = self.classifier(fused_features)

        return logits
