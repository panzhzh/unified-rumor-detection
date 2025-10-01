"""
Cross-modal attention mechanisms for multimodal fusion.
"""

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """
    Cross-modal multi-head attention.

    Allows one modality (query) to attend to another modality (key-value).

    Args:
        dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, dim: int = 1024, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query_features, key_value_features, attention_mask=None):
        """
        Args:
            query_features: [batch_size, query_len, dim]
            key_value_features: [batch_size, kv_len, dim]
            attention_mask: Optional [batch_size, query_len, kv_len]

        Returns:
            attended_features: [batch_size, query_len, dim]
            attention_weights: [batch_size, num_heads, query_len, kv_len]
        """
        batch_size, query_len, _ = query_features.size()
        _, kv_len, _ = key_value_features.size()

        # Multi-head projections
        q = self.query(query_features).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(key_value_features).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(key_value_features).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        attended = torch.matmul(attn, v).transpose(1, 2).contiguous()
        attended = attended.view(batch_size, query_len, self.dim)
        attended = self.out_proj(attended)

        return attended, attn


class DeepFusionLayer(nn.Module):
    """
    Deep fusion layer with bidirectional cross-modal attention.

    Text and image features attend to each other, followed by feed-forward networks.

    Args:
        dim: Hidden dimension
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network dimension (default: 4 * dim)
        dropout: Dropout rate
    """

    def __init__(self, dim: int = 1024, num_heads: int = 8, ffn_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = dim * 4

        # Cross-modal attention
        self.text_to_image_attn = CrossModalAttention(dim, num_heads, dropout)
        self.image_to_text_attn = CrossModalAttention(dim, num_heads, dropout)

        # Feed-forward networks
        self.text_ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout)
        )

        self.image_ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.text_norm1 = nn.LayerNorm(dim)
        self.text_norm2 = nn.LayerNorm(dim)
        self.image_norm1 = nn.LayerNorm(dim)
        self.image_norm2 = nn.LayerNorm(dim)

    def forward(self, text_features, image_features):
        """
        Args:
            text_features: [batch_size, text_len, dim]
            image_features: [batch_size, image_len, dim]

        Returns:
            text_features: [batch_size, text_len, dim] (updated)
            image_features: [batch_size, image_len, dim] (updated)
        """
        # Text attends to image
        text_attended, _ = self.text_to_image_attn(text_features, image_features)
        text_features = self.text_norm1(text_features + text_attended)
        text_features = self.text_norm2(text_features + self.text_ffn(text_features))

        # Image attends to text
        image_attended, _ = self.image_to_text_attn(image_features, text_features)
        image_features = self.image_norm1(image_features + image_attended)
        image_features = self.image_norm2(image_features + self.image_ffn(image_features))

        return text_features, image_features


class EvidenceAttentionPooling(nn.Module):
    """
    Attention-based pooling for evidence images using query.

    Args:
        dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, dim: int = 1024, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, query, evidence_features, num_evidence=None):
        """
        Args:
            query: [batch_size, dim] - Query vector (e.g., from caption or main text)
            evidence_features: [batch_size, num_evidence, dim] - Evidence image features
            num_evidence: Optional list of actual evidence counts per sample

        Returns:
            pooled_evidence: [batch_size, dim] - Attention-pooled evidence features
            attention_weights: [batch_size, num_evidence] - Attention weights
        """
        # Expand query: [batch_size, 1, dim]
        query = query.unsqueeze(1)

        # Apply attention
        attended_evidence, attention_weights = self.attention(
            query,  # [batch_size, 1, dim]
            evidence_features,  # [batch_size, num_evidence, dim]
            evidence_features   # [batch_size, num_evidence, dim]
        )

        # Squeeze: [batch_size, dim]
        pooled_evidence = attended_evidence.squeeze(1)

        # Squeeze weights: [batch_size, num_evidence]
        attention_weights = attention_weights.squeeze(1)

        return pooled_evidence, attention_weights
