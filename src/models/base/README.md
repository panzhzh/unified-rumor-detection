# Base Model Components

Reusable building blocks for multimodal rumor detection models.

## Components

### 1. Encoders (`encoders.py`)

#### TextEncoder
XLM-RoBERTa-based text encoder.

```python
from src.models.base import TextEncoder

encoder = TextEncoder(
    model_name="xlm-roberta-large",  # or "xlm-roberta-base"
    freeze=False  # Set True to freeze weights
)

# Forward pass
last_hidden_state, pooled_output = encoder(input_ids, attention_mask)
# last_hidden_state: [batch_size, seq_len, 1024]
# pooled_output: [batch_size, 1024] (CLS token)
```

#### ImageEncoder
CLIP vision model-based image encoder with optional trainable adapter.

```python
from src.models.base import ImageEncoder

encoder = ImageEncoder(
    model_name="openai/clip-vit-large-patch14",
    freeze_backbone=True,  # Freeze CLIP weights
    use_trainable_adapter=True,  # Add trainable MLP
    adapter_hidden_dim=1024
)

# Single image
last_hidden_state, pooled_output = encoder(pixel_values)
# last_hidden_state: [batch_size, 257, 1024]  # 257 patches (16x16 + CLS)
# pooled_output: [batch_size, 1024]

# Multiple images
pixel_values = torch.randn(8, 5, 3, 224, 224)  # batch=8, num_images=5
last_hidden_state, pooled_output = encoder(pixel_values)
# last_hidden_state: [8, 5, 257, 1024]
# pooled_output: [8, 5, 1024]
```

### 2. Attention Mechanisms (`attention.py`)

#### CrossModalAttention
Allows one modality to attend to another.

```python
from src.models.base import CrossModalAttention

attn = CrossModalAttention(
    dim=1024,
    num_heads=8,
    dropout=0.1
)

# Text attends to image
text_features = torch.randn(8, 128, 1024)  # batch=8, seq_len=128
image_features = torch.randn(8, 257, 1024)  # batch=8, num_patches=257

attended_text, attention_weights = attn(text_features, image_features)
# attended_text: [8, 128, 1024]
# attention_weights: [8, 8, 128, 257]  # [batch, heads, query_len, kv_len]
```

#### DeepFusionLayer
Bidirectional cross-modal fusion layer.

```python
from src.models.base import DeepFusionLayer

fusion = DeepFusionLayer(
    dim=1024,
    num_heads=8,
    ffn_dim=4096,  # Feed-forward dimension (default: 4 * dim)
    dropout=0.1
)

# Fuse text and image
text_features = torch.randn(8, 128, 1024)
image_features = torch.randn(8, 257, 1024)

fused_text, fused_image = fusion(text_features, image_features)
# fused_text: [8, 128, 1024]
# fused_image: [8, 257, 1024]
```

#### EvidenceAttentionPooling
Attention-based pooling for evidence images.

```python
from src.models.base import EvidenceAttentionPooling

pooling = EvidenceAttentionPooling(
    dim=1024,
    num_heads=8,
    dropout=0.1
)

# Pool evidence images using query
query = torch.randn(8, 1024)  # batch=8
evidence_features = torch.randn(8, 5, 1024)  # batch=8, num_evidence=5

pooled_evidence, attention_weights = pooling(query, evidence_features)
# pooled_evidence: [8, 1024]
# attention_weights: [8, 5]
```

## Example: Building a Simple Multimodal Model

```python
import torch
import torch.nn as nn
from src.models.base import TextEncoder, ImageEncoder, DeepFusionLayer

class SimpleMultimodalModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Encoders
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()

        # Fusion
        self.fusion = nn.ModuleList([
            DeepFusionLayer(dim=1024) for _ in range(3)
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),  # 1024 text + 1024 image
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # Encode
        text_hidden, text_cls = self.text_encoder(input_ids, attention_mask)
        image_hidden, image_cls = self.image_encoder(pixel_values)

        # Fuse with multiple layers
        for fusion_layer in self.fusion:
            text_hidden, image_hidden = fusion_layer(text_hidden, image_hidden)

        # Pool and concatenate
        text_pooled = text_hidden[:, 0]  # CLS token
        image_pooled = image_hidden[:, 0]  # CLS token
        fused = torch.cat([text_pooled, image_pooled], dim=1)

        # Classify
        logits = self.classifier(fused)
        return logits

# Usage
model = SimpleMultimodalModel(num_classes=2)
input_ids = torch.randint(0, 1000, (8, 128))
attention_mask = torch.ones(8, 128)
pixel_values = torch.randn(8, 3, 224, 224)

logits = model(input_ids, attention_mask, pixel_values)
print(logits.shape)  # [8, 2]
```

## Design Principles

1. **Modularity**: Each component can be used independently
2. **Flexibility**: Easy to customize dimensions, layers, etc.
3. **Compatibility**: Works with standard PyTorch training loops
4. **Efficiency**: Frozen backbones + trainable adapters reduce memory
5. **Reusability**: Components can be shared across different experiments

## Integration with Experiments

Base components are imported in experiment models:

```python
# In src/models/experiments/my_paper/model.py
from ...base import TextEncoder, ImageEncoder, DeepFusionLayer

class MyExperimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.fusion = DeepFusionLayer()
        # Add your novel components here
```

This approach:
- ✅ Reduces code duplication
- ✅ Ensures consistency across experiments
- ✅ Makes debugging easier
- ✅ Allows rapid prototyping
