"""
Base models and reusable components for rumor detection experiments.

This module contains fundamental building blocks that can be shared across
different papers and experiments.
"""

from .encoders import TextEncoder, ImageEncoder
from .attention import CrossModalAttention, DeepFusionLayer

__all__ = [
    'TextEncoder',
    'ImageEncoder',
    'CrossModalAttention',
    'DeepFusionLayer'
]
