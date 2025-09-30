"""Base model interface for rumor detection"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseRumorDetector(nn.Module, ABC):
    """Abstract base class for rumor detection models"""

    def __init__(self, num_classes: int = 2):
        """
        Args:
            num_classes: Number of classes (2 for binary, 3 for MR2 with unverified)
        """
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass

        Args:
            batch: Dictionary containing input data

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        pass

    def predict(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Get predictions

        Args:
            batch: Dictionary containing input data

        Returns:
            Predicted class indices
        """
        logits = self.forward(batch)
        return torch.argmax(logits, dim=-1)

    def get_probabilities(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Get class probabilities

        Args:
            batch: Dictionary containing input data

        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(batch)
        return torch.softmax(logits, dim=-1)

    def save_pretrained(self, save_path: str):
        """Save model weights"""
        torch.save(self.state_dict(), save_path)

    def load_pretrained(self, load_path: str):
        """Load model weights"""
        self.load_state_dict(torch.load(load_path))

    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)