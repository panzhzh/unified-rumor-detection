"""
Text preprocessing utilities.

This module provides basic text cleaning and processing functions.
For advanced features like translation, language detection, or similarity computation,
consider adding them as needed for specific experiments.
"""

import re
from typing import Optional


class TextProcessor:
    """
    Basic text cleaning and processing for rumor detection.

    Features:
    - HTML tag and entity removal
    - Whitespace normalization
    - Optional custom cleaning functions
    """

    def __init__(self):
        pass

    def clean_html_and_entities(self, text: str) -> str:
        """
        Remove common HTML tags and entities.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove common HTML entities
        text = text.replace('&#39;', "'")
        text = text.replace('&quot;', '"')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&nbsp;', ' ')

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        return text

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace (collapse multiple spaces, remove leading/trailing).

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def clean_text(self, text: str) -> str:
        """
        Main cleaning function - applies all cleaning steps.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return ""

        # Step 1: Clean HTML
        text = self.clean_html_and_entities(text)

        # Step 2: Normalize whitespace
        text = self.normalize_whitespace(text)

        return text

    def truncate_text(self, text: str, max_length: int = 512) -> str:
        """
        Truncate text to maximum length (character-based).

        Args:
            text: Input text
            max_length: Maximum character length

        Returns:
            Truncated text
        """
        if not text:
            return ""

        if len(text) <= max_length:
            return text

        return text[:max_length]


# Optional: Advanced features that can be added if needed
class AdvancedTextProcessor(TextProcessor):
    """
    Extended text processor with advanced features.

    These features require additional dependencies:
    - opencc: Traditional to Simplified Chinese conversion
    - langdetect: Language detection
    - googletrans: Translation (unstable, not recommended)
    - sentence-transformers: Semantic similarity

    Usage:
        processor = AdvancedTextProcessor(use_similarity=True)
    """

    def __init__(
        self,
        use_traditional_to_simplified: bool = False,
        use_similarity: bool = False,
        similarity_model_name: Optional[str] = None
    ):
        super().__init__()

        self.use_t2s = use_traditional_to_simplified
        self.use_similarity = use_similarity

        # Optional: Traditional to Simplified Chinese
        if use_traditional_to_simplified:
            try:
                from opencc import OpenCC
                self.cc = OpenCC('t2s')
            except ImportError:
                print("Warning: opencc not installed. Traditional to Simplified conversion disabled.")
                self.use_t2s = False

        # Optional: Similarity computation
        if use_similarity:
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
                self.similarity_model = SentenceTransformer(
                    similarity_model_name or 'sentence-transformers/all-mpnet-base-v2'
                )
                self.np = np
            except ImportError:
                print("Warning: sentence-transformers not installed. Similarity computation disabled.")
                self.use_similarity = False

    def traditional_to_simplified(self, text: str) -> str:
        """Convert traditional Chinese to simplified Chinese."""
        if not self.use_t2s or not text:
            return text

        try:
            return self.cc.convert(text)
        except Exception as e:
            print(f"Error in traditional to simplified conversion: {e}")
            return text

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        if not self.use_similarity or not text1.strip() or not text2.strip():
            return 0.0

        try:
            embeddings = self.similarity_model.encode([text1, text2])
            similarity = self.np.dot(embeddings[0], embeddings[1]) / (
                self.np.linalg.norm(embeddings[0]) * self.np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0

    def clean_text(self, text: str) -> str:
        """Extended cleaning with traditional to simplified conversion."""
        if not text or not text.strip():
            return ""

        # Base cleaning
        text = super().clean_text(text)

        # Traditional to Simplified
        if self.use_t2s:
            text = self.traditional_to_simplified(text)

        return text
