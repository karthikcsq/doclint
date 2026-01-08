"""Base class for embedding generators."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np


class BaseEmbeddingGenerator(ABC):
    """Abstract base class for embedding generators.

    This base class defines the interface that all embedding generators must implement.
    It follows the same pattern as BaseParser to ensure architectural consistency.

    Subclasses must implement:
    - generate(): Generate embedding for a single text
    - generate_batch(): Generate embeddings for multiple texts
    - get_embedding_dimension(): Return the dimension of embedding vectors

    Example:
        >>> class MyGenerator(BaseEmbeddingGenerator):
        ...     model_name = "my-model"
        ...     def generate(self, text: str) -> np.ndarray:
        ...         return np.random.rand(384)
        ...     def generate_batch(self, texts: list[str], batch_size: int) -> list[np.ndarray]:
        ...         return [self.generate(t) for t in texts]
        ...     def get_embedding_dimension(self) -> int:
        ...         return 384
    """

    # Class-level metadata (similar to BaseParser pattern)
    model_name: ClassVar[str] = ""

    @abstractmethod
    def generate(self, text: str) -> np.ndarray[Any, Any]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[np.ndarray[Any, Any]]:
        """Generate embeddings for a batch of texts.

        Batch processing is typically more efficient than processing texts one by one.
        The batch_size parameter controls how many texts are processed simultaneously.

        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process in each batch (default: 32)

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embedding vectors.

        Returns:
            Embedding vector dimension (e.g., 384 for MiniLM, 768 for BERT-base)
        """
        pass
