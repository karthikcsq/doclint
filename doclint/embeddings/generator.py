"""Sentence transformer-based embedding generator."""

import logging
from typing import Any, ClassVar, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..core.exceptions import EmbeddingError
from ..utils.text import chunk_text
from .base import BaseEmbeddingGenerator

logger = logging.getLogger(__name__)


class SentenceTransformerGenerator(BaseEmbeddingGenerator):
    """Embedding generator using sentence-transformers library.

    This generator uses pre-trained sentence transformer models to create
    semantic embeddings of text. It supports:
    - Lazy model loading (model loaded on first use)
    - Automatic device detection (CPU/CUDA)
    - Batch processing for efficiency
    - Text chunking for long documents
    - Graceful handling of empty text

    Attributes:
        model_name: Name of the sentence-transformers model
        device: Device to use ("cpu", "cuda", or None for auto-detection)
        normalize_embeddings: Whether to L2-normalize embeddings

    Example:
        >>> generator = SentenceTransformerGenerator()
        >>> embedding = generator.generate("Hello world")
        >>> print(embedding.shape)
        (384,)
    """

    model_name: ClassVar[str] = "sentence-transformers"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize sentence transformer generator.

        Args:
            model_name: Name of sentence-transformers model (default: all-MiniLM-L6-v2)
            device: Device to use ("cpu", "cuda", or None for auto-detection)
            normalize_embeddings: Whether to L2-normalize embeddings (default: True)
        """
        self._model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None

        logger.debug(
            f"Initialized SentenceTransformerGenerator "
            f"(model={model_name}, device={device or 'auto'})"
        )

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load sentence transformer model.

        The model is only loaded on first access to avoid unnecessary
        initialization overhead.

        Returns:
            Loaded SentenceTransformer model

        Raises:
            EmbeddingError: If model loading fails
        """
        if self._model is None:
            try:
                # Auto-detect device if not specified
                if self.device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    device = self.device

                logger.info(f"Loading model '{self._model_name}' on device '{device}'...")

                # Load model
                self._model = SentenceTransformer(self._model_name, device=device)

                # Cache dimension
                self._dimension = self._model.get_sentence_embedding_dimension()

                logger.info(f"Model loaded successfully (dimension={self._dimension})")

            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load model '{self._model_name}': {e}",
                    model_name=self._model_name,
                    original_error=e,
                )

        return self._model

    def get_embedding_dimension(self) -> int:
        """Get embedding vector dimension.

        Returns:
            Embedding dimension (e.g., 384 for MiniLM, 768 for BERT-base)
        """
        if self._dimension is None:
            # Trigger lazy loading to get dimension
            _ = self.model
        return self._dimension or 384  # Fallback to 384 if somehow None

    def generate(self, text: str) -> np.ndarray[Any, Any]:
        """Generate embedding for a single text.

        Empty or whitespace-only text returns a zero vector with a warning.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array (float32)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        # Handle empty text gracefully
        if not text or not text.strip():
            logger.warning("Empty text provided, returning zero vector")
            return np.zeros(self.get_embedding_dimension(), dtype=np.float32)

        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )

            return embedding

        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embedding: {e}",
                model_name=self._model_name,
                text_length=len(text),
                original_error=e,
            )

    def generate_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[np.ndarray[Any, Any]]:
        """Generate embeddings for a batch of texts.

        Batch processing is significantly more efficient than processing
        texts individually, especially on GPU.

        Empty texts in the batch return zero vectors.

        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process simultaneously (default: 32)

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        try:
            # Handle empty texts
            processed_texts = []
            empty_indices = []

            for i, text in enumerate(texts):
                if not text or not text.strip():
                    empty_indices.append(i)
                    processed_texts.append("")  # Placeholder
                else:
                    processed_texts.append(text)

            # Generate embeddings for non-empty texts
            embeddings = self.model.encode(
                processed_texts,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                batch_size=batch_size,
                show_progress_bar=False,
            )

            # Convert to list of arrays
            result = [emb for emb in embeddings]

            # Replace empty text embeddings with zero vectors
            if empty_indices:
                logger.warning(f"Found {len(empty_indices)} empty texts in batch")
                zero_vector = np.zeros(self.get_embedding_dimension(), dtype=np.float32)
                for idx in empty_indices:
                    result[idx] = zero_vector

            return result

        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate batch embeddings: {e}",
                model_name=self._model_name,
                text_length=sum(len(t) for t in texts),
                original_error=e,
            )

    def generate_with_chunking(
        self,
        text: str,
        max_length: int = 512,
        overlap: int = 50,
    ) -> np.ndarray[Any, Any]:
        """Generate embedding for long text using chunking and averaging.

        Long texts are split into overlapping chunks, each chunk is embedded,
        and the final embedding is the average of all chunk embeddings.

        Args:
            text: Input text to embed
            max_length: Maximum chunk size in characters (default: 512)
            overlap: Overlap between chunks in characters (default: 50)

        Returns:
            Averaged embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        # Handle empty text
        if not text or not text.strip():
            logger.warning("Empty text provided, returning zero vector")
            return np.zeros(self.get_embedding_dimension(), dtype=np.float32)

        # If text is short enough, embed directly
        if len(text) <= max_length:
            return self.generate(text)

        try:
            # Split into chunks
            chunks = chunk_text(text, chunk_size=max_length, overlap=overlap)

            if not chunks:
                return np.zeros(self.get_embedding_dimension(), dtype=np.float32)

            # Generate embeddings for all chunks
            chunk_embeddings = self.generate_batch(chunks)

            # Average the embeddings
            averaged: np.ndarray[Any, Any] = np.mean(chunk_embeddings, axis=0).astype(np.float32)

            # Renormalize if enabled
            if self.normalize_embeddings:
                norm = np.linalg.norm(averaged)
                if norm > 0:
                    averaged = averaged / norm

            logger.debug(f"Generated chunked embedding from {len(chunks)} chunks")

            return averaged

        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate chunked embedding: {e}",
                model_name=self._model_name,
                text_length=len(text),
                original_error=e,
            )
