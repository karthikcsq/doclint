"""Semantic embedding generation for documents."""

from .base import BaseEmbeddingGenerator
from .generator import SentenceTransformerGenerator
from .processor import DocumentProcessor, get_all_chunks

# Convenience alias for the default generator
EmbeddingGenerator = SentenceTransformerGenerator

__all__ = [
    "BaseEmbeddingGenerator",
    "SentenceTransformerGenerator",
    "EmbeddingGenerator",
    "DocumentProcessor",
    "get_all_chunks",
]
