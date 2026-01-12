"""Core orchestration and models for DocLint."""

from .config import (
    CompletenessDetectorConfig,
    ConflictDetectorConfig,
    DocLintConfig,
    EmbeddingConfig,
)
from .document import Chunk, Document, DocumentMetadata
from .exceptions import (
    CacheError,
    ConfigurationError,
    DocLintError,
    EmbeddingError,
    ParsingError,
)
from .scanner import Issue, Scanner, ScanResult

__all__ = [
    # Document models
    "Document",
    "DocumentMetadata",
    "Chunk",
    # Configuration
    "DocLintConfig",
    "ConflictDetectorConfig",
    "CompletenessDetectorConfig",
    "EmbeddingConfig",
    # Scanner
    "Scanner",
    "ScanResult",
    "Issue",
    # Exceptions
    "DocLintError",
    "ParsingError",
    "EmbeddingError",
    "ConfigurationError",
    "CacheError",
]
