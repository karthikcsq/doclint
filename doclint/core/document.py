"""Core document models."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from ..parsers.base import BaseParser

    EmbeddingArray: TypeAlias = npt.NDArray[np.float32]
else:
    # At runtime, use Any to avoid requiring numpy import
    EmbeddingArray: TypeAlias = Any

from .exceptions import ParsingError


@dataclass
class Chunk:
    """Represents a chunk of a document for embedding and analysis.

    Documents are split into chunks for fine-grained semantic analysis. Each chunk
    has its own embedding and can be independently compared with chunks from other
    documents. This enables detection of conflicts and similarities at a granular level.

    Attributes:
        text: The text content of this chunk
        index: Position of chunk in the document (0-indexed)
        document_path: Path to the parent document
        chunk_hash: SHA-256 hash of the chunk text (for caching)
        embedding: Semantic embedding vector for this chunk
        start_pos: Character position where chunk starts in original document
        end_pos: Character position where chunk ends in original document
    """

    text: str
    index: int
    document_path: Path
    chunk_hash: str
    embedding: Optional[EmbeddingArray] = None
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class DocumentMetadata:
    """Metadata extracted from document.

    All fields are optional as not all document formats support all metadata types.
    Extraction is best-effort and should never fail the parsing process.

    Attributes:
        author: Document author/creator
        created: Creation timestamp
        modified: Last modification timestamp
        version: Document version or revision number
        title: Document title
        tags: List of tags/keywords
        custom: Additional custom metadata as key-value pairs
    """

    author: Optional[str] = None
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    version: Optional[str] = None
    title: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Represents a parsed document with content and metadata.

    This is the core data structure used throughout DocLint. Documents are created
    by parsers and consumed by embeddings, detectors, and reporters.

    Documents are split into chunks for fine-grained semantic analysis. Each chunk
    is embedded independently, enabling document-independent conflict detection across
    the entire corpus (similar to RAG systems).

    Attributes:
        path: Path to the source file
        content: Extracted text content (cleaned)
        metadata: Extracted document metadata
        file_type: Type of document (e.g., "pdf", "docx", "markdown")
        size_bytes: Size of the source file in bytes
        content_hash: SHA256 hash of file content for caching
        chunks: List of Chunk objects with embeddings (populated by processing pipeline)
    """

    path: Path
    content: str
    metadata: DocumentMetadata
    file_type: str
    size_bytes: int
    content_hash: str

    # Computed fields (set later by document processing pipeline)
    chunks: list[Chunk] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: Path, parser: "BaseParser") -> "Document":
        """Create document from file using parser.

        This factory method orchestrates the complete document parsing workflow:
        1. Validates file exists and is readable
        2. Parses content using the provided parser
        3. Extracts metadata (best-effort)
        4. Computes content hash for caching
        5. Returns fully-populated Document object

        Args:
            path: Path to document file
            parser: Parser instance to use for extraction

        Returns:
            Document object with parsed content and metadata

        Raises:
            ParsingError: If file doesn't exist, is not a file, or parsing fails
        """
        if not path.exists():
            raise ParsingError(f"File not found: {path}", path=str(path))

        if not path.is_file():
            raise ParsingError(f"Path is not a file: {path}", path=str(path))

        # Parse content and metadata
        content = parser.parse(path)
        metadata = parser.extract_metadata(path)

        # Compute file hash for caching
        with open(path, "rb") as f:
            content_bytes = f.read()
            content_hash = hashlib.sha256(content_bytes).hexdigest()

        return cls(
            path=path,
            content=content,
            metadata=metadata,
            file_type=parser.file_type,
            size_bytes=len(content_bytes),
            content_hash=content_hash,
        )
