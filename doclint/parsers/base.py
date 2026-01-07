"""Abstract base class for document parsers."""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from ..core.document import DocumentMetadata


class BaseParser(ABC):
    """Abstract base class for all document parsers.

    Subclasses must implement:
        - parse(): Extract text content
        - extract_metadata(): Extract document metadata

    Subclasses must set:
        - file_type: str (e.g., "pdf", "docx")
        - supported_extensions: list[str] (e.g., ['.pdf'])

    The base class provides:
        - can_parse(): Check if parser supports a file extension
        - clean_text(): Utility for cleaning extracted text
    """

    file_type: ClassVar[str] = ""
    supported_extensions: ClassVar[list[str]] = []

    @abstractmethod
    def parse(self, path: Path) -> str:
        """Extract text content from document.

        Args:
            path: Path to document file

        Returns:
            Extracted text content (cleaned)

        Raises:
            ParsingError: If parsing fails
        """
        pass

    @abstractmethod
    def extract_metadata(self, path: Path) -> DocumentMetadata:
        """Extract metadata from document.

        Metadata extraction is best-effort and should never fail the parse.
        If metadata cannot be extracted, return a DocumentMetadata with default values.

        Args:
            path: Path to document file

        Returns:
            DocumentMetadata object (may have None fields if extraction fails)
        """
        pass

    def can_parse(self, path: Path) -> bool:
        """Check if this parser can handle the file.

        Args:
            path: Path to check

        Returns:
            True if parser supports this file extension
        """
        return path.suffix.lower() in self.supported_extensions

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean extracted text.

        Cleaning operations:
        - Remove null bytes
        - Normalize line endings (\\r\\n → \\n, \\r → \\n)
        - Collapse multiple spaces within lines
        - Remove excessive blank lines (max 2 consecutive)
        - Strip leading/trailing whitespace

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove null bytes
        text = text.replace("\x00", "")

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split into lines and clean each line
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            # Collapse multiple spaces within line
            cleaned_line = " ".join(line.split())
            cleaned_lines.append(cleaned_line)

        # Join lines back together
        result = "\n".join(cleaned_lines).strip()

        # Remove excessive blank lines (max 2 consecutive newlines)
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result
