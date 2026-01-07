"""Plain text parser."""

from datetime import datetime
from pathlib import Path
from typing import ClassVar

from ..core.document import DocumentMetadata
from ..core.exceptions import ParsingError
from .base import BaseParser


class TextParser(BaseParser):
    """Parser for plain text files.

    Supports .txt and .text extensions.
    Uses encoding fallback (UTF-8 → latin-1) to handle various text encodings.
    Metadata is extracted from filesystem (dates) and filename (title).
    """

    file_type: ClassVar[str] = "text"
    supported_extensions: ClassVar[list[str]] = [".txt", ".text"]

    def parse(self, path: Path) -> str:
        """Extract text from plain text file.

        Tries UTF-8 encoding first, falls back to latin-1 if UTF-8 fails.

        Args:
            path: Path to text file

        Returns:
            File contents (cleaned)

        Raises:
            ParsingError: If file cannot be read
        """
        try:
            # Try UTF-8 first (most common for modern text files)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            # Fall back to latin-1 (handles most 8-bit encodings)
            try:
                with open(path, "r", encoding="latin-1") as f:
                    text = f.read()
            except Exception as e:
                raise ParsingError(
                    f"Failed to read text file {path}: {e}",
                    path=str(path),
                    original_error=e,
                )
        except Exception as e:
            raise ParsingError(
                f"Failed to parse text file {path}: {e}",
                path=str(path),
                original_error=e,
            )

        return self.clean_text(text)

    def extract_metadata(self, path: Path) -> DocumentMetadata:
        """Extract metadata from text file.

        Text files don't have embedded metadata, so we use file system info.

        Args:
            path: Path to text file

        Returns:
            DocumentMetadata with file system dates and filename as title
        """
        metadata = DocumentMetadata()

        try:
            stat = path.stat()
            metadata.modified = datetime.fromtimestamp(stat.st_mtime)
            metadata.created = datetime.fromtimestamp(stat.st_ctime)

            # Use filename (without extension) as title
            metadata.title = path.stem

        except Exception:
            # Metadata extraction is best-effort, never fail the parse
            pass

        return metadata
