"""PDF parser."""

from datetime import datetime
from pathlib import Path
from typing import ClassVar

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ..core.document import DocumentMetadata
from ..core.exceptions import ParsingError
from .base import BaseParser


class PDFParser(BaseParser):
    """Parser for PDF files.

    Uses pypdf to extract text from all pages.
    Extracts metadata from PDF document info dictionary.
    Handles encrypted PDFs gracefully by raising ParsingError.
    """

    file_type: ClassVar[str] = "pdf"
    supported_extensions: ClassVar[list[str]] = [".pdf"]

    def parse(self, path: Path) -> str:
        """Extract text from PDF file.

        Args:
            path: Path to PDF file

        Returns:
            Extracted text from all pages

        Raises:
            ParsingError: If PDF is encrypted, corrupted, or cannot be read
        """
        try:
            reader = PdfReader(str(path))

            # Check if PDF is encrypted
            if reader.is_encrypted:
                raise ParsingError(
                    f"PDF is encrypted and cannot be read: {path}",
                    path=str(path),
                )

            # Extract text from all pages
            pages_text = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)

            # Join pages with double newline
            full_text = "\n\n".join(pages_text)

            return self.clean_text(full_text)

        except PdfReadError as e:
            raise ParsingError(
                f"Failed to read PDF {path}: {e}",
                path=str(path),
                original_error=e,
            )
        except Exception as e:
            raise ParsingError(
                f"Failed to parse PDF {path}: {e}",
                path=str(path),
                original_error=e,
            )

    def extract_metadata(self, path: Path) -> DocumentMetadata:
        """Extract metadata from PDF file.

        Tries to extract:
        - Title from /Title field
        - Author from /Author field
        - Created date from /CreationDate field
        - Modified date from /ModDate field

        Args:
            path: Path to PDF file

        Returns:
            DocumentMetadata
        """
        metadata = DocumentMetadata()

        try:
            # File system dates (fallback)
            stat = path.stat()
            metadata.modified = datetime.fromtimestamp(stat.st_mtime)
            metadata.created = datetime.fromtimestamp(stat.st_ctime)

            # Read PDF metadata
            reader = PdfReader(str(path))

            if reader.metadata:
                # Title
                if reader.metadata.title:
                    metadata.title = reader.metadata.title

                # Author
                if reader.metadata.author:
                    metadata.author = reader.metadata.author

                # Creation date
                if reader.metadata.creation_date:
                    metadata.created = reader.metadata.creation_date

                # Modification date
                if reader.metadata.modification_date:
                    metadata.modified = reader.metadata.modification_date

                # Subject as custom field
                if reader.metadata.subject:
                    metadata.custom["subject"] = reader.metadata.subject

                # Creator/Producer as custom fields
                if reader.metadata.creator:
                    metadata.custom["creator"] = reader.metadata.creator
                if reader.metadata.producer:
                    metadata.custom["producer"] = reader.metadata.producer

            # If no title from metadata, use filename
            if not metadata.title:
                metadata.title = path.stem

        except Exception:
            # Best-effort
            pass

        return metadata
