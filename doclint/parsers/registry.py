"""Parser registry for automatic parser selection."""

from pathlib import Path
from typing import Dict, List

from ..core.exceptions import ParsingError
from .base import BaseParser
from .docx import DOCXParser
from .html import HTMLParser
from .markdown import MarkdownParser
from .pdf import PDFParser
from .text import TextParser


class ParserRegistry:
    """Registry for managing document parsers.

    Automatically selects the appropriate parser based on file extension.
    Supports custom parser registration for extensibility.
    """

    def __init__(self) -> None:
        """Initialize registry with default parsers."""
        self._parsers: Dict[str, BaseParser] = {}
        self._extension_map: Dict[str, str] = {}

        # Auto-register default parsers
        self.register(TextParser())
        self.register(MarkdownParser())
        self.register(HTMLParser())
        self.register(PDFParser())
        self.register(DOCXParser())

    def register(self, parser: BaseParser) -> None:
        """Register a parser.

        Args:
            parser: Parser instance to register

        Raises:
            ValueError: If parser file_type is already registered
        """
        if parser.file_type in self._parsers:
            raise ValueError(f"Parser for file type '{parser.file_type}' already registered")

        self._parsers[parser.file_type] = parser

        # Map all extensions to this parser's file_type
        for ext in parser.supported_extensions:
            self._extension_map[ext.lower()] = parser.file_type

    def get_parser(self, path: Path) -> BaseParser:
        """Get parser for a file path.

        Args:
            path: Path to file

        Returns:
            Parser instance that can handle the file

        Raises:
            ParsingError: If no parser found for file extension
        """
        ext = path.suffix.lower()

        if ext not in self._extension_map:
            supported = ", ".join(sorted(self._extension_map.keys()))
            raise ParsingError(
                f"No parser found for extension '{ext}'. Supported: {supported}",
                path=str(path),
            )

        file_type = self._extension_map[ext]
        return self._parsers[file_type]

    def can_parse(self, path: Path) -> bool:
        """Check if a parser is available for the file.

        Args:
            path: Path to file

        Returns:
            True if parser available, False otherwise
        """
        ext = path.suffix.lower()
        return ext in self._extension_map

    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions.

        Returns:
            Sorted list of supported extensions (e.g., ['.docx', '.html', ...])
        """
        return sorted(self._extension_map.keys())

    def get_parser_by_type(self, file_type: str) -> BaseParser:
        """Get parser by file type.

        Args:
            file_type: File type identifier (e.g., 'pdf', 'docx')

        Returns:
            Parser instance

        Raises:
            ValueError: If file_type not registered
        """
        if file_type not in self._parsers:
            available = ", ".join(sorted(self._parsers.keys()))
            raise ValueError(f"No parser for type '{file_type}'. Available: {available}")

        return self._parsers[file_type]
