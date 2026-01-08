"""Tests for parser registry."""

from pathlib import Path

import pytest

from doclint.core.document import DocumentMetadata
from doclint.core.exceptions import ParsingError
from doclint.parsers.base import BaseParser
from doclint.parsers.docx import DOCXParser
from doclint.parsers.html import HTMLParser
from doclint.parsers.markdown import MarkdownParser
from doclint.parsers.pdf import PDFParser
from doclint.parsers.registry import ParserRegistry
from doclint.parsers.text import TextParser


class TestParserRegistry:
    """Test suite for ParserRegistry."""

    def test_default_parsers_registered(self) -> None:
        """Test that default parsers are auto-registered."""
        registry = ParserRegistry()

        # Should have all 5 default parsers
        assert isinstance(registry.get_parser_by_type("text"), TextParser)
        assert isinstance(registry.get_parser_by_type("markdown"), MarkdownParser)
        assert isinstance(registry.get_parser_by_type("html"), HTMLParser)
        assert isinstance(registry.get_parser_by_type("pdf"), PDFParser)
        assert isinstance(registry.get_parser_by_type("docx"), DOCXParser)

    def test_get_parser_for_text_file(self) -> None:
        """Test getting parser for text file."""
        registry = ParserRegistry()

        parser = registry.get_parser(Path("test.txt"))
        assert isinstance(parser, TextParser)

        parser = registry.get_parser(Path("test.text"))
        assert isinstance(parser, TextParser)

    def test_get_parser_for_markdown_file(self) -> None:
        """Test getting parser for markdown file."""
        registry = ParserRegistry()

        parser = registry.get_parser(Path("test.md"))
        assert isinstance(parser, MarkdownParser)

        parser = registry.get_parser(Path("test.markdown"))
        assert isinstance(parser, MarkdownParser)

    def test_get_parser_for_html_file(self) -> None:
        """Test getting parser for HTML file."""
        registry = ParserRegistry()

        parser = registry.get_parser(Path("test.html"))
        assert isinstance(parser, HTMLParser)

        parser = registry.get_parser(Path("test.htm"))
        assert isinstance(parser, HTMLParser)

    def test_get_parser_for_pdf_file(self) -> None:
        """Test getting parser for PDF file."""
        registry = ParserRegistry()

        parser = registry.get_parser(Path("test.pdf"))
        assert isinstance(parser, PDFParser)

    def test_get_parser_for_docx_file(self) -> None:
        """Test getting parser for DOCX file."""
        registry = ParserRegistry()

        parser = registry.get_parser(Path("test.docx"))
        assert isinstance(parser, DOCXParser)

    def test_get_parser_case_insensitive(self) -> None:
        """Test that extension matching is case-insensitive."""
        registry = ParserRegistry()

        parser_upper = registry.get_parser(Path("test.PDF"))
        assert isinstance(parser_upper, PDFParser)

        parser_mixed = registry.get_parser(Path("test.Docx"))
        assert isinstance(parser_mixed, DOCXParser)

    def test_get_parser_unsupported_extension(self) -> None:
        """Test that unsupported extension raises ParsingError."""
        registry = ParserRegistry()

        with pytest.raises(ParsingError) as exc_info:
            registry.get_parser(Path("test.xyz"))

        assert "No parser found for extension '.xyz'" in str(exc_info.value)
        assert "Supported:" in str(exc_info.value)

    def test_can_parse_supported_file(self) -> None:
        """Test can_parse returns True for supported files."""
        registry = ParserRegistry()

        assert registry.can_parse(Path("test.txt")) is True
        assert registry.can_parse(Path("test.md")) is True
        assert registry.can_parse(Path("test.html")) is True
        assert registry.can_parse(Path("test.pdf")) is True
        assert registry.can_parse(Path("test.docx")) is True

    def test_can_parse_unsupported_file(self) -> None:
        """Test can_parse returns False for unsupported files."""
        registry = ParserRegistry()

        assert registry.can_parse(Path("test.xyz")) is False
        assert registry.can_parse(Path("test.jpg")) is False
        assert registry.can_parse(Path("test")) is False

    def test_get_supported_extensions(self) -> None:
        """Test getting list of supported extensions."""
        registry = ParserRegistry()

        extensions = registry.get_supported_extensions()

        # Should have all extensions from all parsers
        assert ".txt" in extensions
        assert ".text" in extensions
        assert ".md" in extensions
        assert ".markdown" in extensions
        assert ".html" in extensions
        assert ".htm" in extensions
        assert ".pdf" in extensions
        assert ".docx" in extensions

        # Should be sorted
        assert extensions == sorted(extensions)

    def test_get_parser_by_type(self) -> None:
        """Test getting parser by file type."""
        registry = ParserRegistry()

        assert isinstance(registry.get_parser_by_type("text"), TextParser)
        assert isinstance(registry.get_parser_by_type("markdown"), MarkdownParser)
        assert isinstance(registry.get_parser_by_type("html"), HTMLParser)
        assert isinstance(registry.get_parser_by_type("pdf"), PDFParser)
        assert isinstance(registry.get_parser_by_type("docx"), DOCXParser)

    def test_get_parser_by_type_not_found(self) -> None:
        """Test getting parser by unknown type raises ValueError."""
        registry = ParserRegistry()

        with pytest.raises(ValueError) as exc_info:
            registry.get_parser_by_type("unknown")

        assert "No parser for type 'unknown'" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_register_duplicate_parser_raises_error(self) -> None:
        """Test that registering duplicate file_type raises ValueError."""
        registry = ParserRegistry()

        # Try to register another text parser
        duplicate_parser = TextParser()

        with pytest.raises(ValueError) as exc_info:
            registry.register(duplicate_parser)

        assert "already registered" in str(exc_info.value)

    def test_custom_parser_registration(self) -> None:
        """Test registering a custom parser."""
        registry = ParserRegistry()

        # Create a custom parser
        class CustomParser(BaseParser):
            file_type = "custom"
            supported_extensions = [".custom"]

            def parse(self, path: Path) -> str:
                return "custom content"

            def extract_metadata(self, path: Path) -> DocumentMetadata:
                return DocumentMetadata()

        custom_parser = CustomParser()
        registry.register(custom_parser)

        # Should be able to get it
        parser = registry.get_parser(Path("test.custom"))
        assert isinstance(parser, CustomParser)
        assert registry.can_parse(Path("test.custom")) is True
        assert ".custom" in registry.get_supported_extensions()
