"""Tests for plain text parser."""

from pathlib import Path

import pytest

from doclint.core.document import DocumentMetadata
from doclint.core.exceptions import ParsingError
from doclint.parsers.text import TextParser


class TestTextParser:
    """Test suite for TextParser."""

    def test_can_parse_txt_files(self) -> None:
        """Test parser recognizes .txt files."""
        parser = TextParser()
        assert parser.can_parse(Path("test.txt")) is True
        assert parser.can_parse(Path("test.text")) is True

    def test_cannot_parse_other_files(self) -> None:
        """Test parser rejects non-text files."""
        parser = TextParser()
        assert parser.can_parse(Path("test.pdf")) is False
        assert parser.can_parse(Path("test.docx")) is False
        assert parser.can_parse(Path("test.md")) is False

    def test_parse_simple_text(self, tmp_path: Path) -> None:
        """Test parsing simple text file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        content = "Hello, World!\nThis is a test."
        test_file.write_text(content, encoding="utf-8")

        # Parse
        parser = TextParser()
        result = parser.parse(test_file)

        # Verify
        assert "Hello, World!" in result
        assert "This is a test." in result

    def test_parse_cleans_whitespace(self, tmp_path: Path) -> None:
        """Test that excessive whitespace is cleaned."""
        test_file = tmp_path / "test.txt"
        content = "Line 1  with   extra    spaces\n\n\n\nLine 2"
        test_file.write_text(content, encoding="utf-8")

        parser = TextParser()
        result = parser.parse(test_file)

        # Should clean multiple spaces within lines
        assert "extra    spaces" not in result
        assert "extra spaces" in result

        # Should reduce excessive newlines
        assert "\n\n\n\n" not in result

    def test_parse_handles_utf8(self, tmp_path: Path) -> None:
        """Test parsing UTF-8 encoded text."""
        test_file = tmp_path / "utf8.txt"
        content = "Hello 世界 🌍"
        test_file.write_text(content, encoding="utf-8")

        parser = TextParser()
        result = parser.parse(test_file)

        assert "世界" in result
        assert "🌍" in result

    def test_parse_handles_latin1(self, tmp_path: Path) -> None:
        """Test parsing latin-1 encoded text."""
        test_file = tmp_path / "latin1.txt"
        content = "Café résumé"
        test_file.write_bytes(content.encode("latin-1"))

        parser = TextParser()
        result = parser.parse(test_file)

        # Should parse without error (content may differ due to encoding)
        assert "Caf" in result

    def test_parse_nonexistent_file(self) -> None:
        """Test parsing non-existent file raises error."""
        parser = TextParser()

        with pytest.raises(ParsingError) as exc_info:
            parser.parse(Path("/nonexistent/file.txt"))

        assert "Failed to parse" in str(exc_info.value)

    def test_extract_metadata(self, tmp_path: Path) -> None:
        """Test metadata extraction."""
        test_file = tmp_path / "document.txt"
        test_file.write_text("Test content")

        parser = TextParser()
        metadata = parser.extract_metadata(test_file)

        # Should have filesystem dates
        assert metadata.modified is not None
        assert metadata.created is not None

        # Should use filename as title
        assert metadata.title == "document"

    def test_extract_metadata_best_effort(self) -> None:
        """Test metadata extraction doesn't fail on error."""
        parser = TextParser()

        # Non-existent file should return empty metadata (not raise)
        metadata = parser.extract_metadata(Path("/nonexistent/file.txt"))

        # Should not raise, returns default metadata
        assert isinstance(metadata, DocumentMetadata)

    def test_file_type_is_text(self) -> None:
        """Test parser has correct file_type."""
        parser = TextParser()
        assert parser.file_type == "text"

    def test_supported_extensions(self) -> None:
        """Test parser lists supported extensions."""
        parser = TextParser()
        assert ".txt" in parser.supported_extensions
        assert ".text" in parser.supported_extensions
