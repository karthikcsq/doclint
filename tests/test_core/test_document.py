"""Tests for Document and DocumentMetadata models."""

from pathlib import Path

import pytest

from doclint.core.document import Document, DocumentMetadata
from doclint.core.exceptions import ParsingError
from doclint.parsers.text import TextParser


class TestDocumentMetadata:
    """Test suite for DocumentMetadata."""

    def test_default_metadata(self) -> None:
        """Test creating metadata with defaults."""
        metadata = DocumentMetadata()

        assert metadata.author is None
        assert metadata.created is None
        assert metadata.modified is None
        assert metadata.version is None
        assert metadata.title is None
        assert metadata.tags == []
        assert metadata.custom == {}

    def test_metadata_with_values(self) -> None:
        """Test creating metadata with values."""
        metadata = DocumentMetadata(
            author="John Doe",
            title="Test Document",
            tags=["test", "sample"],
            custom={"category": "technical"},
        )

        assert metadata.author == "John Doe"
        assert metadata.title == "Test Document"
        assert metadata.tags == ["test", "sample"]
        assert metadata.custom == {"category": "technical"}


class TestDocument:
    """Test suite for Document."""

    def test_from_file_creates_document(self, tmp_path: Path) -> None:
        """Test Document.from_file() creates a valid document."""
        # Create test file
        test_file = tmp_path / "test.txt"
        content = "Hello, World!\nThis is a test document."
        test_file.write_text(content, encoding="utf-8")

        # Create document using from_file
        parser = TextParser()
        doc = Document.from_file(test_file, parser)

        # Verify document properties
        assert doc.path == test_file
        assert doc.file_type == "text"
        assert "Hello, World!" in doc.content
        assert "test document" in doc.content
        assert doc.size_bytes > 0
        assert len(doc.content_hash) == 64  # SHA256 hex string
        assert doc.metadata is not None
        assert doc.metadata.title == "test"  # filename without extension

    def test_from_file_computes_content_hash(self, tmp_path: Path) -> None:
        """Test that content hash is computed correctly."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Same content", encoding="utf-8")

        parser = TextParser()
        doc1 = Document.from_file(test_file, parser)
        doc2 = Document.from_file(test_file, parser)

        # Same file should produce same hash
        assert doc1.content_hash == doc2.content_hash

    def test_from_file_different_content_different_hash(self, tmp_path: Path) -> None:
        """Test that different content produces different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("Content A", encoding="utf-8")
        file2.write_text("Content B", encoding="utf-8")

        parser = TextParser()
        doc1 = Document.from_file(file1, parser)
        doc2 = Document.from_file(file2, parser)

        # Different files should have different hashes
        assert doc1.content_hash != doc2.content_hash

    def test_from_file_sets_size_bytes(self, tmp_path: Path) -> None:
        """Test that file size is recorded correctly."""
        test_file = tmp_path / "test.txt"
        content = "Test content"
        test_file.write_text(content, encoding="utf-8")

        parser = TextParser()
        doc = Document.from_file(test_file, parser)

        # Size should match file size (not cleaned content length)
        expected_size = len(content.encode("utf-8"))
        assert doc.size_bytes == expected_size

    def test_from_file_nonexistent_file(self) -> None:
        """Test that from_file raises error for non-existent file."""
        parser = TextParser()

        with pytest.raises(ParsingError) as exc_info:
            Document.from_file(Path("/nonexistent/file.txt"), parser)

        assert "File not found" in str(exc_info.value)
        assert exc_info.value.path == str(Path("/nonexistent/file.txt"))

    def test_from_file_directory_not_file(self, tmp_path: Path) -> None:
        """Test that from_file raises error for directory."""
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()

        parser = TextParser()

        with pytest.raises(ParsingError) as exc_info:
            Document.from_file(test_dir, parser)

        assert "Path is not a file" in str(exc_info.value)

    def test_document_has_default_computed_fields(self, tmp_path: Path) -> None:
        """Test that computed fields have default values."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content", encoding="utf-8")

        parser = TextParser()
        doc = Document.from_file(test_file, parser)

        # Computed fields should be None/empty by default
        assert doc.embedding is None
        assert doc.chunks == []

    def test_from_file_includes_metadata(self, tmp_path: Path) -> None:
        """Test that metadata is extracted and included."""
        test_file = tmp_path / "my-document.txt"
        test_file.write_text("Content", encoding="utf-8")

        parser = TextParser()
        doc = Document.from_file(test_file, parser)

        # Metadata should be populated
        assert doc.metadata is not None
        assert doc.metadata.title == "my-document"
        assert doc.metadata.modified is not None
        assert doc.metadata.created is not None
