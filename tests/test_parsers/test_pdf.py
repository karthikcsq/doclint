"""Tests for PDF parser."""

from pathlib import Path

import pytest
from pypdf import PdfWriter

from doclint.core.document import DocumentMetadata
from doclint.core.exceptions import ParsingError
from doclint.parsers.pdf import PDFParser


class TestPDFParser:
    """Test suite for PDFParser."""

    def test_can_parse_pdf_files(self) -> None:
        """Test parser recognizes PDF extension."""
        parser = PDFParser()
        assert parser.can_parse(Path("test.pdf")) is True

    def test_cannot_parse_other_files(self) -> None:
        """Test parser rejects non-PDF files."""
        parser = PDFParser()
        assert parser.can_parse(Path("test.txt")) is False
        assert parser.can_parse(Path("test.docx")) is False

    def test_parse_simple_pdf(self, tmp_path: Path) -> None:
        """Test parsing a simple PDF file."""
        # Create a simple PDF with pypdf
        test_file = tmp_path / "test.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)

        with open(test_file, "wb") as f:
            writer.write(f)

        parser = PDFParser()
        # Even blank PDF should parse without error
        result = parser.parse(test_file)
        assert isinstance(result, str)

    def test_parse_nonexistent_file(self) -> None:
        """Test parsing non-existent file raises error."""
        parser = PDFParser()

        with pytest.raises(ParsingError) as exc_info:
            parser.parse(Path("/nonexistent/file.pdf"))

        assert "Failed to parse" in str(exc_info.value)

    def test_parse_encrypted_pdf_raises_error(self, tmp_path: Path) -> None:
        """Test that encrypted PDF raises ParsingError."""
        # Create an encrypted PDF
        test_file = tmp_path / "encrypted.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)

        # Encrypt the PDF
        writer.encrypt(user_password="password", owner_password="owner")

        with open(test_file, "wb") as f:
            writer.write(f)

        parser = PDFParser()

        with pytest.raises(ParsingError) as exc_info:
            parser.parse(test_file)

        assert "encrypted" in str(exc_info.value).lower()

    def test_extract_metadata(self, tmp_path: Path) -> None:
        """Test metadata extraction from PDF."""
        test_file = tmp_path / "test.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)

        # Add metadata
        writer.add_metadata(
            {
                "/Title": "Test Document",
                "/Author": "John Doe",
                "/Subject": "Testing",
            }
        )

        with open(test_file, "wb") as f:
            writer.write(f)

        parser = PDFParser()
        metadata = parser.extract_metadata(test_file)

        assert metadata.title == "Test Document"
        assert metadata.author == "John Doe"
        assert metadata.custom.get("subject") == "Testing"

    def test_extract_metadata_fallback_to_filename(self, tmp_path: Path) -> None:
        """Test metadata falls back to filename if no title."""
        test_file = tmp_path / "my-document.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)

        with open(test_file, "wb") as f:
            writer.write(f)

        parser = PDFParser()
        metadata = parser.extract_metadata(test_file)

        assert metadata.title == "my-document"

    def test_extract_metadata_best_effort(self) -> None:
        """Test metadata extraction doesn't fail on error."""
        parser = PDFParser()

        # Non-existent file should return empty metadata (not raise)
        metadata = parser.extract_metadata(Path("/nonexistent/file.pdf"))

        assert isinstance(metadata, DocumentMetadata)

    def test_file_type_is_pdf(self) -> None:
        """Test parser has correct file_type."""
        parser = PDFParser()
        assert parser.file_type == "pdf"

    def test_supported_extensions(self) -> None:
        """Test parser lists supported extensions."""
        parser = PDFParser()
        assert ".pdf" in parser.supported_extensions

    def test_parse_invalid_pdf(self, tmp_path: Path) -> None:
        """Test parsing invalid PDF file raises error."""
        test_file = tmp_path / "invalid.pdf"
        test_file.write_text("Not a PDF file", encoding="utf-8")

        parser = PDFParser()

        with pytest.raises(ParsingError):
            parser.parse(test_file)
