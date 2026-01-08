"""Integration tests for end-to-end document parsing workflows."""

from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfWriter

from doclint.core.document import Document
from doclint.parsers.registry import ParserRegistry


class TestEndToEndParsing:
    """Test end-to-end document parsing workflows."""

    def test_parse_text_file_end_to_end(self, tmp_path: Path) -> None:
        """Test parsing text file from start to finish."""
        # Create sample text file
        test_file = tmp_path / "sample.txt"
        content = "This is a test document.\n\nIt has multiple paragraphs."
        test_file.write_text(content, encoding="utf-8")

        # Parse using registry
        registry = ParserRegistry()
        parser = registry.get_parser(test_file)

        # Create Document object
        doc = Document.from_file(test_file, parser)

        # Verify Document properties
        assert doc.path == test_file
        assert doc.file_type == "text"
        assert "test document" in doc.content
        assert "multiple paragraphs" in doc.content
        assert doc.size_bytes > 0
        assert len(doc.content_hash) == 64  # SHA256
        assert doc.metadata is not None
        assert doc.metadata.title == "sample"
        assert doc.metadata.modified is not None

    def test_parse_markdown_file_end_to_end(self, tmp_path: Path) -> None:
        """Test parsing markdown file from start to finish."""
        # Create sample markdown file
        test_file = tmp_path / "sample.md"
        content = """---
title: Test Document
author: Test Author
tags: [test, markdown]
---

# Test Heading

This is **bold** and *italic* text.
"""
        test_file.write_text(content, encoding="utf-8")

        # Parse using registry
        registry = ParserRegistry()
        parser = registry.get_parser(test_file)

        # Create Document object
        doc = Document.from_file(test_file, parser)

        # Verify Document properties
        assert doc.path == test_file
        assert doc.file_type == "markdown"
        assert "Test Heading" in doc.content
        assert "bold" in doc.content
        assert "italic" in doc.content
        assert doc.metadata.title == "Test Document"
        assert doc.metadata.author == "Test Author"
        assert "test" in doc.metadata.tags
        assert "markdown" in doc.metadata.tags

    def test_parse_html_file_end_to_end(self, tmp_path: Path) -> None:
        """Test parsing HTML file from start to finish."""
        # Create sample HTML file
        test_file = tmp_path / "sample.html"
        content = """
        <html>
        <head>
            <title>Test Page</title>
            <meta name="author" content="Test Author">
        </head>
        <body>
            <h1>Welcome</h1>
            <p>This is a test page.</p>
        </body>
        </html>
        """
        test_file.write_text(content, encoding="utf-8")

        # Parse using registry
        registry = ParserRegistry()
        parser = registry.get_parser(test_file)

        # Create Document object
        doc = Document.from_file(test_file, parser)

        # Verify Document properties
        assert doc.path == test_file
        assert doc.file_type == "html"
        assert "Welcome" in doc.content
        assert "test page" in doc.content
        assert doc.metadata.title == "Test Page"
        assert doc.metadata.author == "Test Author"

    def test_parse_pdf_file_end_to_end(self, tmp_path: Path) -> None:
        """Test parsing PDF file from start to finish."""
        # Create sample PDF file
        test_file = tmp_path / "sample.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.add_metadata({"/Title": "Test PDF", "/Author": "Test Author"})

        with open(test_file, "wb") as f:
            writer.write(f)

        # Parse using registry
        registry = ParserRegistry()
        parser = registry.get_parser(test_file)

        # Create Document object
        doc = Document.from_file(test_file, parser)

        # Verify Document properties
        assert doc.path == test_file
        assert doc.file_type == "pdf"
        assert isinstance(doc.content, str)
        assert doc.metadata.title == "Test PDF"
        assert doc.metadata.author == "Test Author"

    def test_parse_docx_file_end_to_end(self, tmp_path: Path) -> None:
        """Test parsing DOCX file from start to finish."""
        # Create sample DOCX file
        test_file = tmp_path / "sample.docx"
        docx_doc = DocxDocument()
        docx_doc.add_paragraph("This is a test document.")
        docx_doc.core_properties.title = "Test DOCX"
        docx_doc.core_properties.author = "Test Author"
        docx_doc.save(str(test_file))

        # Parse using registry
        registry = ParserRegistry()
        parser = registry.get_parser(test_file)

        # Create Document object
        doc = Document.from_file(test_file, parser)

        # Verify Document properties
        assert doc.path == test_file
        assert doc.file_type == "docx"
        assert "test document" in doc.content
        assert doc.metadata.title == "Test DOCX"
        assert doc.metadata.author == "Test Author"

    def test_parse_multiple_files_sequentially(self, tmp_path: Path) -> None:
        """Test parsing multiple file types in sequence."""
        # Create multiple files
        txt_file = tmp_path / "file1.txt"
        txt_file.write_text("Text content", encoding="utf-8")

        md_file = tmp_path / "file2.md"
        md_file.write_text("# Markdown content", encoding="utf-8")

        html_file = tmp_path / "file3.html"
        html_file.write_text("<html><body>HTML content</body></html>", encoding="utf-8")

        # Parse all files
        registry = ParserRegistry()
        documents = []

        for file_path in [txt_file, md_file, html_file]:
            parser = registry.get_parser(file_path)
            doc = Document.from_file(file_path, parser)
            documents.append(doc)

        # Verify all documents
        assert len(documents) == 3
        assert documents[0].file_type == "text"
        assert documents[1].file_type == "markdown"
        assert documents[2].file_type == "html"

    def test_content_hash_uniqueness(self, tmp_path: Path) -> None:
        """Test that different files have different content hashes."""
        # Create two different files
        file1 = tmp_path / "file1.txt"
        file1.write_text("Content A", encoding="utf-8")

        file2 = tmp_path / "file2.txt"
        file2.write_text("Content B", encoding="utf-8")

        # Parse both files
        registry = ParserRegistry()
        parser = registry.get_parser(file1)

        doc1 = Document.from_file(file1, parser)
        doc2 = Document.from_file(file2, parser)

        # Verify different hashes
        assert doc1.content_hash != doc2.content_hash

    def test_content_hash_stability(self, tmp_path: Path) -> None:
        """Test that same file produces same hash."""
        # Create file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Stable content", encoding="utf-8")

        # Parse twice
        registry = ParserRegistry()
        parser = registry.get_parser(test_file)

        doc1 = Document.from_file(test_file, parser)
        doc2 = Document.from_file(test_file, parser)

        # Verify same hash
        assert doc1.content_hash == doc2.content_hash

    def test_registry_selects_correct_parser_automatically(self, tmp_path: Path) -> None:
        """Test that registry automatically selects correct parser."""
        # Create files with different extensions
        files = {
            "test.txt": "text",
            "test.md": "markdown",
            "test.html": "html",
        }

        registry = ParserRegistry()

        for filename, expected_type in files.items():
            file_path = tmp_path / filename
            file_path.write_text("test content", encoding="utf-8")

            parser = registry.get_parser(file_path)
            doc = Document.from_file(file_path, parser)

            assert doc.file_type == expected_type
