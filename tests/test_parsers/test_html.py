"""Tests for HTML parser."""

from pathlib import Path

from doclint.core.document import DocumentMetadata
from doclint.parsers.html import HTMLParser


class TestHTMLParser:
    """Test suite for HTMLParser."""

    def test_can_parse_html_files(self) -> None:
        """Test parser recognizes HTML extensions."""
        parser = HTMLParser()
        assert parser.can_parse(Path("test.html")) is True
        assert parser.can_parse(Path("test.htm")) is True
        assert parser.can_parse(Path("test.xhtml")) is True

    def test_cannot_parse_other_files(self) -> None:
        """Test parser rejects non-HTML files."""
        parser = HTMLParser()
        assert parser.can_parse(Path("test.txt")) is False
        assert parser.can_parse(Path("test.md")) is False

    def test_parse_simple_html(self, tmp_path: Path) -> None:
        """Test parsing simple HTML."""
        test_file = tmp_path / "test.html"
        content = """
        <html>
        <head><title>Test</title></head>
        <body>
            <h1>Hello World</h1>
            <p>This is a test.</p>
        </body>
        </html>
        """
        test_file.write_text(content, encoding="utf-8")

        parser = HTMLParser()
        result = parser.parse(test_file)

        assert "Hello World" in result
        assert "This is a test" in result

    def test_parse_removes_scripts(self, tmp_path: Path) -> None:
        """Test that script tags are removed."""
        test_file = tmp_path / "test.html"
        content = """
        <html>
        <body>
            <p>Content</p>
            <script>alert('remove me');</script>
        </body>
        </html>
        """
        test_file.write_text(content, encoding="utf-8")

        parser = HTMLParser()
        result = parser.parse(test_file)

        assert "Content" in result
        assert "alert" not in result
        assert "remove me" not in result

    def test_parse_removes_styles(self, tmp_path: Path) -> None:
        """Test that style tags are removed."""
        test_file = tmp_path / "test.html"
        content = """
        <html>
        <head>
            <style>body { color: red; }</style>
        </head>
        <body>
            <p>Content</p>
        </body>
        </html>
        """
        test_file.write_text(content, encoding="utf-8")

        parser = HTMLParser()
        result = parser.parse(test_file)

        assert "Content" in result
        assert "color: red" not in result

    def test_parse_removes_navigation(self, tmp_path: Path) -> None:
        """Test that navigation elements are removed."""
        test_file = tmp_path / "test.html"
        content = """
        <html>
        <body>
            <nav><a href="#">Navigation</a></nav>
            <header>Header Content</header>
            <main><p>Main Content</p></main>
            <footer>Footer Content</footer>
        </body>
        </html>
        """
        test_file.write_text(content, encoding="utf-8")

        parser = HTMLParser()
        result = parser.parse(test_file)

        assert "Main Content" in result
        assert "Navigation" not in result
        assert "Header Content" not in result
        assert "Footer Content" not in result

    def test_extract_metadata_from_tags(self, tmp_path: Path) -> None:
        """Test metadata extraction from HTML meta tags."""
        test_file = tmp_path / "test.html"
        content = """
        <html>
        <head>
            <title>My Document</title>
            <meta name="author" content="John Doe">
            <meta name="keywords" content="test, html, parser">
            <meta name="description" content="A test document">
        </head>
        <body>Content</body>
        </html>
        """
        test_file.write_text(content, encoding="utf-8")

        parser = HTMLParser()
        metadata = parser.extract_metadata(test_file)

        assert metadata.title == "My Document"
        assert metadata.author == "John Doe"
        assert "test" in metadata.tags
        assert "html" in metadata.tags
        assert "parser" in metadata.tags
        assert metadata.custom.get("description") == "A test document"

    def test_extract_metadata_best_effort(self) -> None:
        """Test metadata extraction doesn't fail on error."""
        parser = HTMLParser()

        # Non-existent file should return empty metadata (not raise)
        metadata = parser.extract_metadata(Path("/nonexistent/file.html"))

        assert isinstance(metadata, DocumentMetadata)

    def test_extract_metadata_no_title(self, tmp_path: Path) -> None:
        """Test metadata extraction when no title tag exists."""
        test_file = tmp_path / "test.html"
        content = "<html><body>Content</body></html>"
        test_file.write_text(content, encoding="utf-8")

        parser = HTMLParser()
        metadata = parser.extract_metadata(test_file)

        # Should have filesystem dates but no title
        assert metadata.modified is not None
        assert metadata.title is None

    def test_file_type_is_html(self) -> None:
        """Test parser has correct file_type."""
        parser = HTMLParser()
        assert parser.file_type == "html"

    def test_supported_extensions(self) -> None:
        """Test parser lists supported extensions."""
        parser = HTMLParser()
        assert ".html" in parser.supported_extensions
        assert ".htm" in parser.supported_extensions
        assert ".xhtml" in parser.supported_extensions

    def test_parse_with_nested_tags(self, tmp_path: Path) -> None:
        """Test parsing HTML with nested tags."""
        test_file = tmp_path / "test.html"
        content = """
        <html>
        <body>
            <div>
                <p>Outer <span>nested <strong>deeply</strong> text</span> paragraph</p>
            </div>
        </body>
        </html>
        """
        test_file.write_text(content, encoding="utf-8")

        parser = HTMLParser()
        result = parser.parse(test_file)

        # Should extract all text regardless of nesting
        assert "Outer" in result
        assert "nested" in result
        assert "deeply" in result
        assert "paragraph" in result
