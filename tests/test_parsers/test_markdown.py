"""Tests for Markdown parser."""

from pathlib import Path

from doclint.core.document import DocumentMetadata
from doclint.parsers.markdown import MarkdownParser


class TestMarkdownParser:
    """Test suite for MarkdownParser."""

    def test_can_parse_markdown_files(self) -> None:
        """Test parser recognizes markdown extensions."""
        parser = MarkdownParser()
        assert parser.can_parse(Path("test.md")) is True
        assert parser.can_parse(Path("test.markdown")) is True
        assert parser.can_parse(Path("test.mdown")) is True
        assert parser.can_parse(Path("test.mkd")) is True

    def test_cannot_parse_other_files(self) -> None:
        """Test parser rejects non-markdown files."""
        parser = MarkdownParser()
        assert parser.can_parse(Path("test.txt")) is False
        assert parser.can_parse(Path("test.pdf")) is False

    def test_parse_simple_markdown(self, tmp_path: Path) -> None:
        """Test parsing simple markdown."""
        test_file = tmp_path / "test.md"
        content = """# Hello World

This is a **test** document.

- Item 1
- Item 2
"""
        test_file.write_text(content, encoding="utf-8")

        parser = MarkdownParser()
        result = parser.parse(test_file)

        assert "Hello World" in result
        assert "test document" in result
        assert "Item 1" in result

    def test_parse_strips_markdown_syntax(self, tmp_path: Path) -> None:
        """Test that markdown syntax is converted to plain text."""
        test_file = tmp_path / "test.md"
        content = "**bold** and *italic* and `code`"
        test_file.write_text(content, encoding="utf-8")

        parser = MarkdownParser()
        result = parser.parse(test_file)

        # Should have text without markdown syntax
        assert "bold" in result
        assert "italic" in result
        assert "code" in result
        # Markdown syntax should be removed
        assert "**" not in result
        assert "`" not in result

    def test_parse_with_yaml_frontmatter(self, tmp_path: Path) -> None:
        """Test parsing markdown with YAML frontmatter."""
        test_file = tmp_path / "test.md"
        content = """---
title: My Document
tags: [foo, bar]
---

# Content

This is the actual content.
"""
        test_file.write_text(content, encoding="utf-8")

        parser = MarkdownParser()
        result = parser.parse(test_file)

        # Frontmatter should be stripped
        assert "title:" not in result
        assert "tags:" not in result

        # Content should be present
        assert "Content" in result
        assert "actual content" in result

    def test_parse_with_toml_frontmatter(self, tmp_path: Path) -> None:
        """Test parsing markdown with TOML frontmatter."""
        test_file = tmp_path / "test.md"
        content = """+++
title = "My Document"
tags = ["foo", "bar"]
+++

# Content

This is the actual content.
"""
        test_file.write_text(content, encoding="utf-8")

        parser = MarkdownParser()
        result = parser.parse(test_file)

        # Frontmatter should be stripped
        assert "+++" not in result
        assert 'title = "My Document"' not in result

        # Content should be present
        assert "Content" in result
        assert "actual content" in result

    def test_extract_metadata_from_heading(self, tmp_path: Path) -> None:
        """Test metadata extraction from H1 heading."""
        test_file = tmp_path / "test.md"
        content = "# My Document Title\n\nSome content."
        test_file.write_text(content, encoding="utf-8")

        parser = MarkdownParser()
        metadata = parser.extract_metadata(test_file)

        assert metadata.title == "My Document Title"
        assert metadata.modified is not None

    def test_extract_metadata_fallback_to_filename(self, tmp_path: Path) -> None:
        """Test metadata falls back to filename if no H1."""
        test_file = tmp_path / "my-document.md"
        content = "Just some content without heading."
        test_file.write_text(content, encoding="utf-8")

        parser = MarkdownParser()
        metadata = parser.extract_metadata(test_file)

        assert metadata.title == "my-document"

    def test_extract_metadata_best_effort(self) -> None:
        """Test metadata extraction doesn't fail on error."""
        parser = MarkdownParser()

        # Non-existent file should return empty metadata (not raise)
        metadata = parser.extract_metadata(Path("/nonexistent/file.md"))

        assert isinstance(metadata, DocumentMetadata)

    def test_file_type_is_markdown(self) -> None:
        """Test parser has correct file_type."""
        parser = MarkdownParser()
        assert parser.file_type == "markdown"

    def test_supported_extensions(self) -> None:
        """Test parser lists supported extensions."""
        parser = MarkdownParser()
        assert ".md" in parser.supported_extensions
        assert ".markdown" in parser.supported_extensions
        assert ".mdown" in parser.supported_extensions
        assert ".mkd" in parser.supported_extensions

    def test_parse_with_code_blocks(self, tmp_path: Path) -> None:
        """Test parsing markdown with code blocks."""
        test_file = tmp_path / "test.md"
        content = """# Test

Some text.

```python
def hello():
    print("world")
```

More text.
"""
        test_file.write_text(content, encoding="utf-8")

        parser = MarkdownParser()
        result = parser.parse(test_file)

        # Should include code content
        assert "hello" in result
        assert "world" in result
        assert "Some text" in result
        assert "More text" in result

    def test_extract_metadata_from_yaml_frontmatter(self, tmp_path: Path) -> None:
        """Test metadata extraction from YAML frontmatter."""
        test_file = tmp_path / "test.md"
        content = """---
title: Frontmatter Document
author: Jane Doe
tags:
  - python
  - testing
date: 2024-01-15
category: tutorial
---

# Content

Some content here.
"""
        test_file.write_text(content, encoding="utf-8")

        parser = MarkdownParser()
        metadata = parser.extract_metadata(test_file)

        # Should extract frontmatter metadata
        assert metadata.title == "Frontmatter Document"
        assert metadata.author == "Jane Doe"
        assert "python" in metadata.tags
        assert "testing" in metadata.tags
        assert metadata.created is not None
        assert metadata.created.year == 2024
        assert metadata.created.month == 1
        assert metadata.created.day == 15
        assert metadata.custom.get("category") == "tutorial"

    def test_extract_metadata_with_comma_separated_tags(self, tmp_path: Path) -> None:
        """Test metadata extraction with comma-separated tags string."""
        test_file = tmp_path / "test.md"
        content = """---
title: Test Doc
tags: python, testing, automation
---

Content.
"""
        test_file.write_text(content, encoding="utf-8")

        parser = MarkdownParser()
        metadata = parser.extract_metadata(test_file)

        assert metadata.title == "Test Doc"
        assert "python" in metadata.tags
        assert "testing" in metadata.tags
        assert "automation" in metadata.tags

    def test_extract_metadata_frontmatter_overrides_h1(self, tmp_path: Path) -> None:
        """Test that frontmatter title takes precedence over H1."""
        test_file = tmp_path / "test.md"
        content = """---
title: Frontmatter Title
---

# H1 Title

Content.
"""
        test_file.write_text(content, encoding="utf-8")

        parser = MarkdownParser()
        metadata = parser.extract_metadata(test_file)

        # Frontmatter title should win
        assert metadata.title == "Frontmatter Title"

    def test_extract_metadata_with_modified_date(self, tmp_path: Path) -> None:
        """Test metadata extraction with modified/updated date."""
        test_file = tmp_path / "test.md"
        content = """---
title: Test
date: 2024-01-01
modified: 2024-02-15
---

Content.
"""
        test_file.write_text(content, encoding="utf-8")

        parser = MarkdownParser()
        metadata = parser.extract_metadata(test_file)

        assert metadata.created is not None
        assert metadata.created.year == 2024
        assert metadata.created.month == 1
        assert metadata.modified is not None
        assert metadata.modified.year == 2024
        assert metadata.modified.month == 2

    def test_extract_metadata_malformed_frontmatter(self, tmp_path: Path) -> None:
        """Test that malformed frontmatter doesn't break parsing."""
        test_file = tmp_path / "test.md"
        content = """---
title: Test
invalid yaml here: [unmatched
---

# Fallback Title

Content.
"""
        test_file.write_text(content, encoding="utf-8")

        parser = MarkdownParser()
        metadata = parser.extract_metadata(test_file)

        # Should not crash, returns metadata (possibly empty or with fallbacks)
        assert isinstance(metadata, DocumentMetadata)

    def test_parse_frontmatter_with_datetime_objects(self, tmp_path: Path) -> None:
        """Test that YAML datetime objects are handled correctly."""
        test_file = tmp_path / "test.md"
        content = """---
title: DateTime Test
date: 2024-01-15T10:30:00Z
---

Content.
"""
        test_file.write_text(content, encoding="utf-8")

        parser = MarkdownParser()
        metadata = parser.extract_metadata(test_file)

        assert metadata.title == "DateTime Test"
        assert metadata.created is not None
