"""Markdown parser."""

import re
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import markdown  # type: ignore[import-untyped]

from ..core.document import DocumentMetadata
from ..core.exceptions import ParsingError
from .base import BaseParser


class MarkdownParser(BaseParser):
    """Parser for Markdown files.

    Converts Markdown to HTML, then strips HTML tags to get plain text.
    Preserves paragraph structure while removing formatting syntax.
    Extracts title from first H1 heading or frontmatter.
    """

    file_type: ClassVar[str] = "markdown"
    supported_extensions: ClassVar[list[str]] = [".md", ".markdown", ".mdown", ".mkd"]

    def parse(self, path: Path) -> str:
        """Extract text from Markdown file.

        Converts Markdown to HTML, then strips HTML tags to get plain text.
        Frontmatter (YAML/TOML blocks) is stripped before processing.

        Args:
            path: Path to markdown file

        Returns:
            Plain text content

        Raises:
            ParsingError: If parsing fails
        """
        try:
            # Read file
            with open(path, "r", encoding="utf-8") as f:
                md_content = f.read()

            # Strip frontmatter (YAML/TOML blocks)
            md_content = self._strip_frontmatter(md_content)

            # Convert to HTML
            html = markdown.markdown(md_content, extensions=["extra", "codehilite", "nl2br"])

            # Strip HTML tags to get plain text
            text = self._html_to_text(html)

            return self.clean_text(text)

        except Exception as e:
            raise ParsingError(
                f"Failed to parse Markdown file {path}: {e}",
                path=str(path),
                original_error=e,
            )

    def extract_metadata(self, path: Path) -> DocumentMetadata:
        """Extract metadata from Markdown file.

        Tries to extract:
        - Title from first H1 heading or frontmatter
        - Tags from frontmatter (future enhancement)
        - Dates from file system

        Args:
            path: Path to markdown file

        Returns:
            DocumentMetadata
        """
        metadata = DocumentMetadata()

        try:
            # File system dates
            stat = path.stat()
            metadata.modified = datetime.fromtimestamp(stat.st_mtime)
            metadata.created = datetime.fromtimestamp(stat.st_ctime)

            # Read content
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract title from first H1
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            if title_match:
                metadata.title = title_match.group(1).strip()
            else:
                # Fall back to filename
                metadata.title = path.stem

            # TODO: Parse YAML frontmatter for more metadata

        except Exception:
            # Best-effort
            pass

        return metadata

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        """Remove YAML/TOML frontmatter from Markdown.

        Frontmatter is typically:
        ---
        title: My Document
        tags: [foo, bar]
        ---
        """
        # Remove YAML frontmatter (---...---)
        content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)

        # Remove TOML frontmatter (+++...+++)
        content = re.sub(r"^\+\+\+\s*\n.*?\n\+\+\+\s*\n", "", content, flags=re.DOTALL)

        return content

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Convert HTML to plain text.

        Simple tag stripping with newline preservation for block elements.
        """
        # Remove script and style elements
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)

        # Replace block-level tags with newlines
        html = re.sub(r"</?(p|div|br|h[1-6]|li|tr)[^>]*>", "\n", html)

        # Remove all other tags
        html = re.sub(r"<[^>]+>", "", html)

        # Decode HTML entities
        html = html.replace("&nbsp;", " ")
        html = html.replace("&lt;", "<")
        html = html.replace("&gt;", ">")
        html = html.replace("&amp;", "&")
        html = html.replace("&quot;", '"')
        html = html.replace("&#39;", "'")

        return html
