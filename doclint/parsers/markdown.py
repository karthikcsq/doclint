"""Markdown parser."""

import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, Tuple

import markdown
import yaml

# TOML support (tomllib in 3.11+, tomli for 3.10)
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

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
        Frontmatter (YAML/TOML blocks) is parsed and removed before processing.

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

            # Parse and remove frontmatter (YAML/TOML blocks)
            md_content, _ = self._parse_frontmatter(md_content)

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
        - Title from frontmatter, first H1 heading, or filename (in that order)
        - Author from frontmatter
        - Tags from frontmatter
        - Dates from frontmatter or file system
        - Custom fields from frontmatter

        Args:
            path: Path to markdown file

        Returns:
            DocumentMetadata
        """
        metadata = DocumentMetadata()

        try:
            # File system dates (default)
            stat = path.stat()
            metadata.modified = datetime.fromtimestamp(stat.st_mtime)
            metadata.created = datetime.fromtimestamp(stat.st_ctime)

            # Read content
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse frontmatter
            content_without_fm, frontmatter = self._parse_frontmatter(content)

            # Extract metadata from frontmatter (highest priority)
            if frontmatter:
                # Title
                if "title" in frontmatter:
                    metadata.title = str(frontmatter["title"])

                # Author
                if "author" in frontmatter:
                    metadata.author = str(frontmatter["author"])

                # Tags (can be list or comma-separated string)
                if "tags" in frontmatter:
                    tags = frontmatter["tags"]
                    if isinstance(tags, list):
                        metadata.tags = [str(t) for t in tags]
                    elif isinstance(tags, str):
                        metadata.tags = [t.strip() for t in tags.split(",")]

                # Date (created/published)
                if "date" in frontmatter:
                    date_val = frontmatter["date"]
                    if isinstance(date_val, datetime):
                        metadata.created = date_val
                    elif isinstance(date_val, date):
                        # YAML parses "2024-01-15" as date, not datetime
                        metadata.created = datetime.combine(date_val, datetime.min.time())
                    elif isinstance(date_val, str):
                        # Try to parse date string
                        try:
                            metadata.created = datetime.fromisoformat(
                                date_val.replace("Z", "+00:00")
                            )
                        except (ValueError, AttributeError):
                            pass

                # Modified date
                if "modified" in frontmatter or "updated" in frontmatter:
                    date_val = frontmatter.get("modified") or frontmatter.get("updated")
                    if isinstance(date_val, datetime):
                        metadata.modified = date_val
                    elif isinstance(date_val, date):
                        # YAML parses "2024-01-15" as date, not datetime
                        metadata.modified = datetime.combine(date_val, datetime.min.time())
                    elif isinstance(date_val, str):
                        try:
                            metadata.modified = datetime.fromisoformat(
                                date_val.replace("Z", "+00:00")
                            )
                        except (ValueError, AttributeError):
                            pass

                # Store other frontmatter fields in custom
                for key, value in frontmatter.items():
                    if key not in ["title", "author", "tags", "date", "modified", "updated"]:
                        metadata.custom[key] = value

            # If no title from frontmatter, try H1 heading
            if not metadata.title:
                title_match = re.search(r"^#\s+(.+)$", content_without_fm, re.MULTILINE)
                if title_match:
                    metadata.title = title_match.group(1).strip()
                else:
                    # Fall back to filename
                    metadata.title = path.stem

        except Exception:
            # Best-effort
            pass

        return metadata

    @staticmethod
    def _parse_frontmatter(content: str) -> Tuple[str, Dict[str, Any]]:
        """Parse and remove YAML/TOML frontmatter from Markdown.

        Frontmatter is typically:
        ---
        title: My Document
        tags: [foo, bar]
        date: 2024-01-01
        ---

        Args:
            content: Raw markdown content

        Returns:
            Tuple of (content_without_frontmatter, frontmatter_dict)
        """
        # Try YAML frontmatter (---...---)
        yaml_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, flags=re.DOTALL)
        if yaml_match:
            try:
                frontmatter = yaml.safe_load(yaml_match.group(1))
                content_without_fm = content[yaml_match.end() :]
                # Ensure frontmatter is a dict
                if isinstance(frontmatter, dict):
                    return content_without_fm, frontmatter
            except yaml.YAMLError:
                # Malformed YAML, just strip it
                pass

        # Try TOML frontmatter (+++...+++)
        toml_match = re.match(r"^\+\+\+\s*\n(.*?)\n\+\+\+\s*\n", content, flags=re.DOTALL)
        if toml_match:
            try:
                # Parse TOML frontmatter
                toml_str = toml_match.group(1)
                frontmatter = tomllib.loads(toml_str)
                content_without_fm = content[toml_match.end() :]
                # Ensure frontmatter is a dict
                if isinstance(frontmatter, dict):
                    return content_without_fm, frontmatter
            except Exception:
                # Malformed TOML, just strip it
                pass

        # No frontmatter found
        return content, {}

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
