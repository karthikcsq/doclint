"""HTML parser."""

from datetime import datetime
from pathlib import Path
from typing import ClassVar

from bs4 import BeautifulSoup

from ..core.document import DocumentMetadata
from ..core.exceptions import ParsingError
from .base import BaseParser


class HTMLParser(BaseParser):
    """Parser for HTML files.

    Uses BeautifulSoup to parse HTML and extract text content.
    Removes scripts, styles, and navigation elements.
    Extracts metadata from HTML meta tags.
    """

    file_type: ClassVar[str] = "html"
    supported_extensions: ClassVar[list[str]] = [".html", ".htm", ".xhtml"]

    def parse(self, path: Path) -> str:
        """Extract text from HTML file.

        Uses BeautifulSoup to parse HTML and extract text content.
        Removes scripts, styles, and navigation elements.

        Args:
            path: Path to HTML file

        Returns:
            Plain text content

        Raises:
            ParsingError: If parsing fails
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, "lxml")

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()

            # Extract text
            text = soup.get_text(separator="\n", strip=True)

            return self.clean_text(text)

        except UnicodeDecodeError:
            # Try different encoding
            try:
                with open(path, "r", encoding="latin-1") as f:
                    html_content = f.read()
                soup = BeautifulSoup(html_content, "lxml")
                for element in soup(["script", "style", "nav", "header", "footer"]):
                    element.decompose()
                text = soup.get_text(separator="\n", strip=True)
                return self.clean_text(text)
            except Exception as e:
                raise ParsingError(
                    f"Failed to parse HTML {path}: {e}",
                    path=str(path),
                    original_error=e,
                )
        except Exception as e:
            raise ParsingError(
                f"Failed to parse HTML {path}: {e}", path=str(path), original_error=e
            )

    def extract_metadata(self, path: Path) -> DocumentMetadata:
        """Extract metadata from HTML file.

        Tries to extract:
        - Title from <title> tag
        - Author from <meta name="author">
        - Description from <meta name="description">
        - Keywords as tags from <meta name="keywords">

        Args:
            path: Path to HTML file

        Returns:
            DocumentMetadata
        """
        metadata = DocumentMetadata()

        try:
            # File system dates
            stat = path.stat()
            metadata.modified = datetime.fromtimestamp(stat.st_mtime)
            metadata.created = datetime.fromtimestamp(stat.st_ctime)

            # Parse HTML
            with open(path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "lxml")

            # Extract title
            if soup.title and soup.title.string:
                metadata.title = soup.title.string.strip()

            # Extract meta tags
            author_meta = soup.find("meta", attrs={"name": "author"})
            if author_meta and author_meta.get("content"):
                metadata.author = str(author_meta["content"])

            description_meta = soup.find("meta", attrs={"name": "description"})
            if description_meta and description_meta.get("content"):
                metadata.custom["description"] = str(description_meta["content"])

            # Keywords as tags
            keywords_meta = soup.find("meta", attrs={"name": "keywords"})
            if keywords_meta and keywords_meta.get("content"):
                keywords = str(keywords_meta["content"]).split(",")
                metadata.tags = [k.strip() for k in keywords]

        except Exception:
            # Best-effort
            pass

        return metadata
