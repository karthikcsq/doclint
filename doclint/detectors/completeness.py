"""Completeness detector for validating document metadata and content quality."""

import asyncio
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp

from ..cache import URLCache
from ..core.document import Document
from .base import BaseDetector, Issue, IssueSeverity


class CompletenessDetector(BaseDetector):
    """Detector for incomplete or poorly maintained documents.

    This detector validates:
    1. Metadata completeness - checks for required metadata fields
    2. Content quality - validates minimum content length
    3. Link validation - checks for broken internal file references and external URLs

    External link validation is performed asynchronously in the background to avoid
    blocking the main scan. Results are cached to minimize redundant network requests.

    Example:
        >>> detector = CompletenessDetector(
        ...     required_metadata=["author", "version"],
        ...     min_content_length=100,
        ...     check_external_links=True
        ... )
        >>> issues = await detector.detect(documents)
    """

    name = "completeness"
    description = "Validates document metadata and content quality"

    # Pattern to match markdown/html links
    MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")  # [text](url)
    HTML_LINK_PATTERN = re.compile(r'(?:href|src)=["\']([^"\']+)["\']')  # href="url" or src="url"

    def __init__(
        self,
        required_metadata: List[str] | None = None,
        min_content_length: int = 100,
        check_internal_links: bool = True,
        check_external_links: bool = False,
        base_path: Path | None = None,
        external_link_timeout: int = 5,
        external_link_cache_ttl: int = 86400,
        max_concurrent_requests: int = 10,
    ):
        """Initialize completeness detector.

        Args:
            required_metadata: List of required metadata field names.
                              Valid fields: author, created, modified, version, title
                              Default: ["author", "created"]
            min_content_length: Minimum acceptable content length in characters.
                               Default: 100
            check_internal_links: Whether to check for broken internal file links.
                                 Default: True
            check_external_links: Whether to check external URLs (http/https).
                                 Default: False (opt-in)
            base_path: Base path for resolving relative links. If None, uses document's directory.
            external_link_timeout: Timeout in seconds for external URL requests.
                                  Default: 5
            external_link_cache_ttl: Cache TTL in seconds for URL validation results.
                                    Default: 86400 (24 hours)
            max_concurrent_requests: Maximum concurrent external URL validations.
                                    Default: 10
        """
        self.required_metadata = required_metadata or ["author", "created"]
        self.min_content_length = min_content_length
        self.check_internal_links = check_internal_links
        self.check_external_links = check_external_links
        self.base_path = base_path
        self.external_link_timeout = external_link_timeout
        self.max_concurrent_requests = max_concurrent_requests

        # Initialize URL cache if external link checking is enabled
        self.url_cache: Optional[URLCache] = None
        if self.check_external_links:
            self.url_cache = URLCache(ttl=external_link_cache_ttl)

    async def detect(self, documents: List[Document]) -> List[Issue]:
        """Detect completeness issues in documents.

        Args:
            documents: List of documents to analyze

        Returns:
            List of Issue objects for documents with completeness problems
        """
        issues: List[Issue] = []

        # First pass: check metadata and content (fast, synchronous)
        for doc in documents:
            doc_issues = self._check_document(doc)
            issues.extend(doc_issues)

        # Second pass: validate external links asynchronously if enabled
        if self.check_external_links:
            external_issues = await self._validate_external_links(documents)
            issues.extend(external_issues)

        return issues

    def _check_document(self, doc: Document) -> List[Issue]:
        """Check a single document for completeness issues.

        Args:
            doc: Document to check

        Returns:
            List of issues found in this document
        """
        issues: List[Issue] = []

        # Check metadata completeness
        missing_metadata = self._check_metadata(doc)
        if missing_metadata:
            issues.append(self._create_metadata_issue(doc, missing_metadata))

        # Check content length
        if len(doc.content.strip()) < self.min_content_length:
            issues.append(self._create_content_length_issue(doc))

        # Check for broken links
        if self.check_internal_links:
            broken_links = self._find_broken_links(doc)
            if broken_links:
                issues.append(self._create_broken_links_issue(doc, broken_links))

        return issues

    def _check_metadata(self, doc: Document) -> Set[str]:
        """Check for missing required metadata fields.

        Args:
            doc: Document to check

        Returns:
            Set of missing metadata field names
        """
        missing: Set[str] = set()
        metadata = doc.metadata

        for field_name in self.required_metadata:
            field_value = getattr(metadata, field_name, None)

            # Check if field exists and has a non-empty value
            if field_value is None or (isinstance(field_value, str) and not field_value.strip()):
                missing.add(field_name)

        return missing

    def _find_broken_links(self, doc: Document) -> List[str]:
        """Find broken internal file links in document content.

        This method only checks internal/relative file links.
        External URLs are handled separately by _validate_external_links().

        Args:
            doc: Document to check

        Returns:
            List of broken internal link paths
        """
        broken: List[str] = []
        seen_links: Set[str] = set()

        # Find all markdown links: [text](url)
        for match in self.MARKDOWN_LINK_PATTERN.finditer(doc.content):
            link = match.group(2).strip()  # Group 2 is the URL
            if link and link not in seen_links:
                seen_links.add(link)
                # Only check internal links
                if not self._is_external_link(link) and self._is_broken_internal_link(doc, link):
                    broken.append(link)

        # Find all HTML links: href="url" or src="url"
        for match in self.HTML_LINK_PATTERN.finditer(doc.content):
            link = match.group(1).strip()  # Group 1 is the URL
            if link and link not in seen_links:
                seen_links.add(link)
                # Only check internal links
                if not self._is_external_link(link) and self._is_broken_internal_link(doc, link):
                    broken.append(link)

        return broken

    def _is_external_link(self, link: str) -> bool:
        """Check if a link is an external URL.

        Args:
            link: Link to check

        Returns:
            True if link is external (http/https), False otherwise
        """
        return link.startswith(("http://", "https://"))

    def _is_broken_internal_link(self, doc: Document, link: str) -> bool:
        """Check if an internal file link is broken.

        Args:
            doc: Document containing the link
            link: Internal link path to check

        Returns:
            True if link is broken, False otherwise
        """
        # Skip anchors, data URIs, javascript, mailto, ftp
        if link.startswith(("#", "data:", "javascript:", "mailto:", "ftp://")):
            return False

        # Resolve the link path
        base = self.base_path if self.base_path else doc.path.parent

        try:
            link_path = (base / link).resolve()
            # Check if the file exists
            return not link_path.exists()
        except (ValueError, OSError):
            # Invalid path
            return True

    async def _validate_external_links(self, documents: List[Document]) -> List[Issue]:
        """Validate external URLs asynchronously across all documents.

        This method extracts all external links from documents and validates them
        concurrently using async HTTP requests. Results are cached to avoid
        redundant checks.

        Args:
            documents: List of documents to scan for external links

        Returns:
            List of issues for documents with broken external links
        """
        # Extract external links from all documents
        doc_external_links: Dict[Path, Set[str]] = {}

        for doc in documents:
            external_links: Set[str] = set()

            # Find all markdown links: [text](url)
            for match in self.MARKDOWN_LINK_PATTERN.finditer(doc.content):
                link = match.group(2).strip()
                if link and self._is_external_link(link):
                    external_links.add(link)

            # Find all HTML links: href="url" or src="url"
            for match in self.HTML_LINK_PATTERN.finditer(doc.content):
                link = match.group(1).strip()
                if link and self._is_external_link(link):
                    external_links.add(link)

            if external_links:
                doc_external_links[doc.path] = external_links

        if not doc_external_links:
            return []

        # Collect all unique URLs to validate
        all_urls = set()
        for urls in doc_external_links.values():
            all_urls.update(urls)

        # Validate URLs concurrently
        validation_results = await self._check_urls_async(list(all_urls))

        # Create issues for documents with broken links
        issues: List[Issue] = []
        for doc in documents:
            if doc.path not in doc_external_links:
                continue

            broken_urls = [
                url
                for url in doc_external_links[doc.path]
                if not validation_results.get(url, False)
            ]

            if broken_urls:
                issues.append(self._create_broken_external_links_issue(doc, broken_urls))

        return issues

    async def _check_urls_async(self, urls: List[str]) -> Dict[str, bool]:
        """Check multiple URLs concurrently with rate limiting.

        Args:
            urls: List of URLs to validate

        Returns:
            Dictionary mapping URL to is_valid (True if accessible, False otherwise)
        """
        # Check cache first
        results: Dict[str, bool] = {}
        urls_to_check: List[str] = []

        assert self.url_cache is not None, "URL cache not initialized"

        for url in urls:
            cached = self.url_cache.get(url)
            if cached is not None:
                is_valid, _ = cached
                results[url] = is_valid
            else:
                urls_to_check.append(url)

        if not urls_to_check:
            return results

        # Validate uncached URLs concurrently with semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def check_with_semaphore(url: str) -> Tuple[str, bool]:
            async with semaphore:
                is_valid = await self._check_url(url)
                return (url, is_valid)

        # Execute all checks concurrently
        check_tasks = [check_with_semaphore(url) for url in urls_to_check]
        checked_results = await asyncio.gather(*check_tasks, return_exceptions=True)

        # Process results
        for result in checked_results:
            if isinstance(result, Exception):
                # Unexpected error during check
                continue
            if not isinstance(result, tuple):
                continue
            url, is_valid = result
            results[url] = is_valid

        return results

    async def _check_url(self, url: str) -> bool:
        """Check if a URL is accessible.

        Args:
            url: URL to check

        Returns:
            True if URL is accessible, False otherwise
        """
        assert self.url_cache is not None, "URL cache not initialized"

        try:
            timeout = aiohttp.ClientTimeout(total=self.external_link_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Try HEAD request first (faster)
                try:
                    async with session.head(url, allow_redirects=True) as response:
                        is_valid: bool = 200 <= response.status < 400
                        self.url_cache.set(url, is_valid, response.status)
                        return is_valid
                except aiohttp.ClientResponseError as e:
                    # Some servers don't support HEAD, try GET
                    if e.status in (405, 501):  # Method Not Allowed, Not Implemented
                        async with session.get(url, allow_redirects=True) as response:
                            is_valid_get: bool = 200 <= response.status < 400
                            self.url_cache.set(url, is_valid_get, response.status)
                            return is_valid_get
                    else:
                        # Other client errors (404, 403, etc.)
                        self.url_cache.set(url, False, e.status)
                        return False

        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            UnicodeError,
        ):
            # Network error, timeout, or invalid URL
            self.url_cache.set(url, False, None)
            return False

        except Exception:
            # Unexpected error - treat as broken
            self.url_cache.set(url, False, None)
            return False

    def _create_metadata_issue(self, doc: Document, missing: Set[str]) -> Issue:
        """Create an issue for missing metadata.

        Args:
            doc: Document with missing metadata
            missing: Set of missing field names

        Returns:
            Issue object describing the problem
        """
        missing_list = sorted(missing)
        field_str = ", ".join(missing_list)

        return Issue(
            severity=IssueSeverity.WARNING,
            detector=self.name,
            title="Missing Required Metadata",
            description=f"Document lacks important metadata fields: {field_str}",
            documents=[doc.path],
            details={
                "missing_fields": missing_list,
                "required_fields": self.required_metadata,
            },
        )

    def _create_content_length_issue(self, doc: Document) -> Issue:
        """Create an issue for insufficient content length.

        Args:
            doc: Document with insufficient content

        Returns:
            Issue object describing the problem
        """
        actual_length = len(doc.content.strip())

        # Critical if essentially empty, warning if just short
        severity = IssueSeverity.CRITICAL if actual_length < 10 else IssueSeverity.WARNING

        return Issue(
            severity=severity,
            detector=self.name,
            title="Insufficient Content",
            description=(
                f"Document content is too short ({actual_length} characters, "
                f"minimum: {self.min_content_length})"
            ),
            documents=[doc.path],
            details={
                "content_length": actual_length,
                "min_length": self.min_content_length,
            },
        )

    def _create_broken_links_issue(self, doc: Document, broken_links: List[str]) -> Issue:
        """Create an issue for broken internal links.

        Args:
            doc: Document containing broken links
            broken_links: List of broken link paths

        Returns:
            Issue object describing the problem
        """
        link_count = len(broken_links)
        link_preview = ", ".join(broken_links[:3])
        if link_count > 3:
            link_preview += f" (and {link_count - 3} more)"

        return Issue(
            severity=IssueSeverity.WARNING,
            detector=self.name,
            title="Broken Internal Links Found",
            description=(f"Document contains {link_count} broken internal link(s): {link_preview}"),
            documents=[doc.path],
            details={
                "broken_links": broken_links,
                "link_count": link_count,
                "link_type": "internal",
            },
        )

    def _create_broken_external_links_issue(self, doc: Document, broken_urls: List[str]) -> Issue:
        """Create an issue for broken external URLs.

        Args:
            doc: Document containing broken URLs
            broken_urls: List of broken external URLs

        Returns:
            Issue object describing the problem
        """
        url_count = len(broken_urls)
        url_preview = ", ".join(broken_urls[:3])
        if url_count > 3:
            url_preview += f" (and {url_count - 3} more)"

        return Issue(
            severity=IssueSeverity.WARNING,
            detector=self.name,
            title="Broken External Links Found",
            description=(f"Document contains {url_count} broken external link(s): {url_preview}"),
            documents=[doc.path],
            details={
                "broken_links": broken_urls,
                "link_count": url_count,
                "link_type": "external",
            },
        )
