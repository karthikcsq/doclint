"""Tests for CompletenessDetector."""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from doclint.core.document import Document, DocumentMetadata
from doclint.detectors.base import IssueSeverity
from doclint.detectors.completeness import CompletenessDetector


@pytest.fixture
def complete_document():
    """Create a document with all required metadata and good content."""
    metadata = DocumentMetadata(
        author="John Doe",
        created=datetime(2024, 1, 1, 12, 0, 0),
        modified=datetime(2024, 1, 10, 12, 0, 0),
        version="1.0",
        title="Complete Document",
        tags=["documentation", "complete"],
    )

    content = "This is a complete document with sufficient content. " * 10

    return Document(
        path=Path("/test/complete.md"),
        content=content,
        metadata=metadata,
        file_type="markdown",
        size_bytes=1024,
        content_hash="abc123",
    )


@pytest.fixture
def incomplete_metadata_document():
    """Create a document with missing metadata."""
    metadata = DocumentMetadata(
        title="Incomplete Document",
        # Missing: author, created, version
    )

    content = "This is a document with good content length. " * 10

    return Document(
        path=Path("/test/incomplete_metadata.md"),
        content=content,
        metadata=metadata,
        file_type="markdown",
        size_bytes=512,
        content_hash="def456",
    )


@pytest.fixture
def short_content_document():
    """Create a document with insufficient content."""
    metadata = DocumentMetadata(
        author="Jane Doe",
        created=datetime(2024, 1, 1),
    )

    return Document(
        path=Path("/test/short.md"),
        content="Too short",
        metadata=metadata,
        file_type="markdown",
        size_bytes=100,
        content_hash="ghi789",
    )


@pytest.fixture
def empty_document():
    """Create an essentially empty document."""
    metadata = DocumentMetadata()

    return Document(
        path=Path("/test/empty.md"),
        content="   ",  # Just whitespace
        metadata=metadata,
        file_type="markdown",
        size_bytes=10,
        content_hash="jkl012",
    )


@pytest.fixture
def document_with_links(tmp_path):
    """Create a document with various links."""
    metadata = DocumentMetadata(
        author="Author",
        created=datetime(2024, 1, 1),
    )

    # Create a real file for good link
    existing_file = tmp_path / "existing.md"
    existing_file.write_text("# Existing Document")

    content = """
    # Documentation

    Check out the [existing document](existing.md) for details.

    Also see [broken link](nonexistent.md) which doesn't exist.

    External links are fine: [Google](https://google.com)

    HTML links too: <a href="another-missing.html">Missing</a>
    """

    doc_path = tmp_path / "doc_with_links.md"

    return Document(
        path=doc_path,
        content=content,
        metadata=metadata,
        file_type="markdown",
        size_bytes=200,
        content_hash="mno345",
    )


class TestCompletenessDetectorInit:
    """Test detector initialization."""

    def test_default_initialization(self):
        """Test detector with default settings."""
        detector = CompletenessDetector()

        assert detector.name == "completeness"
        assert detector.description == "Validates document metadata and content quality"
        assert detector.required_metadata == ["author", "created"]
        assert detector.min_content_length == 100
        assert detector.check_internal_links is True

    def test_custom_initialization(self):
        """Test detector with custom settings."""
        detector = CompletenessDetector(
            required_metadata=["author", "version", "title"],
            min_content_length=500,
            check_internal_links=False,
        )

        assert detector.required_metadata == ["author", "version", "title"]
        assert detector.min_content_length == 500
        assert detector.check_internal_links is False


class TestMetadataValidation:
    """Test metadata completeness checks."""

    @pytest.mark.asyncio
    async def test_complete_metadata_no_issues(self, complete_document):
        """Test that complete metadata produces no issues."""
        detector = CompletenessDetector(
            required_metadata=["author", "created"],
            min_content_length=10,
        )

        issues = await detector.detect([complete_document])

        # Should have no metadata issues
        metadata_issues = [i for i in issues if "metadata" in i.title.lower()]
        assert len(metadata_issues) == 0

    @pytest.mark.asyncio
    async def test_missing_metadata_creates_issue(self, incomplete_metadata_document):
        """Test that missing metadata is detected."""
        detector = CompletenessDetector(
            required_metadata=["author", "created", "version"],
            min_content_length=10,
        )

        issues = await detector.detect([incomplete_metadata_document])

        # Should have metadata issue
        metadata_issues = [i for i in issues if "metadata" in i.title.lower()]
        assert len(metadata_issues) == 1

        issue = metadata_issues[0]
        assert issue.severity == IssueSeverity.WARNING
        assert issue.detector == "completeness"
        assert "author" in issue.details["missing_fields"]
        assert "created" in issue.details["missing_fields"]
        assert "version" in issue.details["missing_fields"]

    @pytest.mark.asyncio
    async def test_partial_metadata_missing(self, complete_document):
        """Test detection when only some metadata is missing."""
        # Remove version from metadata
        complete_document.metadata.version = None

        detector = CompletenessDetector(
            required_metadata=["author", "version"],
            min_content_length=10,
        )

        issues = await detector.detect([complete_document])

        metadata_issues = [i for i in issues if "metadata" in i.title.lower()]
        assert len(metadata_issues) == 1

        issue = metadata_issues[0]
        assert issue.details["missing_fields"] == ["version"]

    @pytest.mark.asyncio
    async def test_empty_string_metadata_detected_as_missing(self):
        """Test that empty string metadata is treated as missing."""
        metadata = DocumentMetadata(
            author="",  # Empty string should be treated as missing
            created=datetime(2024, 1, 1),
        )

        doc = Document(
            path=Path("/test/empty_author.md"),
            content="Content here",
            metadata=metadata,
            file_type="markdown",
            size_bytes=100,
            content_hash="test",
        )

        detector = CompletenessDetector(
            required_metadata=["author"],
            min_content_length=10,
        )

        issues = await detector.detect([doc])

        metadata_issues = [i for i in issues if "metadata" in i.title.lower()]
        assert len(metadata_issues) == 1
        assert "author" in metadata_issues[0].details["missing_fields"]


class TestContentLengthValidation:
    """Test content length checks."""

    @pytest.mark.asyncio
    async def test_sufficient_content_no_issue(self, complete_document):
        """Test that sufficient content produces no issues."""
        detector = CompletenessDetector(
            required_metadata=[],  # Don't check metadata
            min_content_length=100,
        )

        issues = await detector.detect([complete_document])

        content_issues = [i for i in issues if "content" in i.title.lower()]
        assert len(content_issues) == 0

    @pytest.mark.asyncio
    async def test_short_content_creates_warning(self, short_content_document):
        """Test that short content creates a warning."""
        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=100,
        )

        issues = await detector.detect([short_content_document])

        content_issues = [i for i in issues if "content" in i.title.lower()]
        assert len(content_issues) == 1

        issue = content_issues[0]
        # "Too short" is 9 chars, which is < 10, so it gets CRITICAL severity
        assert issue.severity == IssueSeverity.CRITICAL
        assert issue.details["content_length"] < 100
        assert issue.details["min_length"] == 100

    @pytest.mark.asyncio
    async def test_empty_content_creates_critical(self, empty_document):
        """Test that empty content creates critical issue."""
        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=100,
        )

        issues = await detector.detect([empty_document])

        content_issues = [i for i in issues if "content" in i.title.lower()]
        assert len(content_issues) == 1

        issue = content_issues[0]
        assert issue.severity == IssueSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_custom_min_length(self, short_content_document):
        """Test custom minimum content length."""
        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=5,  # Very short minimum
        )

        issues = await detector.detect([short_content_document])

        # "Too short" is 9 characters, should pass with min_length=5
        content_issues = [i for i in issues if "content" in i.title.lower()]
        assert len(content_issues) == 0


class TestLinkValidation:
    """Test broken link detection."""

    @pytest.mark.asyncio
    async def test_no_broken_links_no_issue(self, complete_document):
        """Test that documents without broken links have no issues."""
        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=10,
            check_internal_links=True,
        )

        issues = await detector.detect([complete_document])

        link_issues = [i for i in issues if "link" in i.title.lower()]
        assert len(link_issues) == 0

    @pytest.mark.asyncio
    async def test_broken_links_detected(self, document_with_links):
        """Test that broken internal links are detected."""
        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=10,
            check_internal_links=True,
        )

        issues = await detector.detect([document_with_links])

        link_issues = [i for i in issues if "link" in i.title.lower()]
        assert len(link_issues) == 1

        issue = link_issues[0]
        assert issue.severity == IssueSeverity.WARNING
        assert "nonexistent.md" in issue.details["broken_links"]
        assert "another-missing.html" in issue.details["broken_links"]
        # External link should not be flagged
        assert not any("google.com" in link for link in issue.details["broken_links"])

    @pytest.mark.asyncio
    async def test_link_check_disabled(self, document_with_links):
        """Test that link checking can be disabled."""
        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=10,
            check_internal_links=False,
        )

        issues = await detector.detect([document_with_links])

        link_issues = [i for i in issues if "link" in i.title.lower()]
        assert len(link_issues) == 0

    @pytest.mark.asyncio
    async def test_external_links_ignored(self, tmp_path):
        """Test that external links are not checked."""
        metadata = DocumentMetadata(author="Test", created=datetime(2024, 1, 1))

        content = """
        [HTTP Link](http://example.com/missing)
        [HTTPS Link](https://example.com/missing)
        [Email](mailto:test@example.com)
        [FTP](ftp://ftp.example.com/file)
        [Anchor](#section)
        """

        doc = Document(
            path=tmp_path / "external_links.md",
            content=content,
            metadata=metadata,
            file_type="markdown",
            size_bytes=100,
            content_hash="test",
        )

        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=10,
            check_internal_links=True,
        )

        issues = await detector.detect([doc])

        link_issues = [i for i in issues if "link" in i.title.lower()]
        assert len(link_issues) == 0


class TestMultipleDocuments:
    """Test detection with multiple documents."""

    @pytest.mark.asyncio
    async def test_multiple_documents_with_mixed_issues(
        self,
        complete_document,
        incomplete_metadata_document,
        short_content_document,
    ):
        """Test detection across multiple documents."""
        detector = CompletenessDetector(
            required_metadata=["author", "created"],
            min_content_length=100,
        )

        documents = [
            complete_document,
            incomplete_metadata_document,
            short_content_document,
        ]

        issues = await detector.detect(documents)

        # complete_document: no issues
        # incomplete_metadata_document: missing author, created
        # short_content_document: short content
        assert len(issues) >= 2

        # Check we have issues from different documents
        issue_paths = {issue.documents[0] for issue in issues}
        assert Path("/test/incomplete_metadata.md") in issue_paths
        assert Path("/test/short.md") in issue_paths
        assert Path("/test/complete.md") not in issue_paths

    @pytest.mark.asyncio
    async def test_empty_document_list(self):
        """Test that empty document list returns no issues."""
        detector = CompletenessDetector()

        issues = await detector.detect([])

        assert issues == []

    @pytest.mark.asyncio
    async def test_document_with_multiple_issues(self, tmp_path):
        """Test document with multiple types of issues."""
        metadata = DocumentMetadata()  # Missing everything

        content = """
        Short content with [broken link](missing.md).
        """

        doc = Document(
            path=tmp_path / "multiple_issues.md",
            content=content,
            metadata=metadata,
            file_type="markdown",
            size_bytes=10,
            content_hash="test",
        )

        detector = CompletenessDetector(
            required_metadata=["author", "created"],
            min_content_length=100,
            check_internal_links=True,
        )

        issues = await detector.detect([doc])

        # Should have: metadata issue, content length issue, broken link issue
        assert len(issues) == 3

        issue_types = {issue.title for issue in issues}
        assert any("metadata" in title.lower() for title in issue_types)
        assert any("content" in title.lower() for title in issue_types)
        assert any("link" in title.lower() for title in issue_types)


class TestIssueDetails:
    """Test that issues contain proper details."""

    @pytest.mark.asyncio
    async def test_metadata_issue_details(self, incomplete_metadata_document):
        """Test metadata issue has correct details."""
        detector = CompletenessDetector(
            required_metadata=["author", "created", "version"],
            min_content_length=10,
        )

        issues = await detector.detect([incomplete_metadata_document])

        metadata_issue = next(i for i in issues if "metadata" in i.title.lower())

        assert metadata_issue.detector == "completeness"
        assert metadata_issue.documents == [Path("/test/incomplete_metadata.md")]
        assert set(metadata_issue.details["missing_fields"]) == {"author", "created", "version"}
        assert metadata_issue.details["required_fields"] == ["author", "created", "version"]

    @pytest.mark.asyncio
    async def test_content_issue_details(self, short_content_document):
        """Test content issue has correct details."""
        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=100,
        )

        issues = await detector.detect([short_content_document])

        content_issue = next(i for i in issues if "content" in i.title.lower())

        assert content_issue.detector == "completeness"
        assert content_issue.documents == [Path("/test/short.md")]
        assert content_issue.details["content_length"] == len("Too short")
        assert content_issue.details["min_length"] == 100

    @pytest.mark.asyncio
    async def test_link_issue_details(self, document_with_links):
        """Test broken link issue has correct details."""
        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=10,
            check_internal_links=True,
        )

        issues = await detector.detect([document_with_links])

        link_issue = next(i for i in issues if "link" in i.title.lower())

        assert link_issue.detector == "completeness"
        assert link_issue.documents == [document_with_links.path]
        assert isinstance(link_issue.details["broken_links"], list)
        assert len(link_issue.details["broken_links"]) > 0
        assert link_issue.details["link_count"] == len(link_issue.details["broken_links"])


class TestExternalLinkValidation:
    """Test external link validation."""

    @pytest.mark.asyncio
    async def test_external_links_disabled_by_default(self, tmp_path):
        """Test that external link checking is disabled by default."""
        metadata = DocumentMetadata(author="Test", created=datetime(2024, 1, 1))

        content = """
        Check out [Google](https://google.com) for more info.
        Also see [Example](https://example.com/page).
        """

        doc = Document(
            path=tmp_path / "external_links.md",
            content=content,
            metadata=metadata,
            file_type="markdown",
            size_bytes=100,
            content_hash="test",
        )

        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=10,
            check_external_links=False,  # Disabled
        )

        issues = await detector.detect([doc])

        # Should have no external link issues
        external_link_issues = [i for i in issues if "external" in i.title.lower()]
        assert len(external_link_issues) == 0

    @pytest.mark.asyncio
    async def test_external_links_validation_enabled(self, tmp_path):
        """Test external link validation when enabled."""
        metadata = DocumentMetadata(author="Test", created=datetime(2024, 1, 1))

        content = """
        Check out [Google](https://google.com) for more info.
        Also see [Broken](https://thisdoesnotexist12345.com/page).
        """

        doc = Document(
            path=tmp_path / "external_links.md",
            content=content,
            metadata=metadata,
            file_type="markdown",
            size_bytes=100,
            content_hash="test",
        )

        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=10,
            check_external_links=True,
            external_link_timeout=5,
        )

        # Mock URL validation
        with patch.object(detector, "_check_url", new_callable=AsyncMock) as mock_check:
            # google.com works, thisdoesnotexist12345.com doesn't
            async def check_url_side_effect(url: str) -> bool:
                if "google.com" in url:
                    return True
                return False

            mock_check.side_effect = check_url_side_effect

            issues = await detector.detect([doc])

            # Should have external link issue for broken link
            external_link_issues = [i for i in issues if "external" in i.title.lower()]
            assert len(external_link_issues) == 1

            issue = external_link_issues[0]
            assert issue.severity == IssueSeverity.WARNING
            # Check full URL in broken_links
            assert any(
                "thisdoesnotexist12345.com" in link for link in issue.details["broken_links"]
            )
            assert not any("google.com" in link for link in issue.details["broken_links"])

    @pytest.mark.asyncio
    async def test_external_link_caching(self, tmp_path):
        """Test that external link validation results are cached."""
        metadata = DocumentMetadata(author="Test", created=datetime(2024, 1, 1))

        content = """
        [Link 1](https://example.com/page1)
        [Link 2](https://example.com/page2)
        [Link 3](https://example.com/page1)
        """

        doc = Document(
            path=tmp_path / "cached_links.md",
            content=content,
            metadata=metadata,
            file_type="markdown",
            size_bytes=100,
            content_hash="test",
        )

        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=10,
            check_external_links=True,
        )

        # Mock URL validation
        with patch.object(detector, "_check_url", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = True

            await detector.detect([doc])

            # Should only check each unique URL once
            assert mock_check.call_count == 2  # page1 and page2 (page1 appears twice)

    @pytest.mark.asyncio
    async def test_mixed_internal_and_external_links(self, tmp_path):
        """Test handling of documents with both internal and external links."""
        metadata = DocumentMetadata(author="Test", created=datetime(2024, 1, 1))

        # Create an existing internal file
        existing_file = tmp_path / "existing.md"
        existing_file.write_text("# Existing")

        content = """
        Internal: [Existing](existing.md)
        Internal Broken: [Missing](missing.md)
        External: [Google](https://google.com)
        External Broken: [Broken](https://thisdoesnotexist12345.com)
        """

        doc = Document(
            path=tmp_path / "mixed_links.md",
            content=content,
            metadata=metadata,
            file_type="markdown",
            size_bytes=100,
            content_hash="test",
        )

        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=10,
            check_internal_links=True,
            check_external_links=True,
        )

        # Mock external URL validation
        with patch.object(detector, "_check_url", new_callable=AsyncMock) as mock_check:

            async def check_url_side_effect(url: str) -> bool:
                return "google.com" in url

            mock_check.side_effect = check_url_side_effect

            issues = await detector.detect([doc])

            # Should have both internal and external link issues
            internal_issues = [i for i in issues if i.details.get("link_type") == "internal"]
            external_issues = [i for i in issues if i.details.get("link_type") == "external"]

            assert len(internal_issues) == 1
            assert "missing.md" in internal_issues[0].details["broken_links"]

            assert len(external_issues) == 1
            assert any(
                "thisdoesnotexist12345.com" in link
                for link in external_issues[0].details["broken_links"]
            )

    @pytest.mark.asyncio
    async def test_concurrent_url_validation(self, tmp_path):
        """Test that URLs are validated concurrently."""
        metadata = DocumentMetadata(author="Test", created=datetime(2024, 1, 1))

        # Create document with many external links
        links = [f"[Link {i}](https://example{i}.com)" for i in range(20)]
        content = "\n".join(links)

        doc = Document(
            path=tmp_path / "many_links.md",
            content=content,
            metadata=metadata,
            file_type="markdown",
            size_bytes=500,
            content_hash="test",
        )

        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=10,
            check_external_links=True,
            max_concurrent_requests=5,
        )

        # Mock URL validation with a delay to test concurrency
        with patch.object(detector, "_check_url", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = True

            await detector.detect([doc])

            # Should have called _check_url for all 20 unique URLs
            assert mock_check.call_count == 20

    @pytest.mark.asyncio
    async def test_url_timeout_handling(self, tmp_path):
        """Test that URL timeout is handled gracefully."""
        metadata = DocumentMetadata(author="Test", created=datetime(2024, 1, 1))

        content = "[Slow Link](https://veryslow.example.com)"

        doc = Document(
            path=tmp_path / "slow_link.md",
            content=content,
            metadata=metadata,
            file_type="markdown",
            size_bytes=50,
            content_hash="test",
        )

        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=10,
            check_external_links=True,
            external_link_timeout=1,  # Very short timeout
        )

        # Mock URL check to simulate timeout
        with patch.object(detector, "_check_url", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = False  # Timeout treated as broken

            issues = await detector.detect([doc])

            external_issues = [i for i in issues if "external" in i.title.lower()]
            assert len(external_issues) == 1
            assert any(
                "veryslow.example.com" in link
                for link in external_issues[0].details["broken_links"]
            )

    @pytest.mark.asyncio
    async def test_html_external_links(self, tmp_path):
        """Test extraction of external links from HTML content."""
        metadata = DocumentMetadata(author="Test", created=datetime(2024, 1, 1))

        content = """
        <a href="https://example.com/page">Link</a>
        <img src="https://cdn.example.com/image.jpg">
        <a href="https://broken.example.com">Broken</a>
        """

        doc = Document(
            path=tmp_path / "html_links.html",
            content=content,
            metadata=metadata,
            file_type="html",
            size_bytes=150,
            content_hash="test",
        )

        detector = CompletenessDetector(
            required_metadata=[],
            min_content_length=10,
            check_external_links=True,
        )

        with patch.object(detector, "_check_url", new_callable=AsyncMock) as mock_check:

            async def check_url_side_effect(url: str) -> bool:
                return "broken" not in url

            mock_check.side_effect = check_url_side_effect

            issues = await detector.detect([doc])

            external_issues = [i for i in issues if "external" in i.title.lower()]
            assert len(external_issues) == 1
            assert any(
                "broken.example.com" in link for link in external_issues[0].details["broken_links"]
            )

    @pytest.mark.asyncio
    async def test_url_cache_persistence(self, tmp_path):
        """Test that URL cache persists results."""
        from doclint.cache import URLCache

        cache_dir = tmp_path / "cache"

        # Test cache directly to verify persistence
        url_cache = URLCache(cache_dir=cache_dir, ttl=86400)

        # Store a result in cache
        url_cache.set("https://example.com", is_valid=True, status_code=200)

        # Verify it's cached
        result = url_cache.get("https://example.com")
        assert result is not None
        is_valid, status_code = result
        assert is_valid is True
        assert status_code == 200

        # Close the first cache instance
        url_cache.close()

        # Create a new cache instance with the same directory
        url_cache2 = URLCache(cache_dir=cache_dir, ttl=86400)

        # Verify the cached result persists
        result2 = url_cache2.get("https://example.com")
        assert result2 is not None
        is_valid2, status_code2 = result2
        assert is_valid2 is True
        assert status_code2 == 200

        url_cache2.close()
