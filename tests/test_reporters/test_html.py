"""Tests for HTML reporter."""

from pathlib import Path

import pytest

from doclint.core.document import Document, DocumentMetadata
from doclint.core.scanner import ScanResult
from doclint.detectors.base import Issue, IssueSeverity
from doclint.reporters.html import HTMLReporter


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        path=Path("test.txt"),
        content="Test content for the document.",
        metadata=DocumentMetadata(title="Test Document", author="Test Author"),
        file_type="txt",
        size_bytes=100,
        content_hash="abc123",
    )


@pytest.fixture
def sample_issue(sample_document):
    """Create a sample issue for testing."""
    return Issue(
        severity=IssueSeverity.WARNING,
        detector="test_detector",
        title="Test Issue",
        description="This is a test issue description",
        documents=[sample_document.path],
        details={"test_key": "test_value", "count": 42},
    )


@pytest.fixture
def sample_scan_result(sample_document, sample_issue):
    """Create a sample scan result."""
    return ScanResult(
        documents=[sample_document],
        issues={"test_detector": [sample_issue]},
        stats={"path": "/test/path", "total_chunks": 10},
    )


def test_html_reporter_initialization():
    """Test HTML reporter can be initialized."""
    reporter = HTMLReporter()
    assert reporter.name == "html"
    assert reporter.description
    assert reporter.title == "DocLint Report"


def test_html_reporter_custom_title():
    """Test HTML reporter with custom title."""
    reporter = HTMLReporter(title="Custom Report")
    assert reporter.title == "Custom Report"


def test_html_reporter_report(sample_scan_result):
    """Test HTML reporter generates valid HTML."""
    reporter = HTMLReporter()
    output = reporter.report(sample_scan_result)

    # Should be HTML
    assert output.startswith("<!DOCTYPE html>")
    assert "<html" in output
    assert "</html>" in output

    # Should have title
    assert "<title>" in output
    assert "DocLint Report" in output

    # Should have CSS
    assert "<style>" in output
    assert "</style>" in output

    # Should have issue information
    assert "Test Issue" in output
    assert "test_detector" in output


def test_html_reporter_empty_scan(sample_document):
    """Test HTML reporter with empty scan results."""
    reporter = HTMLReporter()
    result = ScanResult(documents=[sample_document], issues={}, stats={"path": "/test"})

    output = reporter.report(result)

    # Should have "no issues" message
    assert "No Issues Found" in output or "no issues found" in output.lower()


def test_html_reporter_file_output(sample_scan_result, tmp_path):
    """Test HTML reporter writes to file."""
    reporter = HTMLReporter()
    output_file = tmp_path / "report.html"

    reporter.report(sample_scan_result, output_path=output_file)

    assert output_file.exists()

    # Verify file content is valid HTML
    content = output_file.read_text(encoding="utf-8")
    assert content.startswith("<!DOCTYPE html>")
    assert "</html>" in content


def test_html_reporter_multiple_severities(sample_document):
    """Test HTML reporter with different severity levels."""
    reporter = HTMLReporter()

    critical_issue = Issue(
        severity=IssueSeverity.CRITICAL,
        detector="detector1",
        title="Critical Issue",
        description="Critical problem",
        documents=[sample_document.path],
    )

    warning_issue = Issue(
        severity=IssueSeverity.WARNING,
        detector="detector2",
        title="Warning Issue",
        description="Warning problem",
        documents=[sample_document.path],
    )

    info_issue = Issue(
        severity=IssueSeverity.INFO,
        detector="detector3",
        title="Info Issue",
        description="Info problem",
        documents=[sample_document.path],
    )

    result = ScanResult(
        documents=[sample_document],
        issues={
            "detector1": [critical_issue],
            "detector2": [warning_issue],
            "detector3": [info_issue],
        },
        stats={"path": "/test"},
    )

    output = reporter.report(result)

    # Should have all three issues
    assert "Critical Issue" in output
    assert "Warning Issue" in output
    assert "Info Issue" in output

    # Should have severity indicators
    assert "🔴" in output  # Critical
    assert "⚠️" in output  # Warning
    assert "ℹ️" in output  # Info


def test_html_reporter_escapes_html(sample_document):
    """Test that HTML characters are escaped in user content."""
    reporter = HTMLReporter()

    issue = Issue(
        severity=IssueSeverity.WARNING,
        detector="test",
        title="Issue with <script>alert('xss')</script>",
        description="Description with <b>HTML</b> tags",
        documents=[sample_document.path],
    )

    result = ScanResult(
        documents=[sample_document],
        issues={"test": [issue]},
        stats={"path": "/test"},
    )

    output = reporter.report(result)

    # User-provided HTML should be escaped - check that HTML entities appear
    # The title contains <script> which should be escaped to &lt;script&gt;
    assert "&lt;script&gt;" in output

    # The description contains <b> which should be escaped to &lt;b&gt;
    assert "&lt;b&gt;" in output

    # Raw unescaped HTML should not appear in the issue content area
    # (We allow <script> in our JavaScript section, but user content should be escaped)
    # Check the issue body section specifically (after issue-title div)
    issue_body_start = output.find('class="issue-body"')
    if issue_body_start != -1:
        issue_body = output[issue_body_start:]
        # In the issue body, raw HTML tags should not appear (only escaped)
        assert "<b>HTML</b>" not in issue_body or "&lt;b&gt;HTML&lt;/b&gt;" in issue_body


def test_html_reporter_statistics(sample_document):
    """Test HTML reporter includes statistics."""
    reporter = HTMLReporter()

    result = ScanResult(
        documents=[sample_document] * 5,  # 5 documents
        issues={
            "detector1": [
                Issue(
                    severity=IssueSeverity.CRITICAL,
                    detector="detector1",
                    title="Critical",
                    description="Critical",
                    documents=[sample_document.path],
                )
            ]
        },
        stats={"path": "/test", "total_chunks": 100},
    )

    output = reporter.report(result)

    # Should have statistics
    assert "5" in output  # Total documents
    assert "1" in output  # Critical count


def test_html_reporter_with_details(sample_document):
    """Test HTML reporter renders issue details."""
    reporter = HTMLReporter()

    issue = Issue(
        severity=IssueSeverity.WARNING,
        detector="test",
        title="Issue with details",
        description="Has details",
        documents=[sample_document.path],
        details={
            "detail_1": "value1",
            "detail_2": 42,
            "long_list": ["a", "b", "c", "d", "e", "f"],
        },
    )

    result = ScanResult(
        documents=[sample_document],
        issues={"test": [issue]},
        stats={"path": "/test"},
    )

    output = reporter.report(result)

    # Should include details
    assert "value1" in output
    assert "42" in output


def test_html_reporter_self_contained(sample_scan_result):
    """Test HTML output is self-contained (no external resources)."""
    reporter = HTMLReporter()
    output = reporter.report(sample_scan_result)

    # Should not reference external stylesheets or scripts
    assert 'rel="stylesheet"' not in output
    assert 'src="http' not in output

    # Should have embedded styles
    assert "<style>" in output
