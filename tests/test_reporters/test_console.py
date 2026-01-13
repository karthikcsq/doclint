"""Tests for console reporter."""

from pathlib import Path

import pytest

from doclint.core.document import Document, DocumentMetadata
from doclint.core.scanner import ScanResult
from doclint.detectors.base import Issue, IssueSeverity
from doclint.reporters.console import ConsoleReporter


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


def test_console_reporter_initialization():
    """Test console reporter can be initialized."""
    reporter = ConsoleReporter()
    assert reporter.name == "console"
    assert reporter.description
    assert reporter.console is not None


def test_console_reporter_with_no_colors():
    """Test console reporter with colors disabled."""
    reporter = ConsoleReporter(use_colors=False)
    assert reporter.console is not None


def test_console_reporter_report(sample_scan_result):
    """Test console reporter generates output."""
    reporter = ConsoleReporter()
    output = reporter.report(sample_scan_result)

    # Console reporter prints directly, returns empty string
    assert output == ""


def test_console_reporter_empty_scan():
    """Test console reporter with empty scan results."""
    reporter = ConsoleReporter()
    empty_result = ScanResult(documents=[], issues={}, stats={})

    output = reporter.report(empty_result)
    assert output == ""


def test_console_reporter_no_issues(sample_document):
    """Test console reporter with no issues found."""
    reporter = ConsoleReporter()
    result = ScanResult(documents=[sample_document], issues={}, stats={"path": "/test"})

    output = reporter.report(result)
    assert output == ""


def test_console_reporter_multiple_severities(sample_document):
    """Test console reporter with issues of different severities."""
    reporter = ConsoleReporter()

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
    assert output == ""


def test_console_reporter_group_issues_by_severity(sample_document):
    """Test issue grouping by severity."""
    reporter = ConsoleReporter()

    issues = {
        "detector1": [
            Issue(
                severity=IssueSeverity.CRITICAL,
                detector="detector1",
                title="Critical",
                description="Critical",
                documents=[sample_document.path],
            )
        ],
        "detector2": [
            Issue(
                severity=IssueSeverity.WARNING,
                detector="detector2",
                title="Warning",
                description="Warning",
                documents=[sample_document.path],
            )
        ],
    }

    grouped = reporter._group_issues_by_severity(issues)

    assert len(grouped[IssueSeverity.CRITICAL]) == 1
    assert len(grouped[IssueSeverity.WARNING]) == 1
    assert len(grouped[IssueSeverity.INFO]) == 0


def test_console_reporter_with_chunks(sample_document):
    """Test console reporter with chunk information."""
    from doclint.core.document import Chunk

    chunk = Chunk(
        text="This is a test chunk with some content.",
        index=0,
        document_path=sample_document.path,
        chunk_hash="chunk123",
    )

    issue = Issue(
        severity=IssueSeverity.WARNING,
        detector="test",
        title="Issue with chunks",
        description="Has chunks",
        documents=[sample_document.path],
        chunks=[chunk],
    )

    result = ScanResult(
        documents=[sample_document],
        issues={"test": [issue]},
        stats={"path": "/test"},
    )

    reporter = ConsoleReporter()
    output = reporter.report(result)
    assert output == ""


def test_console_reporter_file_output(sample_scan_result, tmp_path):
    """Test console reporter can write to file."""
    reporter = ConsoleReporter()
    output_file = tmp_path / "report.txt"

    reporter.report(sample_scan_result, output_path=output_file)

    # File should be created (though content may be minimal)
    assert output_file.exists()
