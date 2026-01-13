"""Tests for JSON reporter."""

import json
from pathlib import Path

import pytest

from doclint.core.document import Document, DocumentMetadata
from doclint.core.scanner import ScanResult
from doclint.detectors.base import Issue, IssueSeverity
from doclint.reporters.json_reporter import JSONReporter


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


def test_json_reporter_initialization():
    """Test JSON reporter can be initialized."""
    reporter = JSONReporter()
    assert reporter.name == "json"
    assert reporter.description
    assert reporter.pretty is True


def test_json_reporter_not_pretty():
    """Test JSON reporter with pretty=False."""
    reporter = JSONReporter(pretty=False)
    assert reporter.pretty is False


def test_json_reporter_with_documents():
    """Test JSON reporter with include_documents=True."""
    reporter = JSONReporter(include_documents=True)
    assert reporter.include_documents is True


def test_json_reporter_report(sample_scan_result):
    """Test JSON reporter generates valid JSON."""
    reporter = JSONReporter()
    output = reporter.report(sample_scan_result)

    # Should be valid JSON
    data = json.loads(output)

    # Check structure
    assert "scan_info" in data
    assert "statistics" in data
    assert "issues" in data

    # Check scan_info
    assert "timestamp" in data["scan_info"]
    assert "path" in data["scan_info"]
    assert "version" in data["scan_info"]
    assert data["scan_info"]["total_documents"] == 1
    assert data["scan_info"]["total_chunks"] == 10

    # Check statistics
    assert data["statistics"]["total_issues"] == 1
    assert data["statistics"]["warning"] == 1
    assert data["statistics"]["critical"] == 0
    assert data["statistics"]["info"] == 0

    # Check issues
    assert len(data["issues"]) == 1
    issue = data["issues"][0]
    assert issue["severity"] == "warning"
    assert issue["detector"] == "test_detector"
    assert issue["title"] == "Test Issue"


def test_json_reporter_empty_scan():
    """Test JSON reporter with empty scan results."""
    reporter = JSONReporter()
    empty_result = ScanResult(documents=[], issues={}, stats={"path": "/test"})

    output = reporter.report(empty_result)
    data = json.loads(output)

    assert data["statistics"]["total_issues"] == 0
    assert data["issues"] == []


def test_json_reporter_file_output(sample_scan_result, tmp_path):
    """Test JSON reporter writes to file."""
    reporter = JSONReporter()
    output_file = tmp_path / "report.json"

    reporter.report(sample_scan_result, output_path=output_file)

    assert output_file.exists()

    # Verify file content is valid JSON
    with open(output_file) as f:
        data = json.load(f)
        assert "scan_info" in data
        assert "statistics" in data
        assert "issues" in data


def test_json_reporter_multiple_issues(sample_document):
    """Test JSON reporter with multiple issues."""
    reporter = JSONReporter()

    issues = [
        Issue(
            severity=IssueSeverity.CRITICAL,
            detector="detector1",
            title="Critical Issue",
            description="Critical problem",
            documents=[sample_document.path],
        ),
        Issue(
            severity=IssueSeverity.WARNING,
            detector="detector2",
            title="Warning Issue",
            description="Warning problem",
            documents=[sample_document.path],
        ),
    ]

    result = ScanResult(
        documents=[sample_document],
        issues={"detector1": [issues[0]], "detector2": [issues[1]]},
        stats={"path": "/test"},
    )

    output = reporter.report(result)
    data = json.loads(output)

    assert data["statistics"]["total_issues"] == 2
    assert data["statistics"]["critical"] == 1
    assert data["statistics"]["warning"] == 1
    assert len(data["issues"]) == 2


def test_json_reporter_issue_sorting(sample_document):
    """Test that issues are sorted by severity."""
    reporter = JSONReporter()

    # Create issues in reverse order
    issues = {
        "detector1": [
            Issue(
                severity=IssueSeverity.INFO,
                detector="detector1",
                title="Info",
                description="Info",
                documents=[sample_document.path],
            )
        ],
        "detector2": [
            Issue(
                severity=IssueSeverity.CRITICAL,
                detector="detector2",
                title="Critical",
                description="Critical",
                documents=[sample_document.path],
            )
        ],
        "detector3": [
            Issue(
                severity=IssueSeverity.WARNING,
                detector="detector3",
                title="Warning",
                description="Warning",
                documents=[sample_document.path],
            )
        ],
    }

    result = ScanResult(documents=[sample_document], issues=issues, stats={"path": "/test"})

    output = reporter.report(result)
    data = json.loads(output)

    # Check that critical comes first
    assert data["issues"][0]["severity"] == "critical"
    assert data["issues"][1]["severity"] == "warning"
    assert data["issues"][2]["severity"] == "info"


def test_json_reporter_with_documents_included(sample_document):
    """Test JSON reporter includes documents when requested."""
    reporter = JSONReporter(include_documents=True)

    result = ScanResult(
        documents=[sample_document],
        issues={},
        stats={"path": "/test"},
    )

    output = reporter.report(result)
    data = json.loads(output)

    assert "documents" in data
    assert len(data["documents"]) == 1
    doc = data["documents"][0]
    assert "path" in doc
    assert "file_type" in doc
    assert doc["file_type"] == "txt"
    assert "chunks" in doc


def test_json_reporter_quality_percentage(sample_document):
    """Test quality percentage calculation."""
    reporter = JSONReporter()

    # No issues = 100%
    result = ScanResult(
        documents=[sample_document],
        issues={},
        stats={"path": "/test"},
    )

    output = reporter.report(result)
    data = json.loads(output)

    assert data["statistics"]["quality_percentage"] == 100.0
    assert data["statistics"]["clean_documents"] == 1


def test_json_reporter_not_pretty_format(sample_scan_result):
    """Test non-pretty JSON output."""
    reporter = JSONReporter(pretty=False)
    output = reporter.report(sample_scan_result)

    # Should not have indentation/newlines
    assert "\n  " not in output

    # But should still be valid JSON
    data = json.loads(output)
    assert "scan_info" in data
