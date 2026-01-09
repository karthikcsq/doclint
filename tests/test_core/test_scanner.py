"""Tests for Scanner orchestration."""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np

from doclint.core.config import DocLintConfig
from doclint.core.document import Document
from doclint.core.scanner import Issue, Scanner, ScanResult
from doclint.parsers.registry import ParserRegistry


class TestIssue:
    """Tests for Issue dataclass."""

    def test_issue_creation(self) -> None:
        """Test creating an issue."""
        issue = Issue(
            issue_type="conflict",
            severity="warning",
            message="Conflicting information found",
            document_path=Path("/test/doc.md"),
            details={"similarity": 0.95},
        )

        assert issue.issue_type == "conflict"
        assert issue.severity == "warning"
        assert issue.message == "Conflicting information found"
        assert issue.document_path == Path("/test/doc.md")
        assert issue.details == {"similarity": 0.95}

    def test_issue_default_details(self) -> None:
        """Test issue with default empty details."""
        issue = Issue(
            issue_type="completeness",
            severity="info",
            message="Document is missing metadata",
            document_path=Path("/test/incomplete.txt"),
        )

        assert issue.details == {}


class TestScanResult:
    """Tests for ScanResult dataclass."""

    def test_scan_result_creation(self) -> None:
        """Test creating a scan result."""
        result = ScanResult(
            documents=[],
            issues={},
            stats={"path": "/test"},
        )

        assert result.documents == []
        assert result.issues == {}
        assert result.stats == {"path": "/test"}

    def test_total_issues(self) -> None:
        """Test total_issues property."""
        issues = {
            "conflict": [
                Issue(
                    issue_type="conflict",
                    severity="warning",
                    message="Test",
                    document_path=Path("/test"),
                ),
                Issue(
                    issue_type="conflict",
                    severity="error",
                    message="Test 2",
                    document_path=Path("/test2"),
                ),
            ],
            "completeness": [
                Issue(
                    issue_type="completeness",
                    severity="info",
                    message="Test",
                    document_path=Path("/test"),
                ),
            ],
        }

        result = ScanResult(documents=[], issues=issues)
        assert result.total_issues == 3

    def test_has_issues(self) -> None:
        """Test has_issues property."""
        result_empty = ScanResult(documents=[], issues={})
        assert result_empty.has_issues is False

        result_with_issues = ScanResult(
            documents=[],
            issues={
                "conflict": [
                    Issue(
                        issue_type="conflict",
                        severity="warning",
                        message="Test",
                        document_path=Path("/test"),
                    )
                ]
            },
        )
        assert result_with_issues.has_issues is True


class TestScanner:
    """Tests for Scanner class."""

    def test_scanner_initialization(self) -> None:
        """Test scanner initialization with defaults."""
        scanner = Scanner()

        assert scanner.config is not None
        assert isinstance(scanner.config, DocLintConfig)
        assert scanner.parser_registry is not None
        assert isinstance(scanner.parser_registry, ParserRegistry)

    def test_scanner_with_custom_config(self) -> None:
        """Test scanner with custom configuration."""
        config = DocLintConfig(recursive=False, max_workers=2)
        scanner = Scanner(config=config)

        assert scanner.config.recursive is False
        assert scanner.config.max_workers == 2

    def test_discover_files_empty_directory(self, tmp_path: Path) -> None:
        """Test file discovery in empty directory."""
        scanner = Scanner()
        files = scanner.discover_files(tmp_path)

        assert files == []

    def test_discover_files_with_supported_files(self, tmp_path: Path) -> None:
        """Test file discovery with supported file types."""
        # Create test files
        (tmp_path / "doc1.txt").write_text("Content 1")
        (tmp_path / "doc2.md").write_text("# Title\nContent")
        (tmp_path / "ignored.xyz").write_text("Ignored")

        scanner = Scanner()
        files = scanner.discover_files(tmp_path)

        assert len(files) == 2
        assert any(f.name == "doc1.txt" for f in files)
        assert any(f.name == "doc2.md" for f in files)

    def test_discover_files_recursive(self, tmp_path: Path) -> None:
        """Test recursive file discovery."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("Root content")
        (subdir / "nested.txt").write_text("Nested content")

        scanner = Scanner()
        files = scanner.discover_files(tmp_path, recursive=True)

        assert len(files) == 2

    def test_discover_files_non_recursive(self, tmp_path: Path) -> None:
        """Test non-recursive file discovery."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("Root content")
        (subdir / "nested.txt").write_text("Nested content")

        scanner = Scanner()
        files = scanner.discover_files(tmp_path, recursive=False)

        assert len(files) == 1
        assert files[0].name == "root.txt"

    def test_discover_single_file(self, tmp_path: Path) -> None:
        """Test file discovery with single file path."""
        test_file = tmp_path / "single.txt"
        test_file.write_text("Content")

        scanner = Scanner()
        files = scanner.discover_files(test_file)

        assert len(files) == 1
        assert files[0] == test_file

    def test_parse_documents(self, tmp_path: Path) -> None:
        """Test document parsing."""
        # Create test files
        (tmp_path / "doc1.txt").write_text("Content 1")
        (tmp_path / "doc2.txt").write_text("Content 2")

        scanner = Scanner()
        files = scanner.discover_files(tmp_path)
        documents = scanner.parse_documents(files)

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)

    def test_parse_documents_with_callback(self, tmp_path: Path) -> None:
        """Test document parsing with progress callback."""
        (tmp_path / "doc.txt").write_text("Content")

        scanner = Scanner()
        files = scanner.discover_files(tmp_path)

        progress_calls: List[tuple[int, int]] = []

        def callback(current: int, total: int) -> None:
            progress_calls.append((current, total))

        scanner.parse_documents(files, progress_callback=callback)

        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == progress_calls[-1][1]  # Final call is complete

    def test_parse_documents_handles_errors(self, tmp_path: Path) -> None:
        """Test that parsing errors don't stop processing."""
        # Create a valid file
        (tmp_path / "valid.txt").write_text("Valid content")

        scanner = Scanner()
        files = [tmp_path / "valid.txt", tmp_path / "nonexistent.txt"]
        documents = scanner.parse_documents(files)

        # Should still get the valid document
        assert len(documents) == 1

    def test_generate_embeddings(self, tmp_path: Path) -> None:
        """Test embedding generation."""
        # Setup mock generator
        mock_generator = MagicMock()
        mock_generator.get_embedding_dimension.return_value = 384
        mock_generator.generate_batch.return_value = [
            np.zeros(384, dtype=np.float32) for _ in range(3)
        ]

        # Create test document
        (tmp_path / "doc.txt").write_text("This is test content for embedding generation.")

        config = DocLintConfig(cache_enabled=False)
        scanner = Scanner(config=config, embedding_generator=mock_generator)

        files = scanner.discover_files(tmp_path)
        documents = scanner.parse_documents(files)
        scanner.generate_embeddings(documents)

        # Documents should have chunks
        assert all(len(doc.chunks) > 0 for doc in documents)

    def test_run_detectors_empty_registry(self, tmp_path: Path) -> None:
        """Test running detectors with empty registry."""
        scanner = Scanner()
        documents: List[Document] = []

        issues = scanner.run_detectors(documents)
        assert issues == {}

    def test_run_detectors_with_mock_detector(self) -> None:
        """Test running detectors with a mock detector."""
        mock_detector = MagicMock()
        mock_detector.enabled = True
        mock_detector.detect.return_value = [
            Issue(
                issue_type="test",
                severity="info",
                message="Test issue",
                document_path=Path("/test"),
            )
        ]

        scanner = Scanner(detector_registry={"test": mock_detector})
        issues = scanner.run_detectors([])

        assert "test" in issues
        assert len(issues["test"]) == 1

    def test_run_detectors_disabled_detector(self) -> None:
        """Test that disabled detectors are skipped."""
        mock_detector = MagicMock()
        mock_detector.enabled = False

        scanner = Scanner(detector_registry={"disabled": mock_detector})
        issues = scanner.run_detectors([])

        # Disabled detector should not be called
        mock_detector.detect.assert_not_called()
        assert "disabled" not in issues

    def test_scan_directory_full_workflow(self, tmp_path: Path) -> None:
        """Test full scan workflow."""
        # Setup mock generator
        mock_generator = MagicMock()
        mock_generator.get_embedding_dimension.return_value = 384
        mock_generator.generate_batch.return_value = [
            np.zeros(384, dtype=np.float32) for _ in range(5)
        ]

        # Create test files
        (tmp_path / "doc1.txt").write_text("First document content for testing.")
        (tmp_path / "doc2.md").write_text("# Second Document\n\nContent here.")

        config = DocLintConfig(cache_enabled=False)
        scanner = Scanner(config=config, embedding_generator=mock_generator)

        result = scanner.scan_directory(tmp_path)

        assert isinstance(result, ScanResult)
        assert len(result.documents) == 2
        assert "files_discovered" in result.stats
        assert "documents_parsed" in result.stats

    def test_scan_directory_empty(self, tmp_path: Path) -> None:
        """Test scanning empty directory."""
        scanner = Scanner()
        result = scanner.scan_directory(tmp_path)

        assert len(result.documents) == 0
        assert result.stats["files_discovered"] == 0

    def test_scanner_context_manager(self) -> None:
        """Test scanner as context manager."""
        with Scanner() as scanner:
            assert isinstance(scanner, Scanner)
        # Should not raise

    def test_scanner_close(self) -> None:
        """Test scanner close method."""
        scanner = Scanner()
        scanner.close()  # Should not raise

    def test_embedding_generator_lazy_loading(self) -> None:
        """Test that embedding generator is lazily loaded."""
        scanner = Scanner()

        # Should not have loaded generator yet
        assert scanner._embedding_generator is None

        # Access would trigger loading (but we don't actually load in test)
        # Just verify the property accessor exists
        assert hasattr(scanner, "embedding_generator")

    def test_cache_manager_lazy_loading(self) -> None:
        """Test that cache manager is lazily loaded."""
        config = DocLintConfig(cache_enabled=True)
        scanner = Scanner(config=config)

        # Should not have loaded cache yet
        assert scanner._cache_manager is None

    def test_cache_manager_disabled(self) -> None:
        """Test that cache manager is None when disabled."""
        config = DocLintConfig(cache_enabled=False)
        scanner = Scanner(config=config)

        # Access the property
        cache = scanner.cache_manager
        assert cache is None
