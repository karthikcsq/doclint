"""Tests for DetectorRegistry."""

from datetime import datetime
from pathlib import Path

import pytest

from doclint.core.document import Document, DocumentMetadata
from doclint.detectors.base import BaseDetector, Issue, IssueSeverity
from doclint.detectors.completeness import CompletenessDetector
from doclint.detectors.registry import DetectorRegistry


class MockDetector(BaseDetector):
    """Mock detector for testing."""

    name = "mock"
    description = "Mock detector for testing"

    def __init__(self, issues_to_return=None):
        """Initialize with specific issues to return."""
        self.issues_to_return = issues_to_return or []
        self.detect_called = False
        self.detect_call_count = 0

    async def detect(self, documents):
        """Return predefined issues."""
        self.detect_called = True
        self.detect_call_count += 1
        return self.issues_to_return


class ErrorDetector(BaseDetector):
    """Detector that raises an error."""

    name = "error"
    description = "Detector that raises errors"

    async def detect(self, documents):
        """Raise an error."""
        raise RuntimeError("Detector error for testing")


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    metadata = DocumentMetadata(
        author="Test Author",
        created=datetime(2024, 1, 1),
    )

    return Document(
        path=Path("/test/sample.md"),
        content="Sample content for testing",
        metadata=metadata,
        file_type="markdown",
        size_bytes=100,
        content_hash="abc123",
    )


class TestRegistryInit:
    """Test registry initialization."""

    def test_empty_registry(self):
        """Test creating an empty registry."""
        registry = DetectorRegistry()

        assert len(registry) == 0
        assert registry.list_detector_names() == []
        assert registry.get_all_detectors() == {}

    def test_repr(self):
        """Test string representation."""
        registry = DetectorRegistry()
        assert "0 detectors" in repr(registry)


class TestRegistration:
    """Test detector registration."""

    def test_register_detector(self):
        """Test registering a detector."""
        registry = DetectorRegistry()
        detector = MockDetector()

        registry.register(detector)

        assert len(registry) == 1
        assert "mock" in registry
        assert registry.has_detector("mock")
        assert registry.get_detector("mock") is detector

    def test_register_multiple_detectors(self):
        """Test registering multiple detectors."""
        registry = DetectorRegistry()
        detector1 = MockDetector()
        detector2 = CompletenessDetector()

        registry.register(detector1)
        registry.register(detector2)

        assert len(registry) == 2
        assert "mock" in registry
        assert "completeness" in registry
        assert set(registry.list_detector_names()) == {"mock", "completeness"}

    def test_register_overwrites_existing(self):
        """Test that re-registering overwrites previous detector."""
        registry = DetectorRegistry()

        detector1 = MockDetector()
        registry.register(detector1)

        detector2 = MockDetector()
        registry.register(detector2)

        assert len(registry) == 1
        assert registry.get_detector("mock") is detector2

    def test_register_empty_name_raises_error(self):
        """Test that registering detector with empty name raises error."""
        registry = DetectorRegistry()

        class BadDetector(BaseDetector):
            name = ""
            description = "Bad detector"

            async def detect(self, documents):
                return []

        with pytest.raises(ValueError, match="non-empty name"):
            registry.register(BadDetector())


class TestRetrieval:
    """Test detector retrieval."""

    def test_get_detector_by_name(self):
        """Test getting detector by name."""
        registry = DetectorRegistry()
        detector = MockDetector()
        registry.register(detector)

        retrieved = registry.get_detector("mock")

        assert retrieved is detector

    def test_get_nonexistent_detector_returns_none(self):
        """Test that getting nonexistent detector returns None."""
        registry = DetectorRegistry()

        assert registry.get_detector("nonexistent") is None

    def test_get_all_detectors(self):
        """Test getting all detectors."""
        registry = DetectorRegistry()
        detector1 = MockDetector()
        detector2 = CompletenessDetector()

        registry.register(detector1)
        registry.register(detector2)

        all_detectors = registry.get_all_detectors()

        assert len(all_detectors) == 2
        assert all_detectors["mock"] is detector1
        assert all_detectors["completeness"] is detector2

    def test_list_detector_names(self):
        """Test listing detector names."""
        registry = DetectorRegistry()
        registry.register(MockDetector())
        registry.register(CompletenessDetector())

        names = registry.list_detector_names()

        assert set(names) == {"mock", "completeness"}

    def test_has_detector(self):
        """Test checking if detector exists."""
        registry = DetectorRegistry()
        registry.register(MockDetector())

        assert registry.has_detector("mock")
        assert not registry.has_detector("nonexistent")

    def test_contains_operator(self):
        """Test 'in' operator."""
        registry = DetectorRegistry()
        registry.register(MockDetector())

        assert "mock" in registry
        assert "nonexistent" not in registry


class TestUnregistration:
    """Test detector unregistration."""

    def test_unregister_detector(self):
        """Test unregistering a detector."""
        registry = DetectorRegistry()
        detector = MockDetector()
        registry.register(detector)

        result = registry.unregister("mock")

        assert result is True
        assert len(registry) == 0
        assert "mock" not in registry

    def test_unregister_nonexistent_returns_false(self):
        """Test unregistering nonexistent detector returns False."""
        registry = DetectorRegistry()

        result = registry.unregister("nonexistent")

        assert result is False

    def test_clear_registry(self):
        """Test clearing all detectors."""
        registry = DetectorRegistry()
        registry.register(MockDetector())
        registry.register(CompletenessDetector())

        registry.clear()

        assert len(registry) == 0
        assert registry.list_detector_names() == []


class TestRunDetectors:
    """Test running detectors."""

    @pytest.mark.asyncio
    async def test_run_all_empty_registry(self, sample_document):
        """Test running all detectors on empty registry."""
        registry = DetectorRegistry()

        results = await registry.run_all([sample_document])

        assert results == {}

    @pytest.mark.asyncio
    async def test_run_all_single_detector(self, sample_document):
        """Test running a single detector."""
        issue = Issue(
            severity=IssueSeverity.WARNING,
            detector="mock",
            title="Test Issue",
            description="Test description",
            documents=[sample_document.path],
        )

        detector = MockDetector(issues_to_return=[issue])
        registry = DetectorRegistry()
        registry.register(detector)

        results = await registry.run_all([sample_document])

        assert "mock" in results
        assert len(results["mock"]) == 1
        assert results["mock"][0] is issue
        assert detector.detect_called

    @pytest.mark.asyncio
    async def test_run_all_multiple_detectors(self, sample_document):
        """Test running multiple detectors."""
        issue1 = Issue(
            severity=IssueSeverity.WARNING,
            detector="mock",
            title="Mock Issue",
            description="From mock",
            documents=[sample_document.path],
        )

        issue2 = Issue(
            severity=IssueSeverity.INFO,
            detector="completeness",
            title="Completeness Issue",
            description="From completeness",
            documents=[sample_document.path],
        )

        detector1 = MockDetector(issues_to_return=[issue1])
        detector2 = MockDetector(issues_to_return=[issue2])
        detector2.name = "completeness"

        registry = DetectorRegistry()
        registry.register(detector1)
        registry.register(detector2)

        results = await registry.run_all([sample_document])

        assert len(results) == 2
        assert "mock" in results
        assert "completeness" in results
        assert len(results["mock"]) == 1
        assert len(results["completeness"]) == 1

    @pytest.mark.asyncio
    async def test_run_all_with_detector_names_filter(self, sample_document):
        """Test running only selected detectors."""
        detector1 = MockDetector()
        detector2 = MockDetector()
        detector2.name = "other"

        registry = DetectorRegistry()
        registry.register(detector1)
        registry.register(detector2)

        # Run only "mock" detector
        results = await registry.run_all([sample_document], detector_names=["mock"])

        assert len(results) == 1
        assert "mock" in results
        assert "other" not in results
        assert detector1.detect_called
        assert not detector2.detect_called

    @pytest.mark.asyncio
    async def test_run_all_with_invalid_detector_name(self, sample_document):
        """Test that requesting nonexistent detector raises error."""
        registry = DetectorRegistry()
        registry.register(MockDetector())

        with pytest.raises(ValueError, match="not registered"):
            await registry.run_all([sample_document], detector_names=["nonexistent"])

    @pytest.mark.asyncio
    async def test_run_all_handles_detector_errors(self, sample_document):
        """Test that detector errors are handled gracefully."""
        error_detector = ErrorDetector()
        good_detector = MockDetector()

        registry = DetectorRegistry()
        registry.register(error_detector)
        registry.register(good_detector)

        # Should not raise, should return empty list for error detector
        results = await registry.run_all([sample_document])

        assert "error" in results
        assert results["error"] == []  # Error detector returns empty list
        assert "mock" in results
        assert isinstance(results["mock"], list)

    @pytest.mark.asyncio
    async def test_run_detector_by_name(self, sample_document):
        """Test running a single detector by name."""
        issue = Issue(
            severity=IssueSeverity.WARNING,
            detector="mock",
            title="Test Issue",
            description="Test description",
            documents=[sample_document.path],
        )

        detector = MockDetector(issues_to_return=[issue])
        registry = DetectorRegistry()
        registry.register(detector)

        issues = await registry.run_detector("mock", [sample_document])

        assert len(issues) == 1
        assert issues[0] is issue

    @pytest.mark.asyncio
    async def test_run_detector_nonexistent_raises_error(self, sample_document):
        """Test that running nonexistent detector raises error."""
        registry = DetectorRegistry()

        with pytest.raises(ValueError, match="not registered"):
            await registry.run_detector("nonexistent", [sample_document])


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_run_all_with_no_documents(self):
        """Test running detectors with no documents."""
        detector = MockDetector()
        registry = DetectorRegistry()
        registry.register(detector)

        results = await registry.run_all([])

        assert "mock" in results
        assert results["mock"] == []

    @pytest.mark.asyncio
    async def test_detector_returning_no_issues(self, sample_document):
        """Test detector that returns no issues."""
        detector = MockDetector(issues_to_return=[])
        registry = DetectorRegistry()
        registry.register(detector)

        results = await registry.run_all([sample_document])

        assert "mock" in results
        assert results["mock"] == []

    def test_get_all_detectors_returns_copy(self):
        """Test that get_all_detectors returns a copy."""
        registry = DetectorRegistry()
        detector = MockDetector()
        registry.register(detector)

        all_detectors = registry.get_all_detectors()
        all_detectors["fake"] = None

        # Original registry should be unchanged
        assert "fake" not in registry
        assert len(registry) == 1
