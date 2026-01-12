"""Integration tests for the full detection pipeline."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from doclint.core.document import Chunk, Document, DocumentMetadata
from doclint.detectors.base import IssueSeverity
from doclint.detectors.completeness import CompletenessDetector
from doclint.detectors.conflicts import ConflictDetector
from doclint.detectors.registry import DetectorRegistry


@pytest.fixture
def documents_with_conflicts():
    """Create a set of documents with semantic conflicts."""
    # Document 1: Claims Python is dynamically typed
    doc1 = Document(
        path=Path("/docs/python_intro.md"),
        content=(
            "Python is a dynamically typed programming language. "
            "Variables don't need explicit type declarations. "
            "Type checking happens at runtime."
        ),
        metadata=DocumentMetadata(
            author="Alice",
            created=datetime(2024, 1, 1),
            title="Python Introduction",
        ),
        file_type="markdown",
        size_bytes=1000,
        content_hash="hash1",
    )

    # Document 2: Claims Python is statically typed (conflict!)
    doc2 = Document(
        path=Path("/docs/python_advanced.md"),
        content=(
            "Python is a statically typed language when using type hints. "
            "All variables must have explicit type annotations. "
            "Type checking happens at compile time with mypy."
        ),
        metadata=DocumentMetadata(
            author="Bob",
            created=datetime(2024, 1, 5),
            title="Python Advanced",
        ),
        file_type="markdown",
        size_bytes=1000,
        content_hash="hash2",
    )

    # Document 3: Unrelated content, no conflict
    doc3 = Document(
        path=Path("/docs/javascript_basics.md"),
        content=(
            "JavaScript is a programming language primarily used for web development. "
            "It runs in browsers and on servers via Node.js. "
            "JavaScript is dynamically typed and interpreted."
        ),
        metadata=DocumentMetadata(
            author="Charlie",
            created=datetime(2024, 1, 10),
            title="JavaScript Basics",
        ),
        file_type="markdown",
        size_bytes=1000,
        content_hash="hash3",
    )

    # Add mock embeddings to simulate semantic similarity
    # In a real scenario, these would come from the embedding generator
    embedding1 = np.random.randn(384).astype(np.float32)
    embedding2 = embedding1 + np.random.randn(384).astype(np.float32) * 0.1  # Very similar
    embedding3 = np.random.randn(384).astype(np.float32)  # Different

    # Add chunks with embeddings
    doc1.chunks = [
        Chunk(
            text=doc1.content,
            index=0,
            document_path=doc1.path,
            chunk_hash="chunk1",
            embedding=embedding1,
            start_pos=0,
            end_pos=len(doc1.content),
        )
    ]

    doc2.chunks = [
        Chunk(
            text=doc2.content,
            index=0,
            document_path=doc2.path,
            chunk_hash="chunk2",
            embedding=embedding2,
            start_pos=0,
            end_pos=len(doc2.content),
        )
    ]

    doc3.chunks = [
        Chunk(
            text=doc3.content,
            index=0,
            document_path=doc3.path,
            chunk_hash="chunk3",
            embedding=embedding3,
            start_pos=0,
            end_pos=len(doc3.content),
        )
    ]

    return [doc1, doc2, doc3]


@pytest.fixture
def documents_with_completeness_issues(tmp_path):
    """Create documents with various completeness issues."""
    # Document 1: Complete, no issues
    doc1 = Document(
        path=Path("/docs/complete.md"),
        content="This is a complete document with sufficient content. " * 10,
        metadata=DocumentMetadata(
            author="Alice",
            created=datetime(2024, 1, 1),
            title="Complete Document",
            version="1.0",
        ),
        file_type="markdown",
        size_bytes=1000,
        content_hash="hash1",
    )

    # Document 2: Missing metadata
    doc2 = Document(
        path=Path("/docs/missing_metadata.md"),
        content="This document is missing important metadata. " * 10,
        metadata=DocumentMetadata(
            title="Incomplete Metadata",
            # Missing: author, created
        ),
        file_type="markdown",
        size_bytes=500,
        content_hash="hash2",
    )

    # Document 3: Short content
    doc3 = Document(
        path=Path("/docs/short_content.md"),
        content="Too short",
        metadata=DocumentMetadata(
            author="Charlie",
            created=datetime(2024, 1, 3),
        ),
        file_type="markdown",
        size_bytes=50,
        content_hash="hash3",
    )

    # Document 4: Broken links
    # Create a test file structure
    existing_file = tmp_path / "existing.md"
    existing_file.write_text("# Existing File")

    doc4_path = tmp_path / "doc_with_links.md"
    doc4 = Document(
        path=doc4_path,
        content="""
        Check [existing file](existing.md) for reference.
        See also [missing file](nonexistent.md).
        External link [Google](https://google.com) is fine.
        """,
        metadata=DocumentMetadata(
            author="Dave",
            created=datetime(2024, 1, 4),
        ),
        file_type="markdown",
        size_bytes=200,
        content_hash="hash4",
    )

    return [doc1, doc2, doc3, doc4]


@pytest.mark.asyncio
async def test_conflict_detector_integration(documents_with_conflicts):
    """Test conflict detection on documents with semantic conflicts."""
    detector = ConflictDetector(similarity_threshold=0.7)

    issues = await detector.detect(documents_with_conflicts)

    # Should detect conflict between doc1 and doc2 about Python typing
    assert len(issues) >= 1

    # Check that issues contain correct information
    for issue in issues:
        assert issue.detector == "conflict"
        # Can be INFO (high similarity), WARNING, or CRITICAL (clear contradiction)
        assert issue.severity in [IssueSeverity.INFO, IssueSeverity.WARNING, IssueSeverity.CRITICAL]
        assert len(issue.documents) >= 2  # Conflict involves multiple documents


@pytest.mark.asyncio
async def test_completeness_detector_integration(documents_with_completeness_issues):
    """Test completeness detection on documents with various issues."""
    detector = CompletenessDetector(
        required_metadata=["author", "created"],
        min_content_length=100,
        check_internal_links=True,
    )

    issues = await detector.detect(documents_with_completeness_issues)

    # Should find:
    # - doc2: missing metadata (author, created)
    # - doc3: short content
    # - doc4: broken link (nonexistent.md)
    assert len(issues) >= 3

    # Verify different types of issues
    issue_titles = {issue.title for issue in issues}
    assert any("metadata" in title.lower() for title in issue_titles)
    assert any("content" in title.lower() for title in issue_titles)
    assert any("link" in title.lower() for title in issue_titles)


@pytest.mark.asyncio
async def test_registry_with_multiple_detectors(
    documents_with_conflicts,
    documents_with_completeness_issues,
):
    """Test running multiple detectors through the registry."""
    registry = DetectorRegistry()

    # Register both detectors
    conflict_detector = ConflictDetector(similarity_threshold=0.7)
    completeness_detector = CompletenessDetector(
        required_metadata=["author", "created"],
        min_content_length=100,
    )

    registry.register(conflict_detector)
    registry.register(completeness_detector)

    # Run on conflict documents
    results = await registry.run_all(documents_with_conflicts)

    assert "conflict" in results
    assert "completeness" in results

    # Conflict detector should find issues
    assert len(results["conflict"]) >= 1

    # Completeness detector should find metadata issues in some docs
    # (doc3 has no author/created)
    assert isinstance(results["completeness"], list)


@pytest.mark.asyncio
async def test_selective_detector_execution(documents_with_completeness_issues):
    """Test running only selected detectors."""
    registry = DetectorRegistry()

    registry.register(ConflictDetector())
    registry.register(CompletenessDetector())

    # Run only completeness detector
    results = await registry.run_all(
        documents_with_completeness_issues,
        detector_names=["completeness"],
    )

    assert "completeness" in results
    assert "conflict" not in results
    assert len(results) == 1


@pytest.mark.asyncio
async def test_empty_document_set():
    """Test detection with no documents."""
    registry = DetectorRegistry()
    registry.register(ConflictDetector())
    registry.register(CompletenessDetector())

    results = await registry.run_all([])

    assert "conflict" in results
    assert "completeness" in results
    assert results["conflict"] == []
    assert results["completeness"] == []


@pytest.mark.asyncio
async def test_documents_with_no_issues():
    """Test detection on documents with no issues."""
    # Create perfect documents
    doc = Document(
        path=Path("/docs/perfect.md"),
        content="This is a perfect document with sufficient content. " * 20,
        metadata=DocumentMetadata(
            author="Alice",
            created=datetime(2024, 1, 1),
            modified=datetime(2024, 1, 2),
            version="1.0",
            title="Perfect Document",
            tags=["complete", "verified"],
        ),
        file_type="markdown",
        size_bytes=2000,
        content_hash="hash_perfect",
    )

    # Add embedding
    embedding = np.random.randn(384).astype(np.float32)
    doc.chunks = [
        Chunk(
            text=doc.content,
            index=0,
            document_path=doc.path,
            chunk_hash="chunk_perfect",
            embedding=embedding,
        )
    ]

    registry = DetectorRegistry()
    registry.register(ConflictDetector())
    registry.register(CompletenessDetector())

    results = await registry.run_all([doc])

    # Should find no issues
    assert results["conflict"] == []
    assert results["completeness"] == []


@pytest.mark.asyncio
async def test_issue_aggregation():
    """Test that issues from multiple detectors are properly aggregated."""
    # Create a document with multiple types of issues
    doc = Document(
        path=Path("/docs/problematic.md"),
        content="Short",  # Too short
        metadata=DocumentMetadata(
            title="Problematic"
            # Missing: author, created
        ),
        file_type="markdown",
        size_bytes=10,
        content_hash="hash_prob",
    )

    # Add embedding
    embedding = np.random.randn(384).astype(np.float32)
    doc.chunks = [
        Chunk(
            text=doc.content,
            index=0,
            document_path=doc.path,
            chunk_hash="chunk_prob",
            embedding=embedding,
        )
    ]

    registry = DetectorRegistry()
    registry.register(ConflictDetector())
    registry.register(
        CompletenessDetector(
            required_metadata=["author", "created"],
            min_content_length=100,
        )
    )

    results = await registry.run_all([doc])

    # Completeness should find: missing metadata + short content
    assert len(results["completeness"]) >= 2

    # Each issue should have proper metadata
    for detector_name, issues in results.items():
        for issue in issues:
            assert issue.detector == detector_name
            assert issue.severity in [
                IssueSeverity.INFO,
                IssueSeverity.WARNING,
                IssueSeverity.CRITICAL,
            ]
            assert issue.title
            assert issue.description
            assert issue.documents


@pytest.mark.asyncio
async def test_large_document_set():
    """Test detection on a larger set of documents."""
    # Create 20 documents with various characteristics
    documents = []

    for i in range(20):
        has_author = i % 2 == 0
        has_created = i % 3 == 0
        is_long = i % 4 == 0

        content = "Content " * (100 if is_long else 5)

        metadata = DocumentMetadata(
            author=f"Author {i}" if has_author else None,
            created=datetime(2024, 1, i + 1) if has_created else None,
            title=f"Document {i}",
        )

        doc = Document(
            path=Path(f"/docs/doc{i}.md"),
            content=content,
            metadata=metadata,
            file_type="markdown",
            size_bytes=len(content),
            content_hash=f"hash{i}",
        )

        # Add embedding
        embedding = np.random.randn(384).astype(np.float32)
        doc.chunks = [
            Chunk(
                text=content,
                index=0,
                document_path=doc.path,
                chunk_hash=f"chunk{i}",
                embedding=embedding,
            )
        ]

        documents.append(doc)

    registry = DetectorRegistry()
    registry.register(ConflictDetector())
    registry.register(CompletenessDetector(required_metadata=["author", "created"]))

    results = await registry.run_all(documents)

    # Should find some completeness issues (missing metadata, short content)
    assert len(results["completeness"]) > 0

    # All results should be valid
    for detector_name, issues in results.items():
        assert isinstance(issues, list)
        for issue in issues:
            assert hasattr(issue, "severity")
            assert hasattr(issue, "detector")
            assert hasattr(issue, "documents")
