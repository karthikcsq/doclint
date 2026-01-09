"""Tests for the conflict detector."""

from pathlib import Path
from typing import List

import numpy as np
import pytest

from doclint.core.document import Chunk, Document, DocumentMetadata
from doclint.detectors.base import ContradictionVerifier, IssueSeverity
from doclint.detectors.conflicts import ConflictDetector


def create_chunk(
    text: str,
    index: int,
    doc_path: Path,
    embedding: np.ndarray | None = None,
) -> Chunk:
    """Helper to create a chunk with optional embedding."""
    if embedding is None:
        embedding = np.random.rand(384).astype(np.float32)
    return Chunk(
        text=text,
        index=index,
        document_path=doc_path,
        chunk_hash=f"hash_{index}_{doc_path.stem}",
        embedding=embedding,
        start_pos=index * 100,
        end_pos=(index + 1) * 100,
    )


def create_document(
    path: Path,
    chunks: List[Chunk],
) -> Document:
    """Helper to create a document with chunks."""
    return Document(
        path=path,
        content=" ".join(c.text for c in chunks),
        metadata=DocumentMetadata(),
        file_type="text",
        size_bytes=1000,
        content_hash=f"doc_hash_{path.stem}",
        chunks=chunks,
    )


def create_similar_embedding(base: np.ndarray, noise: float = 0.05) -> np.ndarray:
    """Create an embedding similar to the base with small noise.

    Uses a blend approach to ensure high cosine similarity:
    result = (1-noise)*base + noise*random, then normalize
    """
    random_vec = np.random.rand(384).astype(np.float32)
    random_vec = random_vec / np.linalg.norm(random_vec)
    # Blend base with random - smaller noise = more similar
    similar = (1 - noise) * base + noise * random_vec
    return (similar / np.linalg.norm(similar)).astype(np.float32)


def create_different_embedding() -> np.ndarray:
    """Create a random different embedding."""
    emb = np.random.rand(384).astype(np.float32)
    return (emb / np.linalg.norm(emb)).astype(np.float32)


class TestConflictDetectorInit:
    """Test ConflictDetector initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        detector = ConflictDetector()
        assert detector.name == "conflict"
        assert detector.similarity_threshold == 0.85
        assert detector.exclude_same_document is True
        assert detector.min_chunk_length == 50
        assert detector.verifier is None

    def test_custom_init(self) -> None:
        """Test custom initialization."""
        detector = ConflictDetector(
            similarity_threshold=0.90,
            exclude_same_document=False,
            min_chunk_length=100,
        )
        assert detector.similarity_threshold == 0.90
        assert detector.exclude_same_document is False
        assert detector.min_chunk_length == 100


class TestConflictDetectorDetect:
    """Test ConflictDetector.detect() method."""

    @pytest.mark.asyncio
    async def test_detect_with_single_document(self) -> None:
        """Test that single document returns no issues."""
        np.random.seed(42)
        doc_path = Path("/docs/doc1.md")
        chunks = [
            create_chunk("This is some text about Python programming.", 0, doc_path),
        ]
        documents = [create_document(doc_path, chunks)]

        detector = ConflictDetector()
        issues = await detector.detect(documents)

        assert issues == []

    @pytest.mark.asyncio
    async def test_detect_no_similar_chunks(self) -> None:
        """Test with documents that have no similar chunks."""
        np.random.seed(42)

        # Create documents with very different embeddings
        doc1_path = Path("/docs/doc1.md")
        doc2_path = Path("/docs/doc2.md")

        emb1 = create_different_embedding()
        emb2 = create_different_embedding()

        chunks1 = [
            create_chunk(
                "Python is a programming language used for web development.",
                0,
                doc1_path,
                emb1,
            ),
        ]
        chunks2 = [
            create_chunk(
                "The weather forecast predicts sunny skies tomorrow.",
                0,
                doc2_path,
                emb2,
            ),
        ]

        documents = [
            create_document(doc1_path, chunks1),
            create_document(doc2_path, chunks2),
        ]

        detector = ConflictDetector(similarity_threshold=0.85)
        issues = await detector.detect(documents)

        assert issues == []

    @pytest.mark.asyncio
    async def test_detect_negation_conflict(self) -> None:
        """Test detection of negation-based conflicts."""
        np.random.seed(42)

        doc1_path = Path("/docs/doc1.md")
        doc2_path = Path("/docs/doc2.md")

        # Create similar embeddings for related content
        base_emb = np.random.rand(384).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)

        chunks1 = [
            create_chunk(
                "JavaScript is single-threaded and cannot run multiple threads.",
                0,
                doc1_path,
                create_similar_embedding(base_emb, 0.02),
            ),
        ]
        chunks2 = [
            create_chunk(
                "JavaScript can run multiple threads using Web Workers.",
                0,
                doc2_path,
                create_similar_embedding(base_emb, 0.02),
            ),
        ]

        documents = [
            create_document(doc1_path, chunks1),
            create_document(doc2_path, chunks2),
        ]

        detector = ConflictDetector(similarity_threshold=0.80)
        issues = await detector.detect(documents)

        assert len(issues) >= 1
        # Should be CRITICAL due to negation
        assert any(issue.severity == IssueSeverity.CRITICAL for issue in issues)

    @pytest.mark.asyncio
    async def test_detect_antonym_conflict(self) -> None:
        """Test detection of antonym-based conflicts."""
        np.random.seed(42)

        doc1_path = Path("/docs/doc1.md")
        doc2_path = Path("/docs/doc2.md")

        # Use the SAME embedding for both chunks to guarantee high similarity
        shared_emb = np.random.rand(384).astype(np.float32)
        shared_emb = (shared_emb / np.linalg.norm(shared_emb)).astype(np.float32)

        chunks1 = [
            create_chunk(
                "The API uses synchronous request handling for all endpoints.",
                0,
                doc1_path,
                shared_emb.copy(),
            ),
        ]
        chunks2 = [
            create_chunk(
                "The API uses asynchronous request handling for all endpoints.",
                0,
                doc2_path,
                shared_emb.copy(),
            ),
        ]

        documents = [
            create_document(doc1_path, chunks1),
            create_document(doc2_path, chunks2),
        ]

        detector = ConflictDetector(similarity_threshold=0.80)
        issues = await detector.detect(documents)

        assert len(issues) >= 1
        # Should be WARNING due to antonyms
        assert any(
            issue.severity in [IssueSeverity.WARNING, IssueSeverity.CRITICAL] for issue in issues
        )

    @pytest.mark.asyncio
    async def test_detect_numerical_mismatch(self) -> None:
        """Test detection of numerical mismatches."""
        np.random.seed(42)

        doc1_path = Path("/docs/doc1.md")
        doc2_path = Path("/docs/doc2.md")

        # Use the SAME embedding for both chunks to guarantee high similarity
        shared_emb = np.random.rand(384).astype(np.float32)
        shared_emb = (shared_emb / np.linalg.norm(shared_emb)).astype(np.float32)

        chunks1 = [
            create_chunk(
                "The API rate limit is 100 requests per minute for all users.",
                0,
                doc1_path,
                shared_emb.copy(),
            ),
        ]
        chunks2 = [
            create_chunk(
                "The API rate limit is 1000 requests per minute for all users.",
                0,
                doc2_path,
                shared_emb.copy(),
            ),
        ]

        documents = [
            create_document(doc1_path, chunks1),
            create_document(doc2_path, chunks2),
        ]

        detector = ConflictDetector(similarity_threshold=0.80)
        issues = await detector.detect(documents)

        assert len(issues) >= 1
        assert any(issue.severity == IssueSeverity.CRITICAL for issue in issues)

    @pytest.mark.asyncio
    async def test_detect_high_similarity_info(self) -> None:
        """Test that high similarity without contradiction is INFO."""
        np.random.seed(42)

        doc1_path = Path("/docs/doc1.md")
        doc2_path = Path("/docs/doc2.md")

        base_emb = np.random.rand(384).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)

        # Similar topic, different phrasing, no contradiction
        chunks1 = [
            create_chunk(
                "Python supports multiple programming paradigms including OOP.",
                0,
                doc1_path,
                create_similar_embedding(base_emb, 0.01),  # Very similar
            ),
        ]
        chunks2 = [
            create_chunk(
                "The Python language allows object-oriented programming styles.",
                0,
                doc2_path,
                create_similar_embedding(base_emb, 0.01),  # Very similar
            ),
        ]

        documents = [
            create_document(doc1_path, chunks1),
            create_document(doc2_path, chunks2),
        ]

        detector = ConflictDetector(similarity_threshold=0.80)
        issues = await detector.detect(documents)

        # Should report INFO for high similarity without conflict
        if issues:
            # Either no issues (text overlap too high) or INFO
            assert all(issue.severity == IssueSeverity.INFO for issue in issues)

    @pytest.mark.asyncio
    async def test_detect_excludes_same_document(self) -> None:
        """Test that same-document conflicts are excluded by default."""
        np.random.seed(42)

        doc_path = Path("/docs/doc1.md")

        base_emb = np.random.rand(384).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)

        # Conflicting chunks in same document
        chunks = [
            create_chunk(
                "JavaScript is single-threaded and cannot run parallel code.",
                0,
                doc_path,
                create_similar_embedding(base_emb, 0.02),
            ),
            create_chunk(
                "JavaScript can run parallel code using Web Workers.",
                1,
                doc_path,
                create_similar_embedding(base_emb, 0.02),
            ),
        ]

        documents = [create_document(doc_path, chunks)]

        detector = ConflictDetector(similarity_threshold=0.80, exclude_same_document=True)
        issues = await detector.detect(documents)

        # Should be empty - same document excluded
        assert issues == []

    @pytest.mark.asyncio
    async def test_detect_includes_same_document_when_configured(self) -> None:
        """Test that same-document conflicts are included when configured."""
        np.random.seed(42)

        doc1_path = Path("/docs/doc1.md")
        doc2_path = Path("/docs/doc2.md")

        base_emb = np.random.rand(384).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)

        # Conflicting chunks in same document
        chunks1 = [
            create_chunk(
                "JavaScript is single-threaded and cannot run parallel code.",
                0,
                doc1_path,
                create_similar_embedding(base_emb, 0.02),
            ),
            create_chunk(
                "JavaScript can run parallel code using Web Workers.",
                1,
                doc1_path,
                create_similar_embedding(base_emb, 0.02),
            ),
        ]
        chunks2 = [
            create_chunk(
                "Some unrelated content about databases and SQL queries.",
                0,
                doc2_path,
                create_different_embedding(),
            ),
        ]

        documents = [
            create_document(doc1_path, chunks1),
            create_document(doc2_path, chunks2),
        ]

        detector = ConflictDetector(similarity_threshold=0.80, exclude_same_document=False)
        issues = await detector.detect(documents)

        # Should find the internal conflict
        assert len(issues) >= 1


class TestConflictDetectorHeuristics:
    """Test individual heuristic methods."""

    def test_jaccard_similarity_identical(self) -> None:
        """Test Jaccard similarity with identical texts."""
        text = "the quick brown fox"
        similarity = ConflictDetector._jaccard_similarity(text, text)
        assert similarity == 1.0

    def test_jaccard_similarity_no_overlap(self) -> None:
        """Test Jaccard similarity with no word overlap."""
        text_a = "apple banana cherry"
        text_b = "dog elephant fox"
        similarity = ConflictDetector._jaccard_similarity(text_a, text_b)
        assert similarity == 0.0

    def test_jaccard_similarity_partial(self) -> None:
        """Test Jaccard similarity with partial overlap."""
        text_a = "the quick brown fox"
        text_b = "the slow brown dog"
        similarity = ConflictDetector._jaccard_similarity(text_a, text_b)
        # Overlap: {the, brown} = 2, Union: {the, quick, brown, fox, slow, dog} = 6
        assert similarity == pytest.approx(2 / 6, rel=0.01)

    def test_negation_pattern_detected(self) -> None:
        """Test negation pattern detection."""
        text_a = "javascript is single-threaded"
        text_b = "javascript is not multi-threaded"

        result = ConflictDetector._has_negation_pattern(text_a, text_b)
        # "is" vs "is not"
        assert result is True

    def test_negation_pattern_cannot(self) -> None:
        """Test cannot/can negation detection."""
        text_a = "the system can handle concurrent requests"
        text_b = "the system cannot handle concurrent requests"

        result = ConflictDetector._has_negation_pattern(text_a, text_b)
        assert result is True

    def test_negation_pattern_not_detected(self) -> None:
        """Test no false positive for non-contradictory text."""
        text_a = "python is dynamically typed"
        text_b = "python is interpreted"

        result = ConflictDetector._has_negation_pattern(text_a, text_b)
        assert result is False

    def test_antonym_pair_detected(self) -> None:
        """Test antonym pair detection."""
        text_a = "the api uses synchronous calls"
        text_b = "the api uses asynchronous calls"

        result = ConflictDetector._has_antonym_pair(text_a, text_b)
        assert result is True

    def test_antonym_pair_true_false(self) -> None:
        """Test true/false antonym detection."""
        text_a = "the setting is true by default"
        text_b = "the setting is false by default"

        result = ConflictDetector._has_antonym_pair(text_a, text_b)
        assert result is True

    def test_antonym_pair_not_detected(self) -> None:
        """Test no false positive for non-antonym text."""
        text_a = "the api uses json format"
        text_b = "the api uses xml format"

        result = ConflictDetector._has_antonym_pair(text_a, text_b)
        assert result is False

    def test_numerical_mismatch_detected(self) -> None:
        """Test numerical mismatch detection."""
        text_a = "the limit is 100 requests per minute"
        text_b = "the limit is 1000 requests per minute"

        result = ConflictDetector._has_numerical_mismatch(text_a, text_b)
        assert result is True

    def test_numerical_mismatch_same_value(self) -> None:
        """Test no mismatch with same values."""
        text_a = "the limit is 100 requests per minute"
        text_b = "rate limiting: 100 requests per minute"

        result = ConflictDetector._has_numerical_mismatch(text_a, text_b)
        assert result is False

    def test_numerical_mismatch_different_context(self) -> None:
        """Test no mismatch with different contexts."""
        # Completely different contexts - shouldn't match
        text_a = "the server has 100 cores available"
        text_b = "users can upload 500 files maximum"

        result = ConflictDetector._has_numerical_mismatch(text_a, text_b)
        assert result is False


class TestConflictDetectorWithVerifier:
    """Test ConflictDetector with LLM verifier hook."""

    @pytest.mark.asyncio
    async def test_verifier_called_for_uncertain_pairs(self) -> None:
        """Test that verifier is called for uncertain pairs."""
        np.random.seed(42)

        class MockVerifier(ContradictionVerifier):
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            async def verify(self, text_a: str, text_b: str) -> tuple[bool, float, str]:
                self.calls.append((text_a, text_b))
                return True, 0.9, "LLM detected contradiction"

        verifier = MockVerifier()

        doc1_path = Path("/docs/doc1.md")
        doc2_path = Path("/docs/doc2.md")

        base_emb = np.random.rand(384).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)

        # Similar content without clear heuristic signals
        chunks1 = [
            create_chunk(
                "The deployment process takes approximately two hours to complete.",
                0,
                doc1_path,
                create_similar_embedding(base_emb, 0.01),
            ),
        ]
        chunks2 = [
            create_chunk(
                "Deployments typically finish within thirty minutes of starting.",
                0,
                doc2_path,
                create_similar_embedding(base_emb, 0.01),
            ),
        ]

        documents = [
            create_document(doc1_path, chunks1),
            create_document(doc2_path, chunks2),
        ]

        detector = ConflictDetector(similarity_threshold=0.80, verifier=verifier)
        issues = await detector.detect(documents)

        # Verifier should have been called
        assert len(verifier.calls) > 0
        # Should have issues from verifier
        assert len(issues) >= 1


class TestIssueDetails:
    """Test Issue object details."""

    @pytest.mark.asyncio
    async def test_issue_contains_chunk_details(self) -> None:
        """Test that issues contain proper chunk details."""
        np.random.seed(42)

        doc1_path = Path("/docs/doc1.md")
        doc2_path = Path("/docs/doc2.md")

        base_emb = np.random.rand(384).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)

        chunks1 = [
            create_chunk(
                "JavaScript cannot execute code in parallel due to single-threading.",
                0,
                doc1_path,
                create_similar_embedding(base_emb, 0.02),
            ),
        ]
        chunks2 = [
            create_chunk(
                "JavaScript can execute code in parallel using worker threads.",
                0,
                doc2_path,
                create_similar_embedding(base_emb, 0.02),
            ),
        ]

        documents = [
            create_document(doc1_path, chunks1),
            create_document(doc2_path, chunks2),
        ]

        detector = ConflictDetector(similarity_threshold=0.80)
        issues = await detector.detect(documents)

        assert len(issues) >= 1
        issue = issues[0]

        # Check issue structure
        assert issue.detector == "conflict"
        assert len(issue.documents) == 2
        assert issue.chunks is not None
        assert len(issue.chunks) == 2

        # Check details
        assert "similarity" in issue.details
        assert "chunk_a" in issue.details
        assert "chunk_b" in issue.details
        assert "text" in issue.details["chunk_a"]
        assert "position" in issue.details["chunk_a"]

    @pytest.mark.asyncio
    async def test_issue_to_dict(self) -> None:
        """Test Issue.to_dict() serialization."""
        np.random.seed(42)

        doc1_path = Path("/docs/doc1.md")
        doc2_path = Path("/docs/doc2.md")

        # Use the SAME embedding for both chunks to guarantee high similarity
        shared_emb = np.random.rand(384).astype(np.float32)
        shared_emb = (shared_emb / np.linalg.norm(shared_emb)).astype(np.float32)

        chunks1 = [
            create_chunk(
                "The feature is enabled by default in all configurations.",
                0,
                doc1_path,
                shared_emb.copy(),
            ),
        ]
        chunks2 = [
            create_chunk(
                "The feature is disabled by default in all configurations.",
                0,
                doc2_path,
                shared_emb.copy(),
            ),
        ]

        documents = [
            create_document(doc1_path, chunks1),
            create_document(doc2_path, chunks2),
        ]

        detector = ConflictDetector(similarity_threshold=0.80)
        issues = await detector.detect(documents)

        assert len(issues) >= 1
        issue_dict = issues[0].to_dict()

        # Check serialization
        assert isinstance(issue_dict, dict)
        assert issue_dict["detector"] == "conflict"
        assert issue_dict["severity"] in ["info", "warning", "critical"]
        assert isinstance(issue_dict["documents"], list)
        assert all(isinstance(d, str) for d in issue_dict["documents"])
