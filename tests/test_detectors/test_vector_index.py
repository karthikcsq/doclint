"""Tests for the FAISS-based vector index."""

from pathlib import Path

import numpy as np
import pytest

from doclint.core.document import Chunk
from doclint.detectors.vector_index import ChunkIndex


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks with mock embeddings for testing."""
    chunks = []
    np.random.seed(42)  # Reproducible randomness

    for i in range(10):
        embedding = np.random.rand(384).astype(np.float32)
        chunk = Chunk(
            text=f"This is test chunk number {i} with some content.",
            index=i,
            document_path=Path(f"/docs/doc{i // 3}.txt"),  # 3 chunks per doc
            chunk_hash=f"hash_{i}",
            embedding=embedding,
            start_pos=i * 100,
            end_pos=(i + 1) * 100,
        )
        chunks.append(chunk)

    return chunks


@pytest.fixture
def similar_chunks() -> list[Chunk]:
    """Create chunks with controlled similarity for testing."""
    base_embedding = np.random.rand(384).astype(np.float32)
    base_embedding = base_embedding / np.linalg.norm(base_embedding)  # Normalize

    chunks = []

    # Chunk 0: Base embedding
    chunks.append(
        Chunk(
            text="JavaScript is single-threaded",
            index=0,
            document_path=Path("/docs/doc_a.md"),
            chunk_hash="hash_0",
            embedding=base_embedding.copy(),
            start_pos=0,
            end_pos=30,
        )
    )

    # Chunk 1: Very similar (small perturbation)
    similar_embedding = base_embedding + 0.05 * np.random.rand(384).astype(np.float32)
    similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)
    chunks.append(
        Chunk(
            text="JavaScript has multi-threading via workers",
            index=0,
            document_path=Path("/docs/doc_b.md"),
            chunk_hash="hash_1",
            embedding=similar_embedding.astype(np.float32),
            start_pos=0,
            end_pos=40,
        )
    )

    # Chunk 2: Somewhat similar (medium perturbation)
    medium_embedding = base_embedding + 0.3 * np.random.rand(384).astype(np.float32)
    medium_embedding = medium_embedding / np.linalg.norm(medium_embedding)
    chunks.append(
        Chunk(
            text="Python uses threads for concurrency",
            index=0,
            document_path=Path("/docs/doc_c.md"),
            chunk_hash="hash_2",
            embedding=medium_embedding.astype(np.float32),
            start_pos=0,
            end_pos=35,
        )
    )

    # Chunk 3: Very different (random)
    different_embedding = np.random.rand(384).astype(np.float32)
    different_embedding = different_embedding / np.linalg.norm(different_embedding)
    chunks.append(
        Chunk(
            text="The weather is sunny today",
            index=0,
            document_path=Path("/docs/doc_d.md"),
            chunk_hash="hash_3",
            embedding=different_embedding.astype(np.float32),
            start_pos=0,
            end_pos=25,
        )
    )

    return chunks


class TestChunkIndexInit:
    """Test ChunkIndex initialization."""

    def test_init_default_dimension(self) -> None:
        """Test initialization with default dimension."""
        index = ChunkIndex()
        assert index.dimension == 384
        assert index.chunks == []
        assert index.index is None

    def test_init_custom_dimension(self) -> None:
        """Test initialization with custom dimension."""
        index = ChunkIndex(dimension=768)
        assert index.dimension == 768


class TestChunkIndexBuild:
    """Test ChunkIndex.build() method."""

    def test_build_with_valid_chunks(self, sample_chunks: list[Chunk]) -> None:
        """Test building index with valid chunks."""
        index = ChunkIndex()
        index.build(sample_chunks)

        assert len(index) == 10
        assert index.index is not None

    def test_build_empty_list(self) -> None:
        """Test building index with empty list."""
        index = ChunkIndex()
        index.build([])

        assert len(index) == 0
        assert index.index is None

    def test_build_with_none_embeddings(self, sample_chunks: list[Chunk]) -> None:
        """Test building index skips chunks without embeddings."""
        # Set some embeddings to None
        sample_chunks[0].embedding = None
        sample_chunks[5].embedding = None

        index = ChunkIndex()
        index.build(sample_chunks)

        assert len(index) == 8  # 10 - 2 = 8

    def test_build_all_none_embeddings(self) -> None:
        """Test building index with all None embeddings."""
        chunks = [
            Chunk(
                text="test",
                index=0,
                document_path=Path("/docs/test.txt"),
                chunk_hash="hash",
                embedding=None,
            )
        ]

        index = ChunkIndex()
        index.build(chunks)

        assert len(index) == 0
        assert index.index is None


class TestChunkIndexFindSimilar:
    """Test ChunkIndex.find_similar() method."""

    def test_find_similar_returns_k_results(self, sample_chunks: list[Chunk]) -> None:
        """Test finding k similar chunks."""
        index = ChunkIndex()
        index.build(sample_chunks)

        results = index.find_similar(sample_chunks[0], k=5)

        assert len(results) <= 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert all(isinstance(idx, int) and isinstance(score, float) for idx, score in results)

    def test_find_similar_excludes_self(self, sample_chunks: list[Chunk]) -> None:
        """Test that find_similar excludes the query chunk itself."""
        index = ChunkIndex()
        index.build(sample_chunks)

        query_chunk = sample_chunks[0]
        results = index.find_similar(query_chunk, k=10, exclude_self=True)

        # Query chunk should not appear in results
        for idx, _ in results:
            result_chunk = index.get_chunk(idx)
            is_same = (
                result_chunk.document_path == query_chunk.document_path
                and result_chunk.index == query_chunk.index
                and result_chunk.chunk_hash == query_chunk.chunk_hash
            )
            assert not is_same

    def test_find_similar_includes_self_when_requested(self, sample_chunks: list[Chunk]) -> None:
        """Test that find_similar can include the query chunk."""
        index = ChunkIndex()
        index.build(sample_chunks)

        query_chunk = sample_chunks[0]
        results = index.find_similar(query_chunk, k=10, exclude_self=False)

        # First result should be the chunk itself with score ~1.0
        # (exact match has highest similarity)
        assert len(results) > 0

    def test_find_similar_empty_index(self) -> None:
        """Test find_similar on empty index returns empty list."""
        index = ChunkIndex()
        index.build([])

        chunk = Chunk(
            text="test",
            index=0,
            document_path=Path("/test.txt"),
            chunk_hash="hash",
            embedding=np.random.rand(384).astype(np.float32),
        )

        results = index.find_similar(chunk, k=5)
        assert results == []

    def test_find_similar_no_embedding_raises(self, sample_chunks: list[Chunk]) -> None:
        """Test find_similar raises error for chunk without embedding."""
        index = ChunkIndex()
        index.build(sample_chunks)

        chunk_no_embedding = Chunk(
            text="test",
            index=0,
            document_path=Path("/test.txt"),
            chunk_hash="hash",
            embedding=None,
        )

        with pytest.raises(ValueError, match="no embedding"):
            index.find_similar(chunk_no_embedding, k=5)


class TestChunkIndexFindAllSimilarPairs:
    """Test ChunkIndex.find_all_similar_pairs() method."""

    def test_find_pairs_with_similar_chunks(self, similar_chunks: list[Chunk]) -> None:
        """Test finding similar pairs with controlled similarity."""
        index = ChunkIndex()
        index.build(similar_chunks)

        # With threshold 0.9, should find the very similar pair (0, 1)
        pairs = index.find_all_similar_pairs(threshold=0.9)

        # Pairs should be sorted by similarity descending
        if len(pairs) > 1:
            for i in range(len(pairs) - 1):
                assert pairs[i][2] >= pairs[i + 1][2]

        # Each pair should have i < j
        for i, j, score in pairs:
            assert i < j
            assert 0 <= score <= 1

    def test_find_pairs_empty_index(self) -> None:
        """Test find_all_similar_pairs on empty index."""
        index = ChunkIndex()
        index.build([])

        pairs = index.find_all_similar_pairs(threshold=0.85)
        assert pairs == []

    def test_find_pairs_single_chunk(self, sample_chunks: list[Chunk]) -> None:
        """Test find_all_similar_pairs with single chunk."""
        index = ChunkIndex()
        index.build(sample_chunks[:1])

        pairs = index.find_all_similar_pairs(threshold=0.85)
        assert pairs == []

    def test_find_pairs_exclude_same_document(self, similar_chunks: list[Chunk]) -> None:
        """Test excluding pairs from the same document."""
        # Put chunks 0 and 1 in the same document
        similar_chunks[1].document_path = similar_chunks[0].document_path

        index = ChunkIndex()
        index.build(similar_chunks)

        pairs_with_same_doc = index.find_all_similar_pairs(
            threshold=0.5, exclude_same_document=False
        )
        pairs_without_same_doc = index.find_all_similar_pairs(
            threshold=0.5, exclude_same_document=True
        )

        # Should have fewer pairs when excluding same document
        assert len(pairs_without_same_doc) <= len(pairs_with_same_doc)

    def test_find_pairs_high_threshold_filters_results(self, sample_chunks: list[Chunk]) -> None:
        """Test that high threshold filters out low similarity pairs."""
        index = ChunkIndex()
        index.build(sample_chunks)

        pairs_low = index.find_all_similar_pairs(threshold=0.3)
        pairs_high = index.find_all_similar_pairs(threshold=0.95)

        # Higher threshold should result in fewer pairs
        assert len(pairs_high) <= len(pairs_low)


class TestChunkIndexUtilities:
    """Test ChunkIndex utility methods."""

    def test_get_chunk(self, sample_chunks: list[Chunk]) -> None:
        """Test getting chunk by index."""
        index = ChunkIndex()
        index.build(sample_chunks)

        chunk = index.get_chunk(5)
        assert chunk == sample_chunks[5]

    def test_get_chunk_out_of_range(self, sample_chunks: list[Chunk]) -> None:
        """Test getting chunk with invalid index raises error."""
        index = ChunkIndex()
        index.build(sample_chunks)

        with pytest.raises(IndexError):
            index.get_chunk(100)

    def test_clear(self, sample_chunks: list[Chunk]) -> None:
        """Test clearing the index."""
        index = ChunkIndex()
        index.build(sample_chunks)

        assert len(index) == 10

        index.clear()

        assert len(index) == 0
        assert index.index is None
        assert index.chunks == []

    def test_len(self, sample_chunks: list[Chunk]) -> None:
        """Test __len__ returns correct count."""
        index = ChunkIndex()
        assert len(index) == 0

        index.build(sample_chunks)
        assert len(index) == 10


class TestChunkIndexPerformance:
    """Performance tests for ChunkIndex with larger datasets."""

    @pytest.mark.parametrize("n_chunks", [100, 1000])
    def test_build_performance(self, n_chunks: int) -> None:
        """Test index building with varying dataset sizes."""
        np.random.seed(42)

        chunks = []
        for i in range(n_chunks):
            chunk = Chunk(
                text=f"Chunk {i}",
                index=i % 10,
                document_path=Path(f"/docs/doc{i // 10}.txt"),
                chunk_hash=f"hash_{i}",
                embedding=np.random.rand(384).astype(np.float32),
            )
            chunks.append(chunk)

        index = ChunkIndex()
        index.build(chunks)

        assert len(index) == n_chunks

    @pytest.mark.parametrize("n_chunks", [100, 1000])
    def test_search_performance(self, n_chunks: int) -> None:
        """Test search performance with varying dataset sizes."""
        np.random.seed(42)

        chunks = []
        for i in range(n_chunks):
            chunk = Chunk(
                text=f"Chunk {i}",
                index=i % 10,
                document_path=Path(f"/docs/doc{i // 10}.txt"),
                chunk_hash=f"hash_{i}",
                embedding=np.random.rand(384).astype(np.float32),
            )
            chunks.append(chunk)

        index = ChunkIndex()
        index.build(chunks)

        # Search should complete quickly
        results = index.find_similar(chunks[0], k=10)
        assert len(results) <= 10
