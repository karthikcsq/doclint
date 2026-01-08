"""Tests for cache manager."""

from pathlib import Path

import numpy as np

from doclint.cache.manager import CacheManager


class TestCacheManager:
    """Test suite for CacheManager class."""

    def test_initialization_with_custom_dir(self, cache_dir: Path) -> None:
        """Test initialization with custom cache directory."""
        manager = CacheManager(cache_dir=cache_dir, model_name="test-model")

        assert manager.cache_dir == cache_dir
        assert manager.model_name == "test-model"
        assert cache_dir.exists()

        manager.close()

    def test_initialization_with_default_dir(self) -> None:
        """Test initialization with platform-specific default directory."""
        manager = CacheManager(model_name="test-model")

        assert manager.cache_dir is not None
        assert manager.cache_dir.exists()
        assert manager.model_name == "test-model"

        manager.close()

    def test_set_and_get_embedding(self, cache_dir: Path) -> None:
        """Test caching and retrieving an embedding."""
        manager = CacheManager(cache_dir=cache_dir)

        # Create test embedding
        embedding = np.random.rand(384).astype(np.float32)
        content_hash = "abc123def456"

        # Cache the embedding
        manager.set_embedding(content_hash, embedding)

        # Retrieve the embedding
        cached = manager.get_embedding(content_hash, dimension=384)

        assert cached is not None
        assert isinstance(cached, np.ndarray)
        assert np.array_equal(cached, embedding)

        manager.close()

    def test_cache_miss_returns_none(self, cache_dir: Path) -> None:
        """Test that cache miss returns None."""
        manager = CacheManager(cache_dir=cache_dir)

        # Try to get non-existent embedding
        result = manager.get_embedding("nonexistent_hash")

        assert result is None

        manager.close()

    def test_cache_key_includes_model_name(self, cache_dir: Path) -> None:
        """Test that cache keys include model name for versioning."""
        embedding = np.random.rand(384).astype(np.float32)
        content_hash = "test_hash"

        # Cache with model A
        manager_a = CacheManager(cache_dir=cache_dir, model_name="model-a")
        manager_a.set_embedding(content_hash, embedding)

        # Try to retrieve with model B
        manager_b = CacheManager(cache_dir=cache_dir, model_name="model-b")
        result = manager_b.get_embedding(content_hash)

        # Should be a cache miss because model name differs
        assert result is None

        # But should work with same model name
        result_a = manager_a.get_embedding(content_hash)
        assert result_a is not None
        assert np.array_equal(result_a, embedding)

        manager_a.close()
        manager_b.close()

    def test_cache_key_includes_dimension(self, cache_dir: Path) -> None:
        """Test that cache keys include embedding dimension."""
        manager = CacheManager(cache_dir=cache_dir)
        content_hash = "test_hash"

        # Cache 384-dim embedding
        embedding_384 = np.random.rand(384).astype(np.float32)
        manager.set_embedding(content_hash, embedding_384)

        # Try to retrieve with different dimension
        result_768 = manager.get_embedding(content_hash, dimension=768)
        assert result_768 is None  # Cache miss due to dimension mismatch

        # Should work with correct dimension
        result_384 = manager.get_embedding(content_hash, dimension=384)
        assert result_384 is not None
        assert np.array_equal(result_384, embedding_384)

        manager.close()

    def test_numpy_array_serialization_roundtrip(self, cache_dir: Path) -> None:
        """Test that numpy arrays survive pickle serialization."""
        manager = CacheManager(cache_dir=cache_dir)

        # Test various array types and shapes
        test_cases = [
            np.random.rand(384).astype(np.float32),  # Float32
            np.random.rand(512).astype(np.float64),  # Float64
            np.array([1, 2, 3, 4], dtype=np.int32),  # Integer
            np.zeros(128, dtype=np.float32),  # Zeros
            np.ones(256, dtype=np.float32),  # Ones
        ]

        for i, embedding in enumerate(test_cases):
            content_hash = f"hash_{i}"
            manager.set_embedding(content_hash, embedding)

            cached = manager.get_embedding(content_hash, dimension=len(embedding))

            assert cached is not None
            assert cached.dtype == embedding.dtype
            assert np.array_equal(cached, embedding)

        manager.close()

    def test_clear_cache(self, cache_dir: Path) -> None:
        """Test clearing all cached embeddings."""
        manager = CacheManager(cache_dir=cache_dir)

        # Cache multiple embeddings
        for i in range(5):
            embedding = np.random.rand(384).astype(np.float32)
            manager.set_embedding(f"hash_{i}", embedding)

        # Verify they're cached
        for i in range(5):
            assert manager.get_embedding(f"hash_{i}") is not None

        # Clear cache
        manager.clear()

        # Verify all gone
        for i in range(5):
            assert manager.get_embedding(f"hash_{i}") is None

        manager.close()

    def test_get_stats(self, cache_dir: Path) -> None:
        """Test retrieving cache statistics."""
        manager = CacheManager(cache_dir=cache_dir, model_name="test-model")

        stats = manager.get_stats()

        assert "cache_dir" in stats
        assert "size_bytes" in stats
        assert "model_name" in stats
        assert stats["cache_dir"] == str(cache_dir)
        assert stats["model_name"] == "test-model"
        assert isinstance(stats["size_bytes"], int)
        assert stats["size_bytes"] >= 0

        manager.close()

    def test_cache_size_increases(self, cache_dir: Path) -> None:
        """Test that cache size increases when adding embeddings."""
        manager = CacheManager(cache_dir=cache_dir)

        initial_stats = manager.get_stats()
        initial_size = initial_stats["size_bytes"]

        # Add large embedding
        large_embedding = np.random.rand(1024).astype(np.float32)
        manager.set_embedding("large_hash", large_embedding)

        final_stats = manager.get_stats()
        final_size = final_stats["size_bytes"]

        # Size should have increased
        assert final_size > initial_size

        manager.close()

    def test_context_manager(self, cache_dir: Path) -> None:
        """Test using manager as context manager."""
        embedding = np.random.rand(384).astype(np.float32)

        with CacheManager(cache_dir=cache_dir) as manager:
            manager.set_embedding("test_hash", embedding)
            cached = manager.get_embedding("test_hash")
            assert np.array_equal(cached, embedding)

        # Manager should be closed after context

    def test_expiration(self, cache_dir: Path) -> None:
        """Test that embeddings can be cached with expiration."""
        manager = CacheManager(cache_dir=cache_dir)
        embedding = np.random.rand(384).astype(np.float32)

        # Cache with short expiration
        manager.set_embedding("temp_hash", embedding, expire=1)

        # Should be available immediately
        cached = manager.get_embedding("temp_hash")
        assert cached is not None

        # After expiration, may or may not be available
        # (diskcache eviction is not immediate)

        manager.close()

    def test_overwrite_existing_embedding(self, cache_dir: Path) -> None:
        """Test overwriting an existing cached embedding."""
        manager = CacheManager(cache_dir=cache_dir)

        # Cache initial embedding
        embedding1 = np.ones(384, dtype=np.float32)
        manager.set_embedding("test_hash", embedding1)

        # Overwrite with new embedding
        embedding2 = np.zeros(384, dtype=np.float32)
        manager.set_embedding("test_hash", embedding2)

        # Should get the new embedding
        cached = manager.get_embedding("test_hash")
        assert cached is not None
        assert np.array_equal(cached, embedding2)
        assert not np.array_equal(cached, embedding1)

        manager.close()

    def test_multiple_embeddings_different_hashes(self, cache_dir: Path) -> None:
        """Test caching multiple embeddings with different content hashes."""
        manager = CacheManager(cache_dir=cache_dir)

        embeddings = {}
        for i in range(10):
            embedding = np.random.rand(384).astype(np.float32)
            content_hash = f"hash_{i}"
            embeddings[content_hash] = embedding
            manager.set_embedding(content_hash, embedding)

        # Verify all embeddings can be retrieved
        for content_hash, expected_embedding in embeddings.items():
            cached = manager.get_embedding(content_hash)
            assert cached is not None
            assert np.array_equal(cached, expected_embedding)

        manager.close()

    def test_set_and_get_chunk_embedding(self, cache_dir: Path) -> None:
        """Test caching and retrieving chunk embeddings."""
        manager = CacheManager(cache_dir=cache_dir)

        doc_hash = "doc_abc123"
        chunk_index = 0
        chunk_hash = "chunk_xyz789"
        embedding = np.random.rand(384).astype(np.float32)

        # Set chunk embedding
        manager.set_chunk_embedding(doc_hash, chunk_index, chunk_hash, embedding)

        # Get chunk embedding
        cached = manager.get_chunk_embedding(doc_hash, chunk_index, chunk_hash)

        assert cached is not None
        assert isinstance(cached, np.ndarray)
        assert cached.shape == (384,)
        assert np.array_equal(cached, embedding)

        manager.close()

    def test_chunk_cache_miss_returns_none(self, cache_dir: Path) -> None:
        """Test that chunk cache miss returns None."""
        manager = CacheManager(cache_dir=cache_dir)

        cached = manager.get_chunk_embedding("nonexistent", 0, "chunk_hash")

        assert cached is None

        manager.close()

    def test_chunk_cache_key_includes_all_components(self, cache_dir: Path) -> None:
        """Test that chunk cache key includes doc_hash, index, chunk_hash, model, dimension."""
        manager = CacheManager(cache_dir=cache_dir, model_name="test-model")

        doc_hash = "doc123"
        chunk_index = 5
        chunk_hash = "chunk456"
        embedding = np.random.rand(384).astype(np.float32)

        manager.set_chunk_embedding(doc_hash, chunk_index, chunk_hash, embedding)

        # Should retrieve with exact same parameters
        cached = manager.get_chunk_embedding(doc_hash, chunk_index, chunk_hash)
        assert cached is not None

        # Different doc_hash should miss
        cached = manager.get_chunk_embedding("different_doc", chunk_index, chunk_hash)
        assert cached is None

        # Different chunk_index should miss
        cached = manager.get_chunk_embedding(doc_hash, 99, chunk_hash)
        assert cached is None

        # Different chunk_hash should miss
        cached = manager.get_chunk_embedding(doc_hash, chunk_index, "different_chunk")
        assert cached is None

        manager.close()

    def test_multiple_chunks_same_document(self, cache_dir: Path) -> None:
        """Test caching multiple chunks from the same document."""
        manager = CacheManager(cache_dir=cache_dir)

        doc_hash = "doc_multi_chunk"
        num_chunks = 10

        embeddings = []
        for i in range(num_chunks):
            embedding = np.random.rand(384).astype(np.float32)
            chunk_hash = f"chunk_{i}"
            embeddings.append((i, chunk_hash, embedding))
            manager.set_chunk_embedding(doc_hash, i, chunk_hash, embedding)

        # Verify all chunks can be retrieved
        for chunk_index, chunk_hash, expected_embedding in embeddings:
            cached = manager.get_chunk_embedding(doc_hash, chunk_index, chunk_hash)
            assert cached is not None
            assert np.array_equal(cached, expected_embedding)

        manager.close()

    def test_chunk_cache_with_different_dimensions(self, cache_dir: Path) -> None:
        """Test that chunk cache keys are dimension-specific."""
        manager = CacheManager(cache_dir=cache_dir)

        doc_hash = "doc_dim_test"
        chunk_index = 0
        chunk_hash = "chunk_dim"

        # Cache embedding with dimension 384
        embedding_384 = np.random.rand(384).astype(np.float32)
        manager.set_chunk_embedding(doc_hash, chunk_index, chunk_hash, embedding_384)

        # Cache embedding with dimension 768
        embedding_768 = np.random.rand(768).astype(np.float32)
        manager.set_chunk_embedding(doc_hash, chunk_index, chunk_hash, embedding_768)

        # Should retrieve different embeddings based on dimension
        cached_384 = manager.get_chunk_embedding(doc_hash, chunk_index, chunk_hash, dimension=384)
        cached_768 = manager.get_chunk_embedding(doc_hash, chunk_index, chunk_hash, dimension=768)

        assert cached_384 is not None
        assert cached_768 is not None
        assert cached_384.shape == (384,)
        assert cached_768.shape == (768,)
        assert np.array_equal(cached_384, embedding_384)
        assert np.array_equal(cached_768, embedding_768)

        manager.close()

    def test_chunk_cache_invalidation_on_chunk_change(self, cache_dir: Path) -> None:
        """Test that changing chunk content invalidates cache (different chunk_hash)."""
        manager = CacheManager(cache_dir=cache_dir)

        doc_hash = "doc_change"
        chunk_index = 0

        # Cache original chunk
        chunk_hash_v1 = "chunk_original"
        embedding_v1 = np.random.rand(384).astype(np.float32)
        manager.set_chunk_embedding(doc_hash, chunk_index, chunk_hash_v1, embedding_v1)

        # Cache modified chunk (different hash)
        chunk_hash_v2 = "chunk_modified"
        embedding_v2 = np.random.rand(384).astype(np.float32)
        manager.set_chunk_embedding(doc_hash, chunk_index, chunk_hash_v2, embedding_v2)

        # Both versions should be cached independently
        cached_v1 = manager.get_chunk_embedding(doc_hash, chunk_index, chunk_hash_v1)
        cached_v2 = manager.get_chunk_embedding(doc_hash, chunk_index, chunk_hash_v2)

        assert cached_v1 is not None
        assert cached_v2 is not None
        assert np.array_equal(cached_v1, embedding_v1)
        assert np.array_equal(cached_v2, embedding_v2)
        assert not np.array_equal(cached_v1, cached_v2)

        manager.close()
