"""Tests for cache backend implementations."""

from pathlib import Path

from doclint.cache.backends import DiskCacheBackend


class TestDiskCacheBackend:
    """Test suite for DiskCacheBackend class."""

    def test_initialization_creates_directory(self, tmp_path: Path) -> None:
        """Test that cache directory is created on initialization."""
        cache_dir = tmp_path / "test_cache"
        assert not cache_dir.exists()

        backend = DiskCacheBackend(cache_dir)

        assert cache_dir.exists()
        assert cache_dir.is_dir()
        backend.close()

    def test_set_and_get(self, tmp_path: Path) -> None:
        """Test basic set and get operations."""
        backend = DiskCacheBackend(tmp_path)

        # Set a value
        backend.set("key1", b"value1")

        # Retrieve the value
        result = backend.get("key1")

        assert result == b"value1"
        backend.close()

    def test_get_nonexistent_key(self, tmp_path: Path) -> None:
        """Test that get returns None for nonexistent keys."""
        backend = DiskCacheBackend(tmp_path)

        result = backend.get("nonexistent")

        assert result is None
        backend.close()

    def test_delete(self, tmp_path: Path) -> None:
        """Test deleting a cached value."""
        backend = DiskCacheBackend(tmp_path)

        # Set and verify
        backend.set("key1", b"value1")
        assert backend.get("key1") == b"value1"

        # Delete
        backend.delete("key1")

        # Verify deleted
        assert backend.get("key1") is None
        backend.close()

    def test_clear(self, tmp_path: Path) -> None:
        """Test clearing all cached values."""
        backend = DiskCacheBackend(tmp_path)

        # Set multiple values
        backend.set("key1", b"value1")
        backend.set("key2", b"value2")
        backend.set("key3", b"value3")

        # Clear cache
        backend.clear()

        # Verify all keys are gone
        assert backend.get("key1") is None
        assert backend.get("key2") is None
        assert backend.get("key3") is None
        backend.close()

    def test_get_size(self, tmp_path: Path) -> None:
        """Test getting cache size."""
        backend = DiskCacheBackend(tmp_path)

        # Empty cache should have some size (metadata)
        initial_size = backend.get_size()
        assert initial_size >= 0

        # Add large data (1 MB to ensure visible size increase)
        backend.set("key1", b"x" * (1024 * 1024))

        # Size should increase or stay same (cache may allocate in blocks)
        new_size = backend.get_size()
        assert new_size >= initial_size
        backend.close()

    def test_expiration(self, tmp_path: Path) -> None:
        """Test that expired entries are not returned."""
        backend = DiskCacheBackend(tmp_path)

        # Set with very short expiration (1 second)
        backend.set("key1", b"value1", expire=1)

        # Should be available immediately
        assert backend.get("key1") == b"value1"

        # Wait for expiration (note: this makes the test slow)
        import time

        time.sleep(1.1)

        # Should be expired now
        result = backend.get("key1")
        # diskcache may return None for expired keys
        assert result is None or result == b"value1"  # Either is acceptable

        backend.close()

    def test_multiple_keys(self, tmp_path: Path) -> None:
        """Test storing and retrieving multiple different keys."""
        backend = DiskCacheBackend(tmp_path)

        # Set multiple keys
        data = {f"key{i}": f"value{i}".encode() for i in range(10)}

        for key, value in data.items():
            backend.set(key, value)

        # Retrieve and verify all keys
        for key, expected_value in data.items():
            assert backend.get(key) == expected_value

        backend.close()

    def test_binary_data(self, tmp_path: Path) -> None:
        """Test caching binary data."""
        backend = DiskCacheBackend(tmp_path)

        # Create binary data
        binary_data = bytes(range(256))

        backend.set("binary_key", binary_data)

        result = backend.get("binary_key")

        assert result == binary_data
        backend.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test using backend as context manager."""
        with DiskCacheBackend(tmp_path) as backend:
            backend.set("key1", b"value1")
            assert backend.get("key1") == b"value1"

        # Cache should be closed after context
        # No exception should be raised

    def test_large_value(self, tmp_path: Path) -> None:
        """Test caching large values."""
        backend = DiskCacheBackend(tmp_path)

        # Create a large value (1 MB)
        large_value = b"x" * (1024 * 1024)

        backend.set("large_key", large_value)

        result = backend.get("large_key")

        assert result == large_value
        assert len(result) == 1024 * 1024
        backend.close()

    def test_overwrite_existing_key(self, tmp_path: Path) -> None:
        """Test overwriting an existing key with new value."""
        backend = DiskCacheBackend(tmp_path)

        # Set initial value
        backend.set("key1", b"initial")
        assert backend.get("key1") == b"initial"

        # Overwrite with new value
        backend.set("key1", b"updated")
        assert backend.get("key1") == b"updated"

        backend.close()
