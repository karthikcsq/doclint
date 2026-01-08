"""Cache backend implementations for persistent storage."""

from pathlib import Path
from typing import Optional

import diskcache


class DiskCacheBackend:
    """Disk-based cache backend using diskcache library.

    This backend provides persistent caching with automatic eviction policies
    and thread-safe operations. It wraps the diskcache.Cache class to provide
    a simplified interface for DocLint's caching needs.

    Attributes:
        cache_dir: Directory where cache files are stored
        cache: Underlying diskcache.Cache instance

    Example:
        >>> backend = DiskCacheBackend("/tmp/my_cache")
        >>> backend.set("key1", b"value1", expire=3600)
        >>> data = backend.get("key1")
        >>> backend.clear()
    """

    def __init__(self, cache_dir: Path) -> None:
        """Initialize disk cache backend.

        Args:
            cache_dir: Directory to store cache files (will be created if needed)
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize diskcache with the specified directory
        self.cache: diskcache.Cache = diskcache.Cache(str(cache_dir))

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve value from cache.

        Args:
            key: Cache key to look up

        Returns:
            Cached value as bytes, or None if key not found or expired

        Example:
            >>> backend = DiskCacheBackend(Path("/tmp/cache"))
            >>> value = backend.get("my_key")
        """
        try:
            value = self.cache.get(key, default=None)
            # Validate that we got bytes or None
            if value is not None and not isinstance(value, bytes):
                return None
            return value
        except Exception:
            # Silently handle any cache read errors
            return None

    def set(self, key: str, value: bytes, expire: int = 2592000) -> None:
        """Store value in cache with optional expiration.

        Args:
            key: Cache key
            value: Value to cache (as bytes)
            expire: Expiration time in seconds (default: 30 days = 2592000 seconds)

        Example:
            >>> backend = DiskCacheBackend(Path("/tmp/cache"))
            >>> backend.set("my_key", b"my_value", expire=3600)
        """
        try:
            self.cache.set(key, value, expire=expire)
        except Exception:
            # Silently handle any cache write errors
            # This ensures caching failures don't break the main workflow
            pass

    def delete(self, key: str) -> None:
        """Delete a key from cache.

        Args:
            key: Cache key to delete

        Example:
            >>> backend = DiskCacheBackend(Path("/tmp/cache"))
            >>> backend.delete("my_key")
        """
        try:
            self.cache.delete(key)
        except Exception:
            # Silently handle delete errors
            pass

    def clear(self) -> None:
        """Clear all entries from the cache.

        Warning:
            This operation removes all cached data permanently.

        Example:
            >>> backend = DiskCacheBackend(Path("/tmp/cache"))
            >>> backend.clear()
        """
        try:
            self.cache.clear()
        except Exception:
            # Silently handle clear errors
            pass

    def get_size(self) -> int:
        """Get total cache size in bytes.

        Returns:
            Total size of cached data in bytes

        Example:
            >>> backend = DiskCacheBackend(Path("/tmp/cache"))
            >>> size = backend.get_size()
            >>> print(f"Cache size: {size / 1024 / 1024:.2f} MB")
        """
        try:
            return self.cache.volume()  # type: ignore
        except Exception:
            # Return 0 if size calculation fails
            return 0

    def close(self) -> None:
        """Close the cache and release resources.

        Example:
            >>> backend = DiskCacheBackend(Path("/tmp/cache"))
            >>> backend.close()
        """
        try:
            self.cache.close()
        except Exception:
            # Silently handle close errors
            pass

    def __enter__(self) -> "DiskCacheBackend":
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit - closes cache."""
        self.close()
