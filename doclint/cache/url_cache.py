"""URL validation cache for external link checking."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from platformdirs import user_cache_dir

from .backends import DiskCacheBackend

logger = logging.getLogger(__name__)


class URLCache:
    """Cache for URL validation results.

    This cache stores the results of URL availability checks to avoid repeated
    network requests to the same URLs. Each cache entry contains:
    - is_valid: Whether the URL was accessible (True/False)
    - status_code: HTTP status code (or None if unreachable)
    - timestamp: When the check was performed

    The cache uses a TTL (time-to-live) to ensure stale results are re-validated.

    Attributes:
        cache_dir: Path to cache directory
        ttl: Time-to-live in seconds for cache entries
        backend: Underlying cache backend

    Example:
        >>> cache = URLCache(ttl=86400)  # 24 hour TTL
        >>> cache.set("https://example.com", is_valid=True, status_code=200)
        >>> result = cache.get("https://example.com")
        >>> if result:
        ...     is_valid, status_code = result
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl: int = 86400,  # 24 hours
    ) -> None:
        """Initialize URL cache.

        Args:
            cache_dir: Custom cache directory (if None, uses platform-specific default)
            ttl: Time-to-live in seconds for cache entries (default: 86400 = 24 hours)
        """
        # Use platform-specific cache directory if not specified
        if cache_dir is None:
            cache_path = user_cache_dir("doclint", "doclint")
            cache_dir = Path(cache_path) / "urls"

        self.cache_dir = cache_dir
        self.ttl = ttl
        self.backend = DiskCacheBackend(cache_dir)

        logger.debug(f"Initialized URLCache at {cache_dir} with TTL={ttl}s")

    def _make_cache_key(self, url: str) -> str:
        """Create cache key for URL.

        Args:
            url: URL to create key for

        Returns:
            Cache key string
        """
        # Use simple prefix to distinguish from embedding cache
        return f"url:v1:{url}"

    def get(self, url: str) -> Optional[Tuple[bool, Optional[int]]]:
        """Retrieve cached URL validation result.

        Args:
            url: URL to look up

        Returns:
            Tuple of (is_valid, status_code) if cached and not expired,
            None if not found or expired
        """
        key = self._make_cache_key(url)

        try:
            # Retrieve serialized data from backend
            data = self.backend.get(key)

            if data is None:
                logger.debug(f"Cache miss for URL: {url}")
                return None

            # Deserialize JSON data
            result = json.loads(data.decode("utf-8"))

            # Validate result structure
            if not isinstance(result, dict) or "timestamp" not in result:
                logger.warning(f"Invalid cache data for URL: {url}")
                return None

            # Check if cache entry has expired
            timestamp = result["timestamp"]
            age = time.time() - timestamp

            if age > self.ttl:
                logger.debug(f"Cache expired for URL: {url} (age: {age:.0f}s)")
                return None

            is_valid = result.get("is_valid", False)
            status_code = result.get("status_code")

            logger.debug(
                f"Cache hit for URL: {url} "
                f"(valid={is_valid}, status={status_code}, age={age:.0f}s)"
            )
            return (is_valid, status_code)

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to deserialize cached URL result: {e}")
            return None

        except Exception as e:
            logger.warning(f"Error retrieving cached URL result: {e}")
            return None

    def set(self, url: str, is_valid: bool, status_code: Optional[int] = None) -> None:
        """Cache URL validation result.

        Args:
            url: URL that was checked
            is_valid: Whether the URL was accessible
            status_code: HTTP status code (None if connection failed)
        """
        key = self._make_cache_key(url)

        try:
            # Create result dictionary
            result: Dict[str, object] = {
                "is_valid": is_valid,
                "status_code": status_code,
                "timestamp": time.time(),
            }

            # Serialize to JSON bytes
            data = json.dumps(result).encode("utf-8")

            # Store in backend with TTL
            self.backend.set(key, data, expire=self.ttl)

            logger.debug(f"Cached URL validation: {url} (valid={is_valid}, status={status_code})")

        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to serialize URL result: {e}")

        except Exception as e:
            logger.warning(f"Error caching URL result: {e}")

    def clear(self) -> None:
        """Clear all cached URL results.

        Warning:
            This operation removes all cached URL data permanently.
        """
        try:
            self.backend.clear()
            logger.info("Cleared all cached URL validations")
        except Exception as e:
            logger.warning(f"Error clearing URL cache: {e}")

    def get_stats(self) -> Dict[str, object]:
        """Get URL cache statistics.

        Returns:
            Dictionary with cache statistics:
                - cache_dir: Path to cache directory
                - size_bytes: Total cache size in bytes
                - ttl: Time-to-live in seconds
        """
        return {
            "cache_dir": str(self.cache_dir),
            "size_bytes": self.backend.get_size(),
            "ttl": self.ttl,
        }

    def close(self) -> None:
        """Close cache and release resources."""
        self.backend.close()

    def __enter__(self) -> "URLCache":
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()
