"""High-level cache manager for embeddings and scan results."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from platformdirs import user_cache_dir

from .backends import DiskCacheBackend

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of embeddings with versioned keys.

    This manager provides a high-level interface for caching embeddings with:
    - Automatic cache path determination using platformdirs
    - Versioned cache keys to handle model changes
    - Pickle serialization for numpy arrays
    - Graceful error handling
    - Cache statistics tracking

    The cache key format is: `emb:v1:{model_name}:{dimension}:{content_hash}`
    This ensures cache invalidation when the model changes.

    Attributes:
        cache_dir: Path to cache directory
        model_name: Name of the embedding model (for cache key versioning)
        backend: Underlying cache backend

    Example:
        >>> manager = CacheManager(model_name="all-MiniLM-L6-v2")
        >>> embedding = np.random.rand(384)
        >>> manager.set_embedding("abc123", embedding)
        >>> cached = manager.get_embedding("abc123")
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize cache manager.

        Args:
            cache_dir: Custom cache directory (if None, uses platform-specific default)
            model_name: Name of embedding model (used in cache keys)
        """
        # Use platform-specific cache directory if not specified
        if cache_dir is None:
            cache_path = user_cache_dir("doclint", "doclint")
            cache_dir = Path(cache_path)

        self.cache_dir = cache_dir
        self.model_name = model_name
        self.backend = DiskCacheBackend(cache_dir)

        logger.debug(f"Initialized CacheManager at {cache_dir}")

    def _make_cache_key(self, content_hash: str, dimension: int) -> str:
        """Create versioned cache key for embedding.

        The key includes:
        - Version prefix (v1) for future schema changes
        - Model name to handle model changes
        - Embedding dimension to handle dimension changes
        - Content hash to identify the document

        Args:
            content_hash: SHA-256 hash of document content
            dimension: Embedding vector dimension

        Returns:
            Versioned cache key string
        """
        return f"emb:v1:{self.model_name}:{dimension}:{content_hash}"

    def get_embedding(
        self, content_hash: str, dimension: int = 384
    ) -> Optional[np.ndarray[Any, Any]]:
        """Retrieve cached embedding.

        Args:
            content_hash: SHA-256 hash of document content
            dimension: Expected embedding dimension (default: 384 for MiniLM)

        Returns:
            Cached embedding array, or None if not found or deserialization fails
        """
        key = self._make_cache_key(content_hash, dimension)

        try:
            # Retrieve serialized data from backend
            data = self.backend.get(key)

            if data is None:
                logger.debug(f"Cache miss for key: {key}")
                return None

            # Deserialize numpy array
            embedding = pickle.loads(data)

            # Validate that we got a numpy array
            if not isinstance(embedding, np.ndarray):
                logger.warning(f"Cached data is not a numpy array: {type(embedding)}")
                return None

            logger.debug(f"Cache hit for key: {key}")
            return embedding

        except (pickle.UnpicklingError, AttributeError, EOFError) as e:
            # Deserialization errors - log warning but don't fail
            logger.warning(f"Failed to deserialize cached embedding: {e}")
            return None

        except Exception as e:
            # Unexpected errors - log warning but don't fail
            logger.warning(f"Error retrieving cached embedding: {e}")
            return None

    def set_embedding(
        self,
        content_hash: str,
        embedding: np.ndarray[Any, Any],
        expire: int = 2592000,
    ) -> None:
        """Cache an embedding vector.

        Args:
            content_hash: SHA-256 hash of document content
            embedding: Embedding vector to cache
            expire: Expiration time in seconds (default: 30 days)
        """
        key = self._make_cache_key(content_hash, embedding.shape[0])

        try:
            # Serialize numpy array using pickle
            data = pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)

            # Store in backend
            self.backend.set(key, data, expire=expire)

            logger.debug(f"Cached embedding with key: {key}")

        except (pickle.PicklingError, AttributeError) as e:
            # Serialization errors - log warning but don't fail
            logger.warning(f"Failed to serialize embedding: {e}")

        except Exception as e:
            # Unexpected errors - log warning but don't fail
            logger.warning(f"Error caching embedding: {e}")

    def _make_chunk_cache_key(
        self,
        doc_hash: str,
        chunk_index: int,
        chunk_hash: str,
        dimension: int,
    ) -> str:
        """Create versioned cache key for chunk embedding.

        The key includes:
        - Prefix identifying chunk embeddings
        - Version prefix (v1) for future schema changes
        - Model name to handle model changes
        - Embedding dimension to handle dimension changes
        - Document hash to group chunks from same document
        - Chunk index for ordering
        - Chunk hash to detect chunk content changes

        Args:
            doc_hash: SHA-256 hash of parent document content
            chunk_index: Index of chunk in document (0-based)
            chunk_hash: SHA-256 hash of chunk text
            dimension: Embedding vector dimension

        Returns:
            Versioned cache key string
        """
        return f"chunk:v1:{self.model_name}:{dimension}:{doc_hash}:{chunk_index}:{chunk_hash}"

    def get_chunk_embedding(
        self,
        doc_hash: str,
        chunk_index: int,
        chunk_hash: str,
        dimension: int = 384,
    ) -> Optional[np.ndarray[Any, Any]]:
        """Retrieve cached chunk embedding.

        Args:
            doc_hash: SHA-256 hash of parent document content
            chunk_index: Index of chunk in document (0-based)
            chunk_hash: SHA-256 hash of chunk text
            dimension: Expected embedding dimension (default: 384 for MiniLM)

        Returns:
            Cached embedding array, or None if not found or deserialization fails
        """
        key = self._make_chunk_cache_key(doc_hash, chunk_index, chunk_hash, dimension)

        try:
            # Retrieve serialized data from backend
            data = self.backend.get(key)

            if data is None:
                logger.debug(f"Cache miss for chunk key: {key}")
                return None

            # Deserialize numpy array
            embedding = pickle.loads(data)

            # Validate that we got a numpy array
            if not isinstance(embedding, np.ndarray):
                logger.warning(f"Cached chunk data is not a numpy array: {type(embedding)}")
                return None

            logger.debug(f"Cache hit for chunk key: {key}")
            return embedding

        except (pickle.UnpicklingError, AttributeError, EOFError) as e:
            # Deserialization errors - log warning but don't fail
            logger.warning(f"Failed to deserialize cached chunk embedding: {e}")
            return None

        except Exception as e:
            # Unexpected errors - log warning but don't fail
            logger.warning(f"Error retrieving cached chunk embedding: {e}")
            return None

    def set_chunk_embedding(
        self,
        doc_hash: str,
        chunk_index: int,
        chunk_hash: str,
        embedding: np.ndarray[Any, Any],
        expire: int = 2592000,
    ) -> None:
        """Cache a chunk embedding vector.

        Args:
            doc_hash: SHA-256 hash of parent document content
            chunk_index: Index of chunk in document (0-based)
            chunk_hash: SHA-256 hash of chunk text
            embedding: Embedding vector to cache
            expire: Expiration time in seconds (default: 30 days)
        """
        key = self._make_chunk_cache_key(doc_hash, chunk_index, chunk_hash, embedding.shape[0])

        try:
            # Serialize numpy array using pickle
            data = pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)

            # Store in backend
            self.backend.set(key, data, expire=expire)

            logger.debug(f"Cached chunk embedding with key: {key}")

        except (pickle.PicklingError, AttributeError) as e:
            # Serialization errors - log warning but don't fail
            logger.warning(f"Failed to serialize chunk embedding: {e}")

        except Exception as e:
            # Unexpected errors - log warning but don't fail
            logger.warning(f"Error caching chunk embedding: {e}")

    def clear(self) -> None:
        """Clear all cached embeddings.

        Warning:
            This operation removes all cached data permanently.
        """
        try:
            self.backend.clear()
            logger.info("Cleared all cached embeddings")
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics:
                - cache_dir: Path to cache directory
                - size_bytes: Total cache size in bytes
                - model_name: Current model name
        """
        return {
            "cache_dir": str(self.cache_dir),
            "size_bytes": self.backend.get_size(),
            "model_name": self.model_name,
        }

    def close(self) -> None:
        """Close cache and release resources."""
        self.backend.close()

    def __enter__(self) -> "CacheManager":
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()
