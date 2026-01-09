"""FAISS-based vector index for efficient chunk similarity search."""

import logging
from typing import List, Optional, Tuple

import numpy as np

from ..core.document import Chunk

logger = logging.getLogger(__name__)

# FAISS import with fallback
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed. Vector similarity search will be unavailable.")


class ChunkIndex:
    """FAISS-based vector index for efficient chunk similarity search.

    This class provides O(log n) similarity search over chunk embeddings using
    Facebook's FAISS library. It supports:
    - Building an index from chunk embeddings
    - Finding k-nearest neighbors for any chunk
    - Finding all chunk pairs above a similarity threshold

    The index uses inner product similarity (equivalent to cosine similarity
    when embeddings are L2 normalized).

    Attributes:
        dimension: Dimensionality of embeddings (default: 384 for MiniLM)
        chunks: List of chunks in the index
        index: FAISS index object

    Example:
        >>> index = ChunkIndex(dimension=384)
        >>> index.build(chunks)
        >>> similar = index.find_similar(chunks[0], k=5)
        >>> for idx, score in similar:
        ...     print(f"Chunk {idx}: similarity {score:.3f}")
    """

    def __init__(self, dimension: int = 384) -> None:
        """Initialize the chunk index.

        Args:
            dimension: Dimensionality of embeddings. Default is 384 for
                      all-MiniLM-L6-v2 model.
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu is required for vector similarity search. "
                "Install with: pip install faiss-cpu"
            )

        self.dimension = dimension
        self.chunks: List[Chunk] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self._embeddings: Optional[np.ndarray] = None

        logger.debug(f"Initialized ChunkIndex with dimension={dimension}")

    def build(self, chunks: List[Chunk]) -> None:
        """Build the FAISS index from chunk embeddings.

        This method normalizes embeddings and builds an inner product index
        for efficient cosine similarity search. Complexity: O(n).

        Args:
            chunks: List of chunks with embeddings. Chunks with None embeddings
                   are skipped with a warning.

        Raises:
            ValueError: If no chunks have valid embeddings
        """
        # Filter chunks with valid embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None]

        if not valid_chunks:
            logger.warning("No chunks with valid embeddings provided")
            self.chunks = []
            self.index = None
            self._embeddings = None
            return

        if len(valid_chunks) < len(chunks):
            logger.warning(f"Skipped {len(chunks) - len(valid_chunks)} chunks without embeddings")

        self.chunks = valid_chunks

        # Stack embeddings into matrix
        embeddings = np.vstack([c.embedding for c in valid_chunks]).astype(np.float32)

        # L2 normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        self._embeddings = embeddings

        # Create index with inner product (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        logger.info(f"Built FAISS index with {len(valid_chunks)} chunks")

    def find_similar(
        self, chunk: Chunk, k: int = 10, exclude_self: bool = True
    ) -> List[Tuple[int, float]]:
        """Find k most similar chunks to the given chunk.

        Uses approximate nearest neighbor search with complexity O(log n).

        Args:
            chunk: Query chunk with embedding
            k: Number of similar chunks to return (default: 10)
            exclude_self: If True, exclude the query chunk from results

        Returns:
            List of (chunk_index, similarity_score) tuples sorted by similarity
            descending. Similarity scores are in range [-1, 1] (cosine similarity).

        Raises:
            ValueError: If index not built or chunk has no embedding
        """
        if self.index is None or len(self.chunks) == 0:
            logger.warning("Index not built or empty")
            return []

        if chunk.embedding is None:
            raise ValueError("Chunk has no embedding")

        # Prepare query vector
        query = chunk.embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        # Search for k+1 to account for potential self-match
        search_k = k + 1 if exclude_self else k
        search_k = min(search_k, len(self.chunks))

        distances, indices = self.index.search(query, search_k)

        # Convert to list of tuples and filter
        results: List[Tuple[int, float]] = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            # Skip self if requested
            if exclude_self:
                if (
                    self.chunks[idx].document_path == chunk.document_path
                    and self.chunks[idx].index == chunk.index
                    and self.chunks[idx].chunk_hash == chunk.chunk_hash
                ):
                    continue

            results.append((int(idx), float(score)))

            if len(results) >= k:
                break

        return results

    def find_all_similar_pairs(
        self,
        threshold: float = 0.85,
        exclude_same_document: bool = False,
    ) -> List[Tuple[int, int, float]]:
        """Find all chunk pairs with similarity above threshold.

        This method efficiently finds all pairs of semantically similar chunks
        across the corpus. It avoids redundant comparisons by only returning
        each pair once (i < j).

        Args:
            threshold: Minimum cosine similarity to consider (default: 0.85)
            exclude_same_document: If True, skip pairs from the same document

        Returns:
            List of (chunk_i_index, chunk_j_index, similarity) tuples
            where i < j, sorted by similarity descending.
        """
        if self.index is None or len(self.chunks) < 2:
            logger.warning("Index not built or has fewer than 2 chunks")
            return []

        pairs: List[Tuple[int, int, float]] = []
        seen_pairs: set[Tuple[int, int]] = set()

        # For each chunk, find its neighbors and filter by threshold
        # Using batch search for efficiency
        n_chunks = len(self.chunks)

        # Determine k based on expected density - start with reasonable value
        # For most corpora, similar chunks are rare, so k=50 is usually enough
        k = min(50, n_chunks)

        logger.debug(f"Searching for similar pairs with threshold={threshold}")

        # Batch search all chunks
        assert self._embeddings is not None
        distances, indices = self.index.search(self._embeddings, k)

        for i in range(n_chunks):
            for j_pos in range(k):
                j = indices[i, j_pos]
                score = distances[i, j_pos]

                if j == -1 or j <= i:  # Skip self and already-seen pairs
                    continue

                if score < threshold:
                    continue  # Below threshold

                # Skip same-document pairs if requested
                if exclude_same_document:
                    if self.chunks[i].document_path == self.chunks[j].document_path:
                        continue

                pair_key = (min(i, j), max(i, j))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pairs.append((i, j, float(score)))

        # Sort by similarity descending
        pairs.sort(key=lambda x: x[2], reverse=True)

        logger.info(f"Found {len(pairs)} similar chunk pairs above threshold {threshold}")
        return pairs

    def get_chunk(self, index: int) -> Chunk:
        """Get chunk by index.

        Args:
            index: Index of chunk in the index

        Returns:
            Chunk at the given index

        Raises:
            IndexError: If index out of range
        """
        return self.chunks[index]

    def clear(self) -> None:
        """Clear the index and release memory."""
        self.chunks = []
        self.index = None
        self._embeddings = None
        logger.debug("Cleared ChunkIndex")

    def __len__(self) -> int:
        """Return number of chunks in the index."""
        return len(self.chunks)
