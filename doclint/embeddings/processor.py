"""Document processing pipeline for chunking and embedding generation."""

import logging
from typing import Optional

from ..cache.manager import CacheManager
from ..core.document import Chunk, Document
from ..utils.hashing import hash_content
from ..utils.text import chunk_text
from .base import BaseEmbeddingGenerator

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes documents by chunking and generating embeddings.

    This processor implements a RAG-like architecture where documents are split
    into chunks, and each chunk is embedded independently. This enables:
    - Fine-grained conflict detection across documents
    - Document-independent similarity search
    - Efficient caching at chunk level
    - Incremental updates when documents change

    Attributes:
        generator: Embedding generator to use
        cache: Cache manager for storing/retrieving embeddings
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks

    Example:
        >>> processor = DocumentProcessor(generator, cache)
        >>> processor.process_document(document)
        >>> # Document now has chunks with embeddings
        >>> len(document.chunks)
        5
        >>> document.chunks[0].embedding.shape
        (384,)
    """

    def __init__(
        self,
        generator: BaseEmbeddingGenerator,
        cache: Optional[CacheManager] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        """Initialize document processor.

        Args:
            generator: Embedding generator to use for chunks
            cache: Optional cache manager (if None, no caching)
            chunk_size: Size of each chunk in characters (default: 512)
            chunk_overlap: Overlap between chunks in characters (default: 50)
        """
        self.generator = generator
        self.cache = cache
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.debug(
            f"Initialized DocumentProcessor " f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
        )

    def process_document(self, document: Document) -> None:
        """Process document by chunking and generating embeddings.

        This method:
        1. Splits document content into overlapping chunks
        2. Creates Chunk objects with metadata
        3. Attempts to load embeddings from cache
        4. Generates embeddings for uncached chunks
        5. Stores new embeddings in cache
        6. Populates document.chunks with embedded chunks

        The document is modified in-place.

        Args:
            document: Document to process (modified in-place)
        """
        logger.info(f"Processing document: {document.path}")

        # Step 1: Split document into text chunks
        text_chunks = chunk_text(
            document.content,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

        if not text_chunks:
            logger.warning(f"No chunks generated for {document.path}")
            return

        logger.debug(f"Split document into {len(text_chunks)} chunks")

        # Step 2: Create Chunk objects
        chunks = []
        uncached_chunks = []
        start_pos = 0

        for i, text in enumerate(text_chunks):
            # Compute chunk hash
            chunk_hash = hash_content(text)

            # Create chunk object
            chunk = Chunk(
                text=text,
                index=i,
                document_path=document.path,
                chunk_hash=chunk_hash,
                start_pos=start_pos,
                end_pos=start_pos + len(text),
            )

            # Step 3: Try to load from cache
            if self.cache is not None:
                cached_embedding = self.cache.get_chunk_embedding(
                    doc_hash=document.content_hash,
                    chunk_index=i,
                    chunk_hash=chunk_hash,
                    dimension=self.generator.get_embedding_dimension(),
                )

                if cached_embedding is not None:
                    chunk.embedding = cached_embedding
                    logger.debug(f"Loaded chunk {i} from cache")
                else:
                    uncached_chunks.append(chunk)
            else:
                uncached_chunks.append(chunk)

            chunks.append(chunk)
            start_pos += len(text) - self.chunk_overlap

        # Step 4: Generate embeddings for uncached chunks (batch processing)
        if uncached_chunks:
            logger.info(f"Generating embeddings for {len(uncached_chunks)}/{len(chunks)} chunks")

            # Extract texts for batch processing
            uncached_texts = [chunk.text for chunk in uncached_chunks]

            # Generate embeddings in batch (more efficient)
            embeddings = self.generator.generate_batch(uncached_texts)

            # Step 5: Assign embeddings and cache them
            for chunk, embedding in zip(uncached_chunks, embeddings):
                chunk.embedding = embedding

                # Cache the embedding
                if self.cache is not None:
                    self.cache.set_chunk_embedding(
                        doc_hash=document.content_hash,
                        chunk_index=chunk.index,
                        chunk_hash=chunk.chunk_hash,
                        embedding=embedding,
                    )
        else:
            logger.info(f"All {len(chunks)} chunks loaded from cache")

        # Step 6: Store chunks in document
        document.chunks = chunks

        logger.info(
            f"Finished processing {document.path}: "
            f"{len(chunks)} chunks, "
            f"{len(uncached_chunks)} generated, "
            f"{len(chunks) - len(uncached_chunks)} cached"
        )

    def process_documents(self, documents: list[Document]) -> None:
        """Process multiple documents.

        Args:
            documents: List of documents to process (modified in-place)
        """
        logger.info(f"Processing {len(documents)} documents...")

        for i, document in enumerate(documents, 1):
            logger.info(f"Processing document {i}/{len(documents)}")
            self.process_document(document)

        # Calculate cache statistics
        total_chunks = sum(len(doc.chunks) for doc in documents)
        logger.info(f"Finished processing {len(documents)} documents ({total_chunks} total chunks)")


def get_all_chunks(documents: list[Document]) -> list[Chunk]:
    """Extract all chunks from a list of documents.

    This is a convenience function for getting a flat list of all chunks
    across all documents. Useful for document-independent similarity search.

    Args:
        documents: List of documents with processed chunks

    Returns:
        Flat list of all chunks from all documents

    Example:
        >>> documents = [doc1, doc2, doc3]
        >>> all_chunks = get_all_chunks(documents)
        >>> # Now can compare any chunk to any other chunk
        >>> for chunk_a in all_chunks:
        ...     for chunk_b in all_chunks:
        ...         similarity = cosine_similarity(chunk_a.embedding, chunk_b.embedding)
    """
    all_chunks = []
    for document in documents:
        all_chunks.extend(document.chunks)
    return all_chunks
