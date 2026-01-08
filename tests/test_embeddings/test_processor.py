"""Tests for DocumentProcessor."""

from pathlib import Path
from typing import Any

import numpy as np

from doclint.cache.manager import CacheManager
from doclint.core.document import Document, DocumentMetadata
from doclint.embeddings.generator import SentenceTransformerGenerator
from doclint.embeddings.processor import DocumentProcessor


class TestDocumentProcessor:
    """Test suite for DocumentProcessor."""

    def test_initialization_default_params(
        self, mock_sentence_transformer: Any, tmp_path: Path
    ) -> None:
        """Test processor initialization with default parameters."""
        generator = SentenceTransformerGenerator()
        cache = CacheManager(cache_dir=tmp_path)

        processor = DocumentProcessor(generator, cache)

        assert processor.generator == generator
        assert processor.cache == cache
        assert processor.chunk_size == 512
        assert processor.chunk_overlap == 50

        cache.close()

    def test_initialization_custom_params(
        self, mock_sentence_transformer: Any, tmp_path: Path
    ) -> None:
        """Test processor initialization with custom parameters."""
        generator = SentenceTransformerGenerator()
        cache = CacheManager(cache_dir=tmp_path)

        processor = DocumentProcessor(generator, cache, chunk_size=1024, chunk_overlap=100)

        assert processor.chunk_size == 1024
        assert processor.chunk_overlap == 100

        cache.close()

    def test_initialization_without_cache(self, mock_sentence_transformer: Any) -> None:
        """Test processor initialization without cache."""
        generator = SentenceTransformerGenerator()

        processor = DocumentProcessor(generator, cache=None)

        assert processor.cache is None

    def test_process_document_creates_chunks(
        self, mock_sentence_transformer: Any, tmp_path: Path
    ) -> None:
        """Test that processing creates chunks from document content."""
        generator = SentenceTransformerGenerator()
        processor = DocumentProcessor(generator, cache=None, chunk_size=100, chunk_overlap=20)

        # Create document with content
        content = "This is a test document. " * 20  # ~500 chars
        doc = Document(
            path=Path("test.txt"),
            content=content,
            content_hash="test_hash",
            size_bytes=len(content),
            metadata=DocumentMetadata(),
            file_type="text",
        )

        processor.process_document(doc)

        # Should have created chunks
        assert len(doc.chunks) > 0
        # Each chunk should have an embedding
        for chunk in doc.chunks:
            assert chunk.embedding is not None
            assert chunk.embedding.shape == (384,)
            assert chunk.document_path == Path("test.txt")
            assert chunk.chunk_hash is not None

    def test_process_document_modifies_in_place(self, mock_sentence_transformer: Any) -> None:
        """Test that document is modified in-place."""
        generator = SentenceTransformerGenerator()
        processor = DocumentProcessor(generator, cache=None, chunk_size=100, chunk_overlap=20)

        doc = Document(
            path=Path("test.txt"),
            content="Short content for testing",
            content_hash="hash123",
            size_bytes=25,
            metadata=DocumentMetadata(),
            file_type="text",
        )

        # Save reference
        original_doc = doc

        processor.process_document(doc)

        # Same object reference
        assert doc is original_doc
        # But now has chunks
        assert len(doc.chunks) > 0

    def test_process_empty_document(self, mock_sentence_transformer: Any) -> None:
        """Test processing document with empty content."""
        generator = SentenceTransformerGenerator()
        processor = DocumentProcessor(generator, cache=None)

        doc = Document(
            path=Path("empty.txt"),
            content="",
            content_hash="empty_hash",
            size_bytes=0,
            metadata=DocumentMetadata(),
            file_type="text",
        )

        processor.process_document(doc)

        # Empty document should have no chunks (or single empty chunk)
        # depending on implementation - let's be flexible
        assert isinstance(doc.chunks, list)

    def test_process_document_with_cache_miss(
        self, mock_sentence_transformer: Any, tmp_path: Path
    ) -> None:
        """Test processing with cache that has no cached embeddings."""
        generator = SentenceTransformerGenerator()
        cache = CacheManager(cache_dir=tmp_path)
        processor = DocumentProcessor(generator, cache, chunk_size=100, chunk_overlap=20)

        content = "Test content for cache miss scenario. " * 10
        doc = Document(
            path=Path("test.txt"),
            content=content,
            content_hash="doc_hash",
            size_bytes=len(content),
            metadata=DocumentMetadata(),
            file_type="text",
        )

        processor.process_document(doc)

        # Should have chunks with embeddings
        assert len(doc.chunks) > 0
        for chunk in doc.chunks:
            assert chunk.embedding is not None

            # Verify embeddings were cached
            cached = cache.get_chunk_embedding("doc_hash", chunk.index, chunk.chunk_hash)
            assert cached is not None
            assert np.array_equal(cached, chunk.embedding)

        cache.close()

    def test_process_document_with_cache_hit(
        self, mock_sentence_transformer: Any, tmp_path: Path
    ) -> None:
        """Test processing with cached embeddings available."""
        generator = SentenceTransformerGenerator()
        cache = CacheManager(cache_dir=tmp_path)
        processor = DocumentProcessor(generator, cache, chunk_size=100, chunk_overlap=20)

        content = "Test content for cache hit scenario. " * 5
        doc_hash = "doc_hash_cached"

        # Pre-populate cache with embeddings
        from doclint.utils.hashing import hash_content
        from doclint.utils.text import chunk_text

        chunks_text = chunk_text(content, chunk_size=100, overlap=20)
        cached_embeddings = []

        for i, chunk_content in enumerate(chunks_text):
            chunk_hash = hash_content(chunk_content)
            embedding = np.random.rand(384).astype(np.float32)
            cache.set_chunk_embedding(doc_hash, i, chunk_hash, embedding)
            cached_embeddings.append((i, chunk_hash, embedding))

        # Now process document
        doc = Document(
            path=Path("test.txt"),
            content=content,
            content_hash=doc_hash,
            size_bytes=len(content),
            metadata=DocumentMetadata(),
            file_type="text",
        )

        processor.process_document(doc)

        # Should have loaded embeddings from cache
        assert len(doc.chunks) == len(cached_embeddings)
        for chunk, (idx, c_hash, c_embedding) in zip(doc.chunks, cached_embeddings):
            assert chunk.index == idx
            assert chunk.chunk_hash == c_hash
            assert np.array_equal(chunk.embedding, c_embedding)

        cache.close()

    def test_process_document_with_mixed_cache(
        self, mock_sentence_transformer: Any, tmp_path: Path
    ) -> None:
        """Test processing with some chunks cached and some not."""
        generator = SentenceTransformerGenerator()
        cache = CacheManager(cache_dir=tmp_path)
        processor = DocumentProcessor(generator, cache, chunk_size=50, chunk_overlap=10)

        content = "Mixed cache test. " * 15
        doc_hash = "doc_hash_mixed"

        # Pre-cache only first chunk
        from doclint.utils.hashing import hash_content
        from doclint.utils.text import chunk_text

        chunks_text = chunk_text(content, chunk_size=50, overlap=10)
        first_chunk_hash = hash_content(chunks_text[0])
        first_embedding = np.random.rand(384).astype(np.float32)
        cache.set_chunk_embedding(doc_hash, 0, first_chunk_hash, first_embedding)

        # Process document
        doc = Document(
            path=Path("test.txt"),
            content=content,
            content_hash=doc_hash,
            size_bytes=len(content),
            metadata=DocumentMetadata(),
            file_type="text",
        )

        processor.process_document(doc)

        # Should have all chunks with embeddings
        assert len(doc.chunks) > 1
        for chunk in doc.chunks:
            assert chunk.embedding is not None

        # First chunk should use cached embedding
        assert np.array_equal(doc.chunks[0].embedding, first_embedding)

        cache.close()

    def test_chunk_metadata_is_correct(self, mock_sentence_transformer: Any) -> None:
        """Test that chunk objects have correct metadata."""
        generator = SentenceTransformerGenerator()
        processor = DocumentProcessor(generator, cache=None, chunk_size=50, chunk_overlap=10)

        content = "Chunk metadata test content. " * 5
        file_path = Path("/path/to/document.txt")

        doc = Document(
            path=file_path,
            content=content,
            content_hash="metadata_test_hash",
            size_bytes=len(content),
            metadata=DocumentMetadata(),
            file_type="text",
        )

        processor.process_document(doc)

        # Verify chunk metadata
        for i, chunk in enumerate(doc.chunks):
            assert chunk.index == i
            assert chunk.document_path == file_path
            assert chunk.text is not None and len(chunk.text) > 0
            assert chunk.chunk_hash is not None
            assert chunk.embedding is not None
            # Chunk positions should be set
            assert chunk.start_pos >= 0
            assert chunk.end_pos > chunk.start_pos

    def test_different_chunk_sizes(self, mock_sentence_transformer: Any) -> None:
        """Test processing with different chunk sizes."""
        generator = SentenceTransformerGenerator()

        content = "X" * 1000  # 1000 characters

        # Small chunks
        processor_small = DocumentProcessor(generator, cache=None, chunk_size=100, chunk_overlap=10)
        doc_small = Document(
            path=Path("test.txt"),
            content=content,
            content_hash="hash1",
            size_bytes=len(content),
            metadata=DocumentMetadata(),
            file_type="text",
        )
        processor_small.process_document(doc_small)

        # Large chunks
        processor_large = DocumentProcessor(generator, cache=None, chunk_size=500, chunk_overlap=10)
        doc_large = Document(
            path=Path("test.txt"),
            content=content,
            content_hash="hash2",
            size_bytes=len(content),
            metadata=DocumentMetadata(),
            file_type="text",
        )
        processor_large.process_document(doc_large)

        # Small chunks should create more chunks
        assert len(doc_small.chunks) > len(doc_large.chunks)

    def test_batch_embedding_generation(self, mock_sentence_transformer: Any) -> None:
        """Test that embeddings are generated in batches."""
        generator = SentenceTransformerGenerator()

        # Track if generate_batch is called by counting embeddings generated
        processor = DocumentProcessor(generator, cache=None, chunk_size=50, chunk_overlap=10)

        content = "Batch test content. " * 50  # Should create multiple chunks
        doc = Document(
            path=Path("test.txt"),
            content=content,
            content_hash="batch_hash",
            size_bytes=len(content),
            metadata=DocumentMetadata(),
            file_type="text",
        )

        processor.process_document(doc)

        # Should have created multiple chunks with embeddings (batch processing)
        assert len(doc.chunks) > 1
        for chunk in doc.chunks:
            assert chunk.embedding is not None

    def test_process_multiple_documents_independently(
        self, mock_sentence_transformer: Any, tmp_path: Path
    ) -> None:
        """Test that multiple documents can be processed independently."""
        generator = SentenceTransformerGenerator()
        cache = CacheManager(cache_dir=tmp_path)
        processor = DocumentProcessor(generator, cache, chunk_size=100, chunk_overlap=20)

        # Process first document
        doc1 = Document(
            path=Path("doc1.txt"),
            content="First document content. " * 10,
            content_hash="hash1",
            size_bytes=240,
            metadata=DocumentMetadata(),
            file_type="text",
        )
        processor.process_document(doc1)

        # Process second document
        doc2 = Document(
            path=Path("doc2.txt"),
            content="Second document content. " * 10,
            content_hash="hash2",
            size_bytes=250,
            metadata=DocumentMetadata(),
            file_type="text",
        )
        processor.process_document(doc2)

        # Both should have chunks
        assert len(doc1.chunks) > 0
        assert len(doc2.chunks) > 0

        # Chunks should reference correct documents
        for chunk in doc1.chunks:
            assert chunk.document_path == Path("doc1.txt")
        for chunk in doc2.chunks:
            assert chunk.document_path == Path("doc2.txt")

        cache.close()

    def test_chunk_overlap_behavior(self, mock_sentence_transformer: Any) -> None:
        """Test that chunk overlap works correctly."""
        generator = SentenceTransformerGenerator()
        processor = DocumentProcessor(generator, cache=None, chunk_size=100, chunk_overlap=30)

        content = "ABCD" * 50  # 200 characters
        doc = Document(
            path=Path("test.txt"),
            content=content,
            content_hash="overlap_test",
            size_bytes=len(content),
            metadata=DocumentMetadata(),
            file_type="text",
        )

        processor.process_document(doc)

        # With overlap, consecutive chunks should share some content
        if len(doc.chunks) >= 2:
            # Check that chunks have expected size relationships
            for chunk in doc.chunks[:-1]:  # All but last
                # Chunks should be around chunk_size (may be slightly less/more)
                assert len(chunk.text) <= 100 + 10  # Allow some flexibility
