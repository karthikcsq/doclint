# 04 - Embeddings Layer

## Overview
Implement semantic embedding generation using sentence-transformers with efficient caching at the **chunk level**. This enables fine-grained conflict detection across document corpora.

## Architecture

### Chunk-Level Embedding Strategy
Instead of generating a single embedding per document, we split documents into chunks and embed each chunk independently. This enables:
- **Fine-grained conflict detection**: Find contradictions buried in different-topic documents
- **Efficient caching**: Only re-embed chunks that changed
- **Scalable similarity search**: Compare chunks across entire corpus

```
Document → Chunking → [Chunk₁, Chunk₂, ..., Chunkₙ]
                           ↓
                    Each chunk embedded independently
                           ↓
                    Cached by chunk hash (not doc hash)
```

### Data Model
```python
@dataclass
class Chunk:
    text: str
    index: int
    document_path: Path
    chunk_hash: str  # SHA-256 of chunk text
    embedding: Optional[np.ndarray] = None
    start_pos: int = 0
    end_pos: int = 0

@dataclass
class Document:
    # ... other fields ...
    chunks: list[Chunk] = field(default_factory=list)
```

## Subtasks

### 4.1 Model Management
- [x] Create `doclint/embeddings/base.py`
- [x] Implement `EmbeddingGenerator` class:
  - Lazy load sentence-transformer model
  - Support model selection (default: "all-MiniLM-L6-v2")
  - Support device selection (CPU/CUDA)
  - Cache loaded model in memory
  - `get_embedding_dimension() -> int` method
- [x] Add model download on first use
- [x] Add error handling for model loading failures

### 4.2 Embedding Generator - Single Text
- [x] Create `doclint/embeddings/generator.py`
- [x] Implement `EmbeddingGenerator` class constructor:
  - Accept model name (default: "all-MiniLM-L6-v2")
  - Accept device ("cpu" or "cuda")
- [x] Implement `generate()` method:
  - Accept text string
  - Generate embedding using model
  - Return numpy array (1D)
  - Handle empty text
  - Add error handling

### 4.3 Embedding Generator - Batch Processing
- [x] Implement `generate_batch()` method:
  - Accept list of text strings
  - Accept batch_size parameter (default: 32)
  - Batch encode using model (much faster)
  - Return list of numpy arrays
  - Handle empty batch
  - Add progress tracking option
- [x] Optimize for large batches
- [x] Add memory management for large datasets

### 4.4 Text Chunking
- [x] Implement `chunk_text()` helper function in `doclint/utils/text.py`:
  - Split text into chunks (default: 512 characters)
  - Support overlapping chunks (default: 50 characters)
  - Return list of chunk strings
- [x] Handle edge cases:
  - Empty text
  - Text shorter than chunk size
  - Unicode boundaries

### 4.5 Document Processor (Chunk-Level Pipeline)
- [x] Create `doclint/embeddings/processor.py`
- [x] Implement `DocumentProcessor` class:
  - Accept `EmbeddingGenerator`, `CacheManager`, chunk_size, chunk_overlap
  - `process_document(document: Document) -> None` method
  - `process_documents(documents: list[Document]) -> None` method
- [x] Processing workflow:
  1. Chunk document content
  2. Create `Chunk` objects with hashes
  3. Check cache for each chunk embedding
  4. Batch generate embeddings for uncached chunks
  5. Store new embeddings in cache
  6. Populate `document.chunks` with embedded chunks
- [x] Implement `get_all_chunks(documents) -> list[Chunk]` utility

### 4.6 Cache Backend - DiskCache
- [x] Create `doclint/cache/backends.py`
- [x] Implement `DiskCacheBackend` class:
  - Initialize with cache directory path
  - Use `diskcache.Cache`
  - `get(key: str) -> Optional[bytes]` method
  - `set(key: str, value: bytes, expire: int)` method
  - `delete(key: str)` method
  - `clear()` method
  - Handle serialization errors

### 4.7 Cache Manager - Document Level
- [x] Create `doclint/cache/manager.py`
- [x] Implement `CacheManager` class:
  - Initialize with cache directory (use platformdirs)
  - Default location: `~/.cache/doclint/` (Linux), `~/Library/Caches/doclint/` (Mac), `%LOCALAPPDATA%\doclint\Cache\` (Windows)
  - Use DiskCacheBackend
  - `get_embedding(content_hash: str) -> Optional[np.ndarray]` method
  - `set_embedding(content_hash: str, embedding: np.ndarray)` method
  - Use pickle for numpy array serialization
  - Set 30-day expiration

### 4.8 Cache Manager - Chunk Level
- [x] Add chunk-level caching methods to `CacheManager`:
  - `get_chunk_embedding(doc_hash, chunk_index, chunk_hash, dimension) -> Optional[np.ndarray]`
  - `set_chunk_embedding(doc_hash, chunk_index, chunk_hash, embedding, expire)`
- [x] Cache key format: `chunk:v1:{model}:{dim}:{doc_hash}:{chunk_idx}:{chunk_hash}`
- [x] Benefits:
  - Detect when individual chunks change
  - Reuse cached chunks even when document structure changes
  - Granular cache invalidation

### 4.9 Cache Statistics
- [x] Implement `get_stats()` method in CacheManager:
  - Return cache size in MB
  - Return number of cached items
  - Calculate hit rate (if tracked)
- [ ] Add cache hit/miss tracking
- [ ] Format stats for CLI display

### 4.10 Cache Invalidation
- [x] Implement cache versioning:
  - Include model name in cache key
  - Include embedding dimension in cache key
  - Auto-invalidate on model change
- [x] Implement `clear()` method:
  - Delete all cache entries
- [ ] Add selective invalidation by age

### 4.11 Utility Functions
- [x] Create `doclint/utils/hashing.py`
- [x] Implement `hash_content(text: str) -> str`:
  - Use SHA-256
  - Return hex digest
  - Handle unicode properly
- [x] Create `doclint/utils/text.py`
- [x] Implement `chunk_text()` function
- [ ] Implement additional text preprocessing utilities:
  - Normalize whitespace
  - Remove special characters (optional)
  - Truncate to max length

### 4.12 Unit Tests - Embedding Generator
- [x] Create `tests/test_embeddings/test_generator.py`
- [x] Test single embedding generation
- [x] Test batch embedding generation
- [x] Test empty text handling
- [x] Test embedding dimensions are correct
- [x] Mock SentenceTransformer to avoid slow model loading

### 4.13 Unit Tests - Document Processor
- [x] Create `tests/test_embeddings/test_processor.py`
- [x] Test document chunking
- [x] Test chunk embedding generation
- [x] Test cache integration
- [x] Test incremental updates (only changed chunks regenerated)

### 4.14 Unit Tests - Cache Manager
- [x] Create `tests/test_cache/test_manager.py`
- [x] Test document-level cache set/get
- [x] Test chunk-level cache set/get
- [x] Test cache miss returns None
- [x] Test numpy array serialization
- [x] Test cache clearing
- [ ] Test cache statistics
- [x] Use temporary directory for tests

### 4.15 Unit Tests - Cache Backend
- [x] Create `tests/test_cache/test_backends.py`
- [x] Test DiskCache operations
- [x] Test error handling
- [ ] Test concurrent access (if needed)

### 4.16 Integration Tests
- [ ] Create `tests/test_integration/test_embeddings_cache.py`
- [ ] Test full embedding + caching workflow
- [ ] Test cache hit/miss behavior with chunks
- [ ] Test performance improvement from caching
- [ ] Measure time difference (first run vs cached run)

## Success Criteria
- ✅ Chunk-level embedding generation working
- ✅ Batch processing implemented and fast
- ✅ Cache system working at chunk granularity
- ✅ Cache hit rate >80% on subsequent scans (when only some chunks change)
- ✅ All unit tests passing (>90% coverage)
- ⬜ Integration tests passing

## Performance Targets
- Generate 100 chunk embeddings in <10 seconds
- Cache lookup <1ms per chunk
- Memory usage <2GB for 10,000 chunks
- 90%+ cache hit rate when editing documents (unchanged chunks reused)

## Cache Efficiency Examples

### Scenario 1: Edit One Section of Document
```
Document (10 chunks):
  [Chunk 0] ─── unchanged ─── Cache HIT  ✓
  [Chunk 1] ─── unchanged ─── Cache HIT  ✓
  [Chunk 2] ─── EDITED ────── Cache MISS → Generate
  [Chunk 3] ─── unchanged ─── Cache HIT  ✓
  ...
  [Chunk 9] ─── unchanged ─── Cache HIT  ✓

Result: 1 embedding generated, 9 from cache (90% hit rate)
```

### Scenario 2: Add New Document
```
Existing corpus: 100 documents, all cached
New document: 10 chunks

Result: Only 10 new embeddings generated
```

## Dependencies
- Requires: Core Module (02)
- Required by: Detectors (05), Scanner integration

## Notes
- Chunk-level architecture enables fine-grained conflict detection
- DiskCache is for **embedding persistence** (hash-based lookup)
- Vector DB (Stage 05) is for **similarity search** (different purpose)
- Use mocked models in tests to avoid slow downloads
- Consider GPU support for large-scale usage (future)
