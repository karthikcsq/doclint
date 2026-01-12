# 05 - Detectors

## Overview
Implement detection algorithms for conflicts and completeness. The conflict detector uses **chunk-level embeddings** with **vector indexing** for efficient similarity search.

## Architecture

### Conflict Detection Strategy
With chunk-level embeddings, we compare chunks across documents rather than whole documents. This catches fine-grained conflicts that document-level embeddings would miss.

```
Problem with Document-Level:
  Doc A: "Python basics... JavaScript is single-threaded"
  Doc B: "SQL guide... JavaScript has multi-threading via workers"

  Document embeddings: [Python-heavy] vs [SQL-heavy]
  Cosine similarity: LOW → Conflict MISSED! ❌

Solution with Chunk-Level:
  Chunk A₅: "JavaScript is single-threaded"
  Chunk B₈: "JavaScript has multi-threading via workers"

  Chunk embeddings: Both about JavaScript
  Cosine similarity: HIGH → Conflict DETECTED! ✅
```

### Scalability with Vector Indexing
Brute-force O(n²) comparison doesn't scale:
- 1,000 docs × 10 chunks = 10,000 chunks → 50M comparisons
- 10,000 docs × 10 chunks = 100,000 chunks → 5B comparisons

Solution: Use **FAISS** for O(n log n) similarity search via approximate nearest neighbors.

```
┌─────────────────────────────────────────────────────────────┐
│                  CONFLICT DETECTION FLOW                     │
├─────────────────────────────────────────────────────────────┤
│  1. Load all chunk embeddings (from DiskCache)              │
│                         ↓                                    │
│  2. Build FAISS index from embeddings → O(n)                │
│                         ↓                                    │
│  3. For each chunk: query k-nearest neighbors → O(log n)    │
│                         ↓                                    │
│  4. Filter pairs above similarity threshold                  │
│                         ↓                                    │
│  5. Check for contradictions (heuristics or LLM)            │
│                         ↓                                    │
│  6. Return conflict Issues                                   │
│                                                              │
│  Total complexity: O(n log n) instead of O(n²)              │
└─────────────────────────────────────────────────────────────┘
```

## Subtasks

### 5.1 Base Detector Interface
- [ ] Create `doclint/detectors/base.py`
- [ ] Define `IssueSeverity` enum:
  - INFO
  - WARNING
  - CRITICAL
- [ ] Define `Issue` dataclass:
  - `severity: IssueSeverity`
  - `detector: str`
  - `title: str`
  - `description: str`
  - `documents: List[Path]`
  - `chunks: Optional[List[Chunk]]` (for chunk-level issues)
  - `details: Dict[str, Any]`
  - `to_dict() -> Dict[str, Any]` method
- [ ] Define `BaseDetector` abstract class:
  - `name: str` class attribute
  - `description: str` class attribute
  - `detect(documents: List[Document]) -> List[Issue]` abstract async method

### 5.2 Vector Index for Similarity Search
- [ ] Add `faiss-cpu` to dependencies in `pyproject.toml`
- [ ] Create `doclint/detectors/vector_index.py`
- [ ] Implement `ChunkIndex` class:
  ```python
  class ChunkIndex:
      """FAISS-based vector index for efficient chunk similarity search."""

      def __init__(self, dimension: int = 384):
          """Initialize index for given embedding dimension."""

      def build(self, chunks: List[Chunk]) -> None:
          """Build index from chunk embeddings. O(n)"""

      def find_similar(self, chunk: Chunk, k: int = 10) -> List[Tuple[int, float]]:
          """Find k most similar chunks. O(log n)"""

      def find_all_similar_pairs(
          self,
          threshold: float = 0.85
      ) -> List[Tuple[int, int, float]]:
          """Find all chunk pairs above similarity threshold."""

      def clear(self) -> None:
          """Clear the index."""
  ```
- [ ] Use `IndexFlatIP` for cosine similarity (with L2 normalization)
- [ ] Handle edge cases:
  - Empty chunk list
  - Single chunk
  - Chunks with None embeddings
- [ ] Add optional `IndexIVFFlat` for corpora >100K chunks (faster but approximate)

### 5.3 Conflict Detector - Core Logic
- [ ] Create `doclint/detectors/conflicts.py`
- [ ] Implement `ConflictDetector` class extending `BaseDetector`
- [ ] Set `name = "conflict"` and description
- [ ] Add constructor parameters:
  - `similarity_threshold: float` (default: 0.85)
  - `k_neighbors: int` (default: 10) - candidates per chunk
- [ ] Implement `detect()` async method:
  - Return early if <2 documents
  - Extract all chunks using `get_all_chunks(documents)`
  - Build `ChunkIndex` from chunk embeddings
  - Find similar chunk pairs above threshold
  - Filter out same-document pairs (optional, configurable)
  - Extract conflict details for each pair
  - Return list of Issue objects

### 5.4 Conflict Detector - Conflict Extraction
- [ ] Implement `_extract_conflict()` method:
  - Accept two chunks and similarity score
  - Check text similarity (not just embedding)
  - Determine if chunks are contradictory vs. just similar
  - Extract excerpts from both chunks
  - Return conflict details dict or None
- [ ] Implement `_text_similarity()` static method:
  - Calculate Jaccard similarity on word level
  - Return float 0-1
- [ ] Implement `_are_contradictory()` method:
  - Use heuristics (negation words, antonyms)
  - Optional: Use LLM for sophisticated contradiction detection
  - Return bool

### 5.5 Conflict Detector - Issue Generation
- [ ] Generate meaningful Issue objects:
  ```python
  Issue(
      severity=IssueSeverity.WARNING,
      detector="conflict",
      title="Potential conflict detected",
      description="Similar content with possible contradiction",
      documents=[chunk_a.document_path, chunk_b.document_path],
      chunks=[chunk_a, chunk_b],
      details={
          "similarity": 0.92,
          "chunk_a_text": "...",
          "chunk_b_text": "...",
          "chunk_a_position": {"start": 1024, "end": 1536},
          "chunk_b_position": {"start": 512, "end": 1024},
      }
  )
  ```
- [ ] Categorize severity:
  - CRITICAL: High similarity + clear contradiction
  - WARNING: High similarity, unclear if contradictory
  - INFO: Moderate similarity, potential duplicate content

### 5.6 Completeness Detector - Metadata Checks
- [ ] Create `doclint/detectors/completeness.py`
- [ ] Implement `CompletenessDetector` class extending `BaseDetector`
- [ ] Set `name = "completeness"` and description
- [ ] Add constructor with configuration:
  - `required_metadata: List[str]` (e.g., ["author", "created", "version"])
  - `min_content_length: int` (default: 100)
- [ ] Implement metadata validation:
  - Check for missing required metadata fields
  - Track missing fields per document

### 5.7 Completeness Detector - Content Checks
- [ ] Implement content validation:
  - Check content length
  - Flag too short documents
  - Check for placeholder text
- [ ] Implement `detect()` async method:
  - For each document:
    - Check metadata completeness
    - Check content quality
    - Collect all issues
    - Create Issue if any problems found
  - Return list of Issues

### 5.8 Completeness Detector - Link Checking (Optional)
- [ ] Implement `_find_broken_links()` method:
  - Extract URLs from content
  - Check internal file references
  - Test external URLs (optional, slow)
  - Return list of broken links
- [ ] Add link checking to detect() method
- [ ] Make link checking optional via config

### 5.9 Detector Registry
- [ ] Create `doclint/detectors/registry.py`
- [ ] Implement `DetectorRegistry` class:
  - Store detectors in dictionary
  - `register(detector: BaseDetector)` method
  - `get_detector(name: str) -> BaseDetector` method
  - `get_all_detectors() -> Dict[str, BaseDetector]` method
  - `run_all(documents: List[Document]) -> Dict[str, List[Issue]]` async method
- [ ] Add selective detector execution
- [ ] Support enable/disable via configuration

### 5.10 Detector Module Initialization
- [ ] Update `doclint/detectors/__init__.py`:
  - Import all detector classes
  - Export base classes and concrete detectors
  - Export `ChunkIndex` for direct use
  - Create default registry instance
  - Auto-register all detectors

### 5.11 Unit Tests - Vector Index
- [ ] Create `tests/test_detectors/test_vector_index.py`
- [ ] Test index building with mock embeddings
- [ ] Test similarity search returns correct neighbors
- [ ] Test threshold filtering
- [ ] Test edge cases (empty, single chunk)
- [ ] Test performance with 10K+ chunks

### 5.12 Unit Tests - Base Detector
- [ ] Create `tests/test_detectors/test_base.py`
- [ ] Test IssueSeverity enum
- [ ] Test Issue dataclass and to_dict()
- [ ] Test BaseDetector interface

### 5.13 Unit Tests - Conflict Detector
- [ ] Create `tests/test_detectors/test_conflicts.py`
- [ ] Test with no conflicts (low similarity)
- [ ] Test with identical chunks (should detect)
- [ ] Test with similar but different chunks
- [ ] Test cross-document conflict detection
- [ ] Test same-document chunk filtering (optional)
- [ ] Test similarity threshold configuration
- [ ] Test with <2 documents
- [ ] Mock embeddings and FAISS for fast tests

### 5.14 Unit Tests - Completeness Detector
- [ ] Create `tests/test_detectors/test_completeness.py`
- [ ] Test with complete documents
- [ ] Test with missing metadata
- [ ] Test with short content
- [ ] Test with broken links (if implemented)
- [ ] Test configuration of required fields

### 5.15 Unit Tests - Detector Registry
- [ ] Create `tests/test_detectors/test_registry.py`
- [ ] Test detector registration
- [ ] Test detector lookup
- [ ] Test running all detectors
- [ ] Test selective detector execution

### 5.16 Integration Tests
- [ ] Create `tests/test_integration/test_detectors.py`
- [ ] Test full detection pipeline with real documents
- [ ] Test chunk-level conflict detection end-to-end
- [ ] Test multiple detectors running together
- [ ] Test issue aggregation
- [ ] Verify performance with 1000+ documents

## Success Criteria
- ✅ All 2 core detectors implemented (conflict, completeness)
- ✅ Vector index (FAISS) integrated for efficient similarity search
- ✅ Chunk-level conflict detection working
- ✅ Detector registry working
- ✅ All unit tests passing (>90% coverage)
- ✅ Integration tests passing
- ✅ Issues correctly categorized by severity
- ✅ Performance acceptable (detect 1000 docs / 10K chunks in <20 seconds)

## Performance Targets
| Operation | Target | Complexity |
|-----------|--------|------------|
| Build FAISS index (10K chunks) | <1 second | O(n) |
| Find k-NN for one chunk | <1 ms | O(log n) |
| Full conflict detection (10K chunks) | <10 seconds | O(n log n) |
| Completeness detection (1K docs) | <1 second | O(n) |

## Dependencies
- Requires: Core Module (02), Embeddings (04)
- Required by: CLI (07), Reporters (06)
- New dependency: `faiss-cpu` (vector similarity search)

## Caching Strategy Summary

| Component | Purpose | Tool |
|-----------|---------|------|
| Embedding Storage | "Get embedding for chunk hash X" | DiskCache (Stage 04) |
| Similarity Search | "Find chunks similar to this one" | FAISS Index (Stage 05) |

**Key insight**: DiskCache and FAISS serve different purposes:
- **DiskCache**: Persist embeddings across runs (avoid regeneration)
- **FAISS**: Efficient similarity search at detection time (avoid O(n²))

## Notes
- Conflict detector is the most complex - requires vector index
- Completeness detector is highly configurable
- Focus on accuracy and meaningful issue detection
- FAISS index is ephemeral (rebuilt each scan) - persistence optional for huge corpora
