# 02 - Core Module

## Overview
Implement the core data models, configuration management, and orchestration layer.

## Subtasks

### 2.1 Document Metadata Model
- [ ] Create `doclint/core/document.py`
- [ ] Implement `DocumentMetadata` dataclass:
  - `author: Optional[str]`
  - `created: Optional[datetime]`
  - `modified: Optional[datetime]`
  - `version: Optional[str]`
  - `title: Optional[str]`
  - `tags: list[str]`
  - `custom: Dict[str, Any]`
- [ ] Add `__post_init__` to initialize empty lists/dicts
- [ ] Add type hints and docstrings

### 2.2 Document Model
- [ ] Implement `Document` dataclass in `doclint/core/document.py`:
  - `path: Path`
  - `content: str`
  - `metadata: DocumentMetadata`
  - `file_type: str`
  - `size_bytes: int`
  - `content_hash: str`
  - `embedding: Optional[np.ndarray]`
  - `chunks: list[str]`
- [ ] Implement `from_file()` classmethod
- [ ] Add content hashing using SHA-256
- [ ] Add `__post_init__` for initialization
- [ ] Add comprehensive docstrings

### 2.3 Configuration Models (Pydantic)
- [ ] Create `doclint/core/config.py`
- [ ] Implement `DetectorConfig` base settings class
- [ ] Implement `ConflictDetectorConfig`:
  - `enabled: bool = True`
  - `similarity_threshold: float = 0.85` (validated 0-1)
- [ ] Implement `CompletenessDetectorConfig`:
  - `enabled: bool = True`
  - `required_metadata: list[str]`
  - `min_content_length: int = 100`

### 2.4 Main Configuration Class
- [ ] Implement `DocLintConfig` in `doclint/core/config.py`:
  - `recursive: bool = True`
  - `cache_enabled: bool = True`
  - `cache_dir: Optional[Path] = None`
  - `max_workers: int = 4` (validated > 0)
  - Nested detector configs
- [ ] Implement `load()` classmethod for TOML loading
- [ ] Add configuration file search order:
  1. Explicit path from CLI
  2. `.doclint.toml` in current directory
  3. `~/.config/doclint/config.toml`
  4. Default config
- [ ] Add TOML parsing with `tomli`
- [ ] Add validation with Pydantic Field validators

### 2.5 Scanner - File Discovery
- [ ] Create `doclint/core/scanner.py`
- [ ] Implement `Scanner` class constructor:
  - Accept `parser_registry`, `embedding_generator`, `detector_registry`, `cache_manager`
  - Accept `max_workers` parameter
- [ ] Implement `_discover_files()` method:
  - Walk directory tree (recursive or not)
  - Filter by parseable extensions
  - Return `List[Path]`
- [ ] Add glob pattern support (`**/*` vs `*`)
- [ ] Add file filtering logic

### 2.6 Scanner - Document Parsing
- [ ] Implement `_parse_single_file()` static method:
  - Accept file path and parser
  - Return Document or None on error
  - Include error handling
- [ ] Implement `_parse_documents()` async method:
  - Use `ProcessPoolExecutor` for CPU-bound parsing
  - Accept files list, progress tracker
  - Submit parsing jobs to worker pool
  - Collect results as they complete
  - Update progress bar
  - Return `List[Document]`

### 2.7 Scanner - Embedding Generation
- [ ] Implement `_generate_embeddings()` async method:
  - Check cache for each document (by content_hash)
  - Collect uncached documents
  - Batch generate embeddings (batch_size=32)
  - Assign embeddings to documents
  - Store new embeddings in cache
  - Support progress tracking

### 2.8 Scanner - Detector Execution
- [ ] Implement `_run_detectors()` async method:
  - Iterate through registered detectors
  - Call `detect()` on each detector
  - Collect results in dictionary
  - Return `Dict[str, List[Issue]]`

### 2.9 Scanner - Main Orchestration
- [ ] Implement `scan_directory()` async method:
  - Accept path, recursive flag, progress instance
  - Call file discovery
  - Call document parsing
  - Call embedding generation
  - Call detector execution
  - Return combined results
- [ ] Add comprehensive error handling
- [ ] Add logging throughout
- [ ] Support progress reporting with rich.Progress

### 2.10 Exception Classes
- [ ] Create `doclint/core/exceptions.py`
- [ ] Define `DocLintError` base exception
- [ ] Define `ParsingError` for parsing failures
- [ ] Define `ConfigurationError` for config issues
- [ ] Define `CacheError` for cache problems
- [ ] Add proper exception hierarchy

### 2.11 Unit Tests - Document Models
- [ ] Create `tests/test_core/test_document.py`
- [ ] Test `DocumentMetadata` initialization
- [ ] Test `Document` creation
- [ ] Test `Document.from_file()` with mock parser
- [ ] Test content hashing
- [ ] Test edge cases (empty content, missing metadata)

### 2.12 Unit Tests - Configuration
- [ ] Create `tests/test_core/test_config.py`
- [ ] Test default configuration
- [ ] Test TOML loading
- [ ] Test configuration validation (invalid ranges)
- [ ] Test nested config loading
- [ ] Test file search order

### 2.13 Unit Tests - Scanner
- [ ] Create `tests/test_core/test_scanner.py`
- [ ] Test file discovery (recursive vs non-recursive)
- [ ] Test parsing orchestration (mock parsers)
- [ ] Test embedding generation (mock embedding generator)
- [ ] Test detector execution (mock detectors)
- [ ] Test error handling
- [ ] Test progress tracking

## Success Criteria
- ✅ Document and Metadata models fully implemented
- ✅ Configuration management with Pydantic working
- ✅ Scanner can discover, parse, embed, and detect
- ✅ All core unit tests passing
- ✅ Type checking passes with MyPy
- ✅ Code formatted with Black

## Dependencies
- Requires: Project Setup (01)
- Required by: Parsers (03), Embeddings (04), Detectors (05)

## Notes
- This is the heart of the application
- Focus on clean abstractions and error handling
- Ensure async/await is used correctly
- Mock heavy dependencies in tests
