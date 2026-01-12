# 08 - Testing Infrastructure

## Overview
Set up comprehensive testing infrastructure with fixtures, mocks, and test utilities.

## Subtasks

### 8.1 Pytest Configuration
- [ ] Configure pytest in `pyproject.toml`:
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  python_files = ["test_*.py"]
  python_classes = ["Test*"]
  python_functions = ["test_*"]
  asyncio_mode = "auto"
  markers = [
      "slow: marks tests as slow",
      "integration: marks tests as integration tests",
      "e2e: marks tests as end-to-end tests",
  ]
  ```
- [ ] Add coverage configuration
- [ ] Set up test output formatting

### 8.2 Conftest - Test Fixtures Base
- [ ] Create `tests/conftest.py`
- [ ] Add pytest plugins imports
- [ ] Add common fixtures available to all tests

### 8.3 Temporary Directory Fixtures
- [ ] Create `tmp_dir` fixture using pytest's tmp_path
- [ ] Create `sample_docs_dir` fixture:
  - Creates temp directory
  - Populates with sample documents
  - Returns path
  - Cleans up after test

### 8.4 Sample Document Fixtures
- [ ] Create `sample_pdf` fixture:
  - Generates a simple PDF
  - Returns path
- [ ] Create `sample_docx` fixture
- [ ] Create `sample_html` fixture
- [ ] Create `sample_markdown` fixture
- [ ] Create `sample_text` fixture
- [ ] Create documents with known content for testing

### 8.5 Problematic Document Fixtures
- [ ] Create `empty_pdf` fixture
- [ ] Create `corrupted_pdf` fixture
- [ ] Create `encrypted_pdf` fixture (if supported)
- [ ] Create `large_document` fixture (>1MB)
- [ ] Create `non_utf8_text` fixture

### 8.6 Conflicting Documents Fixtures
- [ ] Create `conflicting_documents` fixture:
  - Two docs with similar content but different answers
  - Known to trigger conflict detection
  - Returns list of paths
- [ ] Create `identical_documents` fixture:
  - Same content, different files
  - Should NOT trigger conflict

### 8.7 Stale Documents Fixtures
- [ ] Create `old_document` fixture:
  - Document with old timestamp
  - Modify file mtime to be >365 days ago
- [ ] Create `recent_document` fixture:
  - Document modified today

### 8.8 Mock Embedding Generator
- [ ] Create `mock_embedding_generator` fixture:
  - Returns deterministic embeddings
  - Fast (no model loading)
  - Consistent for same input
- [ ] Use for tests that don't need real embeddings

### 8.9 Mock Sentence Transformer Model
- [ ] Create `mock_sentence_transformer` fixture:
  - Mocks SentenceTransformer class
  - Returns random but consistent embeddings
  - Avoids model download
- [ ] Patch in embedding tests

### 8.10 Configuration Fixtures
- [ ] Create `default_config` fixture:
  - Returns DocLintConfig with defaults
- [ ] Create `custom_config` fixture:
  - Returns config with custom settings
- [ ] Create `config_file` fixture:
  - Creates temporary TOML config file
  - Returns path

### 8.11 Cache Fixtures
- [ ] Create `temp_cache` fixture:
  - Creates temporary cache directory
  - Initializes CacheManager
  - Cleans up after test
- [ ] Create `populated_cache` fixture:
  - Cache with some pre-populated embeddings

### 8.12 Scanner Fixtures
- [ ] Create `scanner` fixture:
  - Fully initialized Scanner
  - Uses mock components
  - Ready for testing
- [ ] Create `real_scanner` fixture:
  - Scanner with real components (for integration tests)

### 8.13 Issue Fixtures
- [ ] Create `sample_issue` fixture:
  - Returns a sample Issue object
- [ ] Create `critical_issue` fixture
- [ ] Create `warning_issue` fixture
- [ ] Create `issues_list` fixture (mixed severities)

### 8.14 Test Utilities - File Generation
- [ ] Create `tests/utils/file_generators.py`
- [ ] Implement `generate_pdf(content: str) -> Path`
- [ ] Implement `generate_docx(content: str) -> Path`
- [ ] Implement `generate_html(content: str) -> Path`
- [ ] Implement `generate_markdown(content: str, metadata: dict) -> Path`

### 8.15 Test Utilities - Assertions
- [ ] Create `tests/utils/assertions.py`
- [ ] Implement `assert_issues_equal()`
- [ ] Implement `assert_embeddings_similar()`
- [ ] Implement `assert_valid_json_output()`
- [ ] Implement `assert_valid_html_output()`

### 8.16 Test Utilities - Mocking
- [ ] Create `tests/utils/mocks.py`
- [ ] Implement `MockParser` class
- [ ] Implement `MockDetector` class
- [ ] Implement `MockReporter` class
- [ ] Add helper functions for creating mocks

### 8.17 Integration Test Setup
- [ ] Create `tests/test_integration/conftest.py`
- [ ] Add integration-specific fixtures
- [ ] Create sample document corpus for integration tests
- [ ] Set up shared test data

### 8.18 E2E Test Setup
- [ ] Create `tests/test_e2e/conftest.py`
- [ ] Add CLI runner fixture
- [ ] Add test workspace fixture (full directory structure)
- [ ] Create realistic document corpus

### 8.19 Performance Test Fixtures
- [ ] Create fixtures for performance testing:
  - 100 documents
  - 1000 documents
  - 10000 documents (optional)
- [ ] Add markers for slow tests
- [ ] Set up benchmarking

### 8.20 Coverage Configuration
- [ ] Configure coverage in `pyproject.toml`:
  ```toml
  [tool.coverage.run]
  source = ["doclint"]
  omit = ["tests/*", "*/test_*.py"]

  [tool.coverage.report]
  exclude_lines = [
      "pragma: no cover",
      "def __repr__",
      "raise AssertionError",
      "raise NotImplementedError",
      "if __name__ == .__main__.:",
  ]
  ```
- [ ] Set coverage thresholds

### 8.21 Test Data Organization
- [ ] Organize `tests/fixtures/` directory:
  ```
  tests/fixtures/
  ├── pdfs/
  │   ├── simple.pdf
  │   ├── with_metadata.pdf
  │   └── corrupted.pdf
  ├── docx/
  ├── html/
  ├── markdown/
  └── text/
  ```
- [ ] Add README explaining fixtures
- [ ] Version control sample files

### 8.22 Mock Network Requests
- [ ] Add `responses` or `requests-mock` library
- [ ] Create fixtures for mocking external URLs
- [ ] Mock sentence-transformers model downloads

### 8.23 Test Markers Setup
- [ ] Define test markers in pytest.ini
- [ ] Add `@pytest.mark.slow` for slow tests
- [ ] Add `@pytest.mark.integration` for integration tests
- [ ] Add `@pytest.mark.e2e` for E2E tests
- [ ] Add `@pytest.mark.requires_model` for tests needing ML model

### 8.24 Test Running Scripts
- [ ] Create `scripts/test_all.sh`:
  - Run all tests with coverage
- [ ] Create `scripts/test_unit.sh`:
  - Run only unit tests (fast)
- [ ] Create `scripts/test_integration.sh`:
  - Run integration tests
- [ ] Make scripts cross-platform (Windows compatible)

### 8.25 Test Documentation
- [ ] Create `tests/README.md`:
  - Explain test structure
  - Document how to run tests
  - Explain fixtures
  - Add contribution guidelines for tests
- [ ] Document mocking strategy
- [ ] Add examples of writing tests

## Success Criteria
- ✅ Comprehensive fixture library available
- ✅ All test utilities implemented
- ✅ Test data organized and documented
- ✅ Coverage configuration working
- ✅ Test markers properly defined
- ✅ Easy to write new tests using fixtures
- ✅ Fast unit tests (<10 seconds total)

## Dependencies
- Requires: All component modules
- Required by: All component tests

## Notes
- Good fixtures make writing tests easy and consistent
- Mocking is crucial for fast, reliable tests
- Organize fixtures by component for clarity
- Document fixtures well so others can use them
- Keep unit tests fast by mocking external dependencies
