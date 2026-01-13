# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DocLint** is a Python CLI tool that serves as "ESLint for AI Knowledge Bases" - it detects data quality issues in document collections before they cause AI agents to hallucinate or produce incorrect answers.

**Current Status**: v0.1.0 (Phase 1-5 complete - CLI fully functional with reporting system)

## Development Commands

### Environment Setup
```bash
poetry install              # Install all dependencies
poetry shell               # Activate virtual environment
```

### Testing
```bash
pytest                     # Run all tests with coverage
pytest tests/test_cli/     # Run specific test directory
pytest -v -k "test_name"   # Run specific test by name
```

### Code Quality
```bash
black doclint tests        # Format code (line-length: 100)
ruff check --fix doclint tests  # Lint and auto-fix issues
mypy doclint              # Type check with strict mode
pre-commit run --all-files # Run all pre-commit hooks
```

### Documentation
```bash
mkdocs serve              # Serve docs locally at http://127.0.0.1:8000
mkdocs build              # Build documentation site
```

### CLI Usage
```bash
doclint version           # Display version information
doclint scan <path>       # Scan directory for data quality issues
doclint scan <path> --format json --output report.json  # Export JSON report
doclint scan <path> --format html --output report.html  # Generate HTML report
doclint scan <path> --check-external-links  # Validate external URLs
doclint scan <path> --enable-llm  # Enable LLM-based conflict verification
doclint config-show       # Display current configuration
doclint formats           # List available output formats
python -m doclint         # Alternative entry point
```

## Architecture

DocLint follows a **layered architecture** with parallel component processing:

### CLI Layer (`doclint/cli/`)
- **Framework**: Typer + Rich for beautiful terminal output
- **Entry Point**: `doclint.cli.main:app` (defined in pyproject.toml)
- **Commands**: `version`, `scan`
- **Implementation**: `doclint/cli/main.py`

### Orchestration Layer - **IMPLEMENTED** (not integrated into CLI)
- **Scanner**: `doclint/core/scanner.py`
- Coordinates file discovery, parsing, embedding generation, detection, and reporting
- Parallel document parsing with configurable worker threads
- Progress reporting support
- Handles caching and error recovery
- **Status**: Fully implemented but needs CLI integration

### Component Layers (Parallel Processing)

**Parsers** (`doclint/parsers/`) - **IMPLEMENTED**
- Extract text and metadata from documents
- **Supported formats**: PDF, DOCX, HTML, Markdown, Plain Text
- **Registry system**: Automatic parser selection via `ParserRegistry`
- **Frontmatter support**: YAML and TOML in Markdown/HTML
- **Uses**: pypdf, python-docx, beautifulsoup4, lxml, markdown, toml
- **Key files**:
  - `base.py`: BaseParser abstract class
  - `text.py`: Plain text parser with encoding detection
  - `markdown.py`: Markdown with YAML/TOML frontmatter extraction
  - `html.py`: HTML with metadata and frontmatter extraction
  - `pdf.py`: PDF with metadata extraction
  - `docx.py`: DOCX with metadata and structure extraction
  - `registry.py`: Automatic parser selection by file extension

**Embeddings** (`doclint/embeddings/`) - **IMPLEMENTED**
- Generate semantic embeddings using sentence-transformers
- Model: sentence-transformers v^2.3.1 with torch backend
- Document chunking and processing for efficient embedding generation
- **Components**:
  - `base.py`: BaseEmbeddingGenerator abstract class
  - `generator.py`: SentenceTransformerGenerator implementation
  - `processor.py`: DocumentProcessor for chunking with overlap

**Cache** (`doclint/cache/`) - **IMPLEMENTED**
- Persistent caching layer using DiskCache
- Cross-platform cache paths via platformdirs
- Caches embeddings, scan results, and URL validation results
- **Components**:
  - `backends.py`: DiskCache backend implementation
  - `manager.py`: Cache manager for embeddings with versioning
  - `url_cache.py`: URL validation cache with configurable TTL

**Detectors** (`doclint/detectors/`) - **IMPLEMENTED**
- **BaseDetector** and Issue models with severity levels
- **CompletenessDetector**: Validate metadata, content quality, and links
  - Checks required metadata fields (author, created, etc.)
  - Validates minimum content length
  - Detects broken internal file links
  - **External link validation** (opt-in): Validates external URLs (http/https) asynchronously
    - Concurrent validation with configurable rate limiting
    - Cached results with 24-hour TTL to minimize network requests
    - HEAD request first, fallback to GET if needed
    - Configurable timeout (default: 5s)
- **ConflictDetector**: Find contradictory information using semantic similarity
  - FAISS-based vector similarity search (O(n log n) instead of O(n²))
  - Chunk-level conflict detection for fine-grained analysis
  - Heuristic contradiction detection (negation patterns, antonyms)
  - Optional LLM-based verification for uncertain cases
- **VectorIndex**: FAISS-based chunk indexing for efficient similarity search
- **LLM Verifier**: llama-cpp-python integration for local LLM verification
- **DetectorRegistry**: Dynamic detector management and execution

**Reporters** (`doclint/reporters/`) - **IMPLEMENTED**
- **BaseReporter**: Abstract base class with helper methods for Issue compatibility
- **ConsoleReporter**: Rich-formatted terminal output with severity grouping
- **JSONReporter**: Structured JSON output for CI/CD integration
- **HTMLReporter**: Interactive HTML reports with dark mode toggle
- **ReporterRegistry**: Dynamic reporter discovery and instantiation

## Code Quality Standards

### Black (Formatting)
- Line length: 100 characters
- Target version: Python 3.10+
- Run before committing: `black doclint tests`

### Ruff (Linting)
- Selected rules: E (pycodestyle errors), F (pyflakes), I (isort)
- Auto-fix enabled: `ruff check --fix doclint tests`

### mypy (Type Checking)
- Strict mode enabled
- `warn_return_any = true`
- `disallow_untyped_defs = true`
- `disallow_untyped_decorators = false` (allows third-party decorators)
- Global ignore for untyped imports (configured in pyproject.toml)

### pytest (Testing)
- Auto-coverage reporting: `--cov=doclint --cov-report=term-missing`
- Async mode: auto (via pytest-asyncio)
- Test paths: `tests/`

### Pre-commit Hooks
Configured checks:
- Trailing whitespace removal
- End-of-file fixer
- YAML/TOML validation
- Black formatting
- Ruff linting with auto-fix
- mypy type checking

## Technology Stack

### Core ML/Embeddings
- **sentence-transformers** v^2.3.1: Semantic embedding generation
- **torch** v^2.2.0: Deep learning backend
- **scikit-learn** v^1.4.0: Similarity metrics and clustering
- **numpy** v^1.26.0: Numerical operations

### Document Parsing
- **pypdf** v^4.0.0: PDF parsing
- **python-docx** v^1.1.0: DOCX parsing
- **beautifulsoup4** v^4.12.0 + **lxml** v^5.1.0: HTML parsing
- **markdown** v^3.5.0: Markdown parsing
- **toml** v^0.10.2: TOML frontmatter parsing
- **pyyaml** v^6.0.1: YAML frontmatter parsing
- **python-magic-bin** v^0.4.14: File type detection
- **chardet** v^5.2.0: Character encoding detection

### Caching & Storage
- **diskcache** v^5.6.0: Persistent caching
- **platformdirs** v^4.2.0: Cross-platform cache paths

### CLI & UI
- **typer** v^0.12.0: CLI framework
- **rich** v^13.7.0: Terminal formatting and output
- **tqdm** v^4.66.0: Progress bars

### Configuration
- **pydantic** v^2.6.0: Data validation
- **pydantic-settings** v^2.2.0: Settings management

### Async & Performance
- **aiohttp** v^3.9.0: Async HTTP client for external link validation
- **aiofiles** v^23.2.0: Async file I/O

## Development Status

### Phase 1: Complete (Commit: 5718805)
- Project scaffolding with Poetry
- CLI framework (Typer + Rich)
- Test structure with pytest
- Pre-commit hooks for code quality
- Basic commands: `version`, `scan` (stub)

### Phase 2: Complete (Commit: 6d4a0c6)
**Document parsers fully implemented with comprehensive test coverage:**
- ✅ Core parser infrastructure (`BaseParser`, `Document`, exceptions)
- ✅ Text parser with encoding detection (UTF-8, latin-1 fallback)
- ✅ Markdown parser with YAML and TOML frontmatter extraction
- ✅ HTML parser with metadata and frontmatter support
- ✅ PDF parser with metadata extraction and text rendering
- ✅ DOCX parser with metadata, headings, and structure extraction
- ✅ Parser registry for automatic format detection
- ✅ Integration tests covering multi-format workflows
- ✅ 187 tests across all parsers with edge case coverage

### Phase 3: Embeddings & Core Infrastructure (Complete)
**All core infrastructure implemented:**
- ✅ Embedding generation (sentence-transformers)
- ✅ Document chunking and processing
- ✅ Cache manager for embeddings with versioning
- ✅ Scanner orchestration layer (file discovery, parallel parsing, caching)
- ✅ Configuration system with TOML support

### Phase 4: Detectors (Complete) (Commit: c83f293)
**All detectors fully implemented:**
- ✅ BaseDetector abstract class with Issue model and severity levels
- ✅ **CompletenessDetector** for metadata, content quality, and link validation
  - Metadata completeness validation (required fields: author, created, etc.)
  - Content length validation with configurable minimums
  - Internal file link validation (checks relative paths)
  - **External link validation** (opt-in, async with caching)
    - Concurrent URL validation with rate limiting (default: 10 concurrent)
    - Smart HTTP requests: HEAD first, fallback to GET
    - Persistent caching with 24-hour TTL (URLCache + DiskCache)
    - Configurable timeout (default: 5s)
  - 28 comprehensive tests (82% coverage)
- ✅ **ConflictDetector** for finding contradictory information
  - FAISS-based vector similarity search (O(n log n))
  - Chunk-level conflict detection
  - Heuristic contradiction detection (negation patterns, antonyms)
  - Optional LLM verification support
- ✅ **VectorIndex** for efficient FAISS-based similarity search
- ✅ **LLM Verifier** for optional llama-cpp-python integration
- ✅ **DetectorRegistry** for dynamic detector management
- ✅ Integration tests for detector pipeline

### Phase 5: Reporters & CLI Integration (Complete) (Commit: 2436ec4)
**Full reporting system and CLI integration implemented:**
- ✅ **Reporters** - Complete output formatter system
  - BaseReporter abstract class with helper methods
  - **ConsoleReporter**: Rich-formatted terminal output with grouped issues
  - **JSONReporter**: Structured JSON for CI/CD integration
  - **HTMLReporter**: Beautiful interactive HTML reports with dark mode
  - **ReporterRegistry**: Dynamic reporter management and discovery
  - Comprehensive test coverage for all reporters
- ✅ **CLI Integration** - Scanner wired into CLI
  - `doclint scan <path>` - Full scanning with configurable options
  - Format selection: `--format console|json|html`
  - Output file support: `--output <file>`
  - Detector options: `--check-external-links`, `--enable-llm`
  - Exit codes: 0 (success), 1 (warnings), 2 (critical), 3 (error)
  - `doclint config-show` - Display current configuration
  - `doclint formats` - List available output formats
- ✅ **Issue Compatibility** - Unified interface
  - Added properties to scanner.Issue (title, description, detector, documents)
  - Reporters seamlessly handle both scanner and detector Issue types
  - Type-safe severity handling with enum conversions

### Remaining Work:
- ⬜ **Additional Tests**: ConflictDetector unit tests, end-to-end integration tests
- ⬜ **Documentation**: User guides, API docs, configuration examples
- ⬜ **Optimization**: Performance tuning, incremental scanning
- ⬜ **CI/CD**: GitHub Actions for automated testing and releases

## Key Files

### CLI & Configuration
- **Entry Point**: `doclint/cli/main.py` - CLI commands and Typer app
- **Version**: `doclint/version.py` - Version string ("0.1.0")
- **Package Init**: `doclint/__init__.py` - Package exports
- **Configuration**: `pyproject.toml` - Poetry deps, Black, Ruff, mypy, pytest config
- **Architecture**: `doclint_architecture.md` - Comprehensive implementation guide

### Core Components (Implemented)
- **Document Model**: `doclint/core/document.py` - Core `Document` dataclass
- **Exceptions**: `doclint/core/exceptions.py` - `ParsingError`, `UnsupportedFormatError`
- **Parsers**: `doclint/parsers/` - All parser implementations (text, markdown, HTML, PDF, DOCX)
- **Registry**: `doclint/parsers/registry.py` - Automatic parser selection
- **Cache**:
  - `doclint/cache/backends.py` - DiskCache backend implementation
  - `doclint/cache/manager.py` - Cache manager for embeddings
  - `doclint/cache/url_cache.py` - URL validation cache with TTL
- **Detectors**:
  - `doclint/detectors/base.py` - BaseDetector abstract class and Issue model
  - `doclint/detectors/completeness.py` - Completeness detector with external link validation

### Tests
- **Parser Tests**: `tests/test_parsers/` - Comprehensive tests for all parsers
- **Integration Tests**: `tests/test_parsers/test_integration.py` - Multi-format workflows
- **Registry Tests**: `tests/test_parsers/test_registry.py` - Parser selection logic
- **Detector Tests**: `tests/test_detectors/test_completeness.py` - Completeness detector tests
- **Reporter Tests**: `tests/test_reporters/` - Tests for all reporter implementations
- **Coverage**: 187 parser tests + comprehensive detector/reporter tests with edge cases

## Architecture Reference

For detailed implementation guidance, refer to `doclint_architecture.md`, which includes:
- Detailed component specifications
- Pseudocode examples
- Data flow diagrams
- Performance targets
- Testing strategies
- Success metrics
