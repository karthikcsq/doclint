# DocLint: Architecture Document
**Open Source Data Quality for AI Knowledge Bases**

*Version 1.0 - January 2026*

---

## Table of Contents
1. [Vision & Goals](#vision--goals)
2. [System Architecture](#system-architecture)
3. [Technical Stack](#technical-stack)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [File Format Support](#file-format-support)
7. [Detection Algorithms](#detection-algorithms)
8. [Caching Strategy](#caching-strategy)
9. [CLI Interface](#cli-interface)
10. [Output Formats](#output-formats)
11. [Configuration](#configuration)
12. [Performance Targets](#performance-targets)
13. [Testing Strategy](#testing-strategy)
14. [Deployment & Distribution](#deployment--distribution)
15. [Roadmap](#roadmap)
16. [Success Metrics](#success-metrics)

---

## Vision & Goals

### Vision Statement
DocLint is the **ESLint for AI knowledge bases** - an open source tool that detects data quality issues before they cause AI agents to hallucinate or give wrong answers.

### Core Problem
Companies deploying RAG systems and AI agents face a critical challenge: their source data is messy, contradictory, and stale. This causes:
- AI agents giving conflicting answers
- Hallucinations based on outdated information
- Broken workflows due to missing metadata
- Compliance issues from untracked document changes

### Solution
DocLint scans knowledge bases and detects:
- **Conflicts**: Same question, different answers across documents
- **Incompleteness**: Missing metadata, broken links, incomplete docs
- **Drift**: Meaning changes over time without tracking

### Goals
- **Primary**: Make it dead simple to validate data quality for AI systems
- **Secondary**: Build the industry standard tool (like ESLint for code)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                            │
│  (typer + rich for beautiful terminal output)               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Orchestration Layer                       │
│  (Scanner: coordinates all components)                      │
└─┬──────────┬──────────┬──────────┬──────────┬──────────────┘
  │          │          │          │          │
  ▼          ▼          ▼          ▼          ▼
┌───────┐ ┌──────┐ ┌──────────┐ ┌───────┐ ┌─────────┐
│Parser │ │Cache │ │Embeddings│ │Detect │ │Reporter │
│Layer  │ │Layer │ │  Layer   │ │ Layer │ │  Layer  │
└───────┘ └──────┘ └──────────┘ └───────┘ └─────────┘
    │         │          │          │          │
    ▼         ▼          ▼          ▼          ▼
┌──────────────────────────────────────────────────────────┐
│              Document Corpus (Input)                      │
│  PDFs, DOCX, HTML, Markdown, TXT, etc.                   │
└──────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
User runs: doclint scan ./knowledge-base/

1. CLI Layer
   ↓ Parses arguments, validates paths

2. Scanner (Orchestration)
   ↓ Discovers all files
   ↓ Determines file types

3. Parser Layer
   ↓ Extracts text from each file
   ↓ Extracts metadata (author, date, version)

4. Cache Layer
   ↓ Checks if file has been processed before
   ↓ Returns cached embeddings if available

5. Embeddings Layer
   ↓ Generates semantic embeddings for new/changed files
   ↓ Stores in cache

6. Detection Layer
   ↓ Runs all detectors in parallel:
   ├─ ConflictDetector (semantic similarity)
   ├─ CompletenessDetector (metadata validation)
   └─ DriftDetector (version comparison)

7. Reporter Layer
   ↓ Aggregates results
   ↓ Formats output (console, JSON, HTML)

8. Output
   → Beautiful terminal report with colors/icons
   → Optional JSON/HTML export
```

---

## Technical Stack

### Core Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.10"

# CLI & Terminal UI
typer = "^0.12.0"              # Modern CLI framework
rich = "^13.7.0"                # Beautiful terminal output
click = "^8.1.7"                # typer dependency

# Configuration & Validation
pydantic = "^2.6.0"             # Data validation
pydantic-settings = "^2.2.0"    # Settings management
tomli = "^2.0.1"                # TOML parsing (config files)

# Embeddings & ML
sentence-transformers = "^2.3.1" # Semantic embeddings
torch = "^2.2.0"                # PyTorch (transformers backend)
numpy = "^1.26.0"               # Numerical operations
scikit-learn = "^1.4.0"         # Similarity metrics

# Document Parsing
pypdf = "^4.0.0"                # PDF parsing
python-docx = "^1.1.0"          # Word documents
beautifulsoup4 = "^4.12.0"      # HTML parsing
lxml = "^5.1.0"                 # XML/HTML parser (faster)
markdown = "^3.5.0"             # Markdown parsing
python-magic = "^0.4.27"        # File type detection

# Caching & Storage
diskcache = "^5.6.0"            # Persistent cache
platformdirs = "^4.2.0"         # Cross-platform cache dirs

# Async & Performance
aiofiles = "^23.2.0"            # Async file I/O
tqdm = "^4.66.0"                # Progress bars

# Date/Time utilities
python-dateutil = "^2.8.0"      # Date parsing
```

### Development Dependencies

```toml
[tool.poetry.group.dev.dependencies]
# Testing
pytest = "^8.0.0"
pytest-asyncio = "^0.23.0"
pytest-cov = "^4.1.0"
pytest-mock = "^3.12.0"

# Code Quality
black = "^24.1.0"               # Code formatting
ruff = "^0.2.0"                 # Fast linting
mypy = "^1.8.0"                 # Type checking
pre-commit = "^3.6.0"           # Git hooks

# Documentation
mkdocs = "^1.5.0"
mkdocs-material = "^9.5.0"
```

### Project Structure

```
doclint/
├── pyproject.toml              # Poetry project config
├── README.md                   # Project overview
├── LICENSE                     # MIT License
├── .gitignore
├── .pre-commit-config.yaml
├── mkdocs.yml                  # Documentation config
│
├── doclint/
│   ├── __init__.py
│   ├── __main__.py             # Entry point: python -m doclint
│   ├── version.py              # Version string
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py             # Main CLI app (typer)
│   │   ├── scan.py             # Scan command
│   │   ├── config.py           # Config command
│   │   └── version.py          # Version command
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── scanner.py          # Main orchestration
│   │   ├── document.py         # Document model
│   │   └── config.py           # Configuration management
│   │
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base parser
│   │   ├── pdf.py              # PDF parser
│   │   ├── docx.py             # Word parser
│   │   ├── html.py             # HTML parser
│   │   ├── markdown.py         # Markdown parser
│   │   ├── text.py             # Plain text parser
│   │   └── registry.py         # Parser registry/factory
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── generator.py        # Embedding generation
│   │   ├── models.py           # Model management
│   │   └── cache.py            # Embedding cache
│   │
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base detector
│   │   ├── conflicts.py        # Conflict detection
│   │   ├── completeness.py     # Completeness checks
│   │   ├── drift.py            # Drift detection
│   │   └── registry.py         # Detector registry
│   │
│   ├── reporters/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base reporter
│   │   ├── console.py          # Rich console output
│   │   ├── json.py             # JSON export
│   │   ├── html.py             # HTML report
│   │   └── registry.py         # Reporter registry
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── manager.py          # Cache management
│   │   └── backends.py         # Cache backends
│   │
│   └── utils/
│       ├── __init__.py
│       ├── files.py            # File utilities
│       ├── text.py             # Text processing
│       └── hashing.py          # Content hashing
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── test_cli/
│   ├── test_parsers/
│   ├── test_detectors/
│   ├── test_embeddings/
│   └── fixtures/               # Test documents
│       ├── pdfs/
│       ├── docx/
│       ├── html/
│       └── markdown/
│
├── docs/
│   ├── index.md
│   ├── getting-started.md
│   ├── configuration.md
│   ├── detectors.md
│   ├── contributing.md
│   └── api/
│
└── examples/
    ├── basic_scan.py
    ├── custom_detector.py
    └── ci_integration.py
```

---

## Core Components

### 1. Document Model

```python
# doclint/core/document.py

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib

@dataclass
class DocumentMetadata:
    """Metadata extracted from document."""
    author: Optional[str] = None
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    version: Optional[str] = None
    title: Optional[str] = None
    tags: list[str] = None
    custom: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.custom is None:
            self.custom = {}

@dataclass
class Document:
    """Represents a parsed document."""
    path: Path
    content: str
    metadata: DocumentMetadata
    file_type: str
    size_bytes: int
    content_hash: str

    # Computed fields (set after parsing)
    embedding: Optional[np.ndarray] = None
    chunks: list[str] = None

    @classmethod
    def from_file(cls, path: Path, parser: 'BaseParser') -> 'Document':
        """Create document from file using parser."""
        content = parser.parse(path)
        metadata = parser.extract_metadata(path)

        with open(path, 'rb') as f:
            content_bytes = f.read()
            content_hash = hashlib.sha256(content_bytes).hexdigest()

        return cls(
            path=path,
            content=content,
            metadata=metadata,
            file_type=parser.file_type,
            size_bytes=len(content_bytes),
            content_hash=content_hash,
        )

    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []
```

### 2. Base Parser

```python
# doclint/parsers/base.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from ..core.document import DocumentMetadata

class BaseParser(ABC):
    """Abstract base class for document parsers."""

    file_type: str = None
    supported_extensions: list[str] = []

    @abstractmethod
    def parse(self, path: Path) -> str:
        """Extract text content from document.

        Args:
            path: Path to document

        Returns:
            Extracted text content

        Raises:
            ParsingError: If parsing fails
        """
        pass

    @abstractmethod
    def extract_metadata(self, path: Path) -> DocumentMetadata:
        """Extract metadata from document.

        Args:
            path: Path to document

        Returns:
            Document metadata
        """
        pass

    def can_parse(self, path: Path) -> bool:
        """Check if this parser can handle the file."""
        return path.suffix.lower() in self.supported_extensions

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Remove null bytes
        text = text.replace('\x00', '')
        return text.strip()
```

### 3. PDF Parser Implementation

```python
# doclint/parsers/pdf.py

from pathlib import Path
from datetime import datetime
import pypdf
from .base import BaseParser
from ..core.document import DocumentMetadata

class PDFParser(BaseParser):
    """Parser for PDF documents."""

    file_type = "pdf"
    supported_extensions = ['.pdf']

    def parse(self, path: Path) -> str:
        """Extract text from PDF."""
        try:
            with open(path, 'rb') as f:
                reader = pypdf.PdfReader(f)

                # Extract text from all pages
                text_parts = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)

                full_text = '\n\n'.join(text_parts)
                return self.clean_text(full_text)

        except Exception as e:
            raise ParsingError(f"Failed to parse PDF {path}: {e}")

    def extract_metadata(self, path: Path) -> DocumentMetadata:
        """Extract metadata from PDF."""
        metadata = DocumentMetadata()

        try:
            with open(path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                info = reader.metadata

                if info:
                    metadata.author = info.get('/Author')
                    metadata.title = info.get('/Title')

                    # Parse creation date
                    if '/CreationDate' in info:
                        date_str = info['/CreationDate']
                        metadata.created = self._parse_pdf_date(date_str)

                    # Parse modification date
                    if '/ModDate' in info:
                        date_str = info['/ModDate']
                        metadata.modified = self._parse_pdf_date(date_str)

        except Exception:
            # Metadata extraction is best-effort
            pass

        return metadata

    @staticmethod
    def _parse_pdf_date(date_str: str) -> Optional[datetime]:
        """Parse PDF date format (D:YYYYMMDDHHmmSS)."""
        try:
            # PDF date format: D:20240115143022
            if date_str.startswith('D:'):
                date_str = date_str[2:16]  # Extract YYYYMMDDHHmmSS
                return datetime.strptime(date_str, '%Y%m%d%H%M%S')
        except Exception:
            pass
        return None
```

### 4. Scanner (Orchestration)

```python
# doclint/core/scanner.py

from pathlib import Path
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ProcessPoolExecutor
from rich.progress import Progress, TaskID

from .document import Document
from ..parsers.registry import ParserRegistry
from ..embeddings.generator import EmbeddingGenerator
from ..detectors.registry import DetectorRegistry
from ..cache.manager import CacheManager

class Scanner:
    """Main orchestration class for document scanning."""

    def __init__(
        self,
        parser_registry: ParserRegistry,
        embedding_generator: EmbeddingGenerator,
        detector_registry: DetectorRegistry,
        cache_manager: CacheManager,
        max_workers: int = 4,
    ):
        self.parsers = parser_registry
        self.embeddings = embedding_generator
        self.detectors = detector_registry
        self.cache = cache_manager
        self.max_workers = max_workers

    async def scan_directory(
        self,
        path: Path,
        recursive: bool = True,
        progress: Optional[Progress] = None,
    ) -> Dict[str, Any]:
        """Scan a directory and detect issues.

        Args:
            path: Directory to scan
            recursive: Scan subdirectories
            progress: Optional rich Progress instance

        Returns:
            Dictionary of results from all detectors
        """
        # 1. Discover files
        files = self._discover_files(path, recursive)

        if progress:
            task = progress.add_task("[cyan]Parsing documents...", total=len(files))

        # 2. Parse documents (parallel)
        documents = await self._parse_documents(files, progress, task)

        if progress:
            progress.update(task, description="[cyan]Generating embeddings...")

        # 3. Generate embeddings (with caching)
        await self._generate_embeddings(documents, progress, task)

        if progress:
            progress.update(task, description="[cyan]Running detectors...")

        # 4. Run all detectors (parallel)
        results = await self._run_detectors(documents, progress, task)

        if progress:
            progress.update(task, completed=len(files))

        return results

    def _discover_files(self, path: Path, recursive: bool) -> List[Path]:
        """Discover all parseable files in directory."""
        files = []

        if recursive:
            pattern = '**/*'
        else:
            pattern = '*'

        for file_path in path.glob(pattern):
            if file_path.is_file():
                # Check if any parser can handle this file
                if self.parsers.can_parse(file_path):
                    files.append(file_path)

        return files

    async def _parse_documents(
        self,
        files: List[Path],
        progress: Optional[Progress],
        task: Optional[TaskID],
    ) -> List[Document]:
        """Parse all files into Document objects."""
        documents = []

        # Use process pool for CPU-bound parsing
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for file_path in files:
                parser = self.parsers.get_parser(file_path)
                future = executor.submit(self._parse_single_file, file_path, parser)
                futures.append(future)

            # Collect results as they complete
            for future in asyncio.as_completed(futures):
                doc = await future
                if doc:
                    documents.append(doc)

                if progress and task:
                    progress.advance(task)

        return documents

    @staticmethod
    def _parse_single_file(file_path: Path, parser) -> Optional[Document]:
        """Parse a single file (runs in subprocess)."""
        try:
            return Document.from_file(file_path, parser)
        except Exception as e:
            # Log error but continue
            print(f"Error parsing {file_path}: {e}")
            return None

    async def _generate_embeddings(
        self,
        documents: List[Document],
        progress: Optional[Progress],
        task: Optional[TaskID],
    ) -> None:
        """Generate embeddings for all documents."""
        # Check cache first
        uncached_docs = []

        for doc in documents:
            cached_embedding = self.cache.get_embedding(doc.content_hash)
            if cached_embedding is not None:
                doc.embedding = cached_embedding
            else:
                uncached_docs.append(doc)

        if uncached_docs:
            # Generate embeddings in batch (much faster)
            texts = [doc.content for doc in uncached_docs]
            embeddings = self.embeddings.generate_batch(texts, batch_size=32)

            # Assign embeddings and cache
            for doc, embedding in zip(uncached_docs, embeddings):
                doc.embedding = embedding
                self.cache.set_embedding(doc.content_hash, embedding)

    async def _run_detectors(
        self,
        documents: List[Document],
        progress: Optional[Progress],
        task: Optional[TaskID],
    ) -> Dict[str, Any]:
        """Run all detectors on documents."""
        results = {}

        # Run each detector
        for detector_name, detector in self.detectors.items():
            detector_results = await detector.detect(documents)
            results[detector_name] = detector_results

        return results
```

### 5. Conflict Detector

```python
# doclint/detectors/conflicts.py

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseDetector, Issue, IssueSeverity
from ..core.document import Document

class ConflictDetector(BaseDetector):
    """Detects semantic conflicts between documents."""

    name = "conflict"
    description = "Detects contradictory information across documents"

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Args:
            similarity_threshold: Cosine similarity threshold for conflicts (0-1)
        """
        self.similarity_threshold = similarity_threshold

    async def detect(self, documents: List[Document]) -> List[Issue]:
        """Detect conflicts in documents.

        Strategy:
        1. Compute pairwise similarity matrix
        2. Find pairs with high similarity (>threshold) but different content
        3. Extract conflicting sections
        4. Return as issues
        """
        issues = []

        if len(documents) < 2:
            return issues

        # Get all embeddings
        embeddings = np.array([doc.embedding for doc in documents])

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Find conflicts (high similarity but different documents)
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i][j]

                if similarity >= self.similarity_threshold:
                    # High similarity detected - potential conflict
                    doc1, doc2 = documents[i], documents[j]

                    # Extract conflicting sections
                    conflict = self._extract_conflict(doc1, doc2, similarity)

                    if conflict:
                        issue = Issue(
                            severity=IssueSeverity.CRITICAL,
                            detector=self.name,
                            title=f"Conflict detected: {conflict['topic']}",
                            description=(
                                f"Documents contain similar content but different answers\n"
                                f"Similarity: {similarity:.2%}"
                            ),
                            documents=[doc1.path, doc2.path],
                            details=conflict,
                        )
                        issues.append(issue)

        return issues

    def _extract_conflict(
        self,
        doc1: Document,
        doc2: Document,
        similarity: float,
    ) -> Optional[Dict[str, Any]]:
        """Extract conflicting information from two similar documents.

        This is a simplified version. In production, you'd want:
        - Question extraction (what question is being answered?)
        - Answer extraction from each doc
        - Contradiction detection
        """
        # For MVP: Just show the documents conflict
        # TODO: Implement sophisticated conflict extraction

        # Detect if they're actually different content
        content_similarity = self._text_similarity(doc1.content, doc2.content)

        if content_similarity < 0.95:  # Not identical
            return {
                'topic': self._extract_topic(doc1, doc2),
                'doc1_excerpt': doc1.content[:200] + '...',
                'doc2_excerpt': doc2.content[:200] + '...',
                'similarity': similarity,
            }

        return None

    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """Simple character-level similarity."""
        # Normalize
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        # Check exact match
        if t1 == t2:
            return 1.0

        # Simple Jaccard similarity
        words1 = set(t1.split())
        words2 = set(t2.split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _extract_topic(doc1: Document, doc2: Document) -> str:
        """Extract topic/subject from conflicting documents."""
        # Simple heuristic: use title if available, else first few words
        if doc1.metadata.title:
            return doc1.metadata.title

        # Use first meaningful words
        words = doc1.content.split()[:5]
        return ' '.join(words) + '...'
```

### 6. Base Detector

```python
# doclint/detectors/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional

class IssueSeverity(Enum):
    """Severity levels for detected issues."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Issue:
    """Represents a detected data quality issue."""
    severity: IssueSeverity
    detector: str
    title: str
    description: str
    documents: List[Path]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            'severity': self.severity.value,
            'detector': self.detector,
            'title': self.title,
            'description': self.description,
            'documents': [str(p) for p in self.documents],
            'details': self.details,
        }

class BaseDetector(ABC):
    """Abstract base class for detectors."""

    name: str = None
    description: str = None

    @abstractmethod
    async def detect(self, documents: List['Document']) -> List[Issue]:
        """Run detection on documents.

        Args:
            documents: List of parsed documents

        Returns:
            List of detected issues
        """
        pass
```

### 8. Console Reporter

```python
# doclint/reporters/console.py

from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import box

from .base import BaseReporter
from ..detectors.base import Issue, IssueSeverity

class ConsoleReporter(BaseReporter):
    """Beautiful terminal output using rich."""

    name = "console"

    # Emoji/icon mapping
    SEVERITY_ICONS = {
        IssueSeverity.INFO: "ℹ️ ",
        IssueSeverity.WARNING: "⚠️ ",
        IssueSeverity.CRITICAL: "🔴",
    }

    SEVERITY_COLORS = {
        IssueSeverity.INFO: "blue",
        IssueSeverity.WARNING: "yellow",
        IssueSeverity.CRITICAL: "red",
    }

    def __init__(self):
        self.console = Console()

    def report(self, results: Dict[str, List[Issue]], stats: Dict[str, Any]) -> None:
        """Generate beautiful console report."""
        # Header
        self._print_header(stats)

        # Issues by severity
        critical_issues = []
        warning_issues = []
        info_issues = []

        for detector_name, issues in results.items():
            for issue in issues:
                if issue.severity == IssueSeverity.CRITICAL:
                    critical_issues.append(issue)
                elif issue.severity == IssueSeverity.WARNING:
                    warning_issues.append(issue)
                else:
                    info_issues.append(issue)

        # Print critical issues
        if critical_issues:
            self.console.print("\n")
            self.console.print("🔴 [bold red]CRITICAL ISSUES[/bold red]", style="bold")
            self.console.print()

            for issue in critical_issues:
                self._print_issue(issue)

        # Print warnings
        if warning_issues:
            self.console.print("\n")
            self.console.print("⚠️  [bold yellow]WARNINGS[/bold yellow]", style="bold")
            self.console.print()

            for issue in warning_issues:
                self._print_issue(issue)

        # Summary
        self._print_summary(stats, len(critical_issues), len(warning_issues), len(info_issues))

    def _print_header(self, stats: Dict[str, Any]) -> None:
        """Print scan header."""
        self.console.rule("[bold blue]DocLint Scan Results[/bold blue]")
        self.console.print(f"Scanned {stats['total_documents']} documents")
        self.console.print()

    def _print_issue(self, issue: Issue) -> None:
        """Print a single issue."""
        icon = self.SEVERITY_ICONS[issue.severity]
        color = self.SEVERITY_COLORS[issue.severity]

        # Create panel for issue
        content = f"[bold]{issue.title}[/bold]\n\n"
        content += f"{issue.description}\n\n"
        content += f"[dim]Documents:[/dim]\n"

        for doc_path in issue.documents:
            content += f"  • {doc_path}\n"

        if issue.details:
            content += f"\n[dim]Details:[/dim]\n"
            for key, value in issue.details.items():
                content += f"  {key}: {value}\n"

        panel = Panel(
            content,
            title=f"{icon} {issue.detector}",
            border_style=color,
            box=box.ROUNDED,
        )

        self.console.print(panel)
        self.console.print()

    def _print_summary(
        self,
        stats: Dict[str, Any],
        critical: int,
        warnings: int,
        info: int,
    ) -> None:
        """Print summary statistics."""
        self.console.rule("[bold]Summary[/bold]")
        self.console.print()

        # Create summary table
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")

        table.add_row("Total documents", str(stats['total_documents']))
        table.add_row("Critical issues", f"[red]{critical}[/red]")
        table.add_row("Warnings", f"[yellow]{warnings}[/yellow]")
        table.add_row("Info", f"[blue]{info}[/blue]")

        clean_docs = stats['total_documents'] - (critical + warnings + info)
        if clean_docs > 0:
            percentage = (clean_docs / stats['total_documents']) * 100
            table.add_row(
                "Clean documents",
                f"[green]{clean_docs} ({percentage:.1f}%)[/green]"
            )

        self.console.print(table)
        self.console.print()
```

---

## Data Flow

### Detailed Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. CLI ENTRY POINT                                          │
│    $ doclint scan ./knowledge-base/ --recursive             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. ARGUMENT PARSING (typer)                                 │
│    - Validate paths                                         │
│    - Parse flags (--recursive, --config, etc.)              │
│    - Load configuration                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. INITIALIZE COMPONENTS                                    │
│    scanner = Scanner(                                       │
│        parser_registry=ParserRegistry(),                    │
│        embedding_generator=EmbeddingGenerator(),            │
│        detector_registry=DetectorRegistry(),                │
│        cache_manager=CacheManager(),                        │
│    )                                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. FILE DISCOVERY                                           │
│    - Walk directory tree                                    │
│    - Filter by supported extensions                         │
│    - Result: List[Path] of 1,247 files                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. PARALLEL PARSING (ProcessPoolExecutor)                   │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│    │Worker 1 │  │Worker 2 │  │Worker 3 │  │Worker 4 │      │
│    │PDF      │  │DOCX     │  │HTML     │  │MD       │      │
│    └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘      │
│         └────────────┴─────────────┴────────────┘            │
│                      Results                                │
│              List[Document] (1,247 docs)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. EMBEDDING GENERATION (with caching)                      │
│                                                             │
│    For each document:                                       │
│    1. Check cache (by content hash)                         │
│       ├─ Cache HIT (847 docs) → Use cached embedding       │
│       └─ Cache MISS (400 docs) → Generate new              │
│                                                             │
│    2. Batch generate embeddings (400 docs):                 │
│       model.encode(texts, batch_size=32)                    │
│       → 13 batches × ~2 sec = 26 seconds                    │
│                                                             │
│    3. Store in cache for future runs                        │
│                                                             │
│    Result: All 1,247 docs have embeddings                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. RUN DETECTORS (parallel)                                 │
│                                                             │
│    ┌────────────────┐  ┌────────────────┐                  │
│    │ConflictDetector│  │CompletenessD.  │                  │
│    │                │  │                │                  │
│    │1. Compute sim. │  │1. Check meta   │                  │
│    │   matrix       │  │2. Compare age  │                  │
│    │2. Find pairs   │  │3. Flag old docs│                  │
│    │   >0.85 sim    │  │                │                  │
│    │3. Extract      │  │                │                  │
│    │   conflicts    │  │                │                  │
│    └────┬───────────┘  └────┬───────────┘                  │
│         │                   │                              │
│         ▼                   ▼                              │
│    [3 conflicts]       [12 stale docs]                     │
│                                                             │
│    ┌────────────────┐  ┌────────────────┐                  │
│    │CompletenessDetc│  │DriftDetector   │                  │
│    └────┬───────────┘  └────┬───────────┘                  │
│         │                   │                              │
│         ▼                   ▼                              │
│    [8 incomplete]      [0 drift]                           │
│                                                             │
│    Combined Results: 23 total issues                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. GENERATE REPORT                                          │
│                                                             │
│    ConsoleReporter:                                         │
│    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━               │
│    📊 SCAN COMPLETE                                         │
│                                                             │
│    🔴 CRITICAL ISSUES (3)                                   │
│                                                             │
│      Conflict: Return Policy                                │
│      ├─ docs/policies.pdf: "30-day returns"                 │
│      └─ website/faq.html: "14-day returns"                  │
│      Impact: AI agent will give contradictory answers       │
│                                                             │
│    ⚠️  WARNINGS (20)                                        │
│      ... (stale docs, incomplete metadata)                  │
│                                                             │
│    📈 SUMMARY                                               │
│      Total documents: 1,247                                 │
│      Critical issues: 3                                     │
│      Warnings: 20                                           │
│      Clean documents: 1,224 (98.2%)                         │
│    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━               │
└─────────────────────────────────────────────────────────────┘
```

---

## File Format Support

### Phase 1 (MVP - Week 1-2)

| Format | Extension | Priority | Parsing Library | Notes |
|--------|-----------|----------|-----------------|-------|
| **PDF** | `.pdf` | High | `pypdf` | Most common document format |
| **Markdown** | `.md` | High | `markdown` | Common for docs/wikis |
| **Plain Text** | `.txt` | High | Built-in | Easy baseline |
| **HTML** | `.html`, `.htm` | Medium | `beautifulsoup4` | Web content |

### Phase 2 (Week 3-4)

| Format | Extension | Priority | Parsing Library | Notes |
|--------|-----------|----------|-----------------|-------|
| **Word** | `.docx` | High | `python-docx` | Enterprise documents |
| **Rich Text** | `.rtf` | Low | `striprtf` | Legacy format |
| **JSON** | `.json` | Medium | Built-in | Structured data |
| **CSV** | `.csv` | Low | `pandas` | Tabular data |

### Phase 3 (Post-launch)

| Format | Extension | Priority | Parsing Library | Notes |
|--------|-----------|----------|-----------------|-------|
| **PowerPoint** | `.pptx` | Medium | `python-pptx` | Presentations |
| **Excel** | `.xlsx` | Low | `openpyxl` | Spreadsheets |
| **LaTeX** | `.tex` | Low | Custom | Academic papers |
| **ReStructuredText** | `.rst` | Low | `docutils` | Python docs |

---

## Detection Algorithms

### 1. Conflict Detection Algorithm

**Approach:** Semantic Similarity + Text Comparison

```python
# Pseudocode

def detect_conflicts(documents):
    """
    1. Generate embeddings for all documents
    2. Compute pairwise similarity matrix
    3. For each pair with similarity > threshold:
       a. Check if they're actually different content
       b. Extract topic/question being answered
       c. Extract different answers from each doc
       d. Create conflict issue
    """

    embeddings = [doc.embedding for doc in documents]
    similarity_matrix = cosine_similarity(embeddings)

    conflicts = []
    for i, j in get_high_similarity_pairs(similarity_matrix, threshold=0.85):
        if not identical_content(docs[i], docs[j]):
            conflict = {
                'topic': extract_topic(docs[i], docs[j]),
                'answer1': extract_answer(docs[i]),
                'answer2': extract_answer(docs[j]),
                'docs': [docs[i], docs[j]],
            }
            conflicts.append(conflict)

    return conflicts
```

**Parameters:**
- `similarity_threshold`: 0.85 (configurable)
- `min_content_difference`: 0.05 (5% different = not identical)

**Edge Cases:**
- Identical content (same doc in different locations) → Skip
- Different versions of same doc → Flag as version drift, not conflict
- Multi-language docs → Need language detection

### 2. Completeness Detection Algorithm

**Approach:** Metadata and content validation

```python
def detect_incompleteness(documents):
    """
    1. For each document, check:
       a. Required metadata fields are present
       b. Content meets minimum length requirements
       c. No placeholder text detected
    2. Generate issues for incomplete documents
    """

    incomplete_docs = []

    for doc in documents:
        missing_fields = check_required_metadata(doc)
        content_issues = check_content_quality(doc)

        if missing_fields or content_issues:
            incomplete_docs.append({
                'doc': doc,
                'missing_metadata': missing_fields,
                'content_issues': content_issues,
            })

    return incomplete_docs
```

**Parameters:**
- `required_metadata`: ["title"] (configurable)
- `min_content_length`: 100 (configurable)

**Enhancements (Phase 2):**
- Check if doc is linked/referenced by other docs
- Track access patterns (if available)
- Industry-specific thresholds (legal docs vs marketing)

### 3. Completeness Detection Algorithm

**Approach:** Metadata validation + content analysis

```python
def detect_incompleteness(documents):
    """
    1. Check required metadata fields
    2. Check content quality (too short, missing sections)
    3. Check references (broken links)
    """

    incomplete = []

    for doc in documents:
        issues = []

        # Metadata checks
        if not doc.metadata.author:
            issues.append('Missing author')
        if not doc.metadata.created:
            issues.append('Missing creation date')
        if not doc.metadata.version:
            issues.append('Missing version')

        # Content checks
        if len(doc.content) < 100:
            issues.append('Content too short (<100 chars)')

        # Reference checks
        broken_links = find_broken_links(doc.content)
        if broken_links:
            issues.append(f'{len(broken_links)} broken links')

        if issues:
            incomplete.append({
                'doc': doc,
                'issues': issues,
            })

    return incomplete
```

**Checks:**
- Required metadata: author, date, version
- Content length: min 100 characters
- Links: internal/external broken links
- Structure: missing expected sections (config-driven)

### 4. Drift Detection Algorithm (Phase 2)

**Approach:** Version comparison using git or manual snapshots

```python
def detect_drift(documents, previous_scan=None):
    """
    1. Compare current scan with previous scan
    2. Detect semantic drift (meaning changed)
    3. Detect structural drift (format changed)
    """

    if not previous_scan:
        return []  # First run, no drift to detect

    drift_issues = []

    for doc in documents:
        prev_doc = find_matching_doc(doc, previous_scan)

        if not prev_doc:
            continue  # New document, not drift

        # Semantic drift
        embedding_distance = cosine_distance(doc.embedding, prev_doc.embedding)
        if embedding_distance > 0.2:  # Significant meaning change
            drift_issues.append({
                'doc': doc,
                'type': 'semantic_drift',
                'distance': embedding_distance,
                'previous_version': prev_doc,
            })

        # Structural drift (major content changes)
        content_diff = diff_ratio(doc.content, prev_doc.content)
        if content_diff > 0.5:  # >50% changed
            drift_issues.append({
                'doc': doc,
                'type': 'structural_drift',
                'change_ratio': content_diff,
            })

    return drift_issues
```

**This requires:**
- Storing scan results between runs
- Version control integration (git diff)
- Content diffing library

---

## Caching Strategy

### Why Caching Matters

**Problem:** Embedding generation is slow
- Model loading: ~3 seconds
- Embedding 1 document: ~0.1 seconds
- Embedding 10,000 documents: ~1000 seconds (16 minutes)

**Solution:** Cache embeddings by content hash
- First scan: 16 minutes
- Subsequent scans (no changes): 5 seconds
- Subsequent scans (10% changed): 2 minutes

### Cache Architecture

```python
# doclint/cache/manager.py

from pathlib import Path
from typing import Optional
import numpy as np
import hashlib
import pickle
from diskcache import Cache
from platformdirs import user_cache_dir

class CacheManager:
    """Manages embedding cache."""

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            # Use system cache directory
            cache_dir = Path(user_cache_dir("doclint"))

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = Cache(str(cache_dir))

    def get_embedding(self, content_hash: str) -> Optional[np.ndarray]:
        """Get cached embedding for content hash."""
        key = f"emb:{content_hash}"

        cached = self.cache.get(key)
        if cached:
            return pickle.loads(cached)

        return None

    def set_embedding(self, content_hash: str, embedding: np.ndarray) -> None:
        """Cache embedding for content hash."""
        key = f"emb:{content_hash}"
        value = pickle.dumps(embedding)

        # Store with 30 day expiration
        self.cache.set(key, value, expire=30 * 24 * 60 * 60)

    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'size_mb': self.cache.volume() / (1024 * 1024),
            'item_count': len(self.cache),
        }
```

### Cache Location

**Cross-platform cache directories:**
- Linux: `~/.cache/doclint/`
- macOS: `~/Library/Caches/doclint/`
- Windows: `C:\Users\{user}\AppData\Local\doclint\Cache\`

**Cache Structure:**
```
~/.cache/doclint/
├── cache.db              # diskcache SQLite database
├── 00/                   # Sharded cache files
├── 01/
├── ...
└── ff/
```

### Cache Invalidation

**When to invalidate:**
- File content changes (different hash)
- Model version changes (re-embed everything)
- Manual clear: `doclint cache clear`

**Automatic expiration:** 30 days

---

## CLI Interface

### Main Commands

```bash
# Primary command: scan
doclint scan [PATH] [OPTIONS]

# Configuration
doclint config init              # Create default config
doclint config show              # Show current config

# Cache management
doclint cache stats              # Show cache statistics
doclint cache clear              # Clear cache

# Version
doclint version                  # Show version
```

### Scan Command Options

```bash
doclint scan ./knowledge-base/ \
  --recursive                    # Scan subdirectories (default: true)
  --no-recursive                 # Don't scan subdirectories
  --config ./doclint.toml        # Custom config file
  --format console               # Output format: console, json, html
  --output report.json           # Save report to file
  --detectors conflict,completeness # Run specific detectors
  --severity critical            # Only show critical issues
  --cache-dir ~/.cache/custom    # Custom cache directory
  --no-cache                     # Disable caching
  --workers 8                    # Number of parallel workers
  --verbose                      # Verbose output
  --quiet                        # Minimal output
```

### Example Usage

```bash
# Basic scan
$ doclint scan ./docs/

# Scan with custom config
$ doclint scan ./docs/ --config my-config.toml

# Save JSON report
$ doclint scan ./docs/ --format json --output report.json

# Only check for conflicts
$ doclint scan ./docs/ --detectors conflict

# Show only critical issues
$ doclint scan ./docs/ --severity critical

# Clear cache and rescan
$ doclint cache clear
$ doclint scan ./docs/

# Fast scan with more workers
$ doclint scan ./docs/ --workers 16
```

### Exit Codes

```python
0  # Success, no issues found
1  # Issues found (any severity)
2  # Critical issues found
3  # Error during scan (invalid path, parsing errors)
```

**Use in CI/CD:**
```bash
#!/bin/bash
doclint scan ./docs/ --severity critical
if [ $? -eq 2 ]; then
  echo "Critical data quality issues found!"
  exit 1
fi
```

---

## Output Formats

### 1. Console Output (Default)

**Rich terminal output with colors, icons, and formatting**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DocLint Scan Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Scanned 1,247 documents

🔴 CRITICAL ISSUES (3)

╭─────────────────────────────────────────────────╮
│ 🔴 conflict                                     │
│                                                 │
│ Conflict detected: Return Policy               │
│                                                 │
│ Documents contain similar content but different │
│ answers                                         │
│ Similarity: 87%                                 │
│                                                 │
│ Documents:                                      │
│   • docs/policies.pdf                           │
│   • website/faq.html                            │
│                                                 │
│ Details:                                        │
│   topic: Return Policy                          │
│   doc1_excerpt: All items can be returned...   │
│   doc2_excerpt: Returns are accepted within...  │
╰─────────────────────────────────────────────────╯

⚠️  WARNINGS (12)

... (similar format)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total documents       1,247
Critical issues       3
Warnings              12
Info                  0
Clean documents       1,232 (98.8%)
```

### 2. JSON Output

**Machine-readable format for CI/CD integration**

```json
{
  "scan_info": {
    "timestamp": "2026-01-05T14:30:22Z",
    "path": "./knowledge-base/",
    "version": "0.1.0",
    "total_documents": 1247,
    "scan_duration_seconds": 45.2
  },
  "statistics": {
    "total_issues": 15,
    "critical": 3,
    "warning": 12,
    "info": 0,
    "clean_documents": 1232
  },
  "issues": [
    {
      "severity": "critical",
      "detector": "conflict",
      "title": "Conflict detected: Return Policy",
      "description": "Documents contain similar content but different answers\nSimilarity: 87%",
      "documents": [
        "docs/policies.pdf",
        "website/faq.html"
      ],
      "details": {
        "topic": "Return Policy",
        "doc1_excerpt": "All items can be returned...",
        "doc2_excerpt": "Returns are accepted within...",
        "similarity": 0.87
      }
    },
    ...
  ]
}
```

### 3. HTML Report

**Beautiful web-based report for sharing**

```html
<!DOCTYPE html>
<html>
<head>
  <title>DocLint Report</title>
  <style>/* Beautiful CSS */</style>
</head>
<body>
  <h1>DocLint Scan Results</h1>
  <div class="summary">
    <div class="stat critical">
      <span class="number">3</span>
      <span class="label">Critical Issues</span>
    </div>
    ...
  </div>

  <div class="issues">
    <div class="issue critical">
      <h3>🔴 Conflict detected: Return Policy</h3>
      ...
    </div>
  </div>
</body>
</html>
```

---

## Configuration

### Configuration File Format

**TOML format (like pyproject.toml)**

```toml
# doclint.toml

[doclint]
# General settings
recursive = true
cache_enabled = true
cache_dir = "~/.cache/doclint"
max_workers = 4

[doclint.detectors.conflict]
enabled = true
similarity_threshold = 0.85

[doclint.detectors.completeness]
enabled = true
min_content_length = 100
critical_days = 365

[doclint.detectors.completeness]
enabled = true
required_metadata = ["author", "created", "version"]
min_content_length = 100

[doclint.detectors.drift]
enabled = false  # Disabled by default

[doclint.output]
format = "console"  # console, json, html
colors_enabled = true

[doclint.embeddings]
model = "all-MiniLM-L6-v2"  # sentence-transformers model
batch_size = 32
device = "cpu"  # or "cuda"

[doclint.parsers]
# Enable/disable specific parsers
pdf = true
docx = true
html = true
markdown = true
txt = true

[doclint.ignore]
# Glob patterns to ignore
patterns = [
  "**/.git/**",
  "**/node_modules/**",
  "**/__pycache__/**",
  "**/venv/**",
]
```

### Loading Configuration

**Priority order:**
1. Command-line arguments (highest priority)
2. Custom config file (`--config ./custom.toml`)
3. Local config (`./.doclint.toml`)
4. User config (`~/.config/doclint/config.toml`)
5. Default config (built-in)

```python
# doclint/core/config.py

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import tomli

class DetectorConfig(BaseSettings):
    """Configuration for a detector."""
    enabled: bool = True

class ConflictDetectorConfig(DetectorConfig):
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

class CompletenessDetectorConfig(DetectorConfig):
    required_metadata: List[str] = Field(default=["title"])
    min_content_length: int = Field(default=100, ge=0)

class DocLintConfig(BaseSettings):
    """Main configuration."""
    recursive: bool = True
    cache_enabled: bool = True
    cache_dir: Optional[Path] = None
    max_workers: int = Field(default=4, gt=0)

    # Nested configs
    conflict: ConflictDetectorConfig = ConflictDetectorConfig()
    completeness: CompletenessDetectorConfig = CompletenessDetectorConfig()

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'DocLintConfig':
        """Load configuration from file."""
        if config_path and config_path.exists():
            with open(config_path, 'rb') as f:
                data = tomli.load(f)
                return cls(**data.get('doclint', {}))

        # Try default locations
        for path in [
            Path('.doclint.toml'),
            Path.home() / '.config' / 'doclint' / 'config.toml',
        ]:
            if path.exists():
                with open(path, 'rb') as f:
                    data = tomli.load(f)
                    return cls(**data.get('doclint', {}))

        # Return default config
        return cls()
```

---

## Performance Targets

### Benchmark Goals

**Target performance (for MVP):**

| Metric | Target | Notes |
|--------|--------|-------|
| **100 documents** | <10 seconds | Small project |
| **1,000 documents** | <60 seconds | Medium project |
| **10,000 documents** | <10 minutes | Large enterprise |
| **Memory usage** | <2GB | For 10K documents |
| **Cache effectiveness** | >80% hit rate | On subsequent scans |

### Performance Breakdown

**For 1,000 documents (60 second target):**

```
File discovery:        1 sec   (2%)
Parsing:              10 sec  (17%)
Embedding generation: 30 sec  (50%)  <- Bottleneck
Conflict detection:   15 sec  (25%)
Other detectors:       3 sec   (5%)
Reporting:             1 sec   (2%)
─────────────────────────────
Total:                60 sec
```

### Optimization Strategies

**1. Caching (biggest win):**
- Cache embeddings by content hash
- 80% cache hit rate = 50% time reduction
- First run: 60 sec, Second run: 20 sec

**2. Batching:**
- Batch embedding generation (32 docs at a time)
- 10x faster than one-by-one

**3. Parallelization:**
- Parse files in parallel (4-8 workers)
- 4x speedup on 8-core machine

**4. Smart similarity computation:**
```python
# Bad: O(n²) comparisons
for i in range(n):
    for j in range(n):
        similarity(docs[i], docs[j])

# Good: O(n) with approximate search
embeddings = [doc.embedding for doc in docs]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# Find conflicts efficiently
for query in embeddings:
    neighbors = index.search(query, k=10)
    # Only check top 10 similar docs
```

**5. Lazy loading:**
```python
# Don't load embedding model until needed
class EmbeddingGenerator:
    _model = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(...)
        return self._model
```

---

## Testing Strategy

### Test Pyramid

```
           ╱╲
          ╱  ╲      E2E Tests (5%)
         ╱____╲     - Full CLI integration
        ╱      ╲    - Real document corpus
       ╱        ╲   Integration Tests (15%)
      ╱__________╲  - Component interaction
     ╱            ╲ - Parser + Detector
    ╱______________╲ Unit Tests (80%)
                    - Individual functions
                    - Edge cases
```

### Test Categories

**1. Unit Tests (80% coverage target):**

```python
# tests/test_parsers/test_pdf.py

import pytest
from pathlib import Path
from doclint.parsers.pdf import PDFParser

def test_pdf_parser_can_parse():
    parser = PDFParser()
    assert parser.can_parse(Path("test.pdf")) == True
    assert parser.can_parse(Path("test.docx")) == False

def test_pdf_parser_extracts_text(sample_pdf):
    parser = PDFParser()
    text = parser.parse(sample_pdf)

    assert len(text) > 0
    assert "expected content" in text

def test_pdf_parser_extracts_metadata(sample_pdf):
    parser = PDFParser()
    metadata = parser.extract_metadata(sample_pdf)

    assert metadata.author == "John Doe"
    assert metadata.title == "Test Document"

def test_pdf_parser_handles_encrypted_pdf(encrypted_pdf):
    parser = PDFParser()

    with pytest.raises(ParsingError):
        parser.parse(encrypted_pdf)
```

**2. Integration Tests:**

```python
# tests/test_integration/test_scanner.py

import pytest
from pathlib import Path
from doclint.core.scanner import Scanner

@pytest.mark.integration
async def test_full_scan_workflow(tmp_path, sample_documents):
    # Set up test directory
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()

    # Copy sample documents
    for doc in sample_documents:
        shutil.copy(doc, doc_dir)

    # Initialize scanner
    scanner = Scanner(...)

    # Run scan
    results = await scanner.scan_directory(doc_dir)

    # Verify results
    assert len(results['conflict']) == 2
    assert len(results['completeness']) == 1
```

**3. E2E Tests:**

```python
# tests/test_e2e/test_cli.py

import subprocess
import json

def test_cli_scan_json_output(tmp_path):
    # Run CLI command
    result = subprocess.run(
        ["doclint", "scan", str(tmp_path), "--format", "json"],
        capture_output=True,
        text=True,
    )

    # Check exit code
    assert result.returncode == 0

    # Parse JSON output
    data = json.loads(result.stdout)

    # Verify structure
    assert "scan_info" in data
    assert "issues" in data
```

### Test Fixtures

```python
# tests/conftest.py

import pytest
from pathlib import Path

@pytest.fixture
def sample_pdf(tmp_path):
    """Create a sample PDF for testing."""
    # Use reportlab or copy from fixtures/
    pdf_path = tmp_path / "sample.pdf"
    # ... create PDF
    return pdf_path

@pytest.fixture
def conflicting_documents(tmp_path):
    """Create two documents with conflicting info."""
    doc1 = tmp_path / "policy1.txt"
    doc1.write_text("Return policy: 30 days")

    doc2 = tmp_path / "policy2.txt"
    doc2.write_text("Return policy: 14 days")

    return [doc1, doc2]

@pytest.fixture
def mock_embedding_generator(monkeypatch):
    """Mock embedding generator to avoid slow model loading."""
    class MockGenerator:
        def generate(self, text):
            # Return dummy embedding
            return np.random.rand(384)

    return MockGenerator()
```

### Coverage Goals

```bash
# Run tests with coverage
$ pytest --cov=doclint --cov-report=html

# Target coverage:
doclint/parsers/      95%
doclint/detectors/    90%
doclint/core/         85%
doclint/cli/          70%
Overall:              85%
```

---

## Deployment & Distribution

### Package Structure

```bash
# Build package
$ poetry build

# Output:
dist/
├── doclint-0.1.0-py3-none-any.whl
└── doclint-0.1.0.tar.gz
```

### PyPI Distribution

```toml
# pyproject.toml

[tool.poetry]
name = "doclint"
version = "0.1.0"
description = "Data quality linting for AI knowledge bases"
authors = ["Your Name <you@example.com>"]
license = "MIT"
readme = "README.md"
homepage = "https://github.com/yourusername/doclint"
repository = "https://github.com/yourusername/doclint"
documentation = "https://doclint.readthedocs.io"
keywords = ["ai", "rag", "data-quality", "linting", "llm"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Software Development :: Quality Assurance",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

[tool.poetry.scripts]
doclint = "doclint.cli.main:app"
```

### Installation Methods

**1. From PyPI (end users):**
```bash
pip install doclint
```

**2. From source (contributors):**
```bash
git clone https://github.com/yourusername/doclint.git
cd doclint
poetry install
```

**3. Development mode:**
```bash
pip install -e .
```

### GitHub Actions CI/CD

```yaml
# .github/workflows/ci.yml

name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Poetry
        run: |
          pip install poetry

      - name: Install dependencies
        run: |
          poetry install

      - name: Run linters
        run: |
          poetry run black --check .
          poetry run ruff check .
          poetry run mypy doclint/

      - name: Run tests
        run: |
          poetry run pytest --cov=doclint --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  release:
    needs: test
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install Poetry
        run: pip install poetry

      - name: Build package
        run: poetry build

      - name: Publish to PyPI
        env:
          POETRY_PYPI_TOKEN_PYPI: ${{ secrets.PYPI_TOKEN }}
        run: poetry publish
```

### Docker Image (Optional)

```dockerfile
# Dockerfile

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install doclint
RUN pip install doclint

# Set working directory
WORKDIR /workspace

# Entry point
ENTRYPOINT ["doclint"]
CMD ["--help"]
```

**Usage:**
```bash
docker run -v $(pwd):/workspace doclint scan /workspace/docs
```

---

## Getting Started

### For Contributors

```bash
# 1. Clone repo
git clone https://github.com/yourusername/doclint.git
cd doclint

# 2. Install dependencies
poetry install

# 3. Run tests
poetry run pytest

# 4. Run locally
poetry run doclint scan ./examples/

# 5. Make changes, commit, PR
```

### For Users

```bash
# Install
pip install doclint

# Scan your docs
doclint scan ./knowledge-base/

# Enjoy clean data!
```

---

## Next Steps

**This document should be used as:**
1. Reference during development
2. Onboarding for contributors
3. Specification for implementation
4. Living document (update as you build)

**Start building:**
- Begin with Week 1 tasks
- Follow the architecture
- Ship fast, iterate faster

**Questions?**
- Open an issue
- Join Discord
- Email: you@doclint.dev

---

*This architecture is a living document. Update it as the project evolves.*

**Version History:**
- v1.0 (2026-01-05): Initial architecture document
