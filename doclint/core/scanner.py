"""Scanner module for orchestrating document parsing, embedding, and detection."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.progress import Progress, TaskID

from ..cache.manager import CacheManager
from ..embeddings.base import BaseEmbeddingGenerator
from ..embeddings.processor import DocumentProcessor
from ..parsers.base import BaseParser
from ..parsers.registry import ParserRegistry
from .config import DocLintConfig
from .document import Document
from .exceptions import ParsingError

logger = logging.getLogger(__name__)


@dataclass
class Issue:
    """Represents a detected issue in the knowledge base.

    Attributes:
        issue_type: Type of issue (conflict, completeness)
        severity: Severity level (info, warning, error, critical)
        message: Human-readable description of the issue
        document_path: Path to the affected document
        details: Additional details about the issue
    """

    issue_type: str
    severity: str
    message: str
    document_path: Path
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str:
        """Get issue title from details or default to message."""
        result: str = self.details.get("title", self.message)
        return result

    @property
    def description(self) -> str:
        """Get issue description from details or empty string."""
        result: str = self.details.get("description", "")
        return result

    @property
    def detector(self) -> str:
        """Get detector name from details or default to issue_type."""
        result: str = self.details.get("detector", self.issue_type)
        return result

    @property
    def documents(self) -> List[Path]:
        """Get list of affected documents."""
        docs_from_details = self.details.get("documents", [])
        if docs_from_details:
            return [Path(d) if not isinstance(d, Path) else d for d in docs_from_details]
        # Fallback to single document_path
        return [self.document_path] if self.document_path else []


@dataclass
class ScanResult:
    """Results from a document scan.

    Attributes:
        documents: List of parsed documents
        issues: Dictionary mapping issue type to list of issues
        stats: Statistics about the scan
    """

    documents: List[Document]
    issues: Dict[str, List[Issue]]
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_issues(self) -> int:
        """Get total number of issues across all types."""
        return sum(len(issues) for issues in self.issues.values())

    @property
    def has_issues(self) -> bool:
        """Check if any issues were found."""
        return self.total_issues > 0


def _parse_single_file(file_path: Path, parser: BaseParser) -> Optional[Document]:
    """Parse a single file using the provided parser.

    This function is designed to be called in a subprocess for parallel parsing.

    Args:
        file_path: Path to the file to parse
        parser: Parser instance to use

    Returns:
        Parsed Document or None if parsing failed
    """
    try:
        return Document.from_file(file_path, parser)
    except ParsingError as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing {file_path}: {e}")
        return None


class Scanner:
    """Orchestrates document scanning, embedding, and issue detection.

    The Scanner is the central component that coordinates:
    1. File discovery - Finding documents in directories
    2. Document parsing - Converting files to Document objects
    3. Embedding generation - Creating semantic embeddings for chunks
    4. Issue detection - Running detectors to find problems

    Attributes:
        config: DocLint configuration
        parser_registry: Registry of available parsers
        embedding_generator: Generator for semantic embeddings
        cache_manager: Manager for embedding cache
        detector_registry: Dictionary of detector instances

    Example:
        >>> scanner = Scanner(config)
        >>> result = scanner.scan_directory(Path("./docs"))
        >>> print(f"Found {len(result.documents)} documents")
        >>> print(f"Detected {result.total_issues} issues")
    """

    def __init__(
        self,
        config: Optional[DocLintConfig] = None,
        parser_registry: Optional[ParserRegistry] = None,
        embedding_generator: Optional[BaseEmbeddingGenerator] = None,
        cache_manager: Optional[CacheManager] = None,
        detector_registry: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize scanner with components.

        Args:
            config: DocLint configuration (uses defaults if None)
            parser_registry: Custom parser registry (creates default if None)
            embedding_generator: Custom embedding generator (creates default if None)
            cache_manager: Custom cache manager (creates default based on config if None)
            detector_registry: Dictionary of detector instances (empty if None)
        """
        self.config = config or DocLintConfig()
        self.parser_registry = parser_registry or ParserRegistry()
        self.detector_registry = detector_registry or {}

        # Lazy-load embedding generator to avoid model loading at init
        self._embedding_generator = embedding_generator
        self._cache_manager = cache_manager
        self._document_processor: Optional[DocumentProcessor] = None

        logger.debug(f"Initialized Scanner with config: {self.config.to_dict()}")

    @property
    def embedding_generator(self) -> BaseEmbeddingGenerator:
        """Lazy-load embedding generator."""
        if self._embedding_generator is None:
            from ..embeddings.generator import SentenceTransformerGenerator

            self._embedding_generator = SentenceTransformerGenerator(
                model_name=self.config.embedding.model_name,
                device=self.config.embedding.device,
            )
        return self._embedding_generator

    @property
    def cache_manager(self) -> Optional[CacheManager]:
        """Lazy-load cache manager."""
        if self._cache_manager is None and self.config.cache_enabled:
            self._cache_manager = CacheManager(
                cache_dir=self.config.cache_dir,
                model_name=self.config.embedding.model_name,
            )
        return self._cache_manager

    @property
    def document_processor(self) -> DocumentProcessor:
        """Lazy-load document processor."""
        if self._document_processor is None:
            self._document_processor = DocumentProcessor(
                generator=self.embedding_generator,
                cache=self.cache_manager,
                chunk_size=self.config.embedding.chunk_size,
                chunk_overlap=self.config.embedding.chunk_overlap,
            )
        return self._document_processor

    def discover_files(
        self,
        path: Path,
        recursive: Optional[bool] = None,
    ) -> List[Path]:
        """Discover parseable files in directory.

        Args:
            path: Directory path to scan
            recursive: Whether to scan recursively (uses config if None)

        Returns:
            List of file paths that can be parsed
        """
        if recursive is None:
            recursive = self.config.recursive

        supported_extensions = set(self.parser_registry.get_supported_extensions())
        discovered_files: List[Path] = []

        if path.is_file():
            # Single file
            if path.suffix.lower() in supported_extensions:
                discovered_files.append(path)
        elif path.is_dir():
            # Directory scan
            pattern = "**/*" if recursive else "*"
            for file_path in path.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    discovered_files.append(file_path)

        logger.info(f"Discovered {len(discovered_files)} parseable files in {path}")
        return sorted(discovered_files)

    def parse_documents(
        self,
        files: List[Path],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Document]:
        """Parse multiple files into documents.

        Uses parallel processing for efficiency when parsing many files.

        Args:
            files: List of file paths to parse
            progress_callback: Optional callback for progress updates (current, total)

        Returns:
            List of successfully parsed documents
        """
        if not files:
            return []

        documents: List[Document] = []
        total = len(files)

        # For small numbers of files, parse sequentially
        if total <= 4 or self.config.max_workers <= 1:
            for i, file_path in enumerate(files):
                try:
                    parser = self.parser_registry.get_parser(file_path)
                    doc = Document.from_file(file_path, parser)
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path}: {e}")

                if progress_callback:
                    progress_callback(i + 1, total)
        else:
            # Parse in parallel for larger sets
            # Note: We prepare parser selection here since parsers aren't pickleable
            file_parser_pairs: List[Tuple[Path, str]] = []
            for file_path in files:
                try:
                    parser = self.parser_registry.get_parser(file_path)
                    file_parser_pairs.append((file_path, parser.file_type))
                except Exception as e:
                    logger.warning(f"No parser for {file_path}: {e}")

            # Parse each file (parallel processing would require pickling parsers)
            for i, (file_path, file_type) in enumerate(file_parser_pairs):
                try:
                    parser = self.parser_registry.get_parser_by_type(file_type)
                    doc = Document.from_file(file_path, parser)
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path}: {e}")

                if progress_callback:
                    progress_callback(i + 1, total)

        logger.info(f"Successfully parsed {len(documents)}/{total} files")
        return documents

    def generate_embeddings(
        self,
        documents: List[Document],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Generate embeddings for documents.

        Uses the document processor to chunk documents and generate embeddings.
        Embeddings are cached for efficiency.

        Args:
            documents: List of documents to process (modified in-place)
            progress_callback: Optional callback for progress updates (current, total)
        """
        if not documents:
            return

        total = len(documents)

        for i, document in enumerate(documents):
            self.document_processor.process_document(document)
            if progress_callback:
                progress_callback(i + 1, total)

        # Log cache statistics if available
        if self.cache_manager:
            stats = self.cache_manager.get_stats()
            logger.info(f"Cache stats: {stats}")

    def run_detectors(
        self,
        documents: List[Document],
    ) -> Dict[str, List[Issue]]:
        """Run all enabled detectors on documents.

        Args:
            documents: List of documents with embeddings

        Returns:
            Dictionary mapping detector name to list of issues
        """
        issues: Dict[str, List[Issue]] = {}

        for name, detector in self.detector_registry.items():
            if hasattr(detector, "enabled") and not detector.enabled:
                continue

            try:
                detector_issues = detector.detect(documents)
                if detector_issues:
                    issues[name] = detector_issues
                    logger.info(f"Detector '{name}' found {len(detector_issues)} issues")
            except Exception as e:
                logger.error(f"Detector '{name}' failed: {e}")

        return issues

    def scan_directory(
        self,
        path: Path,
        recursive: Optional[bool] = None,
        progress: Optional[Progress] = None,
    ) -> ScanResult:
        """Scan a directory for documents and detect issues.

        This is the main entry point for scanning. It orchestrates:
        1. File discovery
        2. Document parsing
        3. Embedding generation
        4. Issue detection

        Args:
            path: Path to directory or file to scan
            recursive: Whether to scan recursively (uses config if None)
            progress: Optional Rich progress instance for visual feedback

        Returns:
            ScanResult containing documents, issues, and statistics
        """
        path = Path(path).resolve()
        logger.info(f"Starting scan of {path}")

        stats: Dict[str, Any] = {
            "path": str(path),
            "recursive": recursive if recursive is not None else self.config.recursive,
        }

        # Task IDs for progress tracking
        discover_task: Optional[TaskID] = None
        parse_task: Optional[TaskID] = None
        embed_task: Optional[TaskID] = None
        detect_task: Optional[TaskID] = None

        if progress:
            discover_task = progress.add_task("[cyan]Discovering files...", total=None)

        # Step 1: Discover files
        files = self.discover_files(path, recursive)
        stats["files_discovered"] = len(files)

        if progress and discover_task is not None:
            progress.update(discover_task, completed=True, total=1)

        if not files:
            logger.warning(f"No parseable files found in {path}")
            return ScanResult(documents=[], issues={}, stats=stats)

        # Step 2: Parse documents
        if progress:
            parse_task = progress.add_task("[green]Parsing documents...", total=len(files))

        def parse_progress(current: int, total: int) -> None:
            if progress and parse_task is not None:
                progress.update(parse_task, completed=current)

        documents = self.parse_documents(files, progress_callback=parse_progress)
        stats["documents_parsed"] = len(documents)

        if not documents:
            logger.warning("No documents were successfully parsed")
            return ScanResult(documents=[], issues={}, stats=stats)

        # Step 3: Generate embeddings
        if progress:
            embed_task = progress.add_task("[yellow]Generating embeddings...", total=len(documents))

        def embed_progress(current: int, total: int) -> None:
            if progress and embed_task is not None:
                progress.update(embed_task, completed=current)

        self.generate_embeddings(documents, progress_callback=embed_progress)

        total_chunks = sum(len(doc.chunks) for doc in documents)
        stats["total_chunks"] = total_chunks

        # Step 4: Run detectors
        if progress:
            detect_task = progress.add_task(
                "[magenta]Running detectors...", total=len(self.detector_registry) or 1
            )

        issues = self.run_detectors(documents)
        stats["total_issues"] = sum(len(i) for i in issues.values())

        if progress and detect_task is not None:
            progress.update(detect_task, completed=len(self.detector_registry) or 1)

        logger.info(
            f"Scan complete: {len(documents)} documents, "
            f"{total_chunks} chunks, {stats['total_issues']} issues"
        )

        return ScanResult(documents=documents, issues=issues, stats=stats)

    def close(self) -> None:
        """Close scanner and release resources."""
        if self._cache_manager:
            self._cache_manager.close()
            self._cache_manager = None

    def __enter__(self) -> "Scanner":
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()
