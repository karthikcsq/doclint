"""Base detector classes and data models for issue detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.document import Chunk, Document


class IssueSeverity(Enum):
    """Severity levels for detected issues.

    Attributes:
        INFO: Informational finding, not necessarily a problem
        WARNING: Potential issue that should be reviewed
        CRITICAL: Serious issue that requires attention
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Issue:
    """Represents a detected issue in the document corpus.

    Issues are the output of detectors and contain all information needed
    to understand and address the detected problem.

    Attributes:
        severity: How serious the issue is (INFO, WARNING, CRITICAL)
        detector: Name of the detector that found this issue
        title: Short summary of the issue
        description: Detailed explanation of the issue
        documents: List of document paths involved in this issue
        chunks: Optional list of specific chunks involved (for chunk-level issues)
        details: Additional structured data about the issue
    """

    severity: IssueSeverity
    detector: str
    title: str
    description: str
    documents: List[Path]
    chunks: Optional[List[Chunk]] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to a dictionary representation.

        Returns:
            Dictionary with all issue fields serialized
        """
        result: Dict[str, Any] = {
            "severity": self.severity.value,
            "detector": self.detector,
            "title": self.title,
            "description": self.description,
            "documents": [str(p) for p in self.documents],
            "details": self.details,
        }

        if self.chunks:
            result["chunks"] = [
                {
                    "document_path": str(chunk.document_path),
                    "index": chunk.index,
                    "text_preview": (
                        chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
                    ),
                    "start_pos": chunk.start_pos,
                    "end_pos": chunk.end_pos,
                }
                for chunk in self.chunks
            ]

        return result


class ContradictionVerifier(ABC):
    """Abstract base class for contradiction verification.

    This provides a hook for LLM-based or other advanced contradiction
    detection methods. Implementations can use language models, knowledge
    graphs, or other techniques to verify if two text chunks contradict.

    Example:
        >>> class LLMVerifier(ContradictionVerifier):
        ...     async def verify(self, text_a: str, text_b: str):
        ...         # Call LLM API to check for contradiction
        ...         response = await llm.analyze(text_a, text_b)
        ...         return response.is_contradiction, response.confidence, response.explanation
    """

    @abstractmethod
    async def verify(self, text_a: str, text_b: str) -> tuple[bool, float, str]:
        """Verify if two text chunks contradict each other.

        Args:
            text_a: First text chunk
            text_b: Second text chunk

        Returns:
            Tuple of (is_contradiction, confidence, explanation):
                - is_contradiction: True if texts contradict
                - confidence: Confidence score 0-1
                - explanation: Human-readable explanation of the finding
        """
        pass


class BaseDetector(ABC):
    """Abstract base class for all issue detectors.

    Detectors analyze a corpus of documents and identify potential issues
    such as conflicts, missing information, or drift from previous versions.

    Subclasses must implement:
        - name: Class attribute identifying the detector
        - description: Class attribute describing what the detector does
        - detect(): Async method that performs the detection

    Example:
        >>> class MyDetector(BaseDetector):
        ...     name = "my_detector"
        ...     description = "Detects custom issues"
        ...
        ...     async def detect(self, documents: List[Document]) -> List[Issue]:
        ...         issues = []
        ...         # Detection logic here
        ...         return issues
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    async def detect(self, documents: List[Document]) -> List[Issue]:
        """Detect issues in the given documents.

        This is the main entry point for detection. Subclasses implement
        their specific detection logic here.

        Args:
            documents: List of documents to analyze. Documents should have
                      their chunks populated with embeddings.

        Returns:
            List of Issue objects describing detected problems
        """
        pass
