"""Base reporter classes for output formatting."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

from ..core.scanner import ScanResult


class BaseReporter(ABC):
    """Abstract base class for all output reporters.

    Reporters take scan results and format them for output in various formats
    (console, JSON, HTML, etc.). Each reporter implements the report() method
    to generate output in its specific format.

    Subclasses must implement:
        - name: Class attribute identifying the reporter format
        - description: Class attribute describing the output format
        - report(): Method that generates the formatted output

    Example:
        >>> class MyReporter(BaseReporter):
        ...     name = "my_format"
        ...     description = "Custom output format"
        ...
        ...     def report(self, result: ScanResult, output_path: Optional[Path] = None) -> str:
        ...         # Generate formatted output
        ...         output = self._format_results(result)
        ...         if output_path:
        ...             self.write_to_file(output, output_path)
        ...         return output
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def report(self, result: ScanResult, output_path: Optional[Path] = None) -> str:
        """Generate formatted report from scan results.

        This is the main entry point for report generation. Subclasses implement
        their specific formatting logic here.

        Args:
            result: ScanResult containing documents, issues, and statistics
            output_path: Optional path to write output file. If None, output is
                        returned as string only (for console/stdout display)

        Returns:
            Formatted report as string. For console reporters, this is displayed
            to the terminal. For file-based reporters (JSON, HTML), this is also
            written to output_path if provided.

        Raises:
            IOError: If output_path is provided but file cannot be written
        """
        pass

    def write_to_file(self, content: str, path: Path) -> None:
        """Write report content to a file.

        Helper method for reporters that support file output. Creates parent
        directories if they don't exist.

        Args:
            content: Formatted report content
            path: Path to output file

        Raises:
            IOError: If file cannot be written
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def get_issue_attr(issue: Any, attr: str, default: Any = None) -> Any:
        """Get an attribute from either scanner.Issue or detectors.base.Issue.

        Helper method that works with both Issue types. For scanner.Issue objects,
        falls back to checking the `details` dict.

        Args:
            issue: Issue object (either type)
            attr: Attribute name to retrieve
            default: Default value if attribute not found

        Returns:
            Attribute value or default
        """
        # Try direct attribute access first
        if hasattr(issue, attr):
            return getattr(issue, attr)
        # Fall back to details dict for scanner.Issue
        if hasattr(issue, "details") and isinstance(issue.details, dict):
            return issue.details.get(attr, default)
        return default

    @staticmethod
    def get_issue_documents(issue: Any) -> List[Path]:
        """Get document list from either scanner.Issue or detectors.base.Issue.

        Args:
            issue: Issue object (either type)

        Returns:
            List of document paths
        """
        # For detectors.base.Issue
        if hasattr(issue, "documents") and isinstance(issue.documents, list):
            return issue.documents
        # For scanner.Issue - return document_path as a single-item list
        if hasattr(issue, "document_path"):
            doc_path = issue.document_path
            if doc_path:
                return [doc_path]
        # Fall back to details dict
        if hasattr(issue, "details") and isinstance(issue.details, dict):
            docs = issue.details.get("documents", [])
            return [Path(d) if not isinstance(d, Path) else d for d in docs]
        return []
