"""JSON reporter for machine-readable output."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..core.scanner import ScanResult
from ..detectors.base import IssueSeverity
from ..version import __version__
from .base import BaseReporter


class JSONReporter(BaseReporter):
    """Machine-readable JSON output for CI/CD integration.

    Generates structured JSON output that can be easily consumed by other tools,
    CI/CD pipelines, or processed with utilities like jq.

    Output schema:
        {
            "scan_info": {
                "timestamp": "ISO-8601 datetime",
                "path": "scanned directory path",
                "version": "doclint version",
                "total_documents": int,
                "total_chunks": int
            },
            "statistics": {
                "total_issues": int,
                "critical": int,
                "warning": int,
                "info": int,
                "clean_documents": int,
                "quality_percentage": float
            },
            "issues": [
                {
                    "severity": "critical|warning|info",
                    "detector": "detector name",
                    "title": "issue title",
                    "description": "issue description",
                    "documents": ["path1", "path2"],
                    "chunks": [...],  // optional
                    "details": {...}
                }
            ],
            "documents": [...]  // optional, if include_documents=True
        }

    Example:
        >>> reporter = JSONReporter(pretty=True)
        >>> output = reporter.report(scan_result)
        >>> data = json.loads(output)
        >>> print(data["statistics"]["critical"])
    """

    name = "json"
    description = "Machine-readable JSON output"

    def __init__(self, pretty: bool = True, include_documents: bool = False):
        """Initialize JSON reporter.

        Args:
            pretty: Whether to pretty-print JSON with indentation
            include_documents: Whether to include full document list in output
        """
        self.pretty = pretty
        self.include_documents = include_documents

    def report(self, result: ScanResult, output_path: Optional[Path] = None) -> str:
        """Generate JSON report.

        Args:
            result: Scan results to format
            output_path: Optional file path to save JSON output

        Returns:
            JSON string
        """
        # Build JSON structure
        report_data = self._build_report_structure(result)

        # Serialize to JSON
        if self.pretty:
            json_output = json.dumps(report_data, indent=2, ensure_ascii=False)
        else:
            json_output = json.dumps(report_data, ensure_ascii=False)

        # Write to file if path provided
        if output_path:
            self.write_to_file(json_output, output_path)

        return json_output

    def _build_report_structure(self, result: ScanResult) -> Dict[str, Any]:
        """Build the complete JSON report structure.

        Args:
            result: Scan results

        Returns:
            Dictionary representing the full report
        """
        # Collect statistics
        issues_by_severity = self._count_issues_by_severity(result.issues)
        total_docs = len(result.documents)

        # Calculate unique documents with issues
        docs_with_issues: Set[str] = set()
        for issues in result.issues.values():
            for issue in issues:
                docs_with_issues.update(str(p) for p in issue.documents)
        clean_docs = total_docs - len(docs_with_issues)

        quality_percentage = (clean_docs / total_docs * 100) if total_docs > 0 else 100.0

        report = {
            "scan_info": {
                "timestamp": datetime.now().isoformat(),
                "path": str(result.stats.get("path", "")),
                "version": __version__,
                "total_documents": total_docs,
                "total_chunks": result.stats.get("total_chunks", 0),
                "recursive": result.stats.get("recursive", True),
            },
            "statistics": {
                "total_issues": result.total_issues,
                "critical": issues_by_severity[IssueSeverity.CRITICAL],
                "warning": issues_by_severity[IssueSeverity.WARNING],
                "info": issues_by_severity[IssueSeverity.INFO],
                "clean_documents": clean_docs,
                "quality_percentage": round(quality_percentage, 2),
            },
            "issues": self._serialize_issues(result.issues),
        }

        # Optionally include full document list
        if self.include_documents:
            report["documents"] = [
                {
                    "path": str(doc.path),
                    "file_type": doc.file_type,
                    "size_bytes": doc.size_bytes,
                    "content_hash": doc.content_hash,
                    "chunks": len(doc.chunks),
                    "metadata": self._serialize_metadata(doc.metadata),
                }
                for doc in result.documents
            ]

        return report

    def _serialize_issues(self, issues_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Serialize all issues to JSON-compatible dictionaries.

        Args:
            issues_dict: Dictionary mapping detector name to list of issues

        Returns:
            List of serialized issues
        """
        serialized = []

        for detector_name, issues in issues_dict.items():
            for issue in issues:
                serialized.append(issue.to_dict())

        # Sort by severity (critical first), then by detector name
        severity_order = {
            IssueSeverity.CRITICAL.value: 0,
            IssueSeverity.WARNING.value: 1,
            IssueSeverity.INFO.value: 2,
        }

        serialized.sort(key=lambda x: (severity_order.get(x["severity"], 99), x["detector"]))

        return serialized

    def _serialize_metadata(self, metadata: Any) -> Dict[str, Any]:
        """Serialize document metadata to JSON-compatible dictionary.

        Args:
            metadata: DocumentMetadata object

        Returns:
            Dictionary representation of metadata
        """
        return {
            "author": metadata.author,
            "created": metadata.created.isoformat() if metadata.created else None,
            "modified": metadata.modified.isoformat() if metadata.modified else None,
            "version": metadata.version,
            "title": metadata.title,
            "tags": metadata.tags,
            "custom": metadata.custom,
        }

    def _count_issues_by_severity(
        self, issues_dict: Dict[str, List[Any]]
    ) -> Dict[IssueSeverity, int]:
        """Count issues by severity level.

        Args:
            issues_dict: Dictionary mapping detector name to list of issues

        Returns:
            Dictionary mapping severity to count
        """
        counts = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.WARNING: 0,
            IssueSeverity.INFO: 0,
        }

        for issues in issues_dict.values():
            for issue in issues:
                # Handle both scanner.Issue (with string severity) and detectors.base.Issue
                severity = (
                    issue.severity
                    if isinstance(issue.severity, IssueSeverity)
                    else IssueSeverity(issue.severity)
                )
                counts[severity] += 1

        return counts
