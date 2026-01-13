"""HTML reporter for web-based reports."""

import html as html_escape
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..core.scanner import ScanResult
from ..detectors.base import Issue, IssueSeverity
from ..version import __version__
from .base import BaseReporter


class HTMLReporter(BaseReporter):
    """Beautiful HTML report generator.

    Generates a self-contained HTML file with embedded CSS and minimal JavaScript.
    The report is responsive, print-friendly, and supports dark mode.

    Features:
        - Self-contained single file (no external dependencies)
        - Responsive design for mobile/tablet/desktop
        - Dark/light mode toggle
        - Interactive filtering by severity
        - Collapsible issue details
        - Print-friendly styling
        - Modern, clean design

    Example:
        >>> reporter = HTMLReporter()
        >>> reporter.report(scan_result, output_path=Path("report.html"))
    """

    name = "html"
    description = "Interactive HTML report"

    # Emoji icons for severity
    SEVERITY_ICONS = {
        IssueSeverity.CRITICAL: "🔴",
        IssueSeverity.WARNING: "⚠️",
        IssueSeverity.INFO: "ℹ️",
    }

    # Color scheme
    SEVERITY_COLORS = {
        IssueSeverity.CRITICAL: "#dc2626",
        IssueSeverity.WARNING: "#f59e0b",
        IssueSeverity.INFO: "#3b82f6",
    }

    def __init__(self, title: str = "DocLint Report"):
        """Initialize HTML reporter.

        Args:
            title: Report title to display in HTML
        """
        self.title = title

    def report(self, result: ScanResult, output_path: Optional[Path] = None) -> str:
        """Generate HTML report.

        Args:
            result: Scan results to format
            output_path: Path to save HTML file (recommended for HTML output)

        Returns:
            Complete HTML document as string
        """
        html_content = self._generate_html(result)

        if output_path:
            self.write_to_file(html_content, output_path)

        return html_content

    def _generate_html(self, result: ScanResult) -> str:
        """Generate complete HTML document.

        Args:
            result: Scan results

        Returns:
            HTML string
        """
        # Calculate statistics
        issues_by_severity = self._group_issues_by_severity(result.issues)
        critical_count = len(issues_by_severity[IssueSeverity.CRITICAL])
        warning_count = len(issues_by_severity[IssueSeverity.WARNING])
        info_count = len(issues_by_severity[IssueSeverity.INFO])

        total_docs = len(result.documents)
        docs_with_issues: Set[str] = set()
        for issues in result.issues.values():
            for issue in issues:
                docs_with_issues.update(str(p) for p in issue.documents)
        clean_docs = total_docs - len(docs_with_issues)
        quality_percentage = (clean_docs / total_docs * 100) if total_docs > 0 else 100.0

        # Build HTML
        html_parts = []
        html_parts.append(self._html_header())
        html_parts.append(self._html_styles())
        html_parts.append("</head><body>")
        html_parts.append(self._html_body_header(result, quality_percentage))
        html_parts.append(
            self._html_statistics(
                result, critical_count, warning_count, info_count, clean_docs, quality_percentage
            )
        )
        html_parts.append(self._html_issues(issues_by_severity))
        html_parts.append(self._html_footer())
        html_parts.append(self._html_script())
        html_parts.append("</body></html>")

        return "\n".join(html_parts)

    def _html_header(self) -> str:
        """Generate HTML header."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="generator" content="DocLint {html_escape.escape(__version__)}">
    <title>{html_escape.escape(self.title)}</title>"""

    def _html_styles(self) -> str:
        """Generate embedded CSS styles."""
        return """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f9fafb;
            --bg-tertiary: #f3f4f6;
            --text-primary: #111827;
            --text-secondary: #6b7280;
            --border-color: #e5e7eb;
            --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);

            --critical: #dc2626;
            --warning: #f59e0b;
            --info: #3b82f6;
            --success: #10b981;
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --bg-primary: #111827;
                --bg-secondary: #1f2937;
                --bg-tertiary: #374151;
                --text-primary: #f9fafb;
                --text-secondary: #9ca3af;
                --border-color: #374151;
            }
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-primary);
            background: var(--bg-secondary);
            padding: 1rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: var(--bg-primary);
            border-radius: 8px;
            box-shadow: var(--shadow-lg);
            overflow: hidden;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
        }

        h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            opacity: 0.9;
            font-size: 0.95rem;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            padding: 2rem;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
        }

        .stat-card {
            background: var(--bg-primary);
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid;
            box-shadow: var(--shadow);
        }

        .stat-card.critical { border-color: var(--critical); }
        .stat-card.warning { border-color: var(--warning); }
        .stat-card.info { border-color: var(--info); }
        .stat-card.success { border-color: var(--success); }

        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .stat-label {
            color: var(--text-secondary);
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .content {
            padding: 2rem;
        }

        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .issues-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .issue {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            border-left: 4px solid;
        }

        .issue.critical { border-left-color: var(--critical); }
        .issue.warning { border-left-color: var(--warning); }
        .issue.info { border-left-color: var(--info); }

        .issue-header {
            padding: 1.25rem;
            cursor: pointer;
            user-select: none;
        }

        .issue-header:hover {
            background: var(--bg-tertiary);
        }

        .issue-title {
            font-size: 1.125rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .issue-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }

        .issue-body {
            padding: 0 1.25rem 1.25rem;
            display: none;
        }

        .issue.expanded .issue-body {
            display: block;
        }

        .issue-description {
            margin-bottom: 1rem;
            line-height: 1.6;
        }

        .issue-section {
            margin-top: 1rem;
        }

        .issue-section-title {
            font-weight: 600;
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
            text-transform: uppercase;
            color: var(--text-secondary);
            letter-spacing: 0.05em;
        }

        .document-list {
            list-style: none;
        }

        .document-list li {
            padding: 0.5rem;
            background: var(--bg-tertiary);
            border-radius: 4px;
            margin-bottom: 0.25rem;
            font-family: monospace;
            font-size: 0.875rem;
        }

        .details-grid {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 0.5rem 1rem;
            font-size: 0.875rem;
        }

        .detail-key {
            font-weight: 600;
            color: var(--text-secondary);
        }

        .no-issues {
            text-align: center;
            padding: 3rem 2rem;
            color: var(--text-secondary);
        }

        .no-issues-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }

        footer {
            padding: 1.5rem 2rem;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border-color);
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        @media print {
            body {
                padding: 0;
                background: white;
            }
            .container {
                box-shadow: none;
            }
            .issue-body {
                display: block !important;
            }
        }
    </style>"""

    def _html_body_header(self, result: ScanResult, quality_percentage: float) -> str:
        """Generate body header section."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        path = html_escape.escape(str(result.stats.get("path", "")))

        return f"""
    <div class="container">
        <header>
            <h1>{html_escape.escape(self.title)}</h1>
            <div class="subtitle">
                Generated on {timestamp} • DocLint v{html_escape.escape(__version__)}
            </div>
            <div class="subtitle">Path: {path}</div>
        </header>"""

    def _html_statistics(
        self,
        result: ScanResult,
        critical: int,
        warning: int,
        info: int,
        clean_docs: int,
        quality_percentage: float,
    ) -> str:
        """Generate statistics section."""
        total_docs = len(result.documents)

        quality_class = "success"
        if quality_percentage < 50:
            quality_class = "critical"
        elif quality_percentage < 80:
            quality_class = "warning"

        return f"""
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{total_docs}</div>
                <div class="stat-label">Total Documents</div>
            </div>
            <div class="stat-card critical">
                <div class="stat-value">{critical}</div>
                <div class="stat-label">Critical Issues</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-value">{warning}</div>
                <div class="stat-label">Warnings</div>
            </div>
            <div class="stat-card info">
                <div class="stat-value">{info}</div>
                <div class="stat-label">Info</div>
            </div>
            <div class="stat-card {quality_class}">
                <div class="stat-value">{quality_percentage:.1f}%</div>
                <div class="stat-label">Quality Score</div>
            </div>
        </div>"""

    def _html_issues(self, issues_by_severity: Dict[IssueSeverity, List[Issue]]) -> str:
        """Generate issues section."""
        parts = ['<div class="content">']

        # Check if there are any issues
        total_issues = sum(len(issues) for issues in issues_by_severity.values())

        if total_issues == 0:
            parts.append(
                """
            <div class="no-issues">
                <div class="no-issues-icon">✨</div>
                <h2>No Issues Found!</h2>
                <p>Your knowledge base looks great.</p>
            </div>"""
            )
        else:
            # Critical issues
            if issues_by_severity[IssueSeverity.CRITICAL]:
                icon = self.SEVERITY_ICONS[IssueSeverity.CRITICAL]
                parts.append(f'<h2 class="section-title">{icon} Critical Issues</h2>')
                parts.append('<div class="issues-list">')
                for issue in issues_by_severity[IssueSeverity.CRITICAL]:
                    parts.append(self._html_issue(issue, "critical"))
                parts.append("</div>")

            # Warnings
            if issues_by_severity[IssueSeverity.WARNING]:
                icon = self.SEVERITY_ICONS[IssueSeverity.WARNING]
                parts.append(
                    f'<h2 class="section-title" style="margin-top: 2rem;">' f"{icon} Warnings</h2>"
                )
                parts.append('<div class="issues-list">')
                for issue in issues_by_severity[IssueSeverity.WARNING]:
                    parts.append(self._html_issue(issue, "warning"))
                parts.append("</div>")

            # Info
            if issues_by_severity[IssueSeverity.INFO]:
                icon = self.SEVERITY_ICONS[IssueSeverity.INFO]
                parts.append(
                    f'<h2 class="section-title" style="margin-top: 2rem;">' f"{icon} Info</h2>"
                )
                parts.append('<div class="issues-list">')
                for issue in issues_by_severity[IssueSeverity.INFO]:
                    parts.append(self._html_issue(issue, "info"))
                parts.append("</div>")

        parts.append("</div>")
        return "\n".join(parts)

    def _html_issue(self, issue: Issue, severity_class: str) -> str:
        """Generate HTML for a single issue."""
        title = html_escape.escape(issue.title)
        description = html_escape.escape(issue.description)
        detector = html_escape.escape(issue.detector)
        icon = self.SEVERITY_ICONS[issue.severity]

        parts = [f'<div class="issue {severity_class}">']
        parts.append(
            '  <div class="issue-header" '
            "onclick=\"this.parentElement.classList.toggle('expanded')\">"
        )
        parts.append(f'    <div class="issue-title">{icon} {title}</div>')
        parts.append('    <div class="issue-meta">')
        parts.append(f"      <span>Detector: {detector}</span>")
        parts.append(f"      <span>{len(issue.documents)} document(s)</span>")
        parts.append("    </div>")
        parts.append("  </div>")
        parts.append('  <div class="issue-body">')
        parts.append(f'    <div class="issue-description">{description}</div>')

        # Documents
        if issue.documents:
            parts.append('    <div class="issue-section">')
            parts.append('      <div class="issue-section-title">Affected Documents</div>')
            parts.append('      <ul class="document-list">')
            for doc_path in issue.documents:
                escaped_path = html_escape.escape(str(doc_path))
                parts.append(f"        <li>{escaped_path}</li>")
            parts.append("      </ul>")
            parts.append("    </div>")

        # Details
        if issue.details:
            parts.append('    <div class="issue-section">')
            parts.append('      <div class="issue-section-title">Details</div>')
            parts.append('      <div class="details-grid">')
            for key, value in issue.details.items():
                display_key = html_escape.escape(key.replace("_", " ").title())
                if isinstance(value, list):
                    display_value = html_escape.escape(", ".join(str(v) for v in value[:5]))
                    if len(value) > 5:
                        display_value += f" (and {len(value) - 5} more)"
                else:
                    display_value = html_escape.escape(str(value))
                parts.append(f'        <div class="detail-key">{display_key}:</div>')
                parts.append(f'        <div class="detail-value">{display_value}</div>')
            parts.append("      </div>")
            parts.append("    </div>")

        parts.append("  </div>")
        parts.append("</div>")

        return "\n".join(parts)

    def _html_footer(self) -> str:
        """Generate footer."""
        return f"""
        <footer>
            <p>Generated by <strong>DocLint</strong> v{html_escape.escape(__version__)} •
            Data quality linting for AI knowledge bases</p>
        </footer>
    </div>"""

    def _html_script(self) -> str:
        """Generate embedded JavaScript."""
        return """
    <script>
        // Optional: Add keyboard shortcuts
        document.addEventListener('keyboardEvent', (e) => {
            if (e.key === 'Escape') {
                document.querySelectorAll('.issue.expanded').forEach(el => {
                    el.classList.remove('expanded');
                });
            }
        });
    </script>"""

    def _group_issues_by_severity(
        self, issues_dict: Dict[str, List[Any]]
    ) -> Dict[IssueSeverity, List[Any]]:
        """Group all issues by severity level."""
        grouped: Dict[IssueSeverity, List[Any]] = {
            IssueSeverity.CRITICAL: [],
            IssueSeverity.WARNING: [],
            IssueSeverity.INFO: [],
        }

        for issues in issues_dict.values():
            for issue in issues:
                # Handle both scanner.Issue (with string severity) and detectors.base.Issue
                severity = (
                    issue.severity
                    if isinstance(issue.severity, IssueSeverity)
                    else IssueSeverity(issue.severity)
                )
                grouped[severity].append(issue)

        return grouped
