"""Console reporter with rich terminal output."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.scanner import ScanResult
from ..detectors.base import Issue, IssueSeverity
from .base import BaseReporter


class ConsoleReporter(BaseReporter):
    """Beautiful terminal output using Rich library.

    This reporter generates formatted console output with colors, panels, and tables.
    It groups issues by severity, displays them in attractive panels, and provides
    a summary table at the end.

    Features:
        - Colored output with severity-based styling
        - Emoji icons for visual severity indication
        - Grouped issues (critical → warning → info)
        - Document paths relative to scan directory
        - Summary statistics table
        - Clean percentage indicator

    Example:
        >>> reporter = ConsoleReporter()
        >>> output = reporter.report(scan_result)
        >>> print(output)  # Displays formatted output to terminal
    """

    name = "console"
    description = "Colored terminal output with rich formatting"

    # Emoji/icon mapping for severity levels
    SEVERITY_ICONS = {
        IssueSeverity.CRITICAL: "🔴",
        IssueSeverity.WARNING: "⚠️ ",
        IssueSeverity.INFO: "ℹ️ ",
    }

    # Color scheme for severity levels
    SEVERITY_COLORS = {
        IssueSeverity.CRITICAL: "red",
        IssueSeverity.WARNING: "yellow",
        IssueSeverity.INFO: "blue",
    }

    # Style for severity level text
    SEVERITY_STYLES = {
        IssueSeverity.CRITICAL: "bold red",
        IssueSeverity.WARNING: "bold yellow",
        IssueSeverity.INFO: "bold blue",
    }

    def __init__(self, use_colors: bool = True):
        """Initialize console reporter.

        Args:
            use_colors: Whether to use colored output. Set to False for piped output
                       or environments without color support.
        """
        self.console = Console(highlight=False, force_terminal=use_colors)

    def report(self, result: ScanResult, output_path: Optional[Path] = None) -> str:
        """Generate beautiful console report.

        Args:
            result: Scan results to format
            output_path: Optional file path to save plain text output

        Returns:
            Formatted report string (with ANSI color codes if enabled)
        """
        # Generate output sections
        self._print_header(result)
        self._print_issues_by_severity(result)
        self._print_summary(result)

        # For file output, capture console output
        if output_path:
            # Create a new console for file output (no colors)
            file_console = Console(file=open(output_path, "w"), highlight=False)
            file_console.print(self._generate_text_report(result))

        # Return empty string for console (output already printed)
        return ""

    def _print_header(self, result: ScanResult) -> None:
        """Print report header with scan information.

        Args:
            result: Scan results
        """
        self.console.rule("[bold blue]DocLint Scan Results[/bold blue]")
        self.console.print()

        path = result.stats.get("path", "unknown")
        doc_count = len(result.documents)
        chunk_count = result.stats.get("total_chunks", 0)

        self.console.print(f"📁 Path: {path}")
        self.console.print(f"📄 Documents: {doc_count}")
        self.console.print(f"🔷 Chunks: {chunk_count}")
        self.console.print()

    def _print_issues_by_severity(self, result: ScanResult) -> None:
        """Print issues grouped by severity level.

        Args:
            result: Scan results
        """
        # Group issues by severity
        issues_by_severity = self._group_issues_by_severity(result.issues)

        # Print critical issues
        if issues_by_severity[IssueSeverity.CRITICAL]:
            self._print_severity_section(
                IssueSeverity.CRITICAL, issues_by_severity[IssueSeverity.CRITICAL]
            )

        # Print warnings
        if issues_by_severity[IssueSeverity.WARNING]:
            self._print_severity_section(
                IssueSeverity.WARNING, issues_by_severity[IssueSeverity.WARNING]
            )

        # Print info
        if issues_by_severity[IssueSeverity.INFO]:
            self._print_severity_section(IssueSeverity.INFO, issues_by_severity[IssueSeverity.INFO])

        # If no issues found
        if result.total_issues == 0:
            self.console.print()
            self.console.print(
                Panel(
                    "[bold green]✨ No issues found! Your knowledge base looks great.[/bold green]",
                    border_style="green",
                    box=box.ROUNDED,
                )
            )
            self.console.print()

    def _print_severity_section(self, severity: IssueSeverity, issues: List[Issue]) -> None:
        """Print a section of issues with the same severity.

        Args:
            severity: Severity level
            issues: List of issues with this severity
        """
        icon = self.SEVERITY_ICONS[severity]
        style = self.SEVERITY_STYLES[severity]
        severity_name = severity.value.upper()

        self.console.print()
        self.console.print(f"{icon} [{style}]{severity_name} ({len(issues)})[/{style}]")
        self.console.print()

        for issue in issues:
            self._print_issue(issue)

    def _print_issue(self, issue: Issue) -> None:
        """Print a single issue in a formatted panel.

        Args:
            issue: Issue to display
        """
        color = self.SEVERITY_COLORS[issue.severity]

        # Build panel content
        content_parts = []

        # Title and description
        content_parts.append(f"[bold]{issue.title}[/bold]")
        content_parts.append("")
        content_parts.append(issue.description)
        content_parts.append("")

        # Documents
        if issue.documents:
            content_parts.append("[dim]Documents:[/dim]")
            for doc_path in issue.documents:
                # Show relative path if possible
                try:
                    rel_path = Path(doc_path).relative_to(Path.cwd())
                    content_parts.append(f"  • {rel_path}")
                except ValueError:
                    content_parts.append(f"  • {doc_path}")
            content_parts.append("")

        # Chunks (if present)
        if issue.chunks:
            content_parts.append(f"[dim]Affected chunks:[/dim] {len(issue.chunks)}")
            # Show preview of first chunk
            if issue.chunks:
                first_chunk = issue.chunks[0]
                preview = first_chunk.text[:100]
                if len(first_chunk.text) > 100:
                    preview += "..."
                content_parts.append(f"  [dim]Preview:[/dim] {preview}")
            content_parts.append("")

        # Details
        if issue.details:
            content_parts.append("[dim]Details:[/dim]")
            for key, value in issue.details.items():
                # Format key nicely (snake_case → Title Case)
                display_key = key.replace("_", " ").title()

                # Handle different value types
                if isinstance(value, list):
                    if len(value) <= 3:
                        display_value = ", ".join(str(v) for v in value)
                    else:
                        display_value = (
                            f"{', '.join(str(v) for v in value[:3])} (and {len(value) - 3} more)"
                        )
                elif isinstance(value, (int, float)):
                    display_value = str(value)
                elif isinstance(value, str) and len(value) > 100:
                    display_value = value[:100] + "..."
                else:
                    display_value = str(value)

                content_parts.append(f"  {display_key}: {display_value}")

        content = "\n".join(content_parts)

        # Create panel
        panel = Panel(
            content,
            title=f"[{color}]{issue.detector}[/{color}]",
            border_style=color,
            box=box.ROUNDED,
            padding=(0, 1),
        )

        self.console.print(panel)
        self.console.print()

    def _print_summary(self, result: ScanResult) -> None:
        """Print summary statistics table.

        Args:
            result: Scan results
        """
        self.console.rule("[bold]Summary[/bold]")
        self.console.print()

        # Count issues by severity
        issues_by_severity = self._group_issues_by_severity(result.issues)
        critical_count = len(issues_by_severity[IssueSeverity.CRITICAL])
        warning_count = len(issues_by_severity[IssueSeverity.WARNING])
        info_count = len(issues_by_severity[IssueSeverity.INFO])

        # Calculate clean documents
        total_docs = len(result.documents)
        # Collect unique document paths with issues
        docs_with_issues: Set[str] = set()
        for issues in result.issues.values():
            for issue in issues:
                docs_with_issues.update(str(d) for d in issue.documents)
        clean_docs = total_docs - len(docs_with_issues)

        # Create summary table
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")

        table.add_row("Total documents", str(total_docs))
        table.add_row("Total issues", str(result.total_issues))
        table.add_row("Critical issues", f"[red]{critical_count}[/red]" if critical_count else "0")
        table.add_row("Warnings", f"[yellow]{warning_count}[/yellow]" if warning_count else "0")
        table.add_row("Info", f"[blue]{info_count}[/blue]" if info_count else "0")

        # Calculate and display quality percentage
        if total_docs > 0:
            clean_percentage = (clean_docs / total_docs) * 100
            if clean_percentage == 100:
                quality_str = f"[green]{clean_docs} ({clean_percentage:.0f}%)[/green] ✨"
            elif clean_percentage >= 80:
                quality_str = f"[green]{clean_docs} ({clean_percentage:.1f}%)[/green]"
            elif clean_percentage >= 50:
                quality_str = f"[yellow]{clean_docs} ({clean_percentage:.1f}%)[/yellow]"
            else:
                quality_str = f"[red]{clean_docs} ({clean_percentage:.1f}%)[/red]"
            table.add_row("Clean documents", quality_str)

        self.console.print(table)
        self.console.print()

    def _group_issues_by_severity(
        self, issues_dict: Dict[str, List[Any]]
    ) -> Dict[IssueSeverity, List[Any]]:
        """Group all issues by severity level.

        Args:
            issues_dict: Dictionary mapping detector name to list of issues

        Returns:
            Dictionary mapping severity to list of issues
        """
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

    def _generate_text_report(self, result: ScanResult) -> str:
        """Generate plain text report for file output.

        Args:
            result: Scan results

        Returns:
            Plain text report without color codes
        """
        lines = []
        lines.append("=" * 60)
        lines.append("DocLint Scan Results")
        lines.append("=" * 60)
        lines.append("")

        path = result.stats.get("path", "unknown")
        lines.append(f"Path: {path}")
        lines.append(f"Documents: {len(result.documents)}")
        lines.append(f"Total Issues: {result.total_issues}")
        lines.append("")

        # Group and print issues
        issues_by_severity = self._group_issues_by_severity(result.issues)

        for severity in [IssueSeverity.CRITICAL, IssueSeverity.WARNING, IssueSeverity.INFO]:
            if issues_by_severity[severity]:
                lines.append(f"\n{severity.value.upper()} ({len(issues_by_severity[severity])})")
                lines.append("-" * 60)

                for issue in issues_by_severity[severity]:
                    lines.append(f"\n{issue.title}")
                    lines.append(f"Detector: {issue.detector}")
                    lines.append(f"Description: {issue.description}")
                    if issue.documents:
                        lines.append("Documents:")
                        for doc in issue.documents:
                            lines.append(f"  - {doc}")

        return "\n".join(lines)
