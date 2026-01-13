"""Main CLI application using Typer."""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.progress import Progress

from doclint.core.config import DocLintConfig
from doclint.core.scanner import Scanner
from doclint.detectors.completeness import CompletenessDetector
from doclint.detectors.conflicts import ConflictDetector
from doclint.detectors.registry import DetectorRegistry
from doclint.reporters import get_reporter
from doclint.version import __version__

app = typer.Typer(
    name="doclint",
    help="Data quality linting for AI knowledge bases",
    rich_markup_mode=None,
)

console = Console()


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"[bold blue]DocLint[/bold blue] version {__version__}")


@app.command()
def scan(
    path: str = typer.Argument(
        ...,
        help="Path to the directory or file to scan",
    ),
    format: str = typer.Option(
        "console",
        "--format",
        "-f",
        help="Output format: console, json, or html",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (for json/html formats)",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    recursive: Optional[bool] = typer.Option(
        None,
        "--recursive/--no-recursive",
        help="Scan directories recursively (default: true)",
    ),
    check_external_links: bool = typer.Option(
        False,
        "--check-external-links",
        help="Validate external URLs (opt-in, slower)",
    ),
    enable_llm: bool = typer.Option(
        False,
        "--enable-llm",
        help="Enable LLM-based conflict verification (opt-in)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
) -> None:
    """Scan a directory for data quality issues.

    This command scans your knowledge base for common data quality problems:
    - Conflicts: Contradictory information across documents
    - Completeness: Missing metadata, broken links, insufficient content

    Examples:
        # Basic scan with console output
        $ doclint scan ./docs/

        # Export as JSON for CI/CD
        $ doclint scan ./docs/ --format json --output report.json

        # Generate HTML report
        $ doclint scan ./docs/ --format html --output report.html

        # Check external links (slower)
        $ doclint scan ./docs/ --check-external-links

    Exit codes:
        0 - Success, no issues found
        1 - Issues found (warnings or info)
        2 - Critical issues found
        3 - Error during scan
    """
    try:
        # Load configuration
        config_obj = DocLintConfig.load(config)

        # Override recursive if specified
        if recursive is not None:
            config_obj.recursive = recursive

        # Override LLM verification if specified
        if enable_llm:
            config_obj.conflict.llm_verifier.enabled = True

        # Show scan configuration
        if verbose:
            console.print("[dim]Configuration loaded[/dim]")
            console.print(f"[dim]  Recursive: {config_obj.recursive}[/dim]")
            console.print(f"[dim]  Cache: {config_obj.cache_enabled}[/dim]")
            console.print(f"[dim]  Embedding model: {config_obj.embedding.model_name}[/dim]")

        # Validate path
        scan_path = Path(path).resolve()
        if not scan_path.exists():
            console.print(f"[red]Error:[/red] Path does not exist: {path}")
            raise typer.Exit(3)

        # Initialize detectors
        detector_registry = DetectorRegistry()

        # Add completeness detector
        completeness = CompletenessDetector(
            required_metadata=config_obj.completeness.required_metadata,
            min_content_length=config_obj.completeness.min_content_length,
            check_external_links=check_external_links,
        )
        detector_registry.register(completeness)

        # Add conflict detector
        from doclint.detectors.llm_verifier import LlamaCppVerifier

        verifier = None
        if enable_llm and config_obj.conflict.llm_verifier.enabled:
            verifier = LlamaCppVerifier(
                model_path=config_obj.conflict.llm_verifier.model_path,
                n_ctx=config_obj.conflict.llm_verifier.n_ctx,
            )

        conflict = ConflictDetector(
            similarity_threshold=config_obj.conflict.similarity_threshold,
            verifier=verifier,
        )
        detector_registry.register(conflict)

        if verbose:
            console.print(
                f"[dim]  Detectors: {', '.join(detector_registry.list_detector_names())}[/dim]"
            )
            console.print()

        # Initialize scanner
        scanner = Scanner(
            config=config_obj,
            detector_registry=detector_registry.get_all_detectors(),
        )

        # Run scan with progress
        with Progress() as progress:
            result = scanner.scan_directory(
                scan_path, recursive=config_obj.recursive, progress=progress
            )

        # Run detectors asynchronously
        detector_issues = asyncio.run(detector_registry.run_all(result.documents))
        # Convert to scanner's Issue format
        from doclint.core.scanner import Issue as ScannerIssue

        scanner_issues: Dict[str, List[ScannerIssue]] = {}
        for detector_name, issue_list in detector_issues.items():
            scanner_issues[detector_name] = [
                ScannerIssue(
                    issue_type=detector_name,
                    severity=issue.severity.value,
                    message=f"{issue.title}: {issue.description}",
                    document_path=issue.documents[0] if issue.documents else Path(),
                    details={
                        "title": issue.title,
                        "description": issue.description,
                        "detector": issue.detector,
                        "documents": [str(d) for d in issue.documents],
                    },
                )
                for issue in issue_list
            ]
        result.issues = scanner_issues
        result.stats["total_issues"] = sum(len(i) for i in scanner_issues.values())

        # Generate report
        try:
            reporter = get_reporter(format)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(3)

        # Determine output path
        output_path = Path(output) if output else None

        # For HTML, require output path
        if format == "html" and not output_path:
            console.print("[red]Error:[/red] HTML format requires --output option")
            raise typer.Exit(3)

        # For JSON, output to console if no file specified
        if format == "json" and not output_path:
            json_output = reporter.report(result)
            console.print(json_output)
        else:
            # Generate report
            reporter.report(result, output_path=output_path)

            # Show success message for file output
            if output_path and format != "console":
                console.print()
                console.print(f"[green]✓[/green] Report saved to: {output_path}")

        # Determine exit code
        if result.total_issues == 0:
            exit_code = 0
        else:
            # Check for critical issues
            has_critical = any(
                issue.severity == "critical"
                for issues in result.issues.values()
                for issue in issues
            )
            exit_code = 2 if has_critical else 1

        # Show summary message
        if verbose or format != "console":
            if exit_code == 0:
                console.print("[green]✨ No issues found![/green]")
            elif exit_code == 1:
                console.print(f"[yellow]⚠️  Found {result.total_issues} issue(s)[/yellow]")
            else:
                console.print(
                    f"[red]🔴 Found {result.total_issues} issue(s) including critical[/red]"
                )

        raise typer.Exit(exit_code)

    except typer.Exit:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Scan interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[red]Error during scan:[/red] {e}")
        if verbose:
            import traceback

            console.print("[dim]" + traceback.format_exc() + "[/dim]")
        raise typer.Exit(3)


@app.command()
def config_show() -> None:
    """Show current configuration.

    Displays the configuration that would be used for scanning, including
    default values and any overrides from config files.
    """
    try:
        config_obj = DocLintConfig.load()
        config_dict = config_obj.to_dict()

        console.print("[bold blue]DocLint Configuration[/bold blue]")
        console.print()

        # General settings
        console.print("[bold]General:[/bold]")
        console.print(f"  Recursive: {config_dict['recursive']}")
        console.print(f"  Cache enabled: {config_dict['cache_enabled']}")
        console.print(f"  Cache dir: {config_dict.get('cache_dir', 'default')}")
        console.print(f"  Max workers: {config_dict['max_workers']}")
        console.print()

        # Embedding settings
        console.print("[bold]Embedding:[/bold]")
        console.print(f"  Model: {config_dict['embedding']['model_name']}")
        console.print(f"  Device: {config_dict['embedding'].get('device', 'auto')}")
        console.print(f"  Chunk size: {config_dict['embedding']['chunk_size']}")
        console.print(f"  Chunk overlap: {config_dict['embedding']['chunk_overlap']}")
        console.print()

        # Detector settings
        console.print("[bold]Detectors:[/bold]")
        console.print("  Completeness:")
        console.print(f"    Enabled: {config_dict['completeness']['enabled']}")
        console.print(
            f"    Required metadata: {', '.join(config_dict['completeness']['required_metadata'])}"
        )
        console.print(
            f"    Min content length: {config_dict['completeness']['min_content_length']}"
        )
        console.print()
        console.print("  Conflict:")
        console.print(f"    Enabled: {config_dict['conflict']['enabled']}")
        console.print(
            f"    Similarity threshold: {config_dict['conflict']['similarity_threshold']}"
        )
        console.print(f"    LLM verification: {config_dict['conflict']['llm_verifier']['enabled']}")
        console.print()

    except Exception as e:
        console.print(f"[red]Error loading configuration:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def formats() -> None:
    """List available output formats."""
    from doclint.reporters import list_formats

    available_formats = list_formats()

    console.print("[bold blue]Available Output Formats[/bold blue]")
    console.print()

    for fmt in available_formats:
        console.print(f"  • [cyan]{fmt}[/cyan]")

    console.print()
    console.print("Use with: [dim]doclint scan <path> --format <format>[/dim]")


if __name__ == "__main__":
    app()
