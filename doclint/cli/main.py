"""Main CLI application using Typer."""

import typer
from rich.console import Console

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
def scan(path: str) -> None:
    """Scan a directory for data quality issues.

    Args:
        path: Path to the directory to scan
    """
    console.print(f"[yellow]Scanning {path}...[/yellow]")
    console.print("[dim]Note: Full scanner implementation coming in Phase 2-7[/dim]")
    console.print("[green]OK[/green] Setup complete! DocLint is ready for development.")


if __name__ == "__main__":
    app()
