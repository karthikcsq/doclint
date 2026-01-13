"""Reporter registry for format selection and management."""

from typing import Dict, List, Type

from .base import BaseReporter
from .console import ConsoleReporter
from .html import HTMLReporter
from .json_reporter import JSONReporter


class ReporterRegistry:
    """Registry for managing available reporters.

    Provides a factory pattern for selecting and instantiating reporters based on
    format names. Supports both built-in reporters and custom user-defined reporters.

    Built-in reporters:
        - "console": Rich terminal output (default)
        - "json": Machine-readable JSON output
        - "html": Interactive HTML report

    Example:
        >>> registry = ReporterRegistry()
        >>> reporter = registry.get_reporter("json")
        >>> output = reporter.report(scan_result)

        >>> # Register custom reporter
        >>> registry.register("custom", MyCustomReporter)
        >>> custom = registry.get_reporter("custom")
    """

    def __init__(self) -> None:
        """Initialize registry with built-in reporters."""
        self._reporters: Dict[str, Type[BaseReporter]] = {}

        # Register built-in reporters
        self.register("console", ConsoleReporter)
        self.register("json", JSONReporter)
        self.register("html", HTMLReporter)

    def register(self, name: str, reporter_class: Type[BaseReporter]) -> None:
        """Register a reporter class.

        Args:
            name: Format name (e.g., "json", "html", "custom")
            reporter_class: Reporter class (must inherit from BaseReporter)

        Raises:
            TypeError: If reporter_class is not a BaseReporter subclass
        """
        if not issubclass(reporter_class, BaseReporter):
            raise TypeError(f"{reporter_class} must inherit from BaseReporter")

        self._reporters[name.lower()] = reporter_class

    def get_reporter(self, format_name: str, **kwargs: object) -> BaseReporter:
        """Get a reporter instance by format name.

        Args:
            format_name: Name of the reporter format (e.g., "console", "json")
            **kwargs: Additional keyword arguments passed to reporter constructor

        Returns:
            Instantiated reporter

        Raises:
            ValueError: If format name is not registered
        """
        format_name = format_name.lower()

        if format_name not in self._reporters:
            available = ", ".join(self.list_formats())
            raise ValueError(
                f"Unknown reporter format: '{format_name}'. " f"Available formats: {available}"
            )

        reporter_class = self._reporters[format_name]
        return reporter_class(**kwargs)

    def list_formats(self) -> List[str]:
        """List all available reporter formats.

        Returns:
            List of registered format names
        """
        return sorted(self._reporters.keys())

    def is_registered(self, format_name: str) -> bool:
        """Check if a format is registered.

        Args:
            format_name: Format name to check

        Returns:
            True if format is registered, False otherwise
        """
        return format_name.lower() in self._reporters


# Global registry instance
_default_registry = ReporterRegistry()


def get_reporter(format_name: str, **kwargs: object) -> BaseReporter:
    """Get a reporter from the default registry.

    Convenience function for accessing reporters without creating a registry instance.

    Args:
        format_name: Name of the reporter format
        **kwargs: Additional arguments passed to reporter constructor

    Returns:
        Instantiated reporter

    Example:
        >>> from doclint.reporters import get_reporter
        >>> reporter = get_reporter("json", pretty=True)
    """
    return _default_registry.get_reporter(format_name, **kwargs)


def register_reporter(name: str, reporter_class: Type[BaseReporter]) -> None:
    """Register a custom reporter with the default registry.

    Args:
        name: Format name
        reporter_class: Reporter class

    Example:
        >>> from doclint.reporters import register_reporter
        >>> register_reporter("myformat", MyReporter)
    """
    _default_registry.register(name, reporter_class)


def list_formats() -> List[str]:
    """List available reporter formats from default registry.

    Returns:
        List of format names

    Example:
        >>> from doclint.reporters import list_formats
        >>> print(list_formats())
        ['console', 'json']
    """
    return _default_registry.list_formats()
