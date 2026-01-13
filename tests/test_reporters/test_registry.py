"""Tests for reporter registry."""

import pytest

from doclint.core.scanner import ScanResult
from doclint.reporters import (
    ConsoleReporter,
    HTMLReporter,
    JSONReporter,
    get_reporter,
    list_formats,
    register_reporter,
)
from doclint.reporters.base import BaseReporter
from doclint.reporters.registry import ReporterRegistry


def test_registry_initialization():
    """Test registry initializes with built-in reporters."""
    registry = ReporterRegistry()

    formats = registry.list_formats()
    assert "console" in formats
    assert "json" in formats
    assert "html" in formats


def test_registry_get_reporter():
    """Test getting reporters from registry."""
    registry = ReporterRegistry()

    console = registry.get_reporter("console")
    assert isinstance(console, ConsoleReporter)

    json_rep = registry.get_reporter("json")
    assert isinstance(json_rep, JSONReporter)

    html = registry.get_reporter("html")
    assert isinstance(html, HTMLReporter)


def test_registry_get_reporter_case_insensitive():
    """Test getting reporters is case-insensitive."""
    registry = ReporterRegistry()

    console1 = registry.get_reporter("console")
    console2 = registry.get_reporter("CONSOLE")
    console3 = registry.get_reporter("Console")

    assert type(console1) == type(console2) == type(console3)


def test_registry_get_reporter_with_kwargs():
    """Test getting reporters with constructor arguments."""
    registry = ReporterRegistry()

    json_rep = registry.get_reporter("json", pretty=False)
    assert json_rep.pretty is False

    html = registry.get_reporter("html", title="Custom Title")
    assert html.title == "Custom Title"


def test_registry_get_reporter_unknown_format():
    """Test getting unknown reporter raises ValueError."""
    registry = ReporterRegistry()

    with pytest.raises(ValueError, match="Unknown reporter format"):
        registry.get_reporter("unknown_format")


def test_registry_register_custom_reporter():
    """Test registering a custom reporter."""
    registry = ReporterRegistry()

    class CustomReporter(BaseReporter):
        name = "custom"
        description = "Custom test reporter"

        def report(self, result: ScanResult, output_path=None):
            return "custom output"

    registry.register("custom", CustomReporter)

    reporter = registry.get_reporter("custom")
    assert isinstance(reporter, CustomReporter)

    # Should be in list of formats
    assert "custom" in registry.list_formats()


def test_registry_register_non_reporter_raises_error():
    """Test registering non-BaseReporter class raises TypeError."""
    registry = ReporterRegistry()

    class NotAReporter:
        pass

    with pytest.raises(TypeError, match="must inherit from BaseReporter"):
        registry.register("bad", NotAReporter)


def test_registry_is_registered():
    """Test checking if format is registered."""
    registry = ReporterRegistry()

    assert registry.is_registered("console")
    assert registry.is_registered("json")
    assert registry.is_registered("html")
    assert not registry.is_registered("unknown")


def test_global_get_reporter():
    """Test global get_reporter function."""
    reporter = get_reporter("console")
    assert isinstance(reporter, ConsoleReporter)

    reporter = get_reporter("json", pretty=True)
    assert isinstance(reporter, JSONReporter)
    assert reporter.pretty is True


def test_global_list_formats():
    """Test global list_formats function."""
    formats = list_formats()

    assert "console" in formats
    assert "json" in formats
    assert "html" in formats
    assert isinstance(formats, list)


def test_global_register_reporter():
    """Test global register_reporter function."""

    class GlobalCustomReporter(BaseReporter):
        name = "global_custom"
        description = "Global custom reporter"

        def report(self, result: ScanResult, output_path=None):
            return "global custom"

    register_reporter("global_custom", GlobalCustomReporter)

    # Should be available via global get_reporter
    reporter = get_reporter("global_custom")
    assert isinstance(reporter, GlobalCustomReporter)


def test_reporter_registry_list_formats_sorted():
    """Test that list_formats returns sorted list."""
    registry = ReporterRegistry()
    formats = registry.list_formats()

    # Should be sorted alphabetically
    assert formats == sorted(formats)
