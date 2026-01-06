"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_docs_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test documents."""
    docs_dir = tmp_path / "test_docs"
    docs_dir.mkdir()
    return docs_dir
