"""Pytest configuration and shared fixtures."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def tmp_docs_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test documents."""
    docs_dir = tmp_path / "test_docs"
    docs_dir.mkdir()
    return docs_dir


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Create temporary cache directory for testing."""
    cache = tmp_path / "test_cache"
    cache.mkdir()
    return cache


@pytest.fixture
def sample_texts() -> list[str]:
    """Provide sample text data for embedding tests."""
    return [
        "This is a test document about machine learning.",
        "Python is a popular programming language.",
        "Data quality is important for AI systems.",
        "DocLint helps detect issues in knowledge bases.",
        "",  # Empty string for edge case testing
    ]


@pytest.fixture
def mock_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Mock SentenceTransformer for fast tests without model loading."""

    class MockModel:
        """Mock sentence-transformers model."""

        def __init__(self, model_name: str, device: str = "cpu") -> None:
            self.model_name = model_name
            self.device = device

        def encode(
            self,
            sentences: str | list[str],
            convert_to_numpy: bool = True,
            batch_size: int = 32,
            show_progress_bar: bool = False,
        ) -> np.ndarray | list[np.ndarray]:
            """Mock encode method returning random embeddings."""
            if isinstance(sentences, str):
                # Single sentence - return 1D array
                return np.random.rand(384).astype(np.float32)
            else:
                # Batch of sentences - return list of arrays
                return [np.random.rand(384).astype(np.float32) for _ in sentences]

        def get_sentence_embedding_dimension(self) -> int:
            """Return mock embedding dimension."""
            return 384

    def mock_init(*args: Any, **kwargs: Any) -> MockModel:
        """Mock SentenceTransformer constructor."""
        model_name = args[0] if args else kwargs.get("model_name", "all-MiniLM-L6-v2")
        device = kwargs.get("device", "cpu")
        return MockModel(model_name, device)

    # Patch the SentenceTransformer class
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        mock_init,
    )

    return MockModel("all-MiniLM-L6-v2")
