"""Tests for DocLint configuration management."""

from pathlib import Path

import pytest

from doclint.core.config import (
    CompletenessDetectorConfig,
    ConflictDetectorConfig,
    DocLintConfig,
    DriftDetectorConfig,
    EmbeddingConfig,
)
from doclint.core.exceptions import ConfigurationError


class TestConflictDetectorConfig:
    """Tests for ConflictDetectorConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ConflictDetectorConfig()
        assert config.enabled is True
        assert config.similarity_threshold == 0.85

    def test_custom_threshold(self) -> None:
        """Test custom similarity threshold."""
        config = ConflictDetectorConfig(similarity_threshold=0.9)
        assert config.similarity_threshold == 0.9

    def test_threshold_validation_lower_bound(self) -> None:
        """Test that threshold below 0 is rejected."""
        with pytest.raises(ValueError):
            ConflictDetectorConfig(similarity_threshold=-0.1)

    def test_threshold_validation_upper_bound(self) -> None:
        """Test that threshold above 1 is rejected."""
        with pytest.raises(ValueError):
            ConflictDetectorConfig(similarity_threshold=1.5)

    def test_threshold_edge_cases(self) -> None:
        """Test threshold edge values (0 and 1)."""
        config_zero = ConflictDetectorConfig(similarity_threshold=0.0)
        assert config_zero.similarity_threshold == 0.0

        config_one = ConflictDetectorConfig(similarity_threshold=1.0)
        assert config_one.similarity_threshold == 1.0


class TestCompletenessDetectorConfig:
    """Tests for CompletenessDetectorConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CompletenessDetectorConfig()
        assert config.enabled is True
        assert config.required_metadata == ["title"]
        assert config.min_content_length == 100

    def test_custom_required_metadata(self) -> None:
        """Test custom required metadata fields."""
        config = CompletenessDetectorConfig(required_metadata=["title", "author", "date"])
        assert config.required_metadata == ["title", "author", "date"]

    def test_empty_required_metadata(self) -> None:
        """Test empty required metadata list is allowed."""
        config = CompletenessDetectorConfig(required_metadata=[])
        assert config.required_metadata == []

    def test_custom_min_content_length(self) -> None:
        """Test custom minimum content length."""
        config = CompletenessDetectorConfig(min_content_length=500)
        assert config.min_content_length == 500

    def test_min_content_length_can_be_zero(self) -> None:
        """Test that min_content_length can be zero."""
        config = CompletenessDetectorConfig(min_content_length=0)
        assert config.min_content_length == 0


class TestDriftDetectorConfig:
    """Tests for DriftDetectorConfig."""

    def test_default_disabled(self) -> None:
        """Test that drift detector is disabled by default."""
        config = DriftDetectorConfig()
        assert config.enabled is False

    def test_can_enable(self) -> None:
        """Test that drift detector can be enabled."""
        config = DriftDetectorConfig(enabled=True)
        assert config.enabled is True


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = EmbeddingConfig()
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.device is None
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50

    def test_custom_model(self) -> None:
        """Test custom model name."""
        config = EmbeddingConfig(model_name="all-mpnet-base-v2")
        assert config.model_name == "all-mpnet-base-v2"

    def test_custom_device(self) -> None:
        """Test custom device selection."""
        config = EmbeddingConfig(device="cuda")
        assert config.device == "cuda"

    def test_custom_chunk_settings(self) -> None:
        """Test custom chunk size and overlap."""
        config = EmbeddingConfig(chunk_size=1024, chunk_overlap=100)
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 100

    def test_overlap_must_be_less_than_size(self) -> None:
        """Test that overlap must be less than chunk_size."""
        with pytest.raises(ValueError):
            EmbeddingConfig(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError):
            EmbeddingConfig(chunk_size=100, chunk_overlap=150)


class TestDocLintConfig:
    """Tests for DocLintConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DocLintConfig()
        assert config.recursive is True
        assert config.cache_enabled is True
        assert config.cache_dir is None
        assert config.max_workers == 4

    def test_nested_detector_configs(self) -> None:
        """Test that nested detector configs are created."""
        config = DocLintConfig()
        assert isinstance(config.conflict, ConflictDetectorConfig)
        assert isinstance(config.completeness, CompletenessDetectorConfig)
        assert isinstance(config.drift, DriftDetectorConfig)
        assert isinstance(config.embedding, EmbeddingConfig)

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = DocLintConfig(
            recursive=False,
            cache_enabled=False,
            max_workers=8,
        )
        assert config.recursive is False
        assert config.cache_enabled is False
        assert config.max_workers == 8

    def test_max_workers_validation(self) -> None:
        """Test max_workers validation."""
        with pytest.raises(ValueError):
            DocLintConfig(max_workers=0)

        with pytest.raises(ValueError):
            DocLintConfig(max_workers=-1)

        with pytest.raises(ValueError):
            DocLintConfig(max_workers=100)  # > 32

    def test_to_dict(self) -> None:
        """Test converting config to dictionary."""
        config = DocLintConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "recursive" in config_dict
        assert "conflict" in config_dict
        assert "embedding" in config_dict


class TestDocLintConfigLoad:
    """Tests for loading configuration from files."""

    def test_load_returns_defaults_when_no_file(self, tmp_path: Path) -> None:
        """Test that load returns defaults when no config file exists."""
        # Change to temp directory to avoid loading actual config
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(tmp_path)
            config = DocLintConfig.load()
            assert config.recursive is True
            assert config.max_workers == 4
        finally:
            os.chdir(original_cwd)

    def test_load_from_explicit_path(self, tmp_path: Path) -> None:
        """Test loading from an explicit config path."""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(
            """
recursive = false
max_workers = 2

[conflict]
similarity_threshold = 0.9

[embedding]
model_name = "custom-model"
chunk_size = 256
"""
        )

        config = DocLintConfig.load(config_file)
        assert config.recursive is False
        assert config.max_workers == 2
        assert config.conflict.similarity_threshold == 0.9
        assert config.embedding.model_name == "custom-model"
        assert config.embedding.chunk_size == 256

    def test_load_from_current_directory(self, tmp_path: Path) -> None:
        """Test loading from .doclint.toml in current directory."""
        config_file = tmp_path / ".doclint.toml"
        config_file.write_text(
            """
cache_enabled = false
"""
        )

        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(tmp_path)
            config = DocLintConfig.load()
            assert config.cache_enabled is False
        finally:
            os.chdir(original_cwd)

    def test_load_invalid_toml_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid TOML raises ConfigurationError."""
        config_file = tmp_path / "invalid.toml"
        config_file.write_text(
            """
this is not valid toml [[[
"""
        )

        with pytest.raises(ConfigurationError) as exc_info:
            DocLintConfig.load(config_file)

        assert "Invalid TOML" in str(exc_info.value)
        assert exc_info.value.config_path == str(config_file)

    def test_load_with_detectors_section(self, tmp_path: Path) -> None:
        """Test loading config with [detectors] section."""
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
[detectors.conflict]
similarity_threshold = 0.75

[detectors.completeness]
min_content_length = 200
"""
        )

        config = DocLintConfig.load(config_file)
        assert config.conflict.similarity_threshold == 0.75
        assert config.completeness.min_content_length == 200

    def test_load_nonexistent_explicit_path(self, tmp_path: Path) -> None:
        """Test that loading from nonexistent path returns defaults."""
        nonexistent = tmp_path / "does_not_exist.toml"
        # When explicit path doesn't exist, we fall back to search order
        config = DocLintConfig.load(nonexistent)
        # Should return defaults since file doesn't exist
        assert config.recursive is True
