"""Configuration management for DocLint using Pydantic."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from platformdirs import user_config_dir
from pydantic import BaseModel, Field, field_validator

# TOML support (tomllib in 3.11+, tomli for 3.10)
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from .exceptions import ConfigurationError


class DetectorConfig(BaseModel):
    """Base configuration for all detectors."""

    enabled: bool = True


class LLMVerifierConfig(BaseModel):
    """Configuration for LLM-based contradiction verification.

    Enables local LLM inference via llama-cpp-python for verifying
    potential conflicts detected by heuristics.

    Attributes:
        enabled: Whether to use LLM verification
        type: Verifier type ("llama_cpp_hf", "llama_cpp", or "mock")
        model: Model alias for quick setup (e.g., "phi4-mini", "llama3-3b")
        repo_id: Custom Hugging Face repo ID (overrides model alias)
        filename: GGUF filename in repo (required with repo_id)
        model_path: Path to local GGUF file (for llama_cpp type)
        n_ctx: Context window size
        n_gpu_layers: GPU layers (-1 for all, 0 for CPU-only)
        temperature: Sampling temperature
        max_tokens: Max response tokens

    Available model aliases:
        - phi4-reasoning: Phi-4 Mini Reasoning (default, best for conflicts)
        - phi4-mini: Phi-4 Mini Instruct
        - phi4-mini-q8: Phi-4 Mini Q8 (higher quality)
        - llama3-3b: Llama 3.2 3B
        - qwen2-3b: Qwen 2.5 3B
    """

    enabled: bool = Field(
        default=True,
        description="Whether to use LLM-based verification",
    )
    type: str = Field(
        default="llama_cpp_hf",
        description="Verifier type: llama_cpp_hf (auto-download), llama_cpp (local), or mock",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model alias: phi4-reasoning (default), phi4-mini, llama3-3b, qwen2-3b",
    )
    repo_id: Optional[str] = Field(
        default=None,
        description="Custom Hugging Face repo ID (overrides model alias)",
    )
    filename: Optional[str] = Field(
        default=None,
        description="GGUF filename in the repo (required with repo_id)",
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Path to local GGUF model file (for llama_cpp type)",
    )
    n_ctx: int = Field(
        default=4096,
        gt=0,
        description="Context window size",
    )
    n_gpu_layers: int = Field(
        default=0,
        ge=-1,
        description="GPU layers to offload (-1 for all, 0 for CPU)",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=256,
        gt=0,
        description="Maximum response tokens",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose llama.cpp output",
    )


class ConflictDetectorConfig(DetectorConfig):
    """Configuration for conflict detection.

    Conflict detection identifies contradictory information across documents
    by comparing semantic embeddings of document chunks.

    Attributes:
        enabled: Whether conflict detection is enabled
        similarity_threshold: Minimum similarity score (0-1) to consider chunks related
        llm_verifier: LLM verification configuration
    """

    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for conflict detection",
    )
    llm_verifier: LLMVerifierConfig = Field(
        default_factory=LLMVerifierConfig,
        description="LLM-based verification configuration",
    )

    @field_validator("similarity_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate similarity threshold is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        return v


class CompletenessDetectorConfig(DetectorConfig):
    """Configuration for completeness detection.

    Completeness detection validates that documents have required metadata
    and sufficient content.

    Attributes:
        enabled: Whether completeness detection is enabled
        required_metadata: List of required metadata fields
        min_content_length: Minimum content length in characters
    """

    required_metadata: List[str] = Field(
        default_factory=lambda: ["title"],
        description="List of required metadata fields",
    )
    min_content_length: int = Field(
        default=100,
        ge=0,
        description="Minimum content length in characters",
    )


class DriftDetectorConfig(DetectorConfig):
    """Configuration for drift detection (experimental).

    Drift detection identifies documents that have semantically drifted
    from their original topic or purpose.

    Attributes:
        enabled: Whether drift detection is enabled (default: False, experimental)
    """

    enabled: bool = False


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation.

    Attributes:
        model_name: Name of the sentence-transformers model
        device: Device to use (cpu, cuda, or auto)
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between chunks in characters
    """

    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformers model name",
    )
    device: Optional[str] = Field(
        default=None,
        description="Device to use (cpu, cuda, or None for auto-detection)",
    )
    chunk_size: int = Field(
        default=512,
        gt=0,
        description="Size of text chunks in characters",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Overlap between chunks in characters",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap_less_than_size(cls, v: int, info: Any) -> int:
        """Ensure overlap < chunk_size."""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class DocLintConfig(BaseModel):
    """Main configuration class for DocLint.

    This is the root configuration object that contains all settings for
    scanning, caching, and detection.

    Attributes:
        recursive: Whether to scan directories recursively
        cache_enabled: Whether to use embedding cache
        cache_dir: Custom cache directory (None for platform default)
        max_workers: Maximum number of worker threads for parsing
        conflict: Conflict detector configuration
        completeness: Completeness detector configuration
        drift: Drift detector configuration
        embedding: Embedding generation configuration
    """

    recursive: bool = Field(
        default=True,
        description="Whether to scan directories recursively",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Whether to use embedding cache",
    )
    cache_dir: Optional[Path] = Field(
        default=None,
        description="Custom cache directory (None for platform default)",
    )
    max_workers: int = Field(
        default=4,
        gt=0,
        le=32,
        description="Maximum number of worker threads",
    )

    # Detector configurations
    conflict: ConflictDetectorConfig = Field(default_factory=ConflictDetectorConfig)
    completeness: CompletenessDetectorConfig = Field(default_factory=CompletenessDetectorConfig)
    drift: DriftDetectorConfig = Field(default_factory=DriftDetectorConfig)

    # Embedding configuration
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> "DocLintConfig":
        """Load configuration from TOML file.

        Configuration is searched in the following order:
        1. Explicit path provided as argument
        2. .doclint.toml in current directory
        3. ~/.config/doclint/config.toml (platform-specific)
        4. Default configuration

        Args:
            config_path: Optional explicit path to configuration file

        Returns:
            DocLintConfig instance

        Raises:
            ConfigurationError: If configuration file exists but cannot be parsed
        """
        # Search order for config file
        search_paths: List[Path] = []

        if config_path:
            search_paths.append(Path(config_path))

        # Current directory
        search_paths.append(Path.cwd() / ".doclint.toml")

        # User config directory (platform-specific)
        user_config = Path(user_config_dir("doclint", "doclint")) / "config.toml"
        search_paths.append(user_config)

        # Find first existing config file
        for path in search_paths:
            if path.exists() and path.is_file():
                return cls._load_from_file(path)

        # No config file found, return defaults
        return cls()

    @classmethod
    def _load_from_file(cls, path: Path) -> "DocLintConfig":
        """Load configuration from a specific file.

        Args:
            path: Path to TOML configuration file

        Returns:
            DocLintConfig instance

        Raises:
            ConfigurationError: If file cannot be read or parsed
        """
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)

            return cls._from_dict(data, str(path))

        except tomllib.TOMLDecodeError as e:
            raise ConfigurationError(
                f"Invalid TOML in configuration file {path}: {e}",
                config_path=str(path),
                original_error=e,
            )
        except OSError as e:
            raise ConfigurationError(
                f"Cannot read configuration file {path}: {e}",
                config_path=str(path),
                original_error=e,
            )

    @classmethod
    def _from_dict(cls, data: Dict[str, Any], config_path: str) -> "DocLintConfig":
        """Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values
            config_path: Path to config file (for error messages)

        Returns:
            DocLintConfig instance

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Handle nested detector configs
            config_data: Dict[str, Any] = {}

            # Top-level settings
            for key in ["recursive", "cache_enabled", "cache_dir", "max_workers"]:
                if key in data:
                    config_data[key] = data[key]

            # Detector configs (nested under [detectors] or top-level)
            detectors = data.get("detectors", {})
            for detector in ["conflict", "completeness", "drift"]:
                if detector in detectors:
                    config_data[detector] = detectors[detector]
                elif detector in data:
                    config_data[detector] = data[detector]

            # Embedding config (nested under [embedding] or top-level)
            if "embedding" in data:
                config_data["embedding"] = data["embedding"]

            return cls(**config_data)

        except Exception as e:
            raise ConfigurationError(
                f"Invalid configuration in {config_path}: {e}",
                config_path=config_path,
                original_error=e,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return self.model_dump()
