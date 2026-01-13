"""LLM-based contradiction verification using llama.cpp.

This module provides a ContradictionVerifier implementation that uses locally-run
LLMs via llama-cpp-python for verifying conflicts between text chunks.

Open Source Friendly:
- No API keys required - runs entirely locally
- Auto-downloads models from Hugging Face (or use local GGUF files)
- Supports multiple quantization levels (Q4_K_M recommended for balance)
- Works with Phi-4 Mini, Llama 3, Mistral, and other GGUF models

Example (auto-download from Hugging Face):
    >>> from doclint.detectors.llm_verifier import LlamaCppVerifier
    >>> verifier = LlamaCppVerifier.from_pretrained(
    ...     repo_id="unsloth/Phi-4-mini-reasoning-GGUF",
    ...     filename="Phi-4-mini-reasoning-Q4_K_M.gguf",
    ... )
    >>> is_conflict, confidence, explanation = await verifier.verify(text_a, text_b)

Example (local file):
    >>> verifier = LlamaCppVerifier(model_path="path/to/model.gguf")
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .base import ContradictionVerifier

logger = logging.getLogger(__name__)

# ============================================================================
# MODEL REGISTRY - Popular GGUF models for conflict detection
# ============================================================================
# Users can use these with: LlamaCppVerifier.from_pretrained("phi4-mini")
# Or specify full repo: LlamaCppVerifier.from_pretrained(repo_id="...", filename="...")

KNOWN_MODELS: Dict[str, Dict[str, str]] = {
    # Phi-4 Models (Microsoft) - Excellent for reasoning tasks
    "phi4-reasoning": {
        "repo_id": "unsloth/Phi-4-mini-reasoning-GGUF",
        "filename": "Phi-4-mini-reasoning-Q4_K_M.gguf",
        "description": "Phi-4 Mini Reasoning - Best for conflict detection (default)",
    },
    "phi4-mini": {
        "repo_id": "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "filename": "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        "description": "Phi-4 Mini Instruct - Good general purpose",
    },
    "phi4-mini-q8": {
        "repo_id": "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "filename": "microsoft_Phi-4-mini-instruct-Q8_0.gguf",
        "description": "Phi-4 Mini Instruct Q8 - Higher quality, larger",
    },
    # Llama Models (Meta)
    "llama3-3b": {
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "description": "Llama 3.2 3B - Fast alternative",
    },
    # Qwen Models (Alibaba)
    "qwen2-3b": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "description": "Qwen 2.5 3B - Good multilingual support",
    },
    # Granite Models (IBM)
    "granite-1b": {
        "repo_id": "unsloth/granite-4.0-h-1b-GGUF",
        "filename": "granite-4.0-h-1b-Q5_K_M.gguf",
        "description": "IBM Granite 4.0 H 1B - Excellent reasoning, compact size",
    },
}

# Default model alias
DEFAULT_MODEL = "phi4-reasoning"


def list_available_models() -> Dict[str, str]:
    """List all known models with their descriptions.

    Returns:
        Dictionary mapping model aliases to descriptions.

    Example:
        >>> for name, desc in list_available_models().items():
        ...     print(f"{name}: {desc}")
        phi4-reasoning: Phi-4 Mini Reasoning - Best for conflict detection (default)
        phi4-mini: Phi-4 Mini Instruct - Good general purpose
        ...
    """
    return {name: info["description"] for name, info in KNOWN_MODELS.items()}


# Prompt template for contradiction detection
CONTRADICTION_PROMPT = """You are an expert at detecting contradictions in technical docs.

Analyze these two text chunks and determine if they contain contradictory information.

## Chunk A:
{chunk_a}

## Chunk B:
{chunk_b}

## Instructions:
1. Focus on factual claims, not stylistic differences
2. Consider if both statements could be simultaneously true
3. Numerical differences (e.g., "5 seconds" vs "10 seconds") are contradictions
4. Negations (e.g., "is supported" vs "is not supported") are contradictions
5. Different default values for the same setting are contradictions

## Response Format (JSON only):
{{
  "is_contradiction": true/false,
  "confidence": 0.0-1.0,
  "explanation": "Brief explanation of your finding"
}}

Respond with ONLY the JSON object, no other text."""


class LlamaCppVerifier(ContradictionVerifier):
    """Contradiction verifier using llama.cpp for local LLM inference.

    This verifier runs LLM inference locally using llama-cpp-python bindings,
    making it suitable for open-source deployment without API dependencies.

    Two ways to initialize:
        1. `from_pretrained()` - Auto-downloads from Hugging Face (recommended)
        2. `__init__()` - Use a local GGUF file

    Recommended Models (GGUF format):
        - Phi-4 Mini Reasoning (Q4_K_M): ~2.5GB, excellent for conflict detection
        - Phi-4 Mini (Q8_0): ~4GB, higher quality
        - Llama 3.2 3B: Alternative option
        - Mistral 7B (Q4_K_M): Larger but more capable

    Attributes:
        n_ctx: Context window size (default: 4096)
        n_gpu_layers: Number of layers to offload to GPU (-1 for all)
        temperature: Sampling temperature (lower = more deterministic)
        max_tokens: Maximum tokens in response

    Example (auto-download):
        >>> verifier = LlamaCppVerifier.from_pretrained(
        ...     repo_id="unsloth/Phi-4-mini-reasoning-GGUF",
        ...     filename="Phi-4-mini-reasoning-Q4_K_M.gguf",
        ... )
        >>> detector = ConflictDetector(verifier=verifier)

    Example (local file):
        >>> verifier = LlamaCppVerifier(model_path="models/phi-4-mini.gguf")
    """

    # Default model for from_pretrained when no args given
    DEFAULT_REPO_ID = "unsloth/Phi-4-mini-reasoning-GGUF"
    DEFAULT_FILENAME = "Phi-4-mini-reasoning-Q4_K_M.gguf"

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        temperature: float = 0.1,
        max_tokens: int = 256,
        verbose: bool = False,
        *,
        _from_pretrained_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the LlamaCpp verifier with a local model file.

        For auto-downloading from Hugging Face, use `from_pretrained()` instead.

        Args:
            model_path: Path to GGUF model file. Can be absolute or relative.
                       Required unless using _from_pretrained_config.
            n_ctx: Context window size. Should be at least 2048 for good results.
            n_gpu_layers: Layers to offload to GPU. Use -1 for all layers,
                         0 for CPU-only. Requires CUDA/Metal llama-cpp-python.
            temperature: Sampling temperature. Lower values (0.1-0.3) give more
                        consistent results for classification tasks.
            max_tokens: Maximum tokens in model response.
            verbose: If True, print llama.cpp debug output.
            _from_pretrained_config: Internal use - config from from_pretrained().

        Raises:
            ValueError: If neither model_path nor _from_pretrained_config provided.
        """
        self.model_path = Path(model_path) if model_path else None
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self._from_pretrained_config = _from_pretrained_config

        # Validate that we have a way to load the model
        if not model_path and not _from_pretrained_config:
            raise ValueError(
                "Either model_path or use from_pretrained() class method. "
                "Example: LlamaCppVerifier.from_pretrained()"
            )

        self._llm: Optional[Any] = None
        self._initialized = False

    @classmethod
    def from_pretrained(
        cls,
        model: Optional[str] = None,
        *,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        temperature: float = 0.1,
        max_tokens: int = 256,
        verbose: bool = False,
    ) -> "LlamaCppVerifier":
        """Create a verifier that auto-downloads model from Hugging Face.

        This is the recommended way to create a LlamaCppVerifier as it
        handles model downloading automatically.

        Args:
            model: Model alias (e.g., "phi4-mini", "phi4-reasoning", "llama3-3b")
                  Use list_available_models() to see all options.
                  Ignored if repo_id is specified.
            repo_id: Full Hugging Face repo ID (overrides model alias)
            filename: GGUF filename in the repo (required if using repo_id)
            n_ctx: Context window size.
            n_gpu_layers: GPU layers (-1 for all, 0 for CPU).
            temperature: Sampling temperature.
            max_tokens: Max response tokens.
            verbose: Enable llama.cpp debug output.

        Returns:
            LlamaCppVerifier configured to download from Hugging Face.

        Example:
            >>> # Use default model (phi4-reasoning)
            >>> verifier = LlamaCppVerifier.from_pretrained()

            >>> # Use a model alias
            >>> verifier = LlamaCppVerifier.from_pretrained("phi4-mini")
            >>> verifier = LlamaCppVerifier.from_pretrained("llama3-3b")

            >>> # Or specify full repo details
            >>> verifier = LlamaCppVerifier.from_pretrained(
            ...     repo_id="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
            ...     filename="microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
            ... )
        """
        # If repo_id is specified, use it directly
        if repo_id:
            if not filename:
                raise ValueError("filename is required when specifying repo_id")
            config = {"repo_id": repo_id, "filename": filename}
        else:
            # Use model alias (or default)
            model_name = model or DEFAULT_MODEL

            if model_name not in KNOWN_MODELS:
                available = ", ".join(KNOWN_MODELS.keys())
                raise ValueError(
                    f"Unknown model: '{model_name}'. "
                    f"Available models: {available}\n"
                    f"Or specify repo_id and filename directly."
                )

            model_info = KNOWN_MODELS[model_name]
            config = {
                "repo_id": model_info["repo_id"],
                "filename": model_info["filename"],
            }
            logger.info(f"Using model: {model_name} - {model_info['description']}")

        return cls(
            model_path=None,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            _from_pretrained_config=config,
        )

    def _ensure_initialized(self) -> None:
        """Lazy initialization of the LLM model.

        This is called on first use to avoid loading the model until needed.
        Supports both local file loading and Hugging Face auto-download.

        Raises:
            ImportError: If llama-cpp-python is not installed.
            FileNotFoundError: If local model file doesn't exist.
        """
        if self._initialized:
            return

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for LlamaCppVerifier. "
                "Install it with: pip install llama-cpp-python\n"
                "For GPU support, see: https://github.com/abetlen/llama-cpp-python#installation"
            ) from e

        # Load from Hugging Face using from_pretrained
        if self._from_pretrained_config:
            repo_id = self._from_pretrained_config["repo_id"]
            filename = self._from_pretrained_config["filename"]

            logger.info(f"Loading LLM model from Hugging Face: {repo_id}/{filename}")
            logger.info("First run will download the model (~2-4GB). Please wait...")

            self._llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
            )
        # Load from local file
        else:
            if not self.model_path or not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    "Use from_pretrained() to auto-download, or provide a valid path:\n"
                    "  verifier = LlamaCppVerifier.from_pretrained()  # Auto-downloads Phi-4"
                )

            logger.info(f"Loading LLM model from: {self.model_path}")

            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
            )

        self._initialized = True
        logger.info("LLM model loaded successfully")

    async def verify(self, text_a: str, text_b: str) -> Tuple[bool, float, str]:
        """Verify if two text chunks contradict each other using local LLM.

        Args:
            text_a: First text chunk
            text_b: Second text chunk

        Returns:
            Tuple of (is_contradiction, confidence, explanation):
                - is_contradiction: True if texts contradict
                - confidence: Confidence score 0-1
                - explanation: Human-readable explanation
        """
        # Run in executor to avoid blocking async loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._verify_sync, text_a, text_b)

    def _verify_sync(self, text_a: str, text_b: str) -> Tuple[bool, float, str]:
        """Synchronous verification logic.

        Args:
            text_a: First text chunk
            text_b: Second text chunk

        Returns:
            Tuple of (is_contradiction, confidence, explanation)
        """
        self._ensure_initialized()

        # Truncate long texts to fit context
        max_chunk_len = (self.n_ctx - 500) // 2  # Leave room for prompt and response
        text_a_truncated = text_a[:max_chunk_len] if len(text_a) > max_chunk_len else text_a
        text_b_truncated = text_b[:max_chunk_len] if len(text_b) > max_chunk_len else text_b

        prompt = CONTRADICTION_PROMPT.format(
            chunk_a=text_a_truncated,
            chunk_b=text_b_truncated,
        )

        try:
            if self._llm is None:
                raise RuntimeError("LLM not initialized")

            response = self._llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["```", "\n\n\n"],  # Stop tokens
            )

            # Extract the generated text
            generated_text = response["choices"][0]["text"].strip()

            # Parse JSON response
            return self._parse_response(generated_text)

        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")
            return (False, 0.0, f"LLM verification error: {e}")

    def _parse_response(self, text: str) -> Tuple[bool, float, str]:
        """Parse the LLM's JSON response.

        Args:
            text: Raw text response from LLM

        Returns:
            Tuple of (is_contradiction, confidence, explanation)
        """
        # Try to extract JSON from the response
        # Sometimes models wrap JSON in markdown code blocks
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)

        if not json_match:
            logger.warning(f"Could not find JSON in LLM response: {text[:200]}")
            return (False, 0.0, "Failed to parse LLM response")

        try:
            data = json.loads(json_match.group())

            is_contradiction = bool(data.get("is_contradiction", False))
            confidence = float(data.get("confidence", 0.5))
            explanation = str(data.get("explanation", "No explanation provided"))

            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))

            return (is_contradiction, confidence, explanation)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"JSON parse error: {e}, text: {text[:200]}")
            return (False, 0.0, f"JSON parse error: {e}")


class MockLlamaCppVerifier(ContradictionVerifier):
    """Mock verifier for testing without a real LLM model.

    This verifier always returns configurable responses, useful for:
    - Unit testing the ConflictDetector
    - Development without downloading large model files
    - CI/CD pipelines

    Example:
        >>> verifier = MockLlamaCppVerifier(default_is_contradiction=True)
        >>> is_conflict, conf, expl = await verifier.verify("text1", "text2")
    """

    def __init__(
        self,
        default_is_contradiction: bool = False,
        default_confidence: float = 0.8,
        default_explanation: str = "Mock verification result",
    ) -> None:
        """Initialize mock verifier with default responses.

        Args:
            default_is_contradiction: Default contradiction result
            default_confidence: Default confidence score
            default_explanation: Default explanation text
        """
        self.default_is_contradiction = default_is_contradiction
        self.default_confidence = default_confidence
        self.default_explanation = default_explanation
        self.call_history: list[Tuple[str, str]] = []

    async def verify(self, text_a: str, text_b: str) -> Tuple[bool, float, str]:
        """Return mock verification result.

        Records the call in call_history for test assertions.

        Args:
            text_a: First text chunk (stored in history)
            text_b: Second text chunk (stored in history)

        Returns:
            Configured default response tuple
        """
        self.call_history.append((text_a, text_b))
        return (
            self.default_is_contradiction,
            self.default_confidence,
            self.default_explanation,
        )


def create_verifier_from_config(config: Dict[str, Any]) -> Optional[ContradictionVerifier]:
    """Factory function to create a verifier from configuration.

    This allows configuration-driven verifier creation, useful for CLI
    and configuration file support.

    Args:
        config: Configuration dictionary with keys:
            - type: "llama_cpp_hf", "llama_cpp", "mock", or None
            - For "llama_cpp_hf": model (alias) OR repo_id+filename
            - For "llama_cpp": model_path (required)
            - n_ctx, n_gpu_layers, temperature, etc. (optional)

    Returns:
        ContradictionVerifier instance or None if disabled

    Example (using model alias - simplest):
        >>> config = {
        ...     "type": "llama_cpp_hf",
        ...     "model": "phi4-mini",  # or "phi4-reasoning", "llama3-3b"
        ... }
        >>> verifier = create_verifier_from_config(config)

    Example (custom repo):
        >>> config = {
        ...     "type": "llama_cpp_hf",
        ...     "repo_id": "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        ...     "filename": "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        ... }
        >>> verifier = create_verifier_from_config(config)

    Example (local file):
        >>> config = {
        ...     "type": "llama_cpp",
        ...     "model_path": "models/phi-4-mini.gguf",
        ... }
        >>> verifier = create_verifier_from_config(config)
    """
    verifier_type = config.get("type")

    if not verifier_type or verifier_type == "none":
        return None

    if verifier_type == "mock":
        return MockLlamaCppVerifier(
            default_is_contradiction=config.get("default_is_contradiction", False),
            default_confidence=config.get("default_confidence", 0.8),
            default_explanation=config.get("default_explanation", "Mock result"),
        )

    # Hugging Face auto-download (recommended)
    if verifier_type == "llama_cpp_hf":
        # Check if using model alias or custom repo
        repo_id = config.get("repo_id")
        filename = config.get("filename")
        model_alias = config.get("model")

        if repo_id:
            # Custom repo specified
            return LlamaCppVerifier.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=config.get("n_ctx", 4096),
                n_gpu_layers=config.get("n_gpu_layers", 0),
                temperature=config.get("temperature", 0.1),
                max_tokens=config.get("max_tokens", 256),
                verbose=config.get("verbose", False),
            )
        else:
            # Use model alias (or default)
            return LlamaCppVerifier.from_pretrained(
                model=model_alias,  # None uses default
                n_ctx=config.get("n_ctx", 4096),
                n_gpu_layers=config.get("n_gpu_layers", 0),
                temperature=config.get("temperature", 0.1),
                max_tokens=config.get("max_tokens", 256),
                verbose=config.get("verbose", False),
            )

    # Local file path
    if verifier_type == "llama_cpp":
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("model_path is required for llama_cpp verifier")

        return LlamaCppVerifier(
            model_path=model_path,
            n_ctx=config.get("n_ctx", 4096),
            n_gpu_layers=config.get("n_gpu_layers", 0),
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 256),
            verbose=config.get("verbose", False),
        )

    raise ValueError(f"Unknown verifier type: {verifier_type}")
