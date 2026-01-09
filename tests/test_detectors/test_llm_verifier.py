"""Tests for LLM-based contradiction verification."""

import pytest

from doclint.detectors.llm_verifier import (
    LlamaCppVerifier,
    MockLlamaCppVerifier,
    create_verifier_from_config,
)


class TestMockLlamaCppVerifier:
    """Tests for the mock verifier."""

    @pytest.mark.asyncio
    async def test_returns_configured_defaults(self) -> None:
        """Mock verifier returns configured default values."""
        verifier = MockLlamaCppVerifier(
            default_is_contradiction=True,
            default_confidence=0.95,
            default_explanation="Test explanation",
        )

        is_conflict, confidence, explanation = await verifier.verify("Text A", "Text B")

        assert is_conflict is True
        assert confidence == 0.95
        assert explanation == "Test explanation"

    @pytest.mark.asyncio
    async def test_records_call_history(self) -> None:
        """Mock verifier records calls for test assertions."""
        verifier = MockLlamaCppVerifier()

        await verifier.verify("First text", "Second text")
        await verifier.verify("Another", "Pair")

        assert len(verifier.call_history) == 2
        assert verifier.call_history[0] == ("First text", "Second text")
        assert verifier.call_history[1] == ("Another", "Pair")

    @pytest.mark.asyncio
    async def test_default_returns_no_contradiction(self) -> None:
        """Default mock verifier returns no contradiction."""
        verifier = MockLlamaCppVerifier()

        is_conflict, confidence, _ = await verifier.verify("A", "B")

        assert is_conflict is False
        assert confidence == 0.8


class TestLlamaCppVerifierInit:
    """Tests for LlamaCppVerifier initialization."""

    def test_initialization_is_lazy(self) -> None:
        """Model is not loaded until first use."""
        # Should not raise even with invalid path
        verifier = LlamaCppVerifier(model_path="nonexistent.gguf")
        assert verifier._initialized is False
        assert verifier._llm is None

    def test_stores_configuration(self) -> None:
        """Verifier stores configuration values."""
        verifier = LlamaCppVerifier(
            model_path="test.gguf",
            n_ctx=2048,
            n_gpu_layers=10,
            temperature=0.5,
            max_tokens=128,
        )

        assert str(verifier.model_path) == "test.gguf"
        assert verifier.n_ctx == 2048
        assert verifier.n_gpu_layers == 10
        assert verifier.temperature == 0.5
        assert verifier.max_tokens == 128

    def test_from_pretrained_uses_defaults(self) -> None:
        """from_pretrained uses default repo when not specified."""
        verifier = LlamaCppVerifier.from_pretrained()

        assert verifier._from_pretrained_config is not None
        assert verifier._from_pretrained_config["repo_id"] == "unsloth/Phi-4-mini-reasoning-GGUF"
        assert verifier._from_pretrained_config["filename"] == "Phi-4-mini-reasoning-Q4_K_M.gguf"

    def test_from_pretrained_with_custom_repo(self) -> None:
        """from_pretrained accepts custom repo and filename."""
        verifier = LlamaCppVerifier.from_pretrained(
            repo_id="custom/repo",
            filename="model.gguf",
            n_gpu_layers=-1,
        )

        assert verifier._from_pretrained_config is not None
        assert verifier._from_pretrained_config["repo_id"] == "custom/repo"
        assert verifier._from_pretrained_config["filename"] == "model.gguf"
        assert verifier.n_gpu_layers == -1

    def test_requires_model_path_or_from_pretrained(self) -> None:
        """Raises ValueError if neither model_path nor from_pretrained used."""
        with pytest.raises(ValueError, match="Either model_path"):
            LlamaCppVerifier()


class TestLlamaCppVerifierParsing:
    """Tests for response parsing logic."""

    def test_parse_valid_json(self) -> None:
        """Parses well-formed JSON response."""
        verifier = LlamaCppVerifier(model_path="test.gguf")

        response = '{"is_contradiction": true, "confidence": 0.9, "explanation": "Values differ"}'
        result = verifier._parse_response(response)

        assert result == (True, 0.9, "Values differ")

    def test_parse_json_with_surrounding_text(self) -> None:
        """Extracts JSON from surrounding text."""
        verifier = LlamaCppVerifier(model_path="test.gguf")

        response = (
            "Here is my analysis:\n"
            '{"is_contradiction": false, "confidence": 0.7, "explanation": "No conflict"}'
            "\nDone."
        )
        result = verifier._parse_response(response)

        assert result == (False, 0.7, "No conflict")

    def test_parse_clamps_confidence(self) -> None:
        """Clamps confidence to 0-1 range."""
        verifier = LlamaCppVerifier(model_path="test.gguf")

        # Test over 1
        response = '{"is_contradiction": true, "confidence": 1.5, "explanation": "Test"}'
        _, confidence, _ = verifier._parse_response(response)
        assert confidence == 1.0

        # Test under 0
        response = '{"is_contradiction": true, "confidence": -0.5, "explanation": "Test"}'
        _, confidence, _ = verifier._parse_response(response)
        assert confidence == 0.0

    def test_parse_missing_json(self) -> None:
        """Returns defaults when no JSON found."""
        verifier = LlamaCppVerifier(model_path="test.gguf")

        response = "I couldn't analyze this properly."
        result = verifier._parse_response(response)

        assert result[0] is False  # is_contradiction
        assert result[1] == 0.0  # confidence

    def test_parse_invalid_json(self) -> None:
        """Handles malformed JSON gracefully."""
        verifier = LlamaCppVerifier(model_path="test.gguf")

        response = '{"is_contradiction": true, "confidence": invalid}'
        result = verifier._parse_response(response)

        assert result[0] is False


class TestCreateVerifierFromConfig:
    """Tests for the factory function."""

    def test_returns_none_for_disabled(self) -> None:
        """Returns None when type is 'none' or missing."""
        assert create_verifier_from_config({}) is None
        assert create_verifier_from_config({"type": "none"}) is None
        assert create_verifier_from_config({"type": None}) is None

    def test_creates_mock_verifier(self) -> None:
        """Creates MockLlamaCppVerifier for type='mock'."""
        config = {
            "type": "mock",
            "default_is_contradiction": True,
            "default_confidence": 0.9,
        }

        verifier = create_verifier_from_config(config)

        assert isinstance(verifier, MockLlamaCppVerifier)
        assert verifier.default_is_contradiction is True
        assert verifier.default_confidence == 0.9

    def test_creates_llama_cpp_verifier(self) -> None:
        """Creates LlamaCppVerifier for type='llama_cpp'."""
        config = {
            "type": "llama_cpp",
            "model_path": "/path/to/model.gguf",
            "n_ctx": 2048,
            "n_gpu_layers": -1,
        }

        verifier = create_verifier_from_config(config)

        assert isinstance(verifier, LlamaCppVerifier)
        assert verifier.n_ctx == 2048
        assert verifier.n_gpu_layers == -1

    def test_creates_llama_cpp_hf_verifier(self) -> None:
        """Creates LlamaCppVerifier with from_pretrained for type='llama_cpp_hf'."""
        config = {
            "type": "llama_cpp_hf",
            "repo_id": "custom/repo",
            "filename": "model.gguf",
        }

        verifier = create_verifier_from_config(config)

        assert isinstance(verifier, LlamaCppVerifier)
        assert verifier._from_pretrained_config is not None
        assert verifier._from_pretrained_config["repo_id"] == "custom/repo"

    def test_creates_llama_cpp_hf_with_defaults(self) -> None:
        """llama_cpp_hf uses defaults when repo_id not specified."""
        config = {"type": "llama_cpp_hf"}

        verifier = create_verifier_from_config(config)

        assert isinstance(verifier, LlamaCppVerifier)
        assert verifier._from_pretrained_config is not None
        assert verifier._from_pretrained_config["repo_id"] == "unsloth/Phi-4-mini-reasoning-GGUF"

    def test_raises_for_missing_model_path(self) -> None:
        """Raises ValueError when llama_cpp missing model_path."""
        config = {"type": "llama_cpp"}

        with pytest.raises(ValueError, match="model_path is required"):
            create_verifier_from_config(config)

    def test_raises_for_unknown_type(self) -> None:
        """Raises ValueError for unknown verifier type."""
        config = {"type": "openai"}

        with pytest.raises(ValueError, match="Unknown verifier type"):
            create_verifier_from_config(config)
