"""Tests for SentenceTransformerGenerator."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from doclint.core.exceptions import EmbeddingError
from doclint.embeddings.generator import SentenceTransformerGenerator


class TestSentenceTransformerGenerator:
    """Test suite for SentenceTransformerGenerator."""

    def test_initialization(self) -> None:
        """Test generator initialization with default parameters."""
        gen = SentenceTransformerGenerator()

        assert gen._model_name == "all-MiniLM-L6-v2"
        assert gen.device is None
        assert gen.normalize_embeddings is True
        assert gen._model is None  # Lazy loading - not loaded yet
        assert gen._dimension is None

    def test_initialization_custom_params(self) -> None:
        """Test generator initialization with custom parameters."""
        gen = SentenceTransformerGenerator(
            model_name="custom-model",
            device="cpu",
            normalize_embeddings=False,
        )

        assert gen._model_name == "custom-model"
        assert gen.device == "cpu"
        assert gen.normalize_embeddings is False

    def test_lazy_model_loading(self, mock_sentence_transformer: Any) -> None:
        """Test that model is only loaded on first access."""
        gen = SentenceTransformerGenerator()

        # Model not loaded initially
        assert gen._model is None

        # Access model property triggers loading
        model = gen.model

        assert model is not None
        assert gen._model is not None
        assert gen._dimension == 384

    def test_get_embedding_dimension(self, mock_sentence_transformer: Any) -> None:
        """Test getting embedding dimension."""
        gen = SentenceTransformerGenerator()

        dimension = gen.get_embedding_dimension()

        assert dimension == 384
        assert gen._dimension == 384  # Cached after first call

    def test_generate_single_embedding(self, mock_sentence_transformer: Any) -> None:
        """Test generating embedding for a single text."""
        gen = SentenceTransformerGenerator()

        text = "This is a test document"
        embedding = gen.generate(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_generate_returns_different_embeddings_for_different_text(
        self, mock_sentence_transformer: Any
    ) -> None:
        """Test that different texts produce different embeddings."""
        gen = SentenceTransformerGenerator()

        text1 = "First document"
        text2 = "Second document"

        emb1 = gen.generate(text1)
        emb2 = gen.generate(text2)

        # Different texts should produce different embeddings (with high probability)
        assert not np.array_equal(emb1, emb2)

    def test_generate_empty_text(self, mock_sentence_transformer: Any) -> None:
        """Test handling of empty text returns zero vector."""
        gen = SentenceTransformerGenerator()

        embedding = gen.generate("")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert np.all(embedding == 0)  # Zero vector

    def test_generate_whitespace_only_text(self, mock_sentence_transformer: Any) -> None:
        """Test handling of whitespace-only text returns zero vector."""
        gen = SentenceTransformerGenerator()

        embedding = gen.generate("   \n\t  ")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert np.all(embedding == 0)  # Zero vector

    def test_generate_batch_embeddings(self, mock_sentence_transformer: Any) -> None:
        """Test generating embeddings for a batch of texts."""
        gen = SentenceTransformerGenerator()

        texts = ["First document", "Second document", "Third document"]
        embeddings = gen.generate_batch(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (384,)
            assert emb.dtype == np.float32

    def test_generate_batch_empty_list(self, mock_sentence_transformer: Any) -> None:
        """Test batch generation with empty list returns empty list."""
        gen = SentenceTransformerGenerator()

        embeddings = gen.generate_batch([])

        assert embeddings == []

    def test_generate_batch_with_empty_texts(self, mock_sentence_transformer: Any) -> None:
        """Test batch generation handles empty texts."""
        gen = SentenceTransformerGenerator()

        texts = ["Valid text", "", "Another valid", "   "]
        embeddings = gen.generate_batch(texts)

        assert len(embeddings) == 4
        # Empty texts should have zero vectors
        assert np.all(embeddings[1] == 0)
        assert np.all(embeddings[3] == 0)
        # Valid texts should have non-zero embeddings
        assert not np.all(embeddings[0] == 0)
        assert not np.all(embeddings[2] == 0)

    def test_generate_batch_custom_batch_size(self, mock_sentence_transformer: Any) -> None:
        """Test batch generation with custom batch size."""
        gen = SentenceTransformerGenerator()

        texts = [f"Document {i}" for i in range(10)]
        embeddings = gen.generate_batch(texts, batch_size=3)

        assert len(embeddings) == 10
        for emb in embeddings:
            assert emb.shape == (384,)

    def test_generate_with_chunking_short_text(self, mock_sentence_transformer: Any) -> None:
        """Test chunking with text shorter than max_length."""
        gen = SentenceTransformerGenerator()

        short_text = "Short text"
        embedding = gen.generate_with_chunking(short_text, max_length=512)

        # Should call generate() directly for short text
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_generate_with_chunking_long_text(self, mock_sentence_transformer: Any) -> None:
        """Test chunking with text longer than max_length."""
        gen = SentenceTransformerGenerator()

        # Create long text that exceeds max_length
        long_text = "word " * 200  # 1000 characters
        embedding = gen.generate_with_chunking(long_text, max_length=100, overlap=10)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        # Averaged embedding should still be normalized
        if gen.normalize_embeddings:
            # Check if normalized (L2 norm should be close to 1)
            norm = np.linalg.norm(embedding)
            assert 0.9 < norm < 1.1

    def test_generate_with_chunking_empty_text(self, mock_sentence_transformer: Any) -> None:
        """Test chunking with empty text returns zero vector."""
        gen = SentenceTransformerGenerator()

        embedding = gen.generate_with_chunking("")

        assert isinstance(embedding, np.ndarray)
        assert np.all(embedding == 0)

    def test_model_loading_failure(self) -> None:
        """Test error handling when model loading fails."""
        with patch("doclint.embeddings.generator.SentenceTransformer") as mock_st_class:
            mock_st_class.side_effect = Exception("Model not found")

            gen = SentenceTransformerGenerator()

            with pytest.raises(EmbeddingError) as exc_info:
                _ = gen.model

            assert "Failed to load model" in str(exc_info.value)
            assert exc_info.value.model_name == "all-MiniLM-L6-v2"
            assert exc_info.value.original_error is not None

    def test_generate_failure_raises_embedding_error(self, mock_sentence_transformer: Any) -> None:
        """Test that generation failures raise EmbeddingError."""
        gen = SentenceTransformerGenerator()

        # Mock the encode method to raise an exception
        gen.model.encode = MagicMock(side_effect=RuntimeError("GPU out of memory"))

        with pytest.raises(EmbeddingError) as exc_info:
            gen.generate("Test text")

        assert "Failed to generate embedding" in str(exc_info.value)
        assert exc_info.value.model_name == "all-MiniLM-L6-v2"
        assert exc_info.value.text_length == 9
        assert exc_info.value.original_error is not None

    def test_device_auto_detection_cuda_available(self) -> None:
        """Test device auto-detection when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("doclint.embeddings.generator.SentenceTransformer") as mock_st_class:
                mock_model = MagicMock()
                mock_model.get_sentence_embedding_dimension.return_value = 384
                mock_st_class.return_value = mock_model

                gen = SentenceTransformerGenerator()
                _ = gen.model

                # Should be called with cuda device
                mock_st_class.assert_called_once_with("all-MiniLM-L6-v2", device="cuda")

    def test_device_auto_detection_cuda_unavailable(self) -> None:
        """Test device auto-detection when CUDA is unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("doclint.embeddings.generator.SentenceTransformer") as mock_st_class:
                mock_model = MagicMock()
                mock_model.get_sentence_embedding_dimension.return_value = 384
                mock_st_class.return_value = mock_model

                gen = SentenceTransformerGenerator()
                _ = gen.model

                # Should be called with cpu device
                mock_st_class.assert_called_once_with("all-MiniLM-L6-v2", device="cpu")

    def test_device_explicit_setting(self) -> None:
        """Test that explicit device setting is respected."""
        with patch("doclint.embeddings.generator.SentenceTransformer") as mock_st_class:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st_class.return_value = mock_model

            gen = SentenceTransformerGenerator(device="cuda:1")
            _ = gen.model

            # Should use the explicitly specified device
            mock_st_class.assert_called_once_with("all-MiniLM-L6-v2", device="cuda:1")
