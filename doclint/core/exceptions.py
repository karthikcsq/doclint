"""Custom exceptions for DocLint."""

from typing import Optional


class DocLintError(Exception):
    """Base exception for all DocLint errors."""

    pass


class ParsingError(DocLintError):
    """Raised when document parsing fails.

    Attributes:
        path: Path to the file that failed to parse
        original_error: The underlying exception that caused the parsing failure
    """

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """Initialize parsing error.

        Args:
            message: Error message describing the parsing failure
            path: Optional path to the file that failed to parse
            original_error: Optional underlying exception that caused the failure
        """
        self.path = path
        self.original_error = original_error
        super().__init__(message)


class EmbeddingError(DocLintError):
    """Raised when embedding generation fails.

    Attributes:
        model_name: Name of the embedding model being used
        text_length: Length of the text that failed to embed
        original_error: The underlying exception that caused the embedding failure
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        text_length: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """Initialize embedding error.

        Args:
            message: Error message describing the embedding failure
            model_name: Optional name of the embedding model being used
            text_length: Optional length of the text that failed to embed
            original_error: Optional underlying exception that caused the failure
        """
        self.model_name = model_name
        self.text_length = text_length
        self.original_error = original_error
        super().__init__(message)
