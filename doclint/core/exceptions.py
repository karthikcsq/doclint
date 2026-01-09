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


class ConfigurationError(DocLintError):
    """Raised when configuration is invalid or cannot be loaded.

    Attributes:
        config_path: Path to the configuration file that caused the error
        field_name: Name of the configuration field that is invalid
        original_error: The underlying exception that caused the configuration failure
    """

    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        field_name: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """Initialize configuration error.

        Args:
            message: Error message describing the configuration failure
            config_path: Optional path to the configuration file
            field_name: Optional name of the invalid configuration field
            original_error: Optional underlying exception that caused the failure
        """
        self.config_path = config_path
        self.field_name = field_name
        self.original_error = original_error
        super().__init__(message)


class CacheError(DocLintError):
    """Raised when cache operations fail.

    Attributes:
        cache_dir: Path to the cache directory
        key: Cache key that caused the error
        original_error: The underlying exception that caused the cache failure
    """

    def __init__(
        self,
        message: str,
        cache_dir: Optional[str] = None,
        key: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """Initialize cache error.

        Args:
            message: Error message describing the cache failure
            cache_dir: Optional path to the cache directory
            key: Optional cache key that caused the error
            original_error: Optional underlying exception that caused the failure
        """
        self.cache_dir = cache_dir
        self.key = key
        self.original_error = original_error
        super().__init__(message)
