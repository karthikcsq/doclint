"""Text preprocessing utilities for document processing."""

import re


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text by collapsing multiple spaces and newlines.

    This function replaces multiple consecutive whitespace characters (spaces,
    tabs, newlines) with a single space, and strips leading/trailing whitespace.

    Args:
        text: Input text to normalize

    Returns:
        Text with normalized whitespace

    Examples:
        >>> normalize_whitespace("Hello    world")
        'Hello world'

        >>> normalize_whitespace("Line1\\n\\n\\nLine2")
        'Line1 Line2'

        >>> normalize_whitespace("  spaced  \\t text  ")
        'spaced text'
    """
    # Replace multiple whitespace characters with a single space
    normalized = re.sub(r"\s+", " ", text)

    # Strip leading and trailing whitespace
    return normalized.strip()


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length.

    If the text exceeds max_length, it will be truncated and an ellipsis (...)
    will be appended. The ellipsis counts towards the max_length.

    Args:
        text: Input text to truncate
        max_length: Maximum length of the returned text

    Returns:
        Truncated text with ellipsis if needed

    Examples:
        >>> truncate_text("Hello world", 20)
        'Hello world'

        >>> truncate_text("Hello world", 8)
        'Hello...'

        >>> truncate_text("Short", 100)
        'Short'
    """
    if len(text) <= max_length:
        return text

    # Reserve 3 characters for the ellipsis
    if max_length < 3:
        return text[:max_length]

    return text[: max_length - 3] + "..."


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for processing.

    This function splits long text into smaller chunks with configurable overlap.
    Overlap helps maintain context across chunk boundaries. Empty chunks are
    filtered out.

    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters (default: 512)
        overlap: Number of overlapping characters between chunks (default: 50)

    Returns:
        List of text chunks with overlap

    Raises:
        ValueError: If chunk_size <= 0 or overlap < 0 or overlap >= chunk_size

    Examples:
        >>> chunks = chunk_text("A" * 1000, chunk_size=100, overlap=20)
        >>> len(chunks)
        11

        >>> chunk_text("Short text", chunk_size=100)
        ['Short text']

        >>> chunk_text("", chunk_size=100)
        []
    """
    # Validate parameters
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")

    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    # Handle empty text
    if not text or not text.strip():
        return []

    # If text is shorter than chunk_size, return as single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    stride = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)

        # Move to next chunk position
        start += stride

        # Break if we've covered the entire text
        if end >= len(text):
            break

    return chunks
