"""Hashing utilities for content identification and caching."""

import hashlib


def hash_content(text: str) -> str:
    """Compute SHA-256 hash of text content.

    This function is used to generate cache keys and identify document content
    for deduplication and caching purposes. The hash is computed on the UTF-8
    encoded bytes of the text.

    Args:
        text: Text content to hash

    Returns:
        Hexadecimal SHA-256 hash string (64 characters)

    Examples:
        >>> hash_content("Hello, world!")
        '315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3'

        >>> hash_content("")
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    """
    # Encode text as UTF-8 bytes
    content_bytes = text.encode("utf-8")

    # Compute SHA-256 hash
    content_hash = hashlib.sha256(content_bytes).hexdigest()

    return content_hash
