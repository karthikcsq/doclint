"""Caching layer for embeddings and scan results."""

from .backends import DiskCacheBackend
from .manager import CacheManager

__all__ = ["CacheManager", "DiskCacheBackend"]
