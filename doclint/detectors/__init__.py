"""Issue detectors for data quality problems.

This module provides detectors that analyze document corpora for:
- Conflicts: Semantically similar but contradictory content
- Completeness: Missing metadata or incomplete content
- Drift: Changes from previous versions (future)

The detection pipeline uses FAISS for efficient similarity search,
enabling O(n log n) conflict detection instead of O(n²).
"""

from .base import BaseDetector, Issue, IssueSeverity
from .vector_index import ChunkIndex

__all__ = [
    "BaseDetector",
    "Issue",
    "IssueSeverity",
    "ChunkIndex",
]
