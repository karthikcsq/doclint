"""Issue detectors for data quality problems.

This module provides detectors that analyze document corpora for:
- Conflicts: Semantically similar but contradictory content
- Completeness: Missing metadata or incomplete content (future)
- Drift: Changes from previous versions (future)

The detection pipeline uses FAISS for efficient similarity search,
enabling O(n log n) conflict detection instead of O(n²).
"""

from .base import BaseDetector, ContradictionVerifier, Issue, IssueSeverity
from .conflicts import ConflictDetector
from .vector_index import ChunkIndex

__all__ = [
    # Base classes
    "BaseDetector",
    "ContradictionVerifier",
    "Issue",
    "IssueSeverity",
    # Detectors
    "ConflictDetector",
    # Utilities
    "ChunkIndex",
]
