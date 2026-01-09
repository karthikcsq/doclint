"""Issue detectors for data quality problems.

This module provides detectors that analyze document corpora for:
- Conflicts: Semantically similar but contradictory content
- Completeness: Missing metadata or incomplete content (future)
- Drift: Changes from previous versions (future)

The detection pipeline uses FAISS for efficient similarity search,
enabling O(n log n) conflict detection instead of O(n²).

LLM-based verification is available via LlamaCppVerifier (requires llama-cpp-python).
Install with: pip install doclint[llm]
"""

from .base import BaseDetector, ContradictionVerifier, Issue, IssueSeverity
from .conflicts import ConflictDetector
from .llm_verifier import (
    KNOWN_MODELS,
    LlamaCppVerifier,
    MockLlamaCppVerifier,
    create_verifier_from_config,
    list_available_models,
)
from .vector_index import ChunkIndex

__all__ = [
    # Base classes
    "BaseDetector",
    "ContradictionVerifier",
    "Issue",
    "IssueSeverity",
    # Detectors
    "ConflictDetector",
    # LLM Verifiers
    "LlamaCppVerifier",
    "MockLlamaCppVerifier",
    "create_verifier_from_config",
    "list_available_models",
    "KNOWN_MODELS",
    # Utilities
    "ChunkIndex",
]
