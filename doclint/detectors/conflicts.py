"""Conflict detector for finding contradictory content across documents."""

import logging
import re
from typing import List, Optional, Set, Tuple

from ..core.document import Chunk, Document
from ..embeddings.processor import get_all_chunks
from .base import BaseDetector, ContradictionVerifier, Issue, IssueSeverity
from .vector_index import ChunkIndex

logger = logging.getLogger(__name__)


# Negation pattern pairs: (pattern_a, pattern_b) where presence of both suggests contradiction
NEGATION_PATTERNS: List[Tuple[str, str]] = [
    (r"\bis\s+not\b", r"\bis\b"),
    (r"\bare\s+not\b", r"\bare\b"),
    (r"\bdoes\s+not\b", r"\bdoes\b"),
    (r"\bdo\s+not\b", r"\bdo\b"),
    (r"\bcannot\b", r"\bcan\b"),
    (r"\bcan't\b", r"\bcan\b"),
    (r"\bwon't\b", r"\bwill\b"),
    (r"\bwill\s+not\b", r"\bwill\b"),
    (r"\bshould\s+not\b", r"\bshould\b"),
    (r"\bshouldn't\b", r"\bshould\b"),
    (r"\bnever\b", r"\balways\b"),
    (r"\bdisabled\b", r"\benabled\b"),
    (r"\bunsupported\b", r"\bsupported\b"),
    (r"\bdeprecated\b", r"\brecommended\b"),
    (r"\brequired\b", r"\boptional\b"),
    (r"\bmust\s+not\b", r"\bmust\b"),
]

# Antonym pairs that suggest contradiction when both appear
ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("true", "false"),
    ("yes", "no"),
    ("synchronous", "asynchronous"),
    ("sync", "async"),
    ("single", "multi"),
    ("blocking", "non-blocking"),
    ("mutable", "immutable"),
    ("public", "private"),
    ("static", "dynamic"),
    ("required", "optional"),
    ("mandatory", "optional"),
    ("allow", "deny"),
    ("accept", "reject"),
    ("include", "exclude"),
    ("enable", "disable"),
]


class ConflictDetector(BaseDetector):
    """Detects conflicting information across document chunks.

    This detector finds semantically similar chunks across different documents
    and analyzes them for potential contradictions. It uses:
    - FAISS-based vector similarity search for efficient chunk comparison
    - Heuristic contradiction detection (negation patterns, antonyms)
    - Optional LLM-based verification for uncertain cases

    Attributes:
        name: Detector identifier ("conflict")
        description: Human-readable description
        similarity_threshold: Minimum embedding similarity to consider (default: 0.85)
        exclude_same_document: Skip conflicts within same document (default: True)
        verifier: Optional LLM-based contradiction verifier

    Example:
        >>> detector = ConflictDetector(similarity_threshold=0.85)
        >>> issues = await detector.detect(documents)
        >>> for issue in issues:
        ...     print(f"{issue.severity}: {issue.title}")
    """

    name = "conflict"
    description = "Detects contradictory information across documents"

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        exclude_same_document: bool = True,
        min_chunk_length: int = 50,
        verifier: Optional[ContradictionVerifier] = None,
    ) -> None:
        """Initialize the conflict detector.

        Args:
            similarity_threshold: Minimum cosine similarity to consider chunks
                                 as potentially conflicting (default: 0.85)
            exclude_same_document: If True, skip conflicts within the same
                                  document (default: True)
            min_chunk_length: Minimum chunk text length to analyze (default: 50)
            verifier: Optional ContradictionVerifier for LLM-based verification
        """
        self.similarity_threshold = similarity_threshold
        self.exclude_same_document = exclude_same_document
        self.min_chunk_length = min_chunk_length
        self.verifier = verifier

        logger.debug(
            f"Initialized ConflictDetector "
            f"(threshold={similarity_threshold}, exclude_same_doc={exclude_same_document})"
        )

    async def detect(self, documents: List[Document]) -> List[Issue]:
        """Detect conflicts across document chunks.

        Args:
            documents: List of documents with populated chunks and embeddings

        Returns:
            List of Issue objects for detected conflicts
        """
        if len(documents) < 2:
            logger.info("Less than 2 documents, skipping conflict detection")
            return []

        # Extract all chunks from documents
        all_chunks = get_all_chunks(documents)

        # Filter out tiny chunks
        valid_chunks = [c for c in all_chunks if len(c.text) >= self.min_chunk_length]

        if len(valid_chunks) < 2:
            logger.info("Less than 2 valid chunks, skipping conflict detection")
            return []

        logger.info(f"Analyzing {len(valid_chunks)} chunks for conflicts")

        # Build vector index
        index = ChunkIndex(dimension=384)  # MiniLM default
        index.build(valid_chunks)

        # Find similar pairs
        similar_pairs = index.find_all_similar_pairs(
            threshold=self.similarity_threshold,
            exclude_same_document=self.exclude_same_document,
        )

        logger.info(f"Found {len(similar_pairs)} similar chunk pairs")

        # Analyze each pair for conflicts
        issues: List[Issue] = []
        for i, j, similarity in similar_pairs:
            chunk_a = index.get_chunk(i)
            chunk_b = index.get_chunk(j)

            issue = await self._analyze_pair(chunk_a, chunk_b, similarity)
            if issue:
                issues.append(issue)

        # Sort by severity (CRITICAL first, then WARNING, then INFO)
        severity_order = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.WARNING: 1,
            IssueSeverity.INFO: 2,
        }
        issues.sort(key=lambda x: (severity_order[x.severity], -x.details.get("similarity", 0)))

        logger.info(f"Detected {len(issues)} conflict issues")
        return issues

    async def _analyze_pair(
        self, chunk_a: Chunk, chunk_b: Chunk, similarity: float
    ) -> Optional[Issue]:
        """Analyze a pair of similar chunks for conflicts.

        Args:
            chunk_a: First chunk
            chunk_b: Second chunk
            similarity: Embedding similarity score

        Returns:
            Issue if conflict detected, None otherwise
        """
        text_a = chunk_a.text.lower()
        text_b = chunk_b.text.lower()

        # Calculate text overlap (Jaccard similarity)
        text_overlap = self._jaccard_similarity(text_a, text_b)

        # Check for contradiction signals FIRST (before text overlap filter)
        has_negation = self._has_negation_pattern(text_a, text_b)
        has_antonyms = self._has_antonym_pair(text_a, text_b)
        numerical_mismatch = self._has_numerical_mismatch(chunk_a.text, chunk_b.text)

        # Determine severity based on signals
        if has_negation or numerical_mismatch:
            severity = IssueSeverity.CRITICAL
            reason = "negation_detected" if has_negation else "numerical_mismatch"
        elif has_antonyms:
            severity = IssueSeverity.WARNING
            reason = "antonym_detected"
        elif text_overlap > 0.7:
            # High text overlap without contradiction = likely duplicate, skip
            logger.debug(
                f"Skipping pair with high text overlap ({text_overlap:.2f}) "
                "and no contradiction signals"
            )
            return None
        elif self.verifier:
            # Use LLM verifier for uncertain cases
            is_conflict, confidence, explanation = await self.verifier.verify(
                chunk_a.text, chunk_b.text
            )
            if is_conflict and confidence > 0.7:
                severity = IssueSeverity.WARNING
                reason = "llm_verified"
            elif similarity > 0.90:
                severity = IssueSeverity.INFO
                reason = "high_similarity"
            else:
                return None
        elif similarity > 0.90:
            # High similarity but no clear contradiction signal
            severity = IssueSeverity.INFO
            reason = "high_similarity"
        else:
            return None

        return self._create_issue(chunk_a, chunk_b, similarity, severity, reason, text_overlap)

    def _create_issue(
        self,
        chunk_a: Chunk,
        chunk_b: Chunk,
        similarity: float,
        severity: IssueSeverity,
        reason: str,
        text_overlap: float,
    ) -> Issue:
        """Create an Issue object for a detected conflict.

        Args:
            chunk_a: First conflicting chunk
            chunk_b: Second conflicting chunk
            similarity: Embedding similarity score
            severity: Issue severity level
            reason: Detection reason code
            text_overlap: Text overlap score

        Returns:
            Issue object with conflict details
        """
        # Generate title based on severity
        if severity == IssueSeverity.CRITICAL:
            title = "Contradiction detected"
        elif severity == IssueSeverity.WARNING:
            title = "Potential conflict detected"
        else:
            title = "Similar content found"

        # Generate description
        doc_a = chunk_a.document_path.name
        doc_b = chunk_b.document_path.name
        description = (
            f"Found similar content in '{doc_a}' and '{doc_b}' "
            f"with {similarity:.0%} semantic similarity. "
            f"Review for potential contradictions."
        )

        return Issue(
            severity=severity,
            detector=self.name,
            title=title,
            description=description,
            documents=[chunk_a.document_path, chunk_b.document_path],
            chunks=[chunk_a, chunk_b],
            details={
                "similarity": round(similarity, 4),
                "text_overlap": round(text_overlap, 4),
                "reason": reason,
                "chunk_a": {
                    "text": chunk_a.text,
                    "document": str(chunk_a.document_path),
                    "index": chunk_a.index,
                    "position": {"start": chunk_a.start_pos, "end": chunk_a.end_pos},
                },
                "chunk_b": {
                    "text": chunk_b.text,
                    "document": str(chunk_b.document_path),
                    "index": chunk_b.index,
                    "position": {"start": chunk_b.start_pos, "end": chunk_b.end_pos},
                },
            },
        )

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """Calculate Jaccard similarity between two texts.

        Jaccard similarity = |intersection| / |union| of word sets.

        Args:
            text_a: First text (should be lowercase)
            text_b: Second text (should be lowercase)

        Returns:
            Similarity score between 0 and 1
        """
        # Tokenize into words
        words_a = set(re.findall(r"\b\w+\b", text_a))
        words_b = set(re.findall(r"\b\w+\b", text_b))

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union)

    @staticmethod
    def _has_negation_pattern(text_a: str, text_b: str) -> bool:
        """Check if texts contain opposing negation patterns.

        Looks for patterns where one text has a negation and the other
        has the positive form (e.g., "is not" vs "is").

        Args:
            text_a: First text (lowercase)
            text_b: Second text (lowercase)

        Returns:
            True if negation pattern detected
        """
        for neg_pattern, pos_pattern in NEGATION_PATTERNS:
            # Check if one has negation and other has positive
            a_has_neg = bool(re.search(neg_pattern, text_a, re.IGNORECASE))
            b_has_neg = bool(re.search(neg_pattern, text_b, re.IGNORECASE))
            a_has_pos = bool(re.search(pos_pattern, text_a, re.IGNORECASE))
            b_has_pos = bool(re.search(pos_pattern, text_b, re.IGNORECASE))

            # Contradiction: one has negation, other has positive (not negation)
            if (a_has_neg and b_has_pos and not b_has_neg) or (
                b_has_neg and a_has_pos and not a_has_neg
            ):
                return True

        return False

    @staticmethod
    def _has_antonym_pair(text_a: str, text_b: str) -> bool:
        """Check if texts contain antonym pairs.

        Args:
            text_a: First text (lowercase)
            text_b: Second text (lowercase)

        Returns:
            True if antonym pair detected across texts
        """
        words_a: Set[str] = set(re.findall(r"\b\w+\b", text_a))
        words_b: Set[str] = set(re.findall(r"\b\w+\b", text_b))

        for word1, word2 in ANTONYM_PAIRS:
            # One text has word1, other has word2
            if (word1 in words_a and word2 in words_b) or (word2 in words_a and word1 in words_b):
                return True

        return False

    @staticmethod
    def _has_numerical_mismatch(text_a: str, text_b: str) -> bool:
        """Check if texts contain conflicting numerical values.

        Looks for numbers that appear in similar contexts but with
        different values. Uses extended context (2 words before, 2 after)
        to reduce false positives.

        Args:
            text_a: First text (original case)
            text_b: Second text (original case)

        Returns:
            True if numerical mismatch detected
        """
        # Extract numbers with extended context (2 words before and after)
        # Pattern: word1 word2 NUMBER word3 word4
        number_pattern = (
            r"(?:(\b\w+\b)\s+)?(\b\w+\b)\s+(\d+(?:\.\d+)?)\s+(\b\w+\b)(?:\s+(\b\w+\b))?"
        )

        matches_a = re.findall(number_pattern, text_a.lower())
        matches_b = re.findall(number_pattern, text_b.lower())

        if not matches_a or not matches_b:
            return False

        # Look for same context but different numbers
        for w1a, w2a, num_a, w3a, w4a in matches_a:
            for w1b, w2b, num_b, w3b, w4b in matches_b:
                if num_a == num_b:
                    continue

                # Count matching context words (need at least 2 matches)
                context_matches = sum(
                    [
                        w1a == w1b and w1a != "",
                        w2a == w2b,
                        w3a == w3b,
                        w4a == w4b and w4a != "",
                    ]
                )

                # Need strong context match (at least 2 words)
                if context_matches >= 2:
                    try:
                        val_a = float(num_a)
                        val_b = float(num_b)
                        # Significant difference (not just rounding)
                        if abs(val_a - val_b) / max(val_a, val_b, 1) > 0.1:
                            return True
                    except ValueError:
                        continue

        return False
