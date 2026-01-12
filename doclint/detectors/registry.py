"""Detector registry for managing and executing detectors."""

import logging
from typing import Dict, List, Optional

from ..core.document import Document
from .base import BaseDetector, Issue

logger = logging.getLogger(__name__)


class DetectorRegistry:
    """Registry for managing issue detectors.

    The registry allows registering detectors, retrieving them by name,
    and running multiple detectors on a corpus of documents.

    Example:
        >>> registry = DetectorRegistry()
        >>> registry.register(ConflictDetector())
        >>> registry.register(CompletenessDetector())
        >>> results = await registry.run_all(documents)
        >>> results
        {
            'conflict': [Issue(...)],
            'completeness': [Issue(...)]
        }
    """

    def __init__(self) -> None:
        """Initialize an empty detector registry."""
        self._detectors: Dict[str, BaseDetector] = {}

    def register(self, detector: BaseDetector) -> None:
        """Register a detector.

        Args:
            detector: Detector instance to register

        Raises:
            ValueError: If detector name is empty or already registered
        """
        if not detector.name:
            raise ValueError("Detector must have a non-empty name")

        if detector.name in self._detectors:
            logger.warning(f"Detector '{detector.name}' already registered, overwriting")

        self._detectors[detector.name] = detector
        logger.debug(f"Registered detector: {detector.name}")

    def get_detector(self, name: str) -> Optional[BaseDetector]:
        """Get a detector by name.

        Args:
            name: Name of the detector to retrieve

        Returns:
            Detector instance if found, None otherwise
        """
        return self._detectors.get(name)

    def get_all_detectors(self) -> Dict[str, BaseDetector]:
        """Get all registered detectors.

        Returns:
            Dictionary mapping detector names to detector instances
        """
        return self._detectors.copy()

    def list_detector_names(self) -> List[str]:
        """Get list of registered detector names.

        Returns:
            List of detector names
        """
        return list(self._detectors.keys())

    def has_detector(self, name: str) -> bool:
        """Check if a detector is registered.

        Args:
            name: Detector name to check

        Returns:
            True if detector is registered, False otherwise
        """
        return name in self._detectors

    def unregister(self, name: str) -> bool:
        """Unregister a detector by name.

        Args:
            name: Name of detector to remove

        Returns:
            True if detector was removed, False if not found
        """
        if name in self._detectors:
            del self._detectors[name]
            logger.debug(f"Unregistered detector: {name}")
            return True
        return False

    async def run_all(
        self, documents: List[Document], detector_names: Optional[List[str]] = None
    ) -> Dict[str, List[Issue]]:
        """Run all (or selected) detectors on documents.

        Args:
            documents: List of documents to analyze
            detector_names: Optional list of detector names to run.
                          If None, runs all registered detectors.

        Returns:
            Dictionary mapping detector names to their detected issues

        Raises:
            ValueError: If a requested detector name is not registered
        """
        # Determine which detectors to run
        if detector_names is None:
            detectors_to_run = self._detectors.copy()
        else:
            detectors_to_run = {}
            for name in detector_names:
                if name not in self._detectors:
                    raise ValueError(f"Detector '{name}' is not registered")
                detectors_to_run[name] = self._detectors[name]

        # Run each detector
        results: Dict[str, List[Issue]] = {}

        for name, detector in detectors_to_run.items():
            logger.info(f"Running detector: {name}")
            try:
                issues = await detector.detect(documents)
                results[name] = issues
                logger.info(f"Detector '{name}' found {len(issues)} issue(s)")
            except Exception as e:
                logger.error(f"Error running detector '{name}': {e}", exc_info=True)
                # Continue with other detectors even if one fails
                results[name] = []

        return results

    async def run_detector(self, detector_name: str, documents: List[Document]) -> List[Issue]:
        """Run a single detector by name.

        Args:
            detector_name: Name of detector to run
            documents: List of documents to analyze

        Returns:
            List of issues found by the detector

        Raises:
            ValueError: If detector is not registered
        """
        detector = self.get_detector(detector_name)
        if detector is None:
            raise ValueError(f"Detector '{detector_name}' is not registered")

        logger.info(f"Running detector: {detector_name}")
        issues = await detector.detect(documents)
        logger.info(f"Detector '{detector_name}' found {len(issues)} issue(s)")

        return issues

    def clear(self) -> None:
        """Remove all registered detectors."""
        self._detectors.clear()
        logger.debug("Cleared all detectors from registry")

    def __len__(self) -> int:
        """Get number of registered detectors."""
        return len(self._detectors)

    def __contains__(self, name: str) -> bool:
        """Check if detector is registered (supports 'in' operator)."""
        return name in self._detectors

    def __repr__(self) -> str:
        """String representation of the registry."""
        names = ", ".join(self._detectors.keys())
        return f"DetectorRegistry({len(self)} detectors: {names})"
