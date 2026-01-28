"""
PII Detection Module

Uses Presidio for detecting and redacting personally identifiable information
in text data before LLM processing.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import Presidio, fall back to mock if not available
try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logger.warning("Presidio not available. Using mock PII detection.")


@dataclass
class PIIResult:
    """Result of PII detection."""
    original_text: str
    redacted_text: str
    entities_found: List[Dict[str, Any]]
    has_pii: bool


class PIIDetector:
    """
    PII detection and redaction using Presidio.

    Detects sensitive information like:
    - Email addresses
    - Phone numbers
    - SSN/National IDs
    - Credit card numbers
    - Names
    - IP addresses (custom)
    - Location data
    """

    # Entity types to detect
    DEFAULT_ENTITIES = [
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "PERSON",
        "LOCATION",
        "IP_ADDRESS",
        "US_DRIVER_LICENSE",
        "US_PASSPORT",
    ]

    def __init__(
        self,
        entities: Optional[List[str]] = None,
        score_threshold: float = 0.5,
        language: str = "en",
    ):
        """
        Initialize the PII detector.

        Args:
            entities: List of entity types to detect (None = all defaults)
            score_threshold: Minimum confidence score for detection
            language: Language code for analysis
        """
        self.entities = entities or self.DEFAULT_ENTITIES
        self.score_threshold = score_threshold
        self.language = language

        if PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            logger.info("Presidio PII detector initialized")
        else:
            self.analyzer = None
            self.anonymizer = None
            logger.info("Mock PII detector initialized (Presidio not available)")

    def detect(self, text: str) -> PIIResult:
        """
        Detect PII in text.

        Args:
            text: Input text to analyze

        Returns:
            PIIResult with detection details
        """
        if not text or not text.strip():
            return PIIResult(
                original_text=text,
                redacted_text=text,
                entities_found=[],
                has_pii=False,
            )

        if PRESIDIO_AVAILABLE and self.analyzer:
            return self._detect_presidio(text)
        else:
            return self._detect_mock(text)

    def _detect_presidio(self, text: str) -> PIIResult:
        """Detect PII using Presidio."""
        # Analyze text
        results: List[RecognizerResult] = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language=self.language,
        )

        # Filter by score threshold
        results = [r for r in results if r.score >= self.score_threshold]

        # Convert to dict format
        entities_found = [
            {
                "type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": r.score,
                "text": text[r.start:r.end],
            }
            for r in results
        ]

        # Redact if PII found
        if results:
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators={
                    "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})
                },
            )
            redacted_text = anonymized.text
        else:
            redacted_text = text

        return PIIResult(
            original_text=text,
            redacted_text=redacted_text,
            entities_found=entities_found,
            has_pii=len(entities_found) > 0,
        )

    def _detect_mock(self, text: str) -> PIIResult:
        """Mock PII detection using simple patterns."""
        import re

        entities_found = []
        redacted_text = text

        # Simple pattern matching for common PII
        patterns = {
            "EMAIL_ADDRESS": r'\b[\w.-]+@[\w.-]+\.\w+\b',
            "PHONE_NUMBER": r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
            "IP_ADDRESS": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            "US_SSN": r'\b\d{3}-\d{2}-\d{4}\b',
            "CREDIT_CARD": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        }

        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                entities_found.append({
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.85,  # Mock confidence
                    "text": match.group(),
                })

        # Redact found entities (process in reverse order to preserve positions)
        for entity in sorted(entities_found, key=lambda x: x["start"], reverse=True):
            redacted_text = (
                redacted_text[:entity["start"]] +
                "[REDACTED]" +
                redacted_text[entity["end"]:]
            )

        return PIIResult(
            original_text=text,
            redacted_text=redacted_text,
            entities_found=entities_found,
            has_pii=len(entities_found) > 0,
        )

    def redact(self, text: str) -> str:
        """
        Redact PII from text.

        Convenience method that returns only the redacted text.

        Args:
            text: Input text

        Returns:
            Text with PII redacted
        """
        result = self.detect(text)
        return result.redacted_text

    def redact_dict(
        self,
        data: Dict[str, Any],
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Redact PII from dictionary fields.

        Args:
            data: Dictionary with potentially sensitive values
            fields: Specific fields to check (None = all string fields)

        Returns:
            Dictionary with PII redacted
        """
        result = data.copy()

        for key, value in data.items():
            if fields and key not in fields:
                continue

            if isinstance(value, str):
                result[key] = self.redact(value)
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value, fields)
            elif isinstance(value, list):
                result[key] = [
                    self.redact(item) if isinstance(item, str) else item
                    for item in value
                ]

        return result


class PrivacyMiddleware:
    """
    Middleware for applying privacy controls to requests/responses.

    Can be used with FastAPI to automatically redact PII from logs
    and responses.
    """

    def __init__(
        self,
        detector: Optional[PIIDetector] = None,
        log_detections: bool = True,
    ):
        self.detector = detector or PIIDetector()
        self.log_detections = log_detections

    def process_for_llm(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process text before sending to LLM.

        Args:
            text: Text to process
            context: Optional context dict to also process

        Returns:
            Dict with processed text and metadata
        """
        result = self.detector.detect(text)

        if self.log_detections and result.has_pii:
            logger.warning(
                f"PII detected and redacted: {len(result.entities_found)} entities "
                f"({', '.join(set(e['type'] for e in result.entities_found))})"
            )

        processed = {
            "text": result.redacted_text,
            "pii_detected": result.has_pii,
            "entity_count": len(result.entities_found),
        }

        if context:
            processed["context"] = self.detector.redact_dict(context)

        return processed


if __name__ == "__main__":
    # Quick test
    print("Testing PII detector...")

    detector = PIIDetector()

    test_texts = [
        "Contact John Smith at john.smith@example.com or call 555-123-4567",
        "User ID: user_12345, IP: 192.168.1.100, Device: laptop_001",
        "SSN: 123-45-6789, Credit Card: 4111-1111-1111-1111",
        "Normal login from San Francisco at 3pm",
    ]

    for text in test_texts:
        result = detector.detect(text)
        print(f"\nOriginal: {text}")
        print(f"Redacted: {result.redacted_text}")
        print(f"PII found: {result.has_pii} ({len(result.entities_found)} entities)")
        if result.entities_found:
            for e in result.entities_found:
                print(f"  - {e['type']}: '{e['text']}' (score: {e['score']:.2f})")
