# ADR 005: Privacy & PII Protection

## Status
Accepted

## Context
The identity risk platform handles sensitive user data:
- Email addresses
- IP addresses
- Device identifiers
- Location information
- Login patterns

Before sending any data to LLMs or logging, PII must be detected and redacted to:
- Comply with privacy regulations (GDPR, CCPA)
- Prevent data leakage to external APIs
- Minimize exposure in logs and monitoring

## Decision
Use **Presidio** for PII detection with regex fallback.

### Architecture
```
┌──────────────────┐
│   User Query /   │
│   Event Data     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   PIIDetector    │
│  ┌────────────┐  │
│  │ Presidio   │◄─┼── If available
│  │ Analyzer   │  │
│  └────────────┘  │
│  ┌────────────┐  │
│  │   Regex    │◄─┼── Fallback
│  │  Patterns  │  │
│  └────────────┘  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Redacted Text   │
│  [REDACTED]      │
└──────────────────┘
```

### Detected Entity Types
| Entity | Pattern | Example |
|--------|---------|---------|
| EMAIL_ADDRESS | RFC 5322 | john@example.com → [REDACTED] |
| PHONE_NUMBER | Various formats | 555-123-4567 → [REDACTED] |
| IP_ADDRESS | IPv4/IPv6 | 192.168.1.1 → [REDACTED] |
| US_SSN | XXX-XX-XXXX | 123-45-6789 → [REDACTED] |
| CREDIT_CARD | Luhn-valid | 4111-1111-1111-1111 → [REDACTED] |

## Implementation

### PIIDetector Class
```python
class PIIDetector:
    def detect(self, text: str) -> PIIResult:
        if PRESIDIO_AVAILABLE:
            return self._detect_presidio(text)
        return self._detect_mock(text)

    def redact(self, text: str) -> str:
        result = self.detect(text)
        return result.redacted_text
```

### Privacy Middleware
```python
class PrivacyMiddleware:
    def process_for_llm(self, text: str, context: dict) -> dict:
        result = self.detector.detect(text)

        if result.has_pii:
            logger.warning(f"PII redacted: {result.entity_count} entities")

        return {
            "text": result.redacted_text,
            "pii_detected": result.has_pii,
            "context": self.detector.redact_dict(context),
        }
```

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **Presidio (chosen)** | Comprehensive, extensible | Requires spaCy models |
| **spaCy NER only** | Lightweight | Limited entity types |
| **Regex only** | Fast, no dependencies | False positives/negatives |
| **Cloud DLP APIs** | High accuracy | Cost, latency, privacy irony |

## Consequences

### Positive
- Prevents PII leakage to LLMs
- Consistent redaction across the platform
- Audit trail of PII detections
- Graceful fallback without Presidio

### Negative
- False positives may over-redact
- Performance overhead for large texts
- Presidio models add ~500MB to dependencies

## Usage Guidelines

1. **Always redact before LLM calls**
```python
processed = privacy.process_for_llm(user_query)
response = llm.invoke(processed["text"])
```

2. **Redact in logs**
```python
logger.info(f"Query: {detector.redact(query)}")
```

3. **Preserve IDs that aren't PII**
```python
# user_id like "user_001" is not PII
# email like "john@example.com" is PII
```

## References
- [Microsoft Presidio](https://microsoft.github.io/presidio/)
- [GDPR Article 4](https://gdpr-info.eu/art-4-gdpr/)
