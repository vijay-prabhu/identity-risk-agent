# ADR 006: Testing Strategy & Quality Gates

## Status
Accepted

## Context
ML systems require testing beyond traditional software:
- **Model quality**: Accuracy, AUC, fairness metrics
- **Data quality**: Schema validation, drift detection
- **Integration**: End-to-end pipeline testing
- **Regression**: Ensure changes don't degrade performance

The CI/CD pipeline must enforce quality gates that prevent deploying degraded models.

## Decision
Implement a **multi-layer testing strategy** with automated quality gates.

### Testing Pyramid
```
                    ┌─────────┐
                    │  E2E    │  ← Full pipeline tests
                    │  Tests  │
                    └────┬────┘
                         │
                ┌────────┴────────┐
                │  Integration    │  ← API + component tests
                │     Tests       │
                └────────┬────────┘
                         │
        ┌────────────────┴────────────────┐
        │          Unit Tests             │  ← Function-level tests
        │   (features, tools, agents)     │
        └─────────────────────────────────┘
```

### Quality Gates

#### 1. Model Performance Gate
```yaml
# .github/workflows/ci.yml
- name: Model Quality Gate
  run: |
    python -c "
    results = train_and_evaluate(df)
    assert results['auc'] >= 0.85, 'AUC below threshold'
    "
```

**Thresholds:**
| Metric | Threshold | Action if Failed |
|--------|-----------|------------------|
| AUC | >= 0.85 | Block deployment |
| Accuracy | >= 0.80 | Warning |
| Precision | >= 0.75 | Warning |
| Recall | >= 0.70 | Warning |

#### 2. Data Quality Checks
```python
def validate_features(df):
    # No nulls in key columns
    assert df[['user_id', 'risk_score']].isnull().sum().sum() == 0

    # Risk score in valid range
    assert (df['risk_score'] >= 0).all()
    assert (df['risk_score'] <= 1).all()

    # Fraud rate within bounds
    fraud_rate = df['is_fraud'].mean()
    assert 0.01 <= fraud_rate <= 0.20
```

#### 3. Test Categories
```python
# tests/test_models/test_risk_model.py

class TestQualityGate:
    """Quality gate tests that fail CI if model performance drops."""

    def test_quality_gate_auc_threshold(self):
        results = train_and_evaluate(df)
        assert results["auc"] >= 0.85

    def test_quality_gate_low_risk_normal_login(self):
        # Normal login should be low risk
        score = scorer.score(normal_features)
        assert score["risk_level"] == "low"

    def test_quality_gate_high_risk_suspicious_login(self):
        # Suspicious login should be high risk
        score = scorer.score(suspicious_features)
        assert score["risk_level"] in ["high", "critical"]
```

## CI/CD Pipeline

```yaml
jobs:
  lint:        # Fast: code style
  test:        # Medium: unit + integration
  quality-gate: # Slow: train model, check AUC
  security:    # Medium: vulnerability scan
  build:       # Only on main, after gates pass
```

## Test Coverage

| Component | Coverage Target | Current |
|-----------|-----------------|---------|
| src/core | 90% | ✓ |
| src/features | 85% | ✓ |
| src/models | 90% | ✓ |
| src/agents | 80% | ✓ |
| api/ | 85% | ✓ |

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **pytest + custom gates** | Flexible, integrated | More code to maintain |
| **Great Expectations** | Rich data validation | Heavy dependency |
| **MLflow Model Validation** | Built-in | Limited customization |
| **Manual review** | Thorough | Doesn't scale, slow |

## Consequences

### Positive
- Automated quality enforcement
- Prevents model regression
- Fast feedback in PRs
- Documented test cases

### Negative
- CI pipeline takes longer
- May need to adjust thresholds over time
- Flaky tests from randomness in training

## Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Quality gate only
make test-gate

# Fast subset
make test-fast
```

## References
- [Testing ML Systems](https://madewithml.com/courses/mlops/testing/)
- [pytest Documentation](https://docs.pytest.org/)
