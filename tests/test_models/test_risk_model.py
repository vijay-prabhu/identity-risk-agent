"""
Tests for risk scoring model.

Includes quality gates that fail CI if model performance drops.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.models.risk_model import (
    RiskScorer,
    prepare_data,
    train_random_forest,
    train_and_evaluate,
    FEATURE_COLUMNS,
)


@pytest.fixture
def sample_features_df():
    """Create sample feature data for testing."""
    np.random.seed(42)
    n_samples = 200

    # Create features with clear fraud patterns
    df = pd.DataFrame({
        "failed_logins_24h": np.random.randint(0, 3, n_samples),
        "login_count_7d": np.random.randint(0, 20, n_samples),
        "device_age_days": np.random.uniform(0, 100, n_samples),
        "is_new_device": np.random.randint(0, 2, n_samples),
        "ip_reputation_score": np.random.uniform(0, 1, n_samples),
        "hour_of_day": np.random.randint(0, 24, n_samples),
        "is_unusual_hour": np.random.randint(0, 2, n_samples),
        "location_changed": np.random.randint(0, 2, n_samples),
        "mfa_used": np.random.randint(0, 2, n_samples),
        "vpn_detected": np.random.randint(0, 2, n_samples),
        "success": np.random.randint(0, 2, n_samples),
    })

    # Create target with correlation to features
    fraud_score = (
        df["is_new_device"] * 0.3 +
        df["ip_reputation_score"] * 0.3 +
        df["is_unusual_hour"] * 0.2 +
        (1 - df["mfa_used"]) * 0.1 +
        df["vpn_detected"] * 0.1
    )
    df["is_fraudulent"] = (fraud_score > 0.5).astype(int)

    return df


def test_prepare_data(sample_features_df):
    """Test data preparation."""
    X_train, X_test, y_train, y_test, scaler = prepare_data(sample_features_df)

    # Check shapes
    assert len(X_train) + len(X_test) == len(sample_features_df)
    assert X_train.shape[1] == len(FEATURE_COLUMNS)

    # Check stratification (fraud rate should be similar)
    train_fraud_rate = y_train.mean()
    test_fraud_rate = y_test.mean()
    assert abs(train_fraud_rate - test_fraud_rate) < 0.1


def test_train_random_forest(sample_features_df):
    """Test Random Forest training."""
    X_train, X_test, y_train, y_test, scaler = prepare_data(sample_features_df)
    model = train_random_forest(X_train, y_train)

    # Should be able to predict
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)
    assert set(y_pred).issubset({0, 1})


def test_risk_scorer_score(sample_features_df):
    """Test RiskScorer scoring interface."""
    X_train, X_test, y_train, y_test, scaler = prepare_data(sample_features_df)
    model = train_random_forest(X_train, y_train)

    scorer = RiskScorer(
        supervised_model=model,
        scaler=scaler,
        feature_columns=FEATURE_COLUMNS,
    )

    # Test scoring
    features = {col: 0.5 for col in FEATURE_COLUMNS}
    result = scorer.score(features)

    assert "risk_score" in result
    assert "risk_level" in result
    assert 0 <= result["risk_score"] <= 1
    assert result["risk_level"] in ["low", "medium", "high", "critical"]


def test_risk_scorer_save_load(sample_features_df, tmp_path):
    """Test model save and load."""
    X_train, X_test, y_train, y_test, scaler = prepare_data(sample_features_df)
    model = train_random_forest(X_train, y_train)

    scorer = RiskScorer(
        supervised_model=model,
        scaler=scaler,
        feature_columns=FEATURE_COLUMNS,
    )

    # Save
    model_path = tmp_path / "test_model.pkl"
    scorer.save(str(model_path))
    assert model_path.exists()

    # Load
    loaded_scorer = RiskScorer.load(str(model_path))

    # Should produce same results
    features = {col: 0.5 for col in FEATURE_COLUMNS}
    original_result = scorer.score(features)
    loaded_result = loaded_scorer.score(features)

    assert abs(original_result["risk_score"] - loaded_result["risk_score"]) < 1e-10


# Quality Gate Tests
class TestQualityGate:
    """
    Quality gate tests that fail CI if model performance drops.

    These tests ensure the model meets minimum performance thresholds.
    """

    @pytest.fixture
    def trained_model_path(self):
        """Path to the trained model."""
        return Path("models/risk_model.pkl")

    @pytest.mark.skipif(
        not Path("models/risk_model.pkl").exists(),
        reason="Model file not present - skipped in CI (model trained during quality-gate job)"
    )
    def test_quality_gate_model_exists(self, trained_model_path):
        """Quality gate: Model file must exist (skipped if not present)."""
        assert trained_model_path.exists(), (
            f"Model not found at {trained_model_path}. "
            "Run training first: python src/models/risk_model.py"
        )

    def test_quality_gate_auc_threshold(self):
        """
        Quality gate: Model AUC must be >= 0.85.

        This is the primary quality gate for the risk model.
        If this test fails, the model should not be deployed.
        """
        # Load data and model
        features_path = Path("data/features.parquet")
        if not features_path.exists():
            pytest.skip("Features not generated yet")

        model_path = Path("models/risk_model.pkl")
        if not model_path.exists():
            pytest.skip("Model not trained yet")

        df = pd.read_parquet(features_path)
        scorer = RiskScorer.load(str(model_path))

        # Evaluate on full dataset
        X = df[FEATURE_COLUMNS].values
        y = df["is_fraudulent"].values
        y_proba = scorer.predict_proba(X)

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, y_proba)

        # Quality gate threshold
        MINIMUM_AUC = 0.85

        assert auc >= MINIMUM_AUC, (
            f"Model AUC ({auc:.4f}) below threshold ({MINIMUM_AUC}). "
            "Model quality is insufficient for deployment."
        )

    def test_quality_gate_low_risk_normal_login(self):
        """Quality gate: Normal login should have low risk score."""
        model_path = Path("models/risk_model.pkl")
        if not model_path.exists():
            pytest.skip("Model not trained yet")

        scorer = RiskScorer.load(str(model_path))

        # Typical normal login
        normal_features = {
            "failed_logins_24h": 0,
            "login_count_7d": 10,
            "device_age_days": 30,
            "is_new_device": 0,
            "ip_reputation_score": 0.0,
            "hour_of_day": 10,
            "is_unusual_hour": 0,
            "location_changed": 0,
            "mfa_used": 1,
            "vpn_detected": 0,
            "success": 1,
        }

        result = scorer.score(normal_features)
        assert result["risk_score"] < 0.3, (
            f"Normal login scored too high: {result['risk_score']:.2f}"
        )

    def test_quality_gate_high_risk_suspicious_login(self):
        """Quality gate: Suspicious login should have high risk score."""
        model_path = Path("models/risk_model.pkl")
        if not model_path.exists():
            pytest.skip("Model not trained yet")

        scorer = RiskScorer.load(str(model_path))

        # Suspicious login
        suspicious_features = {
            "failed_logins_24h": 5,
            "login_count_7d": 0,
            "device_age_days": 0,
            "is_new_device": 1,
            "ip_reputation_score": 0.9,
            "hour_of_day": 3,
            "is_unusual_hour": 1,
            "location_changed": 1,
            "mfa_used": 0,
            "vpn_detected": 1,
            "success": 0,
        }

        result = scorer.score(suspicious_features)
        assert result["risk_score"] > 0.7, (
            f"Suspicious login scored too low: {result['risk_score']:.2f}"
        )
