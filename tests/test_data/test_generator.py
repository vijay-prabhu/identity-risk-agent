"""
Tests for synthetic data generator.
"""

import pandas as pd
import pytest

from src.core.data_generator import generate_logins, generate_user_pool


def test_generate_user_pool():
    """Test user pool generation."""
    users = generate_user_pool(n_users=10)
    assert len(users) == 10

    for user in users:
        assert "user_id" in user
        assert "tenant_id" in user
        assert "devices" in user
        assert len(user["devices"]) >= 1


def test_generate_logins_shape():
    """Test that generated data has correct shape."""
    df = generate_logins(n_events=100, n_users=20, output_path=None)
    assert len(df) == 100
    assert df["user_id"].nunique() <= 20


def test_generate_logins_schema():
    """Test that generated data has correct schema."""
    df = generate_logins(n_events=100, output_path=None)

    expected_columns = [
        "event_id", "user_id", "tenant_id", "timestamp", "ip",
        "device_id", "location_country", "location_city",
        "success", "mfa_used", "vpn_detected", "is_fraudulent"
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"


def test_generate_logins_fraud_rate():
    """Test that fraud rate is approximately correct."""
    df = generate_logins(n_events=1000, fraud_rate=0.10, output_path=None)
    actual_fraud_rate = df["is_fraudulent"].mean()

    # Allow 2% tolerance
    assert abs(actual_fraud_rate - 0.10) < 0.02, f"Fraud rate {actual_fraud_rate} not close to 0.10"


def test_generate_logins_tenant_distribution():
    """Test that data spans multiple tenants."""
    df = generate_logins(n_events=500, output_path=None)
    assert df["tenant_id"].nunique() >= 3, "Expected at least 3 tenants"


def test_fraud_patterns():
    """Test that fraudulent logins have distinct patterns."""
    df = generate_logins(n_events=1000, fraud_rate=0.20, output_path=None)

    fraud = df[df["is_fraudulent"]]
    normal = df[~df["is_fraudulent"]]

    # Fraudulent logins should have lower success rate
    assert fraud["success"].mean() < normal["success"].mean()

    # Fraudulent logins should have higher VPN usage
    assert fraud["vpn_detected"].mean() > normal["vpn_detected"].mean()

    # Fraudulent logins should have lower MFA usage
    assert fraud["mfa_used"].mean() < normal["mfa_used"].mean()
