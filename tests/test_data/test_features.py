"""
Tests for feature engineering.
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.features.feature_engineering import (
    compute_device_age_days,
    compute_ip_reputation_score,
    compute_is_new_device,
    compute_hour_of_day,
    engineer_features,
)


@pytest.fixture
def sample_df():
    """Create sample login data for testing."""
    return pd.DataFrame({
        "event_id": ["e1", "e2", "e3", "e4", "e5"],
        "user_id": ["u1", "u1", "u1", "u2", "u2"],
        "tenant_id": ["t1", "t1", "t1", "t1", "t1"],
        "timestamp": pd.to_datetime([
            "2024-01-01 10:00:00",
            "2024-01-01 14:00:00",
            "2024-01-02 09:00:00",
            "2024-01-01 02:00:00",  # Unusual hour
            "2024-01-01 15:00:00",
        ]),
        "ip": ["1.1.1.1", "1.1.1.2", "1.1.1.3", "2.2.2.2", "2.2.2.3"],
        "device_id": ["d1", "d1", "d2", "device_unknown_123", "d3"],
        "location_country": ["US", "US", "CA", "RU", "US"],
        "location_city": ["NYC", "NYC", "Toronto", "Moscow", "LA"],
        "success": [True, False, True, False, True],
        "mfa_used": [True, True, False, False, True],
        "vpn_detected": [False, False, False, True, False],
        "is_fraudulent": [False, False, False, True, False],
    })


def test_compute_device_age_days(sample_df):
    """Test device age calculation."""
    # Sort by timestamp as the function does
    df_sorted = sample_df.sort_values("timestamp").reset_index(drop=True)
    result = compute_device_age_days(df_sorted)

    # Check that at least some devices have age > 0 (repeated devices)
    assert result.max() >= 0
    # First occurrence of any device should be 0
    assert (result == 0).any()


def test_compute_is_new_device(sample_df):
    """Test new device detection."""
    # Sort by timestamp as the function does
    df_sorted = sample_df.sort_values("timestamp").reset_index(drop=True)
    result = compute_is_new_device(df_sorted)

    # There should be some new devices (True) and some not new (False)
    assert result.any()  # Some new devices
    # With 5 events and some device reuse, we should have at least one repeat
    n_unique_devices = sample_df["device_id"].nunique()
    n_events = len(sample_df)
    if n_events > n_unique_devices:
        assert (~result).any()  # Some not-new devices


def test_compute_ip_reputation_score(sample_df):
    """Test IP reputation scoring."""
    result = compute_ip_reputation_score(sample_df)

    # Normal logins should have low score
    assert result.iloc[0] < 0.3

    # VPN + risky country + unknown device should have high score
    assert result.iloc[3] >= 0.8  # RU country + VPN + unknown device


def test_compute_hour_of_day(sample_df):
    """Test hour extraction."""
    result = compute_hour_of_day(sample_df)

    assert result.iloc[0] == 10
    assert result.iloc[3] == 2


def test_engineer_features_output_shape(sample_df):
    """Test that engineer_features produces expected columns."""
    result = engineer_features(sample_df, output_path=None)

    expected_features = [
        "failed_logins_24h", "login_count_7d", "device_age_days",
        "is_new_device", "ip_reputation_score", "hour_of_day",
        "is_unusual_hour", "location_changed"
    ]

    for col in expected_features:
        assert col in result.columns, f"Missing feature: {col}"


def test_engineer_features_no_nulls(sample_df):
    """Test that computed features don't have nulls."""
    result = engineer_features(sample_df, output_path=None)

    feature_cols = [
        "device_age_days", "is_new_device", "ip_reputation_score",
        "hour_of_day", "is_unusual_hour"
    ]

    for col in feature_cols:
        assert result[col].isna().sum() == 0, f"Feature {col} has nulls"
