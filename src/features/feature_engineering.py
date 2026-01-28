"""
Feature Engineering for Identity Risk Scoring

Computes behavioral, temporal, and network features from login events.
"""

from typing import Optional

import numpy as np
import pandas as pd

# Risk country list for IP reputation mock
RISK_COUNTRIES = {"RU", "CN", "KP", "IR", "NG"}


def compute_failed_logins_24h(df: pd.DataFrame) -> pd.Series:
    """
    Count failed logins in the 24 hours prior to each event for the same user.

    Args:
        df: DataFrame with 'user_id', 'timestamp', 'success' columns

    Returns:
        Series with failed login counts
    """
    df = df.sort_values("timestamp").copy()
    failed_counts = []

    for idx, row in df.iterrows():
        user_id = row["user_id"]
        current_time = row["timestamp"]
        window_start = current_time - pd.Timedelta(hours=24)

        # Get failed logins for this user in the window
        mask = (
            (df["user_id"] == user_id) &
            (df["timestamp"] >= window_start) &
            (df["timestamp"] < current_time) &
            (~df["success"])
        )
        failed_counts.append(mask.sum())

    return pd.Series(failed_counts, index=df.index)


def compute_failed_logins_24h_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    Optimized version using rolling windows.
    """
    df = df.sort_values(["user_id", "timestamp"]).copy()

    # Create failure indicator
    df["is_failure"] = (~df["success"]).astype(int)

    # Group by user and compute rolling count
    def user_rolling_count(group):
        group = group.set_index("timestamp").sort_index()
        # Rolling 24h window, excluding current row
        rolling = group["is_failure"].rolling("24h", closed="left").sum()
        return rolling.fillna(0).astype(int)

    result = df.groupby("user_id", group_keys=False).apply(user_rolling_count, include_groups=False)
    return result.reindex(df.index).fillna(0).astype(int)


def compute_login_frequency(df: pd.DataFrame, days: int = 7) -> pd.Series:
    """
    Compute average logins per day over the past N days for each user.

    Args:
        df: DataFrame with 'user_id', 'timestamp' columns
        days: Lookback window in days

    Returns:
        Series with login frequency (logins per day)
    """
    df = df.sort_values(["user_id", "timestamp"]).copy()

    frequencies = []
    for idx, row in df.iterrows():
        user_id = row["user_id"]
        current_time = row["timestamp"]
        window_start = current_time - pd.Timedelta(days=days)

        # Count logins in window
        mask = (
            (df["user_id"] == user_id) &
            (df["timestamp"] >= window_start) &
            (df["timestamp"] < current_time)
        )
        login_count = mask.sum()
        frequency = login_count / days
        frequencies.append(frequency)

    return pd.Series(frequencies, index=df.index)


def compute_device_age_days(df: pd.DataFrame) -> pd.Series:
    """
    Compute days since device was first seen for each user-device pair.

    Args:
        df: DataFrame with 'user_id', 'device_id', 'timestamp' columns

    Returns:
        Series with device age in days
    """
    df = df.sort_values("timestamp").copy()

    # Track first seen date for each user-device pair
    first_seen = {}
    device_ages = []

    for idx, row in df.iterrows():
        key = (row["user_id"], row["device_id"])
        current_time = row["timestamp"]

        if key not in first_seen:
            first_seen[key] = current_time
            device_ages.append(0.0)
        else:
            age = (current_time - first_seen[key]).total_seconds() / 86400
            device_ages.append(age)

    return pd.Series(device_ages, index=df.index)


def compute_is_new_device(df: pd.DataFrame) -> pd.Series:
    """
    Check if device is new (first time seen for this user).

    Args:
        df: DataFrame with 'user_id', 'device_id', 'timestamp' columns

    Returns:
        Series with boolean indicating new device
    """
    df = df.sort_values("timestamp").copy()

    seen_devices = {}
    is_new = []

    for idx, row in df.iterrows():
        key = (row["user_id"], row["device_id"])

        if key not in seen_devices:
            seen_devices[key] = True
            is_new.append(True)
        else:
            is_new.append(False)

    return pd.Series(is_new, index=df.index)


def compute_ip_reputation_score(df: pd.DataFrame) -> pd.Series:
    """
    Mock IP reputation score based on VPN detection and location.

    In production, this would call external APIs like AbuseIPDB.

    Args:
        df: DataFrame with 'vpn_detected', 'location_country' columns

    Returns:
        Series with reputation score (0-1, higher = riskier)
    """
    scores = []

    for idx, row in df.iterrows():
        score = 0.0

        # VPN adds risk
        if row["vpn_detected"]:
            score += 0.3

        # Risky country adds risk
        if row["location_country"] in RISK_COUNTRIES:
            score += 0.5

        # Unknown device pattern (device_id contains 'unknown')
        if "unknown" in row["device_id"]:
            score += 0.2

        scores.append(min(score, 1.0))

    return pd.Series(scores, index=df.index)


def compute_hour_of_day(df: pd.DataFrame) -> pd.Series:
    """Extract hour of day from timestamp."""
    return df["timestamp"].dt.hour


def compute_is_unusual_hour(df: pd.DataFrame) -> pd.Series:
    """
    Check if login is at unusual hour (outside 6am-10pm).
    """
    hour = df["timestamp"].dt.hour
    return (hour < 6) | (hour > 22)


def compute_location_change(df: pd.DataFrame) -> pd.Series:
    """
    Check if location changed from previous login for the same user.

    Args:
        df: DataFrame with 'user_id', 'timestamp', 'location_country' columns

    Returns:
        Series with boolean indicating location change
    """
    df = df.sort_values(["user_id", "timestamp"]).copy()

    # Get previous location for each user
    df["prev_location"] = df.groupby("user_id")["location_country"].shift(1)
    location_changed = (df["location_country"] != df["prev_location"]) & df["prev_location"].notna()

    return location_changed


def engineer_features(
    df: pd.DataFrame,
    output_path: Optional[str] = "data/features.parquet",
) -> pd.DataFrame:
    """
    Compute all features for the login events dataset.

    Args:
        df: Raw login events DataFrame
        output_path: Path to save features (None to skip)

    Returns:
        DataFrame with computed features
    """
    print("Computing features...")
    df = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort by timestamp for temporal features
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute features
    print("  - failed_logins_24h...")
    df["failed_logins_24h"] = compute_failed_logins_24h_vectorized(df)

    print("  - login_frequency_7d...")
    # Simplified: use rolling count approximation
    df["login_count_7d"] = df.groupby("user_id").cumcount()

    print("  - device_age_days...")
    df["device_age_days"] = compute_device_age_days(df)

    print("  - is_new_device...")
    df["is_new_device"] = compute_is_new_device(df)

    print("  - ip_reputation_score...")
    df["ip_reputation_score"] = compute_ip_reputation_score(df)

    print("  - hour_of_day...")
    df["hour_of_day"] = compute_hour_of_day(df)

    print("  - is_unusual_hour...")
    df["is_unusual_hour"] = compute_is_unusual_hour(df)

    print("  - location_changed...")
    df["location_changed"] = compute_location_change(df)

    # Convert booleans to int for model compatibility
    bool_cols = ["is_new_device", "is_unusual_hour", "location_changed", "success", "mfa_used", "vpn_detected"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Summary
    print(f"\nFeature Summary:")
    feature_cols = [
        "failed_logins_24h", "login_count_7d", "device_age_days",
        "is_new_device", "ip_reputation_score", "hour_of_day",
        "is_unusual_hour", "location_changed"
    ]
    print(df[feature_cols].describe().round(2))

    # Save
    if output_path:
        df.to_parquet(output_path, index=False)
        print(f"\nSaved features to {output_path}")

    return df


if __name__ == "__main__":
    # Load raw data
    df = pd.read_parquet("data/logins.parquet")
    print(f"Loaded {len(df)} events")

    # Engineer features
    df_features = engineer_features(df)

    # Show correlation with target
    print("\nFeature correlation with is_fraudulent:")
    feature_cols = [
        "failed_logins_24h", "login_count_7d", "device_age_days",
        "is_new_device", "ip_reputation_score", "hour_of_day",
        "is_unusual_hour", "location_changed", "mfa_used", "vpn_detected", "success"
    ]
    correlations = df_features[feature_cols + ["is_fraudulent"]].corr()["is_fraudulent"].drop("is_fraudulent")
    print(correlations.sort_values(ascending=False).round(3))
