"""
Synthetic Login Data Generator

Generates realistic login events for identity risk scoring model training.
"""

import random
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# Configuration
TENANTS = ["tenant_a", "tenant_b", "tenant_c", "tenant_d", "tenant_e"]
DEVICE_TYPES = ["mobile_ios", "mobile_android", "desktop_windows", "desktop_mac", "desktop_linux"]
COUNTRIES = ["US", "CA", "UK", "DE", "FR", "JP", "AU", "IN", "BR", "MX"]
RISK_COUNTRIES = ["RU", "CN", "KP", "IR", "NG"]  # Higher risk regions


def generate_user_pool(n_users: int = 500) -> list[dict]:
    """Generate a pool of users with consistent attributes."""
    users = []
    for i in range(n_users):
        user_id = f"user_{i:04d}"
        tenant_id = random.choice(TENANTS)
        # Each user has 1-3 typical devices
        n_devices = random.randint(1, 3)
        devices = [f"device_{user_id}_{j}" for j in range(n_devices)]
        # Each user has a home location
        home_country = random.choice(COUNTRIES)
        home_city = fake.city()

        users.append({
            "user_id": user_id,
            "tenant_id": tenant_id,
            "devices": devices,
            "home_country": home_country,
            "home_city": home_city,
            "typical_hour_start": random.randint(6, 10),
            "typical_hour_end": random.randint(17, 22),
        })
    return users


def generate_ip_address(is_vpn: bool = False, is_risky: bool = False) -> str:
    """Generate a realistic IP address."""
    if is_risky:
        # Generate IP from risky ranges
        return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    if is_vpn:
        # VPN IPs often come from known ranges
        vpn_prefixes = ["104.238", "185.199", "45.33", "66.115"]
        prefix = random.choice(vpn_prefixes)
        return f"{prefix}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    # Regular IP
    return fake.ipv4_public()


def generate_login_event(
    user: dict,
    timestamp: datetime,
    is_fraudulent: bool = False,
) -> dict:
    """Generate a single login event."""

    if is_fraudulent:
        # Fraudulent logins have suspicious patterns
        device_id = f"device_unknown_{random.randint(1000, 9999)}"
        vpn_detected = random.random() < 0.7  # 70% use VPN
        location_country = random.choice(RISK_COUNTRIES) if random.random() < 0.6 else random.choice(COUNTRIES)
        location_city = fake.city()
        mfa_used = random.random() < 0.2  # Usually bypass MFA
        success = random.random() < 0.4  # Often fail
        ip = generate_ip_address(is_vpn=vpn_detected, is_risky=True)
        # Unusual hours
        hour = random.randint(0, 5) if random.random() < 0.5 else random.randint(0, 23)
        timestamp = timestamp.replace(hour=hour)
    else:
        # Normal login behavior
        device_id = random.choice(user["devices"])
        vpn_detected = random.random() < 0.15  # 15% use VPN normally
        location_country = user["home_country"] if random.random() < 0.85 else random.choice(COUNTRIES)
        location_city = user["home_city"] if location_country == user["home_country"] else fake.city()
        mfa_used = random.random() < 0.7  # 70% use MFA
        success = random.random() < 0.95  # 95% success rate
        ip = generate_ip_address(is_vpn=vpn_detected)
        # Normal working hours
        hour = random.randint(user["typical_hour_start"], user["typical_hour_end"])
        timestamp = timestamp.replace(hour=hour, minute=random.randint(0, 59))

    return {
        "user_id": user["user_id"],
        "tenant_id": user["tenant_id"],
        "timestamp": timestamp,
        "ip": ip,
        "device_id": device_id,
        "location_country": location_country,
        "location_city": location_city,
        "success": success,
        "mfa_used": mfa_used,
        "vpn_detected": vpn_detected,
        "is_fraudulent": is_fraudulent,
    }


def generate_logins(
    n_events: int = 10000,
    n_users: int = 500,
    fraud_rate: float = 0.10,
    days_back: int = 30,
    output_path: Optional[str] = "data/logins.parquet",
) -> pd.DataFrame:
    """
    Generate synthetic login events dataset.

    Args:
        n_events: Number of login events to generate
        n_users: Number of unique users in the pool
        fraud_rate: Percentage of fraudulent logins (0.0-1.0)
        days_back: Number of days of history to generate
        output_path: Path to save parquet file (None to skip saving)

    Returns:
        DataFrame with login events
    """
    print(f"Generating {n_events} login events with {fraud_rate*100:.0f}% fraud rate...")

    # Generate user pool
    users = generate_user_pool(n_users)

    # Generate timestamps spread over the time period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    events = []
    n_fraud = int(n_events * fraud_rate)
    n_normal = n_events - n_fraud

    # Generate normal events
    for _ in range(n_normal):
        user = random.choice(users)
        timestamp = fake.date_time_between(start_date=start_date, end_date=end_date)
        events.append(generate_login_event(user, timestamp, is_fraudulent=False))

    # Generate fraudulent events
    for _ in range(n_fraud):
        user = random.choice(users)
        timestamp = fake.date_time_between(start_date=start_date, end_date=end_date)
        events.append(generate_login_event(user, timestamp, is_fraudulent=True))

    # Create DataFrame
    df = pd.DataFrame(events)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Add event_id
    df.insert(0, "event_id", [f"evt_{i:06d}" for i in range(len(df))])

    # Print summary
    print(f"\nDataset Summary:")
    print(f"  Total events: {len(df)}")
    print(f"  Unique users: {df['user_id'].nunique()}")
    print(f"  Unique tenants: {df['tenant_id'].nunique()}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Fraud rate: {df['is_fraudulent'].mean()*100:.1f}%")
    print(f"  Success rate: {df['success'].mean()*100:.1f}%")
    print(f"  MFA usage: {df['mfa_used'].mean()*100:.1f}%")
    print(f"  VPN detected: {df['vpn_detected'].mean()*100:.1f}%")

    # Save to parquet
    if output_path:
        df.to_parquet(output_path, index=False)
        print(f"\nSaved to {output_path}")

    return df


if __name__ == "__main__":
    df = generate_logins()
    print("\nSample rows:")
    print(df.head(10).to_string())
