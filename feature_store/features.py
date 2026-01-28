"""
Feast Feature Definitions for Identity Risk Scoring

Defines entities, feature views, and feature services for the risk scoring platform.
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, FeatureService, ValueType
from feast.types import Float32, Int64, String

# =============================================================================
# Entities
# =============================================================================

# User entity - primary entity for risk scoring
user = Entity(
    name="user",
    join_keys=["user_id"],
    value_type=ValueType.STRING,
    description="User entity for identity risk scoring",
)

# Tenant entity - for multi-tenant isolation
tenant = Entity(
    name="tenant",
    join_keys=["tenant_id"],
    value_type=ValueType.STRING,
    description="Tenant entity for multi-tenant isolation",
)

# =============================================================================
# Data Sources
# =============================================================================

# Login features source (computed features from feature engineering)
login_features_source = FileSource(
    name="login_features_source",
    path="../data/features.parquet",
    timestamp_field="timestamp",
)

# =============================================================================
# Feature Views
# =============================================================================

# User login features - behavioral features per user
user_login_features = FeatureView(
    name="user_login_features",
    entities=[user],
    ttl=timedelta(days=1),  # Features valid for 1 day
    schema=[
        Field(name="failed_logins_24h", dtype=Int64),
        Field(name="login_count_7d", dtype=Int64),
        Field(name="device_age_days", dtype=Float32),
        Field(name="is_new_device", dtype=Int64),
        Field(name="ip_reputation_score", dtype=Float32),
        Field(name="hour_of_day", dtype=Int64),
        Field(name="is_unusual_hour", dtype=Int64),
        Field(name="location_changed", dtype=Int64),
        Field(name="mfa_used", dtype=Int64),
        Field(name="vpn_detected", dtype=Int64),
        Field(name="success", dtype=Int64),
    ],
    source=login_features_source,
    online=True,
    tags={"team": "identity", "version": "v1"},
)

# =============================================================================
# Feature Services
# =============================================================================

# Risk scoring feature service - all features needed for risk scoring
risk_scoring_service = FeatureService(
    name="risk_scoring",
    features=[user_login_features],
    description="Features for identity risk scoring model",
    tags={"model": "risk_scorer", "version": "v1"},
)
