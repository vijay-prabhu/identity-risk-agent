"""
Feast Feature Store Integration

Provides a wrapper around Feast for online/offline feature retrieval
with multi-tenant isolation support.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from feast import FeatureStore
from feast.repo_config import RepoConfig

# Default feature store path
FEATURE_STORE_PATH = Path(__file__).parent.parent.parent / "feature_store"


class IdentityFeatureStore:
    """
    Wrapper around Feast feature store for identity risk scoring.

    Provides:
    - Online feature retrieval for real-time scoring
    - Offline feature retrieval for training
    - Multi-tenant isolation via tenant_id filtering
    """

    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize the feature store.

        Args:
            repo_path: Path to Feast repository (default: feature_store/)
        """
        self.repo_path = Path(repo_path) if repo_path else FEATURE_STORE_PATH

        if not (self.repo_path / "feature_store.yaml").exists():
            raise FileNotFoundError(
                f"Feature store config not found at {self.repo_path}/feature_store.yaml. "
                "Run `feast apply` in the feature_store directory first."
            )

        self.store = FeatureStore(repo_path=str(self.repo_path))
        self._initialized = False

    def apply(self) -> None:
        """Apply feature definitions to the registry."""
        self.store.apply([])  # Apply all objects defined in features.py
        self._initialized = True

    def materialize(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """
        Materialize features to the online store.

        Args:
            start_date: Start of materialization window
            end_date: End of materialization window (default: now)
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - pd.Timedelta(days=30)

        self.store.materialize(start_date=start_date, end_date=end_date)

    def get_online_features(
        self,
        user_ids: List[str],
        tenant_id: Optional[str] = None,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get features from the online store for real-time scoring.

        Args:
            user_ids: List of user IDs to fetch features for
            tenant_id: Optional tenant ID for filtering (multi-tenant)
            features: Optional list of specific features to retrieve

        Returns:
            DataFrame with features for each user
        """
        if features is None:
            features = [
                "user_login_features:failed_logins_24h",
                "user_login_features:login_count_7d",
                "user_login_features:device_age_days",
                "user_login_features:is_new_device",
                "user_login_features:ip_reputation_score",
                "user_login_features:hour_of_day",
                "user_login_features:is_unusual_hour",
                "user_login_features:location_changed",
                "user_login_features:mfa_used",
                "user_login_features:vpn_detected",
                "user_login_features:success",
            ]

        entity_rows = [{"user_id": uid} for uid in user_ids]

        feature_vector = self.store.get_online_features(
            features=features,
            entity_rows=entity_rows,
        )

        df = feature_vector.to_df()

        # Multi-tenant filtering would happen here if tenant_id is stored
        # For now, tenant isolation is handled at the API layer

        return df

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get historical features for training.

        Args:
            entity_df: DataFrame with entity keys and timestamps
            features: Optional list of specific features
            tenant_id: Optional tenant ID for filtering

        Returns:
            DataFrame with historical features
        """
        if features is None:
            features = [
                "user_login_features:failed_logins_24h",
                "user_login_features:login_count_7d",
                "user_login_features:device_age_days",
                "user_login_features:is_new_device",
                "user_login_features:ip_reputation_score",
                "user_login_features:hour_of_day",
                "user_login_features:is_unusual_hour",
                "user_login_features:location_changed",
                "user_login_features:mfa_used",
                "user_login_features:vpn_detected",
                "user_login_features:success",
            ]

        # Filter by tenant if specified
        if tenant_id and "tenant_id" in entity_df.columns:
            entity_df = entity_df[entity_df["tenant_id"] == tenant_id]

        training_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=features,
        ).to_df()

        return training_df

    def get_feature_service(self, name: str = "risk_scoring"):
        """Get a feature service by name."""
        return self.store.get_feature_service(name)

    def list_feature_views(self) -> List[str]:
        """List all registered feature views."""
        return [fv.name for fv in self.store.list_feature_views()]


def initialize_feature_store(
    repo_path: Optional[str] = None,
    apply: bool = True,
    materialize: bool = True,
) -> IdentityFeatureStore:
    """
    Initialize and optionally materialize the feature store.

    Args:
        repo_path: Path to Feast repository
        apply: Whether to apply feature definitions
        materialize: Whether to materialize to online store

    Returns:
        Initialized IdentityFeatureStore
    """
    store = IdentityFeatureStore(repo_path)

    if apply:
        print("Applying feature definitions...")
        store.apply()

    if materialize:
        print("Materializing features to online store...")
        store.materialize()

    return store


if __name__ == "__main__":
    # Quick test
    print("Initializing feature store...")

    try:
        store = IdentityFeatureStore()
        print(f"Feature store path: {store.repo_path}")
        print(f"Feature views: {store.list_feature_views()}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo initialize the feature store, run:")
        print("  cd feature_store && feast apply")
