"""
Multi-Tenant Isolation for Identity Risk Platform

Provides tenant isolation for:
- API requests via middleware
- Feature store queries
- Model serving
- Audit logging
"""

import logging
from contextvars import ContextVar
from functools import wraps
from typing import Callable, Dict, List, Optional, Any

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Context variable for current tenant
_current_tenant: ContextVar[Optional[str]] = ContextVar("current_tenant", default=None)


# =============================================================================
# Tenant Context Management
# =============================================================================


def get_current_tenant() -> Optional[str]:
    """Get the current tenant from context."""
    return _current_tenant.get()


def set_current_tenant(tenant_id: str) -> None:
    """Set the current tenant in context."""
    _current_tenant.set(tenant_id)


class TenantContext:
    """
    Context manager for tenant isolation.

    Usage:
        with TenantContext("tenant_a"):
            # All operations here are scoped to tenant_a
            features = feature_store.get_features(user_id)
    """

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.previous_tenant = None

    def __enter__(self):
        self.previous_tenant = get_current_tenant()
        set_current_tenant(self.tenant_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_tenant:
            set_current_tenant(self.previous_tenant)
        else:
            _current_tenant.set(None)
        return False


def tenant_required(func: Callable) -> Callable:
    """
    Decorator that requires a tenant to be set in context.

    Usage:
        @tenant_required
        def get_user_features(user_id: str):
            tenant = get_current_tenant()
            # ... tenant-scoped operation
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        tenant = get_current_tenant()
        if tenant is None:
            raise ValueError("No tenant set in context. Use TenantContext or set_current_tenant().")
        return func(*args, **kwargs)

    return wrapper


# =============================================================================
# API Key to Tenant Mapping
# =============================================================================

# In production, this would be in a database
# Format: {api_key: {"tenant_id": str, "name": str, "permissions": list}}
API_KEY_TENANT_MAP: Dict[str, Dict[str, Any]] = {
    "sk_test_tenant_a_123": {
        "tenant_id": "tenant_a",
        "name": "Tenant A (Test)",
        "permissions": ["read", "write", "score"],
    },
    "sk_test_tenant_b_456": {
        "tenant_id": "tenant_b",
        "name": "Tenant B (Test)",
        "permissions": ["read", "score"],
    },
    "sk_test_tenant_c_789": {
        "tenant_id": "tenant_c",
        "name": "Tenant C (Test)",
        "permissions": ["read", "write", "score", "admin"],
    },
    # Default/demo key for testing without authentication
    "sk_demo": {
        "tenant_id": "default",
        "name": "Demo Tenant",
        "permissions": ["read", "score"],
    },
}


def validate_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """
    Validate an API key and return tenant info.

    Args:
        api_key: The API key to validate

    Returns:
        Tenant info dict or None if invalid
    """
    return API_KEY_TENANT_MAP.get(api_key)


def get_tenant_from_api_key(api_key: str) -> Optional[str]:
    """
    Get tenant ID from an API key.

    Args:
        api_key: The API key

    Returns:
        Tenant ID or None if invalid
    """
    tenant_info = validate_api_key(api_key)
    if tenant_info:
        return tenant_info["tenant_id"]
    return None


# =============================================================================
# FastAPI Middleware
# =============================================================================


class TenantMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for tenant isolation.

    Extracts tenant from:
    1. X-API-Key header (maps to tenant)
    2. X-Tenant-ID header (direct tenant ID, for testing)
    3. Query parameter tenant_id
    4. Request body tenant_id field

    Sets the tenant in context for the duration of the request.
    """

    def __init__(self, app, require_auth: bool = False):
        super().__init__(app)
        self.require_auth = require_auth

    async def dispatch(self, request: Request, call_next):
        tenant_id = None

        # Try to get tenant from various sources
        # 1. X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            tenant_id = get_tenant_from_api_key(api_key)
            if tenant_id is None and self.require_auth:
                raise HTTPException(status_code=401, detail="Invalid API key")

        # 2. X-Tenant-ID header (for testing/internal use)
        if not tenant_id:
            tenant_id = request.headers.get("X-Tenant-ID")

        # 3. Query parameter
        if not tenant_id:
            tenant_id = request.query_params.get("tenant_id")

        # Default tenant if none specified
        if not tenant_id:
            tenant_id = "default"

        # Set tenant in context
        set_current_tenant(tenant_id)

        # Add tenant to request state for easy access in endpoints
        request.state.tenant_id = tenant_id

        # Process request
        response = await call_next(request)

        # Add tenant to response headers for debugging
        response.headers["X-Tenant-ID"] = tenant_id

        return response


# =============================================================================
# Tenant-Scoped Operations
# =============================================================================


class TenantScopedStore:
    """
    Base class for tenant-scoped data stores.

    Ensures all operations are isolated to the current tenant.
    """

    def _get_tenant_key(self, key: str) -> str:
        """Create a tenant-scoped key."""
        tenant = get_current_tenant()
        if tenant:
            return f"{tenant}:{key}"
        return key

    def _filter_by_tenant(self, data: List[Dict], tenant_id: Optional[str] = None) -> List[Dict]:
        """Filter data by tenant."""
        tenant = tenant_id or get_current_tenant()
        if tenant and tenant != "default":
            return [d for d in data if d.get("tenant_id") == tenant]
        return data


# =============================================================================
# Audit Logging
# =============================================================================


class TenantAuditLogger:
    """
    Audit logger for tenant operations.

    Logs all operations with tenant context for compliance and debugging.
    """

    def __init__(self, name: str = "tenant_audit"):
        self.logger = logging.getLogger(name)

    def log_operation(
        self,
        operation: str,
        user_id: Optional[str] = None,
        details: Optional[Dict] = None,
        level: str = "INFO",
    ) -> None:
        """
        Log a tenant-scoped operation.

        Args:
            operation: Name of the operation (e.g., "score_login", "get_features")
            user_id: Optional user ID involved in the operation
            details: Optional additional details
            level: Log level
        """
        tenant = get_current_tenant()
        log_data = {
            "tenant_id": tenant,
            "operation": operation,
            "user_id": user_id,
            "details": details or {},
        }

        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(f"AUDIT: {log_data}")


# Global audit logger instance
audit_logger = TenantAuditLogger()


def log_tenant_operation(operation: str):
    """
    Decorator to log tenant operations.

    Usage:
        @log_tenant_operation("score_login")
        def score_login(user_id: str, ...):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to extract user_id from kwargs
            user_id = kwargs.get("user_id") or (args[0] if args else None)
            audit_logger.log_operation(operation, user_id=str(user_id) if user_id else None)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Tenant Configuration
# =============================================================================


class TenantConfig:
    """
    Per-tenant configuration.

    Allows different settings per tenant (e.g., risk thresholds, features enabled).
    """

    # Default configuration
    DEFAULT_CONFIG = {
        "risk_threshold_high": 0.7,
        "risk_threshold_critical": 0.9,
        "mfa_required_on_high_risk": True,
        "block_critical_risk": True,
        "features_enabled": ["scoring", "explanation"],
    }

    # Tenant-specific overrides
    TENANT_CONFIGS: Dict[str, Dict] = {
        "tenant_a": {
            "risk_threshold_high": 0.6,  # More sensitive
            "features_enabled": ["scoring", "explanation", "quarantine"],
        },
        "tenant_b": {
            "risk_threshold_critical": 0.95,  # Less aggressive blocking
            "block_critical_risk": False,
        },
    }

    @classmethod
    def get_config(cls, tenant_id: Optional[str] = None) -> Dict:
        """
        Get configuration for a tenant.

        Args:
            tenant_id: Tenant ID (uses current tenant if None)

        Returns:
            Merged configuration dict
        """
        tenant = tenant_id or get_current_tenant() or "default"

        # Start with defaults
        config = cls.DEFAULT_CONFIG.copy()

        # Apply tenant-specific overrides
        if tenant in cls.TENANT_CONFIGS:
            config.update(cls.TENANT_CONFIGS[tenant])

        return config

    @classmethod
    def get_risk_thresholds(cls, tenant_id: Optional[str] = None) -> Dict[str, float]:
        """Get risk thresholds for a tenant."""
        config = cls.get_config(tenant_id)
        return {
            "high": config["risk_threshold_high"],
            "critical": config["risk_threshold_critical"],
        }
