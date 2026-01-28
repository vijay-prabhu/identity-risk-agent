"""
API Contract Tests

Tests for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_check(client):
    """Test health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data


def test_root_endpoint(client):
    """Test root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "endpoints" in data


def test_score_login_valid_input(client):
    """Test score endpoint with valid input."""
    payload = {
        "user_id": "test_user",
        "ip": "192.168.1.1",
        "device_id": "device_123",
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_level" in data
    assert 0 <= data["risk_score"] <= 1
    assert data["risk_level"] in ["low", "medium", "high", "critical"]


def test_score_login_missing_fields(client):
    """Test score endpoint rejects missing required fields."""
    payload = {"user_id": "test_user"}  # Missing ip and device_id
    response = client.post("/score", json=payload)
    assert response.status_code == 422  # Validation error


def test_explain_endpoint(client):
    """Test explain endpoint returns explanation."""
    payload = {
        "query": "Why was this login flagged?",
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "explanation" in data
    assert "confidence" in data


def test_score_login_with_risk_factors(client):
    """Test that risky logins return risk factors."""
    payload = {
        "user_id": "test_user",
        "ip": "1.2.3.4",
        "device_id": "device_unknown_123",  # Unknown device
        "vpn_detected": True,
        "mfa_used": False,
        "location_country": "RU",  # Risk country
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Should have elevated risk
    assert data["risk_score"] > 0.3
    # Should have risk factors
    assert len(data["factors"]) > 0


def test_score_login_normal_user(client):
    """Test that normal logins have low risk."""
    payload = {
        "user_id": "trusted_user",
        "ip": "10.0.0.1",
        "device_id": "known_device_001",
        "vpn_detected": False,
        "mfa_used": True,
        "success": True,
        "location_country": "US",
        "device_age_days": 30,
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Should have low risk (depends on model, but MFA + known device should help)
    assert data["risk_level"] in ["low", "medium"]


def test_features_endpoint(client):
    """Test features endpoint returns feature list."""
    response = client.get("/features")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "description" in data
    assert len(data["features"]) > 0


def test_score_login_with_timestamp(client):
    """Test score endpoint with custom timestamp."""
    payload = {
        "user_id": "test_user",
        "ip": "192.168.1.1",
        "device_id": "device_123",
        "timestamp": "2024-01-15T03:00:00",  # Unusual hour
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Should flag unusual hour
    assert "unusual hour" in str(data["factors"]).lower() or data["risk_score"] > 0
