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
    assert data["status"] == "healthy"


def test_root_endpoint(client):
    """Test root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data


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
