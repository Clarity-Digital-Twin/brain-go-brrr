"""Integration test for API root endpoint."""

import pytest
from fastapi.testclient import TestClient

from brain_go_brrr.api.main import app


class TestAPIRoot:
    """Test the root endpoint and basic API functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test that root endpoint returns API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        # Check basic structure
        assert "message" in data
        assert "Welcome to Brain-Go-Brrr API" in data["message"]
        assert "version" in data
        assert "endpoints" in data

        # Check endpoints are listed
        endpoints = data["endpoints"]
        assert "docs" in endpoints
        assert "redoc" in endpoints
        assert "health" in endpoints
        assert "qc_analyze" in endpoints
        assert "sleep_analyze" in endpoints
        assert "jobs" in endpoints

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "service" in data
        assert data["service"] == "brain-go-brrr-api"

    def test_ready_endpoint(self, client):
        """Test readiness check endpoint."""
        response = client.get("/api/v1/ready")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] == "ready"
        assert "timestamp" in data
