"""Unit tests to boost API coverage with FastAPI TestClient.

These are simple smoke tests that verify basic endpoint behavior
without requiring real services or data.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from brain_go_brrr.api.app import create_app


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    app = create_app()
    return TestClient(app)


class TestAPIEndpointSmokes:
    """Basic smoke tests for API endpoints to boost coverage."""

    def test_health_check(self, client):
        """Test health endpoint returns OK."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "ok"

    def test_root_redirect(self, client):
        """Test root redirects to docs."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == status.HTTP_307_TEMPORARY_REDIRECT
        assert response.headers["location"] == "/docs"

    @patch("brain_go_brrr.api.routers.resources.psutil.cpu_percent")
    @patch("brain_go_brrr.api.routers.resources.psutil.virtual_memory")
    def test_resources_endpoint(self, mock_memory, mock_cpu, client):
        """Test resources endpoint returns system metrics."""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0, available=1024*1024*1024)
        
        response = client.get("/api/v1/resources")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "cpu_percent" in data
        assert "memory_percent" in data

    def test_cache_stats_no_cache(self, client):
        """Test cache stats with no cache configured."""
        response = client.get("/api/v1/cache/stats")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["type"] == "memory"
        assert data["stats"]["hits"] == 0
        assert data["stats"]["misses"] == 0

    def test_invalid_endpoint_404(self, client):
        """Test that invalid endpoints return 404."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch("brain_go_brrr.api.routers.eegpt.Path")
    def test_eegpt_analyze_missing_file(self, mock_path, client):
        """Test EEGPT analyze with missing file."""
        mock_path.return_value.exists.return_value = False
        
        response = client.post(
            "/api/v1/eeg/analyze",
            json={"file_path": "/fake/path.edf"}
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_eegpt_health(self, client):
        """Test EEGPT health endpoint."""
        response = client.get("/api/v1/eeg/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    @patch("brain_go_brrr.api.routers.qc.Path")
    def test_qc_analyze_missing_file(self, mock_path, client):
        """Test QC analyze with missing file."""
        mock_path.return_value.exists.return_value = False
        
        response = client.post(
            "/api/v1/qc/analyze",
            json={"file_path": "/fake/path.edf"}
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_qc_health(self, client):
        """Test QC health endpoint."""
        response = client.get("/api/v1/qc/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"

    def test_cache_clear(self, client):
        """Test cache clear endpoint."""
        response = client.post("/api/v1/cache/clear")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "success"
        assert "entries_cleared" in data

    @patch("brain_go_brrr.api.routers.eegpt.torch.cuda.is_available")
    def test_gpu_status(self, mock_cuda, client):
        """Test GPU status in EEGPT health."""
        mock_cuda.return_value = False
        
        response = client.get("/api/v1/eeg/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["gpu_available"] is False

    def test_api_version_header(self, client):
        """Test API version header is present."""
        response = client.get("/health")
        assert "X-API-Version" in response.headers

    def test_cors_headers(self, client):
        """Test CORS headers are configured."""
        response = client.options("/health")
        assert response.status_code == status.HTTP_200_OK

    def test_request_id_header(self, client):
        """Test request ID header is generated."""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers

    def test_content_type_json(self, client):
        """Test endpoints return JSON content type."""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"