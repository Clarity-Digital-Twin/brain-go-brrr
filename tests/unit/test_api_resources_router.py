"""CLEAN tests for API resources router - FastAPI TestClient."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from brain_go_brrr.api.routers.resources import router


class TestResourcesRouterClean:
    """Test resources router endpoints with TestClient."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with resources router."""
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_get_memory_resources(self, client):
        """Test getting memory resources."""
        # Mock memory stats
        with patch("psutil.virtual_memory") as mock_memory:
            mock_obj = MagicMock()
            mock_obj.used = 8 * 1024**3  # 8 GB used
            mock_obj.available = 8 * 1024**3  # 8 GB available
            mock_obj.percent = 50.0
            mock_obj.total = 16 * 1024**3  # 16 GB total
            mock_obj.free = 8 * 1024**3
            mock_memory.return_value = mock_obj

            response = client.get("/resources/memory")

        assert response.status_code == 200
        data = response.json()

        assert "used" in data
        assert "available" in data
        assert "percent" in data
        assert data["percent"] == 50.0
        assert "total" in data
        assert "free" in data

    def test_get_gpu_resources_with_gputil(self, client):
        """Test getting GPU resources when GPUtil is available."""
        # Mock GPU object
        mock_gpu = MagicMock()
        mock_gpu.id = 0
        mock_gpu.name = "RTX 3090"
        mock_gpu.memoryUsed = 1024  # MB
        mock_gpu.memoryTotal = 24576  # MB
        mock_gpu.memoryFree = 23552  # MB
        mock_gpu.load = 0.15  # 15% load
        mock_gpu.temperature = 45  # Celsius

        with (
            patch("brain_go_brrr.api.routers.resources.HAS_GPUTIL", True),
            patch("brain_go_brrr.api.routers.resources.GPUtil") as mock_gputil,
        ):
            mock_gputil.getGPUs.return_value = [mock_gpu]
            response = client.get("/resources/gpu")

        assert response.status_code == 200
        data = response.json()

        assert "gpus" in data
        assert len(data["gpus"]) == 1

        gpu_info = data["gpus"][0]
        assert gpu_info["id"] == 0
        assert gpu_info["name"] == "RTX 3090"
        assert gpu_info["memory_used"] == 1024
        assert gpu_info["memory_total"] == 24576
        assert gpu_info["gpu_load"] == 15.0  # Converted to percentage

    def test_get_gpu_resources_without_gputil(self, client):
        """Test getting GPU resources when GPUtil is not installed."""
        with patch("brain_go_brrr.api.routers.resources.HAS_GPUTIL", False):
            response = client.get("/resources/gpu")

        assert response.status_code == 200
        data = response.json()

        assert "gpus" in data
        assert data["gpus"] == []
        assert "message" in data
        assert "GPUtil not installed" in data["message"]

    def test_get_gpu_resources_with_error(self, client):
        """Test GPU resources when GPU access fails."""
        with (
            patch("brain_go_brrr.api.routers.resources.HAS_GPUTIL", True),
            patch("brain_go_brrr.api.routers.resources.GPUtil") as mock_gputil,
        ):
            mock_gputil.getGPUs.side_effect = RuntimeError("No GPU available")
            response = client.get("/resources/gpu")

        assert response.status_code == 200
        data = response.json()

        assert "gpus" in data
        assert data["gpus"] == []
        assert "error" in data
        assert "GPU not available" in data["error"]

    def test_get_memory_high_usage(self, client):
        """Test memory resources with high usage."""
        with patch("psutil.virtual_memory") as mock_memory:
            mock_obj = MagicMock()
            mock_obj.used = 15 * 1024**3  # 15 GB used
            mock_obj.available = 1 * 1024**3  # 1 GB available
            mock_obj.percent = 93.75
            mock_obj.total = 16 * 1024**3  # 16 GB total
            mock_obj.free = 1 * 1024**3
            mock_memory.return_value = mock_obj

            response = client.get("/resources/memory")

        assert response.status_code == 200
        data = response.json()
        assert data["percent"] == 93.75
        assert data["available"] == 1 * 1024**3

    def test_get_gpu_multiple_gpus(self, client):
        """Test getting resources for multiple GPUs."""
        # Mock two GPUs
        mock_gpu1 = MagicMock()
        mock_gpu1.id = 0
        mock_gpu1.name = "RTX 3090"
        mock_gpu1.memoryUsed = 2048
        mock_gpu1.memoryTotal = 24576
        mock_gpu1.memoryFree = 22528
        mock_gpu1.load = 0.25
        mock_gpu1.temperature = 50

        mock_gpu2 = MagicMock()
        mock_gpu2.id = 1
        mock_gpu2.name = "RTX 3080"
        mock_gpu2.memoryUsed = 512
        mock_gpu2.memoryTotal = 10240
        mock_gpu2.memoryFree = 9728
        mock_gpu2.load = 0.10
        mock_gpu2.temperature = 42

        with (
            patch("brain_go_brrr.api.routers.resources.HAS_GPUTIL", True),
            patch("brain_go_brrr.api.routers.resources.GPUtil") as mock_gputil,
        ):
            mock_gputil.getGPUs.return_value = [mock_gpu1, mock_gpu2]
            response = client.get("/resources/gpu")

        assert response.status_code == 200
        data = response.json()

        assert len(data["gpus"]) == 2
        assert data["gpus"][0]["name"] == "RTX 3090"
        assert data["gpus"][1]["name"] == "RTX 3080"
        assert data["gpus"][0]["gpu_load"] == 25.0
        assert data["gpus"][1]["gpu_load"] == 10.0

    def test_get_memory_low_memory_scenario(self, client):
        """Test memory resources in low memory scenario."""
        with patch("psutil.virtual_memory") as mock_memory:
            mock_obj = MagicMock()
            mock_obj.used = 512 * 1024**2  # 512 MB used
            mock_obj.available = 512 * 1024**2  # 512 MB available
            mock_obj.percent = 50.0
            mock_obj.total = 1024 * 1024**2  # 1 GB total
            mock_obj.free = 512 * 1024**2
            mock_memory.return_value = mock_obj

            response = client.get("/resources/memory")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1024 * 1024**2  # 1 GB system

    def test_get_gpu_attribute_error(self, client):
        """Test GPU resources when GPUtil has attribute issues."""
        with (
            patch("brain_go_brrr.api.routers.resources.HAS_GPUTIL", True),
            patch("brain_go_brrr.api.routers.resources.GPUtil") as mock_gputil,
        ):
            mock_gputil.getGPUs.side_effect = AttributeError("getGPUs not found")
            response = client.get("/resources/gpu")

        assert response.status_code == 200
        data = response.json()
        assert data["gpus"] == []
        assert "error" in data
