"""Clean tests for resources router - test real logic without torch import issues."""

import importlib.util
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient


def load_resources_module():
    """Load resources module directly without triggering torch import."""
    spec = importlib.util.spec_from_file_location(
        "resources", "src/brain_go_brrr/api/routers/resources.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestResourcesRouter:
    """Test resources monitoring endpoints."""

    def test_get_memory_resources(self):
        """Test memory resources endpoint returns system memory info."""
        resources = load_resources_module()
        app = FastAPI()
        app.include_router(resources.router)
        client = TestClient(app)

        response = client.get("/resources/memory")

        assert response.status_code == 200
        data = response.json()

        # Verify all expected fields are present
        assert "used" in data
        assert "available" in data
        assert "percent" in data
        assert "total" in data
        assert "free" in data

        # Verify values are reasonable
        assert data["total"] > 0
        assert data["used"] > 0
        assert 0 <= data["percent"] <= 100

    def test_get_gpu_resources_no_gputil(self):
        """Test GPU endpoint when GPUtil is not installed."""
        resources = load_resources_module()

        # Mock HAS_GPUTIL as False
        resources.HAS_GPUTIL = False

        app = FastAPI()
        app.include_router(resources.router)
        client = TestClient(app)

        response = client.get("/resources/gpu")

        assert response.status_code == 200
        data = response.json()

        assert data["gpus"] == []
        assert data["message"] == "GPUtil not installed"

    def test_get_gpu_resources_with_gpus(self):
        """Test GPU endpoint when GPUs are available."""
        resources = load_resources_module()

        # Create mock GPU
        mock_gpu = MagicMock()
        mock_gpu.id = 0
        mock_gpu.name = "NVIDIA GeForce RTX 3090"
        mock_gpu.memoryUsed = 1024
        mock_gpu.memoryTotal = 24576
        mock_gpu.memoryFree = 23552
        mock_gpu.load = 0.15
        mock_gpu.temperature = 45

        # Mock GPUtil
        mock_gputil = MagicMock()
        mock_gputil.getGPUs.return_value = [mock_gpu]

        resources.HAS_GPUTIL = True
        resources.GPUtil = mock_gputil

        app = FastAPI()
        app.include_router(resources.router)
        client = TestClient(app)

        response = client.get("/resources/gpu")

        assert response.status_code == 200
        data = response.json()

        assert len(data["gpus"]) == 1
        gpu_info = data["gpus"][0]

        assert gpu_info["id"] == 0
        assert gpu_info["name"] == "NVIDIA GeForce RTX 3090"
        assert gpu_info["memory_used"] == 1024
        assert gpu_info["memory_total"] == 24576
        assert gpu_info["memory_free"] == 23552
        assert gpu_info["gpu_load"] == 15.0
        assert gpu_info["temperature"] == 45

    def test_get_gpu_resources_error(self):
        """Test GPU endpoint handles errors gracefully."""
        resources = load_resources_module()

        # Mock GPUtil to raise error
        mock_gputil = MagicMock()
        mock_gputil.getGPUs.side_effect = RuntimeError("NVIDIA driver not found")

        resources.HAS_GPUTIL = True
        resources.GPUtil = mock_gputil

        app = FastAPI()
        app.include_router(resources.router)
        client = TestClient(app)

        response = client.get("/resources/gpu")

        assert response.status_code == 200
        data = response.json()

        assert data["gpus"] == []
        assert "error" in data
        assert "GPU not available" in data["error"]
        assert "NVIDIA driver not found" in data["error"]

    def test_get_gpu_resources_empty_list(self):
        """Test GPU endpoint when no GPUs found."""
        resources = load_resources_module()

        # Mock GPUtil to return empty list
        mock_gputil = MagicMock()
        mock_gputil.getGPUs.return_value = []

        resources.HAS_GPUTIL = True
        resources.GPUtil = mock_gputil

        app = FastAPI()
        app.include_router(resources.router)
        client = TestClient(app)

        response = client.get("/resources/gpu")

        assert response.status_code == 200
        data = response.json()

        assert data["gpus"] == []
        assert "error" not in data  # No error, just empty list
