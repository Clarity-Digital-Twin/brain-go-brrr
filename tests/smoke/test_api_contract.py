"""Smoke test for API contract endpoints."""

import os
import signal
import subprocess
import time

import httpx


def test_api_health_ready_root():
    """Test that core API endpoints respond correctly."""
    # Start the API server
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "brain_go_brrr.api.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            "8010",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Give server time to start
        time.sleep(2.0)

        # Test each endpoint
        client = httpx.Client(base_url="http://127.0.0.1:8010", timeout=5.0)

        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200, f"Root failed: {response.status_code} - {response.text}"
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert data["status"] == "ok"

        # Test health endpoint
        response = client.get("/health")
        assert (
            response.status_code == 200
        ), f"Health failed: {response.status_code} - {response.text}"
        data = response.json()
        assert data["status"] == "ok"

        # Test ready endpoint
        response = client.get("/ready")
        assert (
            response.status_code == 200
        ), f"Ready failed: {response.status_code} - {response.text}"
        data = response.json()
        assert data["status"] == "ready"

    finally:
        # Clean up - terminate the server
        os.kill(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
