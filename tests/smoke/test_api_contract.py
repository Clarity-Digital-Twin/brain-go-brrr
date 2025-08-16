"""Smoke test for API contract endpoints."""

from fastapi.testclient import TestClient

from brain_go_brrr.api.app import create_app


def test_api_health_ready_root():
    """Test that core API endpoints respond correctly."""
    # Use TestClient for in-process testing (no ports, no subprocess)
    app = create_app()
    
    with TestClient(app) as client:
        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "brain-go-brrr"
        assert "version" in data
        
        # Test health endpoint at root
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        
        # Test ready endpoint at root
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        
        # Test versioned health endpoint (allow 404 if not exported)
        response = client.get("/api/v1/health")
        assert response.status_code in (200, 404)  # Allow absence if not routed
