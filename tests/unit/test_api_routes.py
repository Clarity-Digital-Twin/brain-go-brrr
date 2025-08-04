"""Tests for API routes configuration."""

import pytest

from brain_go_brrr.api.routes import APIRoutes, routes


class TestAPIRoutes:
    """Test API routes configuration."""

    def test_api_routes_instance_exists(self):
        """Test that the routes singleton instance exists."""
        assert routes is not None
        assert isinstance(routes, APIRoutes)

    def test_api_v1_prefix(self):
        """Test API v1 prefix is correctly defined."""
        assert routes.API_V1 == "/api/v1"

    def test_health_endpoints(self):
        """Test health check endpoints."""
        assert routes.HEALTH == "/api/v1/health"
        assert routes.READY == "/api/v1/ready"

    def test_job_management_endpoints(self):
        """Test job management endpoints."""
        assert routes.JOBS_CREATE == "/api/v1/jobs/create"
        assert routes.JOBS_STATUS == "/api/v1/jobs/{job_id}/status"
        assert routes.JOBS_LIST == "/api/v1/jobs"

    def test_eeg_analysis_endpoints(self):
        """Test EEG analysis endpoints."""
        assert routes.QC_ANALYZE == "/api/v1/eeg/analyze"
        assert routes.SLEEP_ANALYZE == "/api/v1/eeg/sleep/analyze"

    def test_queue_endpoints(self):
        """Test queue management endpoints."""
        assert routes.QUEUE_STATUS == "/api/v1/queue/status"

    def test_cache_endpoints(self):
        """Test cache operation endpoints."""
        assert routes.CACHE_STATUS == "/api/v1/cache/status"

    def test_documentation_endpoints(self):
        """Test documentation endpoints."""
        assert routes.DOCS == "/api/docs"
        assert routes.REDOC == "/api/redoc"

    def test_routes_immutability(self):
        """Test that APIRoutes is immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            routes.API_V1 = "/api/v2"

    def test_job_status_route_formatting(self):
        """Test that job status route can be formatted with job_id."""
        job_id = "test-job-123"
        formatted = routes.JOBS_STATUS.format(job_id=job_id)
        assert formatted == f"/api/v1/jobs/{job_id}/status"

    def test_all_routes_start_correctly(self):
        """Test that all routes start with expected prefixes."""
        # Get all route attributes
        route_attrs = [attr for attr in dir(routes) if attr.isupper() and not attr.startswith('_')]
        
        for attr_name in route_attrs:
            route_value = getattr(routes, attr_name)
            if isinstance(route_value, str) and route_value.startswith("/"):
                # Skip API_V1 constant and documentation routes
                if attr_name == "API_V1":
                    assert route_value == "/api/v1"
                elif attr_name in ["DOCS", "REDOC"]:
                    assert route_value.startswith("/api/")
                else:
                    # All other routes should start with /api/v1/
                    assert route_value.startswith("/api/v1/"), f"{attr_name} doesn't start with /api/v1/"

    def test_create_new_routes_instance(self):
        """Test creating a new APIRoutes instance."""
        new_routes = APIRoutes()
        
        # Should have same values as singleton
        assert new_routes.API_V1 == routes.API_V1
        assert new_routes.HEALTH == routes.HEALTH
        assert new_routes.QC_ANALYZE == routes.QC_ANALYZE
        
        # But should be different instance
        assert new_routes is not routes