"""Tests for Redis caching of EEG analysis results."""

import hashlib
from unittest.mock import Mock

import pytest


class TestRedisCaching:
    """Test Redis caching for repeated EEG analyses."""

    @pytest.fixture
    def mock_qc_controller(self):
        """Mock QC controller."""
        mock_controller = Mock()
        mock_controller.run_full_qc_pipeline = Mock(
            return_value={
                "quality_metrics": {
                    "bad_channels": ["T3"],
                    "bad_channel_ratio": 0.05,
                    "abnormality_score": 0.3,
                    "artifact_ratio": 0.1,
                    "quality_grade": "GOOD",
                },
                "processing_info": {"confidence": 0.85},
            }
        )
        return mock_controller

    @pytest.fixture
    def client_for_cache_tests(self, client_with_cache, mock_qc_controller):
        """Enhance the client_with_cache fixture with QC controller."""
        import brain_go_brrr.api.main as api_main
        import brain_go_brrr.api.routers.qc as qc_router

        # Inject the mock QC controller
        api_main.qc_controller = mock_qc_controller
        qc_router.qc_controller = mock_qc_controller

        yield client_with_cache

    def test_cache_hit_on_repeated_analysis(
        self,
        client_for_cache_tests,
        valid_edf_content,
        dummy_cache,
        mock_qc_controller,
    ):
        """Test that repeated analysis uses cache instead of reprocessing."""
        files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}

        # First call - should store in cache
        response1 = client_for_cache_tests.post("/api/v1/eeg/analyze", files=files)
        assert response1.status_code == 200
        result1 = response1.json()
        assert "cached" not in result1 or not result1.get("cached")

        # Verify cache was set
        assert any(call[0] == "set" for call in dummy_cache.mock_calls), (
            "First call must prime the cache"
        )

        # Reset mocks
        dummy_cache.reset_mock()
        mock_qc_controller.run_full_qc_pipeline.reset_mock()

        # Second call - should hit cache
        response2 = client_for_cache_tests.post("/api/v1/eeg/analyze", files=files)
        assert response2.status_code == 200
        result2 = response2.json()
        assert result2.get("cached") is True

        # Verify cache was used, not set again
        assert not any(call[0] == "set" for call in dummy_cache.mock_calls), (
            "Second call should use cache"
        )
        assert any(call[0] == "get" for call in dummy_cache.mock_calls), (
            "Second call should get from cache"
        )

        # QC pipeline should NOT be called again
        mock_qc_controller.run_full_qc_pipeline.assert_not_called()

    def test_cache_key_generation(self, client_for_cache_tests, valid_edf_content, dummy_cache):
        """Test that cache keys are properly generated from file content."""
        files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}
        response = client_for_cache_tests.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200

        # Check that generate_cache_key was called with correct params
        gen_key_calls = [call for call in dummy_cache.mock_calls if call[0] == "generate_cache_key"]
        assert len(gen_key_calls) == 1
        assert gen_key_calls[0][1] == valid_edf_content  # content
        assert gen_key_calls[0][2] == "basic"  # analysis_type

        # Check that set was called with the generated key
        expected_key = f"eeg_analysis:{hashlib.sha256(valid_edf_content).hexdigest()}:basic"
        set_calls = dummy_cache.set_calls()
        assert len(set_calls) == 1
        assert set_calls[0][1] == expected_key  # key parameter

    def test_cache_expiration(self, client_for_cache_tests, valid_edf_content, dummy_cache):
        """Test that cached results have proper expiration."""
        files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}
        response = client_for_cache_tests.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200

        # Check that set was called with expiry
        set_calls = dummy_cache.set_calls()
        assert len(set_calls) == 1
        # set_calls format: ('set', key, value, kwargs)
        kwargs = set_calls[0][3]
        assert kwargs.get("expiry") == 3600  # 1 hour

    def test_cache_invalidation_on_different_file(
        self, client_for_cache_tests, dummy_cache, valid_edf_content
    ):
        """Test that different files generate different cache keys."""
        # First file
        files1 = {"edf_file": ("test1.edf", valid_edf_content, "application/octet-stream")}
        response1 = client_for_cache_tests.post("/api/v1/eeg/analyze", files=files1)
        assert response1.status_code == 200

        # Create a second valid EDF file with different data
        import tempfile

        import numpy as np
        from pyedflib import EdfWriter

        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            writer = EdfWriter(tmp.name, n_channels=1)
            writer.setSignalHeader(
                0,
                {
                    "label": "EEG C3-C4",  # Different channel name
                    "dimension": "uV",
                    "sample_frequency": 256,
                    "physical_max": 300,  # Different range
                    "physical_min": -300,
                    "digital_max": 2047,
                    "digital_min": -2048,
                    "prefilter": "HP:0.5Hz LP:70Hz",
                    "transducer": "AgAgCl electrode",
                },
            )
            # Write different data pattern
            data = np.sin(np.linspace(0, 2 * np.pi, 256)) * 1000
            writer.writeDigitalSamples(data.astype(np.int32))
            writer.close()

            # Read the different file
            from pathlib import Path

            different_edf_content = Path(tmp.name).read_bytes()
            Path(tmp.name).unlink()

        # Second file with different content
        files2 = {"edf_file": ("test2.edf", different_edf_content, "application/octet-stream")}
        response2 = client_for_cache_tests.post("/api/v1/eeg/analyze", files=files2)
        assert response2.status_code == 200

        # Check that two different cache keys were used
        set_calls = dummy_cache.set_calls()
        cache_keys = [call[1] for call in set_calls]  # Extract keys

        assert len(cache_keys) == 2, f"Expected 2 set calls, got {len(cache_keys)}"
        assert len(set(cache_keys)) == 2, (
            f"Different files must generate different cache keys. Keys: {cache_keys}"
        )

    def test_cache_disabled_when_redis_unavailable(
        self, client_for_cache_tests, valid_edf_content, dummy_cache
    ):
        """Test that analysis works even when cache is unavailable."""
        # Simulate Redis being disconnected
        dummy_cache.connected = False

        files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}
        response = client_for_cache_tests.post("/api/v1/eeg/analyze", files=files)

        # Should still work without cache
        assert response.status_code == 200
        result = response.json()
        assert "flag" in result
        assert "confidence" in result
        # When cache is disabled, cached should be False if present
        assert result.get("cached", False) is False

        # Cache operations should not be called when disconnected
        assert not any(call[0] == "set" for call in dummy_cache.mock_calls)
        assert not any(call[0] == "get" for call in dummy_cache.mock_calls)

    def test_detailed_endpoint_caching(
        self, client_for_cache_tests, valid_edf_content, dummy_cache
    ):
        """Test caching on the detailed analysis endpoint."""
        files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}
        response = client_for_cache_tests.post(
            "/api/v1/eeg/analyze/detailed", files=files, data={"include_report": "false"}
        )

        assert response.status_code == 200

        # Should use "detailed" analysis type for cache key
        gen_key_calls = [call for call in dummy_cache.mock_calls if call[0] == "generate_cache_key"]
        assert len(gen_key_calls) == 1
        assert gen_key_calls[0][2] == "detailed"  # analysis_type

    @pytest.mark.parametrize(
        "endpoint,field_name,analysis_type",
        [
            ("/api/v1/eeg/analyze", "edf_file", "basic"),
            ("/api/v1/eeg/analyze/detailed", "edf_file", "detailed"),  # Both use edf_file now
        ],
    )
    def test_cache_behavior_across_endpoints(
        self,
        client_for_cache_tests,
        valid_edf_content,
        dummy_cache,
        endpoint,
        field_name,
        analysis_type,
    ):
        """Test that cache works correctly for both endpoints."""
        files = {field_name: ("test.edf", valid_edf_content, "application/octet-stream")}
        data = {"include_report": "false"} if "detailed" in endpoint else None

        # First request - should cache
        response1 = client_for_cache_tests.post(endpoint, files=files, data=data)
        assert response1.status_code == 200
        assert any(call[0] == "set" for call in dummy_cache.mock_calls), (
            f"{endpoint} should cache results"
        )

        # Reset and make second request
        dummy_cache.reset_mock()
        response2 = client_for_cache_tests.post(endpoint, files=files, data=data)
        assert response2.status_code == 200

        # Both endpoints should use cache on second call
        assert not any(call[0] == "set" for call in dummy_cache.mock_calls), (
            "Should not set cache again"
        )
        assert any(call[0] == "get" for call in dummy_cache.mock_calls), "Should get from cache"

    def test_cache_stats_endpoint(self, client_for_cache_tests, dummy_cache):
        """Test the cache stats endpoint."""
        response = client_for_cache_tests.get("/api/v1/cache/stats")
        assert response.status_code == 200

        stats = response.json()
        assert stats["connected"] is True
        assert "total_keys" in stats
        assert "memory_usage" in stats
