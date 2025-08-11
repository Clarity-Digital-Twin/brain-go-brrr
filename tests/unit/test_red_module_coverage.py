"""Sharp tests targeting red (low coverage) modules.

Based on auditor feedback, these tests specifically target:
1. WindowExtractor edge cases 
2. LinearProbe initialization
3. EDF streaming scenarios
4. Cache operations
5. Error handling paths
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open


class TestWindowExtractorEdgeCases:
    """Sharp tests for window extractor edge cases."""
    
    def test_window_extractor_empty_data(self):
        """Test WindowExtractor handles empty data gracefully."""
        from brain_go_brrr.core.window_extractor import WindowExtractor
        
        extractor = WindowExtractor(window_seconds=4.0, overlap_seconds=2.0)
        
        # Empty array should return empty windows
        empty_data = np.array([]).reshape(0, 0)
        windows = extractor.extract(empty_data, sfreq=256)
        assert len(windows) == 0
    
    def test_window_extractor_single_sample(self):
        """Test WindowExtractor with data shorter than window."""
        from brain_go_brrr.core.window_extractor import WindowExtractor
        
        extractor = WindowExtractor(window_seconds=4.0, overlap_seconds=2.0)
        
        # Single sample - too short for any window
        short_data = np.random.randn(19, 100)  # Less than 4s at 256Hz
        windows = extractor.extract(short_data, sfreq=256)
        assert len(windows) == 0
    
    def test_window_extractor_exact_window(self):
        """Test WindowExtractor with exactly one window of data."""
        from brain_go_brrr.core.window_extractor import WindowExtractor
        
        extractor = WindowExtractor(window_seconds=4.0, overlap_seconds=0.0)
        
        # Exactly 4 seconds at 256Hz
        exact_data = np.random.randn(19, 1024)
        windows = extractor.extract(exact_data, sfreq=256)
        assert len(windows) == 1
        assert windows[0].shape == (19, 1024)


class TestLinearProbeInit:
    """Sharp tests for LinearProbe initialization paths."""
    
    def test_linear_probe_default_init(self):
        """Test LinearProbeHead with default initialization."""
        from brain_go_brrr.models.linear_probe import LinearProbeHead
        
        probe = LinearProbeHead(input_dim=2048, num_classes=2)
        assert probe.input_dim == 2048
        assert probe.num_classes == 2
        assert hasattr(probe, 'classifier')
        
        # Test forward pass
        x = torch.randn(16, 2048)
        output = probe(x)
        assert output.shape == (16, 2)
    
    def test_linear_probe_custom_init(self):
        """Test LinearProbeHead with custom initialization."""
        from brain_go_brrr.models.linear_probe import LinearProbeHead
        
        probe = LinearProbeHead(input_dim=512, num_classes=5, dropout=0.3)
        assert probe.input_dim == 512
        assert probe.num_classes == 5
        
        # Check dropout is applied in training mode
        probe.train()
        x = torch.randn(8, 512)
        output1 = probe(x)
        output2 = probe(x)
        # Outputs should differ due to dropout
        assert not torch.allclose(output1, output2)
    
    def test_linear_probe_eval_mode(self):
        """Test LinearProbeHead behavior in eval mode."""
        from brain_go_brrr.models.linear_probe import LinearProbeHead
        
        probe = LinearProbeHead(input_dim=128, num_classes=2, dropout=0.5)
        probe.eval()
        
        x = torch.randn(4, 128)
        output1 = probe(x)
        output2 = probe(x)
        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)


class TestEDFValidatorEdgeCases:
    """Sharp tests for EDF validation scenarios."""
    
    def test_edf_validator_empty_file(self):
        """Test EDF validator handles empty files."""
        from brain_go_brrr.core.edf_validator import EDFValidator
        
        validator = EDFValidator()
        result = validator.validate(b"")
        assert result.is_valid is False
        assert "empty" in result.error.lower()
    
    def test_edf_validator_invalid_header(self):
        """Test EDF validator handles invalid headers."""
        from brain_go_brrr.core.edf_validator import EDFValidator
        
        validator = EDFValidator()
        # EDF files must start with "0       " (8 bytes)
        invalid_content = b"INVALID HEADER" + b"\x00" * 1000
        result = validator.validate(invalid_content)
        assert result.is_valid is False
        assert "header" in result.error.lower() or "invalid" in result.error.lower()
    
    def test_edf_validator_minimum_size(self):
        """Test EDF validator checks minimum file size."""
        from brain_go_brrr.core.edf_validator import EDFValidator
        
        validator = EDFValidator()
        # EDF header is at least 256 bytes
        too_small = b"0       " + b"\x00" * 100  # Less than minimum
        result = validator.validate(too_small)
        assert result.is_valid is False


class TestCacheOperations:
    """Sharp tests for cache operations."""
    
    def test_cache_manager_key_generation(self):
        """Test cache key generation is deterministic."""
        from brain_go_brrr.api.cache import CacheManager
        
        manager = CacheManager()
        
        # Same inputs should generate same key
        key1 = manager.generate_key("test", {"a": 1, "b": 2})
        key2 = manager.generate_key("test", {"a": 1, "b": 2})
        assert key1 == key2
        
        # Different order should still match (sorted)
        key3 = manager.generate_key("test", {"b": 2, "a": 1})
        assert key1 == key3
        
        # Different values should differ
        key4 = manager.generate_key("test", {"a": 1, "b": 3})
        assert key1 != key4
    
    def test_cache_manager_ttl_handling(self):
        """Test cache TTL is properly set."""
        from brain_go_brrr.api.cache import CacheManager
        from tests.fakes import FakeRedis
        
        fake_redis = FakeRedis()
        manager = CacheManager(redis_client=fake_redis)
        
        # Store with TTL
        manager.set("test_key", {"data": "value"}, ttl=300)
        
        # Check it was stored
        assert fake_redis.storage.get("test_key") is not None
        assert fake_redis.call_count['set'] == 1
    
    def test_cache_invalidation(self):
        """Test cache invalidation works correctly."""
        from brain_go_brrr.api.cache import CacheManager
        from tests.fakes import FakeRedis
        
        fake_redis = FakeRedis()
        manager = CacheManager(redis_client=fake_redis)
        
        # Store data
        manager.set("test_key", {"data": "value"})
        assert manager.get("test_key") is not None
        
        # Invalidate
        manager.invalidate("test_key")
        assert manager.get("test_key") is None
        assert fake_redis.call_count['delete'] == 1


class TestErrorHandlingPaths:
    """Sharp tests for error handling paths."""
    
    def test_quality_controller_handle_bad_data(self):
        """Test EEGQualityController handles bad data."""
        from brain_go_brrr.core.quality import EEGQualityController
        
        controller = EEGQualityController()
        
        # Empty data should be handled
        result = controller.assess_quality(np.array([]))
        assert result is not None
        assert result.get("quality_grade") == "INVALID"
    
    def test_abnormality_detector_handle_insufficient_data(self):
        """Test AbnormalityDetector handles insufficient data."""
        from brain_go_brrr.core.abnormal.detector import AbnormalityDetector
        
        detector = AbnormalityDetector()
        
        # Too short data (less than 4 seconds)
        short_data = np.random.randn(19, 100)  # Less than 4s at 256Hz
        result = detector.detect_abnormality(short_data)
        assert result is not None
        assert "error" in result or result["confidence"] == 0.0
    
    def test_api_validation_rejects_non_edf(self):
        """Test API validation properly rejects non-EDF files."""
        from brain_go_brrr.core.edf_validator import EDFValidator
        
        validator = EDFValidator()
        
        # Test various invalid inputs
        result = validator.validate(b"Not an EDF file")
        assert result.is_valid is False
        
        result = validator.validate(b"")
        assert result.is_valid is False
        
        result = validator.validate(None)
        assert result.is_valid is False


class TestConfigurationEdgeCases:
    """Sharp tests for configuration edge cases."""
    
    def test_config_from_env(self):
        """Test config loads from environment variables."""
        from brain_go_brrr.core.config import Settings
        
        with patch.dict('os.environ', {'BGB_LOG_LEVEL': 'DEBUG'}):
            settings = Settings()
            assert settings.log_level == 'DEBUG'
    
    def test_config_defaults(self):
        """Test config uses sensible defaults."""
        from brain_go_brrr.core.config import Settings
        
        settings = Settings()
        assert settings.api_port == 8000
        assert settings.redis_host == 'localhost'
        assert settings.redis_port == 6379
    
    def test_config_validation(self):
        """Test config validates constraints."""
        from brain_go_brrr.core.config import Settings
        
        # Port must be valid
        with pytest.raises(ValueError):
            Settings(api_port=-1)
        
        with pytest.raises(ValueError):
            Settings(api_port=70000)