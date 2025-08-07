"""Tests for TUAB dataset - Marked for integration testing."""

import pytest

# TUAB dataset tests should be in integration suite due to file I/O
pytestmark = pytest.mark.integration


class TestTUABDataset:
    """Test TUAB dataset - runs in integration suite."""
    
    def test_placeholder(self):
        """Placeholder for integration tests."""
        pass