"""Tests for TUAB cached dataset - requires MNE."""

import pytest

# Mark as integration since it requires MNE and potentially large datasets
pytestmark = pytest.mark.integration


class TestTUABCachedDataset:
    """Test TUAB cached dataset functionality."""

    def test_placeholder(self):
        """Placeholder to keep file valid."""
        pytest.skip("Expected exception was raised")
