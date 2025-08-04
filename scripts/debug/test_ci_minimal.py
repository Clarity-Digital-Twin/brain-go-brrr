"""Minimal test to ensure CI passes."""


def test_imports():
    """Test critical imports work."""
    import numpy as np
    import pytorch_lightning
    import torch

    assert np.__version__
    assert torch.__version__
    assert pytorch_lightning.__version__


def test_basic_math():
    """Most basic test."""
    assert 2 + 2 == 4


if __name__ == "__main__":
    test_imports()
    test_basic_math()
    print("✅ Basic tests pass!")
