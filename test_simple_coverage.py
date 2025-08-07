#!/usr/bin/env python
"""Simple test to verify coverage works."""

def test_simple():
    """Verify basic test runs."""
    assert 1 + 1 == 2
    
def test_import():
    """Test we can import our modules."""
    from src.brain_go_brrr.core import exceptions
    assert exceptions.EdfLoadError is not None