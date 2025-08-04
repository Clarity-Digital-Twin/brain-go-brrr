#!/usr/bin/env python3
"""Test imports to find hanging issue."""

import sys
import time

def test_import(module_name):
    """Test importing a module."""
    print(f"Importing {module_name}...", end=" ", flush=True)
    start = time.time()
    try:
        __import__(module_name)
        elapsed = time.time() - start
        print(f"✓ {elapsed:.2f}s")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

modules = [
    "brain_go_brrr.preprocessing.flexible_preprocessor",
    "brain_go_brrr.training.sleep_probe_trainer",
    "brain_go_brrr.models.eegpt_model",
    "brain_go_brrr.core.edf_loader",
]

print("Testing imports...")
for module in modules:
    test_import(module)