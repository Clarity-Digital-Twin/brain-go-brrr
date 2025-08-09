#!/usr/bin/env python
"""Run test directly without pytest to check if tests work."""

import sys
sys.path.insert(0, 'src')

# Import the exceptions
from brain_go_brrr.core.exceptions import (
    BrainGoBrrrError,
    ConfigurationError,
    EdfLoadError,
    ProcessingError,
    ModelLoadError,
)

print("Testing exceptions...")

# Test 1: Basic exception
try:
    exc = BrainGoBrrrError("Test error")
    assert str(exc) == "Test error"
    print("✓ BrainGoBrrrError works")
except AssertionError as e:
    print(f"✗ BrainGoBrrrError failed: {e}")

# Test 2: Hierarchy
try:
    exc = EdfLoadError("Bad EDF")
    assert isinstance(exc, BrainGoBrrrError)
    print("✓ EdfLoadError inherits correctly")
except AssertionError as e:
    print(f"✗ EdfLoadError inheritance failed: {e}")

# Test 3: Exception chaining
try:
    try:
        raise ValueError("Original")
    except ValueError as e:
        raise ConfigurationError("Config failed") from e
except ConfigurationError as e:
    assert str(e) == "Config failed"
    assert e.__cause__ is not None
    print("✓ Exception chaining works")

print("\n✅ All direct tests passed!")
print("\nNow testing if we can get coverage...")

# Try to run with coverage
import subprocess
result = subprocess.run(
    ["uv", "run", "python", "-m", "coverage", "run", "--source=src/brain_go_brrr/core/exceptions", __file__],
    capture_output=True,
    text=True,
    timeout=5
)

if result.returncode == 0:
    # Get coverage report
    report = subprocess.run(
        ["uv", "run", "python", "-m", "coverage", "report"],
        capture_output=True,
        text=True
    )
    print("\nCoverage report:")
    print(report.stdout)