#!/usr/bin/env python
"""Get coverage statistics quickly."""

import subprocess
import sys
from pathlib import Path

def get_coverage():
    """Run minimal tests to get coverage baseline."""
    
    # Test files that should run fast
    fast_tests = [
        "tests/unit/test_config.py",
        "tests/unit/test_yasa_adapter.py", 
        "tests/unit/test_linear_probe.py",
        "tests/unit/test_preprocessing.py",
        "tests/unit/test_spectral_features.py",
    ]
    
    # Find which exist
    existing = []
    for test in fast_tests:
        if Path(test).exists():
            existing.append(test)
            print(f"✓ Found: {test}")
        else:
            print(f"✗ Missing: {test}")
    
    if not existing:
        print("No test files found!")
        return
    
    print(f"\n🧪 Running {len(existing)} test files for coverage...")
    
    # Run tests with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        *existing,
        "--no-cov",  # Disable default coverage
        "--cov=src/brain_go_brrr",  # Add our coverage
        "--cov-report=term:skip-covered",
        "--cov-report=",
        "-q",
        "--tb=no",
        "-o", "addopts=''",  # Override pytest.ini
        "--timeout=30",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output for coverage
    lines = result.stdout.split("\n")
    for line in lines:
        if "TOTAL" in line or "brain_go_brrr" in line:
            print(line)
    
    # Get the TOTAL line
    for line in lines:
        if "TOTAL" in line and "%" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if "%" in part:
                    coverage = part.strip("%")
                    print(f"\n📊 TOTAL COVERAGE: {coverage}%")
                    return float(coverage)
    
    print("\n❌ Could not determine coverage percentage")
    print("Raw output:", result.stdout[-500:] if result.stdout else "No output")

if __name__ == "__main__":
    get_coverage()