#!/usr/bin/env python3
"""Find hanging tests by running them one by one."""

import subprocess
import sys
from pathlib import Path

def run_test(test_file):
    """Run a single test file with timeout."""
    cmd = ["uv", "run", "pytest", str(test_file), "-v", "-x"]
    try:
        result = subprocess.run(cmd, timeout=10, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {test_file}")
            return True
        else:
            print(f"✗ {test_file} - FAILED")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏱ {test_file} - TIMEOUT (hanging test!)")
        return False

def main():
    """Find all test files and run them."""
    test_dir = Path("tests/unit")
    test_files = sorted(test_dir.glob("test_*.py"))
    
    print(f"Found {len(test_files)} test files")
    print("-" * 50)
    
    hanging_tests = []
    failed_tests = []
    
    for test_file in test_files:
        if not run_test(test_file):
            if "TIMEOUT" in str(test_file):
                hanging_tests.append(test_file)
            else:
                failed_tests.append(test_file)
    
    print("\n" + "=" * 50)
    print(f"Summary: {len(test_files) - len(hanging_tests) - len(failed_tests)} passed")
    
    if hanging_tests:
        print(f"\nHanging tests ({len(hanging_tests)}):")
        for test in hanging_tests:
            print(f"  - {test}")
    
    if failed_tests:
        print(f"\nFailed tests ({len(failed_tests)}):")
        for test in failed_tests:
            print(f"  - {test}")

if __name__ == "__main__":
    main()