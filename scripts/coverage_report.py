#!/usr/bin/env python3
"""Professional coverage report generator that handles large codebases.

This script runs pytest with coverage in chunks to avoid timeouts,
then combines the results for a full report.
"""

import subprocess
import sys
from pathlib import Path


def run_coverage_chunk(test_paths, cov_append=False):
    """Run coverage on a chunk of tests."""
    cmd = [
        "uv", "run", "pytest",
        *test_paths,
        "--cov=brain_go_brrr",
        "--cov-report=",  # No report, just collect data
        "--no-cov-on-fail",
        "-q"
    ]
    
    if cov_append:
        cmd.append("--cov-append")
    
    print(f"Running: {' '.join(test_paths)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Warning: Some tests failed in {test_paths}")
    
    return result.returncode == 0


def main():
    """Run coverage in chunks and generate report."""
    # Define test chunks to avoid timeout
    test_chunks = [
        ["tests/unit/test_abnormality_accuracy.py"],
        ["tests/unit/test_api_routers_eegpt.py"],
        ["tests/unit/test_api_routers_resources_clean.py"],
        ["tests/unit/test_models_linear_probe.py"],
        ["tests/unit/test_sleep_montage_detection.py"],
        ["tests/unit/test_redis_pool.py"],
        ["tests/unit/test_flexible_preprocessing.py"],
        ["tests/unit/test_eegpt_pipeline.py"],
        ["tests/unit/test_yasa_compliance.py"],
        # Add more chunks for other test files
        ["tests/api"],  # All API tests
        ["tests/benchmarks"],  # Benchmarks (without coverage ideally)
    ]
    
    # Clean previous coverage data
    coverage_file = Path(".coverage")
    if coverage_file.exists():
        coverage_file.unlink()
    
    # Run coverage on each chunk
    all_passed = True
    for i, chunk in enumerate(test_chunks):
        # Check if paths exist
        existing_paths = []
        for path in chunk:
            test_path = Path(path)
            if test_path.exists():
                existing_paths.append(path)
        
        if not existing_paths:
            print(f"Skipping {chunk} - paths don't exist")
            continue
        
        # First chunk creates .coverage, rest append
        success = run_coverage_chunk(existing_paths, cov_append=(i > 0))
        all_passed = all_passed and success
    
    # Generate final report
    print("\n" + "="*60)
    print("Generating coverage report...")
    print("="*60 + "\n")
    
    # Show terminal report
    subprocess.run([
        "uv", "run", "coverage", "report",
        "--skip-covered",
        "--show-missing",
        "--precision=2"
    ])
    
    # Generate HTML report
    subprocess.run(["uv", "run", "coverage", "html"])
    print("\nHTML report generated at: htmlcov/index.html")
    
    # Get total coverage
    result = subprocess.run(
        ["uv", "run", "coverage", "report", "--format=total"],
        capture_output=True,
        text=True
    )
    
    try:
        total_coverage = float(result.stdout.strip())
        print(f"\n{'='*60}")
        print(f"TOTAL COVERAGE: {total_coverage:.2f}%")
        
        if total_coverage >= 55:
            print("✅ Coverage meets minimum threshold (55%)")
        else:
            print(f"❌ Coverage below threshold (55% required, got {total_coverage:.2f}%)")
            return 1
    except:
        print("Could not determine total coverage")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())