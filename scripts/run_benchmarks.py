#!/usr/bin/env python3
"""Standalone script to run EEGPT performance benchmarks.

Usage:
    python scripts/run_benchmarks.py [--cpu-only] [--quick] [--report-format markdown]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Run EEGPT benchmarks with various options."""
    parser = argparse.ArgumentParser(description="Run EEGPT performance benchmarks")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run only CPU benchmarks (skip GPU tests)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks only (skip slow tests)",
    )
    parser.add_argument(
        "--report-format",
        choices=["markdown", "json", "text"],
        default="markdown",
        help="Benchmark report format",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for benchmark results",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(exist_ok=True)

    # Build pytest command
    cmd = [
        "uv",
        "run",
        "pytest",
        "tests/benchmarks/",
        "--benchmark-only",
        "--benchmark-sort=mean",
        f"--benchmark-json={args.output_dir}/benchmark_results.json",
        "-v" if args.verbose else "-q",
    ]

    # Add marker-based filtering
    markers = ["benchmark"]

    if args.cpu_only:
        # Skip GPU tests
        markers.append("not (gpu and cuda)")

    if args.quick:
        # Skip slow tests
        markers.append("not slow")

    if len(markers) > 1:
        cmd.extend(["-m", " and ".join(markers)])
    else:
        cmd.extend(["-m", "benchmark"])

    print(f"Running benchmarks with command: {' '.join(cmd)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Report format: {args.report_format}")

    try:
        # Run benchmarks
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        print("✅ Benchmarks completed successfully!")
        print("\nBenchmark output:")
        print(result.stdout)

        if result.stderr:
            print("\nWarnings/Errors:")
            print(result.stderr)

        # Check if JSON results were generated
        json_results = args.output_dir / "benchmark_results.json"
        if json_results.exists():
            print(f"\n📊 Detailed results saved to: {json_results}")

        return 0

    except subprocess.CalledProcessError as e:
        print(f"❌ Benchmarks failed with return code {e.returncode}")
        print("\nOutput:")
        print(e.stdout)
        print("\nError output:")
        print(e.stderr)
        return e.returncode

    except FileNotFoundError:
        print("❌ Error: Could not find 'uv' command. Make sure uv is installed and in PATH.")
        print("Alternative: Run directly with pytest:")
        print(f"  pytest {' '.join(cmd[3:])}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
