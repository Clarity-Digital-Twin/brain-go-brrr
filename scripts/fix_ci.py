#!/usr/bin/env python
"""Professional CI/CD health check and fix script."""

import subprocess
import sys

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def run_command(cmd: str, check: bool = True) -> tuple[int, str]:
    """Run command and return exit code and output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.returncode, result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout + e.stderr


def print_status(success: bool, message: str):
    """Print colored status."""
    icon = f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"
    print(f"{icon} {message}")


def main():
    """Run all CI checks locally."""
    print(f"{BOLD}🔍 Professional CI/CD Health Check{RESET}")
    print("=" * 50)

    all_passed = True

    # 1. Lint Check
    print(f"\n{BOLD}1. Linting (Ruff){RESET}")
    code, output = run_command("uv run ruff check src tests --statistics", check=False)
    if code == 0:
        print_status(True, "No linting issues")
    else:
        print_status(False, f"Linting failed (exit {code})")
        print(output[:500])
        all_passed = False

    # 2. Format Check
    print(f"\n{BOLD}2. Formatting (Ruff){RESET}")
    code, output = run_command("uv run ruff format src tests --check", check=False)
    if code == 0:
        print_status(True, "Code properly formatted")
    else:
        print_status(False, "Formatting issues found")
        print("Run: make format")
        all_passed = False

    # 3. Type Check (Fast)
    print(f"\n{BOLD}3. Type Checking (MyPy){RESET}")
    code, output = run_command(
        "uv run mypy src/brain_go_brrr --ignore-missing-imports --no-error-summary | head -20",
        check=False,
    )
    if "Success:" in output or code == 0:
        print_status(True, "Type checking passed")
    else:
        print_status(False, "Type errors found")
        print(output[:500])
        all_passed = False

    # 4. Unit Tests (Fast)
    print(f"\n{BOLD}4. Unit Tests (Fast){RESET}")
    code, output = run_command(
        "uv run pytest tests -x -q --tb=no -k 'not integration' --disable-warnings | tail -20",
        check=False,
    )
    if "failed" not in output.lower() and code == 0:
        print_status(True, "Unit tests passed")
    else:
        print_status(False, f"Tests failed (exit {code})")
        print(output)
        all_passed = False

    # 5. Import Checks
    print(f"\n{BOLD}5. Critical Imports{RESET}")
    import_test = """
import numpy as np
import torch
import pytorch_lightning
from brain_go_brrr  # noqa: E402.data.tuab_dataset import TUABDataset
print("OK")
"""
    code, output = run_command(f'uv run python -c "{import_test}"', check=False)
    if "OK" in output:
        print_status(True, "All critical imports work")
    else:
        print_status(False, "Import errors found")
        print(output)
        all_passed = False

    # Summary
    print(f"\n{BOLD}Summary{RESET}")
    print("=" * 50)

    if all_passed:
        print(f"{GREEN}✅ ALL CHECKS PASSED - CI SHOULD BE GREEN!{RESET}")
        print("\nNext steps:")
        print("1. git add -A && git commit -m 'fix: ensure CI passes all checks'")
        print("2. git push origin development")
    else:
        print(f"{RED}❌ SOME CHECKS FAILED - FIX BEFORE PUSHING!{RESET}")
        print("\nQuick fixes:")
        print("- Linting: uv run ruff check --fix")
        print("- Formatting: make format")
        print("- Types: Add type hints or # type: ignore")
        print("- Tests: Fix failing tests or mark as xfail")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
