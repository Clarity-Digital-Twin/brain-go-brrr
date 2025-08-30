"""Smoke test for CLI version consistency."""

import re
import subprocess
import sys
from importlib.metadata import version as pkg_version


def test_cli_version_matches_package():
    """Ensure CLI version matches package metadata version."""
    # Get package version from metadata
    package_version = pkg_version("brain-go-brrr")

    # Get CLI version
    result = subprocess.run(
        [sys.executable, "-m", "brain_go_brrr.cli", "version"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Extract version from CLI output
    match = re.search(r"(\d+\.\d+\.\d+)", result.stdout)
    assert match, f"Could not find version in CLI output: {result.stdout}"
    cli_version = match.group(1)

    # Versions should match
    assert cli_version == package_version, (
        f"Version mismatch: CLI={cli_version}, Package={package_version}"
    )
