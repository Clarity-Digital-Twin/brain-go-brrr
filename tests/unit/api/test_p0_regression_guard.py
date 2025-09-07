"""ATTACK ON TITANS REGRESSION GUARD - Prevent P0 dimension bugs from returning!

This test ensures that ALL probe-feeding paths ALWAYS use summary=False
and proper flattening to (B,2048). Any future code that violates this
will be caught by these tests.
"""

import re
from pathlib import Path

import pytest


class TestP0RegressionGuard:
    """Guard against future P0 dimension violations."""

    def test_all_extract_features_to_probes_use_summary_false(self):
        """Ensure ALL extract_features calls that feed probes use summary=False."""
        # Files that feed probes
        probe_feeding_files = [
            "src/brain_go_brrr/api/routers/eegpt.py",
            "src/brain_go_brrr/api/routers/sleep.py",
            "src/brain_go_brrr/application/training/sleep_probe_trainer.py",
        ]

        root = Path(__file__).parent.parent.parent.parent

        for file_path in probe_feeding_files:
            full_path = root / file_path
            if not full_path.exists():
                pytest.skip(f"File {file_path} not found")

            content = full_path.read_text()

            # Find all extract_features calls
            # Pattern matches extract_features or extract_features_batch
            pattern = r'\.extract_features(?:_batch)?\([^)]*\)'
            matches = re.findall(pattern, content)

            for match in matches:
                # Skip if it's in a comment
                if match.startswith('#'):
                    continue

                # Check if summary=False is present
                if 'summary=' not in match:
                    pytest.fail(
                        f"P0 VIOLATION in {file_path}:\n"
                        f"Found extract_features call without summary parameter:\n"
                        f"{match}\n"
                        f"ALL probe-feeding paths MUST use summary=False!"
                    )

                if 'summary=True' in match:
                    pytest.fail(
                        f"P0 VIOLATION in {file_path}:\n"
                        f"Found extract_features with summary=True:\n"
                        f"{match}\n"
                        f"Probe-feeding paths MUST use summary=False!"
                    )

    def test_all_probe_paths_use_adapter_or_proper_flatten(self):
        """Ensure all probe paths either use adapter or flatten to 2048."""
        probe_feeding_files = [
            "src/brain_go_brrr/api/routers/eegpt.py",
            "src/brain_go_brrr/api/routers/sleep.py",
            "src/brain_go_brrr/application/training/sleep_probe_trainer.py",
        ]

        root = Path(__file__).parent.parent.parent.parent

        for file_path in probe_feeding_files:
            full_path = root / file_path
            if not full_path.exists():
                pytest.skip(f"File {file_path} not found")

            content = full_path.read_text()

            # Check for prepare_probe_features usage (the adapter)
            # Handle both single-line and multi-line imports
            has_adapter_import = (
                'from brain_go_brrr.utils.probe_utils import prepare_probe_features' in content
                or ('from brain_go_brrr.utils.probe_utils import' in content 
                    and 'prepare_probe_features' in content)
            )
            uses_adapter = 'prepare_probe_features(' in content

            # Check for manual flattening (legacy but acceptable if to 2048)
            has_manual_flatten = '.flatten()' in content or '.flatten(1)' in content

            if not has_adapter_import and probe_feeding_files:
                pytest.fail(
                    f"P0 BEST PRACTICE VIOLATION in {file_path}:\n"
                    f"File should import prepare_probe_features adapter for DRY principle.\n"
                    f"Add: from brain_go_brrr.utils.probe_utils import prepare_probe_features"
                )

            if not uses_adapter and not has_manual_flatten:
                pytest.fail(
                    f"P0 VIOLATION in {file_path}:\n"
                    f"File must either use prepare_probe_features adapter or manual flatten to 2048!"
                )

    def test_probe_adapter_exists_at_correct_location(self):
        """Ensure the probe adapter exists at the P0 doc specified location."""
        root = Path(__file__).parent.parent.parent.parent

        # P0 doc specifies this location
        correct_path = root / "src/brain_go_brrr/utils/probe_utils.py"

        if not correct_path.exists():
            pytest.fail(
                f"P0 VIOLATION: Probe adapter not found at required location!\n"
                f"Expected: {correct_path}\n"
                f"The P0_CRITICAL_FIXES.md doc requires the adapter at utils/probe_utils.py"
            )

        # Verify it has the prepare_probe_features function
        content = correct_path.read_text()
        if 'def prepare_probe_features' not in content:
            pytest.fail(
                "P0 VIOLATION: probe_utils.py exists but missing prepare_probe_features function!"
            )

        # Verify it enforces 2048 dimensions
        if 'shape[-1] != 2048' not in content and '2048' not in content:
            pytest.fail("P0 VIOLATION: prepare_probe_features doesn't verify 2048 dimensions!")

    def test_no_probe_receives_512_dimensions(self):
        """Ensure NO probe ever receives 512-dimensional input."""
        # This is a smoke test - if any probe gets 512-d input, we have a P0 bug
        # We check by looking for patterns that would indicate 512-d usage

        probe_files = [
            "src/brain_go_brrr/infra/ml_models/linear_probe.py",
            "src/brain_go_brrr/infra/ml_models/eegpt_probe_unified.py",
        ]

        root = Path(__file__).parent.parent.parent.parent

        for file_path in probe_files:
            full_path = root / file_path
            if not full_path.exists():
                continue

            content = full_path.read_text()

            # Check for any hardcoded 512 in input_dim
            if 'input_dim: int = 512' in content or 'input_dim=512' in content:
                # Check if it's not in a comment
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'input_dim' in line and '512' in line and not line.strip().startswith('#'):
                        pytest.fail(
                            f"P0 VIOLATION in {file_path} line {i + 1}:\n"
                            f"Found probe with 512-d input dimension!\n"
                            f"ALL probes must expect 2048 dimensions (4x512 flattened)"
                        )


if __name__ == "__main__":
    # Run the regression guard
    pytest.main([__file__, "-xvs"])
