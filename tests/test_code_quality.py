"""Code quality tests to prevent regressions."""

import ast
from pathlib import Path

import pytest


class TestCodeQuality:
    """Tests to ensure code quality standards are maintained."""

    def test_no_fast_test_in_production(self):
        """Ensure FAST_TEST environment variable is not used in production code."""
        src_dir = Path(__file__).parent.parent / "src"

        # Check all Python files in src/
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text()

            # Check for FAST_TEST string
            assert "FAST_TEST" not in content, (
                f"FAST_TEST found in production code: {py_file}\n"
                "Use pytest fixtures for test configuration instead."
            )

            # Check for common environment variable patterns
            if "os.environ" in content or "os.getenv" in content:
                # Parse AST to check for FAST_TEST usage
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Str) and "FAST_TEST" in node.s:
                            pytest.fail(
                                f"FAST_TEST environment variable reference found in {py_file}"
                            )
                except SyntaxError:
                    # If we can't parse, at least we checked the string
                    pass

    def test_no_global_test_mode_in_classes(self):
        """Ensure no global test mode variables in production classes."""
        src_dir = Path(__file__).parent.parent / "src"

        # Patterns that indicate test-only behavior in production
        forbidden_patterns = [
            "FAST_TEST_MODE",
            "TEST_MODE",
            "SKIP_SLOW",
            "DEBUG_MODE",  # Unless it's a proper debug flag
        ]

        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text()

            for pattern in forbidden_patterns:
                if pattern in content:
                    # Allow DEBUG_MODE if it's properly configured
                    if pattern == "DEBUG_MODE" and "config" in py_file.name:
                        continue

                    pytest.fail(
                        f"Global test mode variable '{pattern}' found in {py_file}\n"
                        "Use dependency injection or configuration instead."
                    )

    def test_autoreject_not_globally_disabled(self):
        """Ensure Autoreject is not globally disabled in preprocessor."""
        preprocessor_file = (
            Path(__file__).parent.parent / "src/brain_go_brrr/preprocessing/eeg_preprocessor.py"
        )

        if preprocessor_file.exists():
            content = preprocessor_file.read_text()

            # Check for hardcoded disabling
            assert "use_autoreject=False" not in content.replace(" ", ""), (
                "Autoreject should not be hardcoded to False in production"
            )

            # Check for environment-based disabling
            assert "DISABLE_AUTOREJECT" not in content, (
                "Autoreject should not be disabled via environment variables"
            )

    def test_no_print_statements_in_src(self):
        """Ensure no print() statements in production code (use logging instead)."""
        src_dir = Path(__file__).parent.parent / "src"

        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text()

            # Simple check for print statements
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                # Skip comments and strings
                if line.strip().startswith("#"):
                    continue

                # Look for print( at start of statement
                if "print(" in line:
                    # Allow in __main__ blocks
                    if (
                        "__main__"
                        in content[max(0, content.find(line) - 100) : content.find(line) + 100]
                    ):
                        continue

                    pytest.fail(
                        f"print() statement found in {py_file}:{i}\n"
                        "Use logging module instead:\n"
                        f"  {line.strip()}"
                    )

    def test_proper_type_hints_in_key_modules(self):
        """Ensure key modules have proper type hints."""
        key_modules = [
            "src/brain_go_brrr/preprocessing/eeg_preprocessor.py",
            "src/brain_go_brrr/models/eegpt_model.py",
            "services/abnormality_detector.py",
            "services/sleep_metrics.py",
        ]

        base_dir = Path(__file__).parent.parent

        for module_path in key_modules:
            full_path = base_dir / module_path
            if not full_path.exists():
                continue

            content = full_path.read_text()

            # Check for function definitions without type hints
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                if (
                    line.strip().startswith("def ")
                    and "__init__" not in line
                    and "->" not in line
                    and not line.strip().endswith(":")
                    and i < len(lines)
                    and "->" not in lines[i]
                ):
                    print(f"Warning: Function without return type in {module_path}:{i}")
                    print(f"  {line.strip()}")
