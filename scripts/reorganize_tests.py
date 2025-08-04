#!/usr/bin/env python
"""Reorganize test suite with proper structure based on first principles."""

import re
import shutil
from pathlib import Path

# Define test categories based on naming and imports
TEST_CATEGORIES = {
    "unit": {
        "patterns": [
            r"test_(model|module|function|class)_",
            r"test_.*_(loading|serialization|utils|config|logger)",
            r"test_(preprocessor|extractor|adapter|fallbacks)",
            r"test_eegpt_(?!.*integration)",  # EEGPT tests but not integration
            r"test_(sleep_preprocessor|train_sleep|yasa)",
            r"test_(encoder|code_quality|redis_caching)",
        ],
        "imports": ["unittest", "pytest", "mock", "MagicMock"],
        "excludes": ["integration", "e2e", "api", "benchmark", "pipeline", "real"],
    },
    "integration": {
        "patterns": [
            r"test_.*_integration",
            r"test_.*_pipeline",
            r"test_real_.*",
            r"test_cli_.*",
            r".*autoreject.*",  # All autoreject tests are integration
        ],
        "imports": ["real", "pipeline", "integration"],
        "excludes": ["benchmark"],
    },
    "api": {
        "patterns": [
            r"test_api_",
            r"test_.*_api",
            r"test_(auth|endpoints|routers)",
            r"test_job_queue",
        ],
        "imports": ["fastapi", "httpx", "TestClient", "api"],
        "excludes": ["unit"],
    },
    "e2e": {
        "patterns": [
            r"test_e2e_",
            r"test_full_.*",
            r"test_.*_workflow",
        ],
        "imports": ["e2e", "workflow", "scenario"],
        "excludes": [],
    },
    "benchmarks": {
        "patterns": [
            r"test_.*_performance",
            r"test_.*_benchmark",
            r"benchmark_.*",
        ],
        "imports": ["benchmark", "performance", "timeit"],
        "excludes": [],
    },
}

# Valid modules based on actual codebase structure
VALID_MODULES = {
    "models": ["eegpt", "linear_probe"],
    "preprocessing": ["autoreject", "sleep", "flexible", "eeg"],
    "data": ["tuab", "edf_streaming"],
    "core": [
        "config",
        "logger",
        "exceptions",
        "abnormal",
        "sleep",
        "quality",
        "edf",
        "pipeline",
        "jobs",
    ],
    "api": ["auth", "cache", "routers", "schemas", "endpoints", "app", "main"],
    "tasks": ["abnormality_detection"],
    "utils": ["time"],
    "visualization": ["pdf_report", "markdown_report"],
    "infra": ["redis", "cache", "serialization"],
    "cli": ["cli"],
}

# Additional valid keywords that indicate valid tests
VALID_KEYWORDS = ["yasa", "parallel", "encoder", "accuracy", "embedding", "job_queue", "cli"]


def analyze_test_file(file_path: Path) -> dict[str, any]:
    """Analyze a test file to determine its category and validity."""
    content = file_path.read_text()
    filename = file_path.name

    # Check imports
    imports = re.findall(r"(?:from|import)\s+([^\s]+)", content)

    # Check what's being tested
    tested_modules = []
    for imp in imports:
        if "brain_go_brrr" in imp:
            parts = imp.split(".")
            if len(parts) > 2:
                module = parts[2]
                if module in VALID_MODULES:
                    tested_modules.append(module)

    # Determine category
    category = "unknown"
    for cat, rules in TEST_CATEGORIES.items():
        # Check filename patterns
        if any(re.match(pattern, filename) for pattern in rules["patterns"]):
            category = cat
            break
        # Check imports
        if any(
            imp_pattern in " ".join(imports).lower() for imp_pattern in rules["imports"]
        ) and not any(excl in filename for excl in rules["excludes"]):
            category = cat
            break

    # Check if test is for valid module
    is_valid = False
    if (
        tested_modules
        or any(
            module in filename.lower()
            for module_list in VALID_MODULES.values()
            for module in module_list
        )
        or any(keyword in filename.lower() for keyword in VALID_KEYWORDS)
    ):
        is_valid = True
    elif "brain_go_brrr" in " ".join(imports):
        # If it imports from our package, it's probably valid
        is_valid = True

    return {
        "category": category,
        "is_valid": is_valid,
        "tested_modules": tested_modules,
        "imports": imports[:5],  # First 5 imports for debugging
    }


def reorganize_tests(dry_run: bool = True):
    """Reorganize test files into proper structure."""
    test_root = Path("tests")

    # Collect all test files
    test_files = []
    for file in test_root.rglob("*.py"):
        if (
            file.name.startswith(("test_", "_test")) or file.name.endswith("_test.py")
        ) and "__pycache__" not in str(file):
            test_files.append(file)

    # Analyze each file
    moves = []
    deletes = []

    for file in test_files:
        analysis = analyze_test_file(file)

        if not analysis["is_valid"]:
            deletes.append((file, "Invalid - no matching module in codebase"))
            continue

        # Determine target directory
        category = analysis["category"]
        if category == "unknown" and analysis["tested_modules"]:
            category = "unit"  # Default to unit if we can identify modules

        if category != "unknown":
            target_dir = test_root / category
            target_path = target_dir / file.name

            # Only move if not already in correct location
            if file.parent != target_dir:
                moves.append((file, target_path, category))

    # Print analysis
    print("TEST REORGANIZATION PLAN")
    print("=" * 80)
    print(f"Total test files found: {len(test_files)}")
    print(f"Files to move: {len(moves)}")
    print(f"Files to delete: {len(deletes)}")
    print()

    if moves:
        print("FILES TO MOVE:")
        print("-" * 80)
        for src, dst, cat in sorted(moves):
            print(f"{src.relative_to(test_root)} -> {cat}/{dst.name}")
        print()

    if deletes:
        print("FILES TO DELETE (invalid/obsolete):")
        print("-" * 80)
        for file, reason in sorted(deletes):
            print(f"{file.relative_to(test_root)} - {reason}")
        print()

    if not dry_run:
        print("EXECUTING CHANGES...")

        # Create directories
        for cat in TEST_CATEGORIES:
            (test_root / cat).mkdir(exist_ok=True)

        # Move files
        for src, dst, _ in moves:
            dst.parent.mkdir(exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"Moved: {src.name} -> {dst.parent.name}/")

        # Delete invalid files
        for file, _ in deletes:
            file.unlink()
            print(f"Deleted: {file.name}")

        print("\nReorganization complete!")
    else:
        print("DRY RUN - No changes made. Run with --execute to apply changes.")


def fix_imports():
    """Fix imports in all test files after reorganization."""
    test_root = Path("tests")
    fixes = 0

    for file in test_root.rglob("*.py"):
        if file.name.startswith(("test_", "_test")) or file.name.endswith("_test.py"):
            content = file.read_text()
            original = content

            # Fix common import issues
            # 1. Update relative imports
            content = re.sub(r"from \.\. import", "from brain_go_brrr import", content)
            content = re.sub(r"from \.\.\. import", "from brain_go_brrr import", content)

            # 2. Fix test fixture imports
            content = re.sub(r"from tests\.fixtures import", "from ..fixtures import", content)

            # 3. Fix conftest imports
            if "conftest" not in file.name:
                depth = len(file.relative_to(test_root).parts) - 1
                if depth > 0:
                    rel_import = "../" * depth
                    content = re.sub(
                        r"from conftest import", f"from {rel_import}conftest import", content
                    )

            if content != original:
                file.write_text(content)
                fixes += 1
                print(f"Fixed imports in: {file.relative_to(test_root)}")

    print(f"\nFixed imports in {fixes} files")


if __name__ == "__main__":
    import sys

    if "--execute" in sys.argv:
        print("Executing reorganization...")
        reorganize_tests(dry_run=False)
        print("\nFixing imports...")
        fix_imports()
    else:
        reorganize_tests(dry_run=True)
