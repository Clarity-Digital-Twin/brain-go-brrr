#!/usr/bin/env python3
"""Fix type errors systematically."""

import re
from pathlib import Path


def fix_fastapi_decorators(file_path: Path) -> None:
    """Add type annotations to FastAPI route decorators."""
    content = file_path.read_text()
    lines = content.split("\n")

    modified = False
    new_lines = []

    for i, line in enumerate(lines):
        # Check if this is a FastAPI decorator
        if re.match(r"^\s*@router\.(get|post|put|delete|patch)", line):
            # Check if the next line has a function def
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if "async def" in next_line or "def" in next_line:
                    # Extract function name
                    func_match = re.search(r"def\s+(\w+)", next_line)
                    if func_match and "-> " not in next_line:
                        # Function missing return type
                        print(
                            f"Warning: {file_path}:{i + 2} - Function {func_match.group(1)} missing return type"
                        )

        new_lines.append(line)

    if modified:
        file_path.write_text("\n".join(new_lines))
        print(f"Fixed: {file_path}")


def fix_unused_type_ignores(file_path: Path) -> None:
    """Remove unused type: ignore comments."""
    content = file_path.read_text()

    # Remove unused type: ignore comments
    new_content = re.sub(
        r"\s*#\s*type:\s*ignore(?:\[[^\]]+\])?\s*$", "", content, flags=re.MULTILINE
    )

    if new_content != content:
        file_path.write_text(new_content)
        print(f"Fixed unused type ignores: {file_path}")


def add_missing_return_types(file_path: Path) -> None:
    """Add -> None to functions missing return types."""
    content = file_path.read_text()
    lines = content.split("\n")

    modified = False
    new_lines = []

    for _i, line in enumerate(lines):
        # Check for __init__ without return type
        if re.match(r"^\s*def\s+__init__\s*\([^)]*\)\s*:", line):
            new_line = re.sub(r"(\)\s*):", r"\1 -> None:", line)
            if new_line != line:
                modified = True
                line = new_line

        new_lines.append(line)

    if modified:
        file_path.write_text("\n".join(new_lines))
        print(f"Fixed return types: {file_path}")


def main():
    """Main function to fix type errors."""
    # Read type errors
    with open("/tmp/type_errors.txt") as f:
        errors = f.readlines()

    # Group errors by file
    file_errors = {}
    for error in errors:
        match = re.match(r"(.*\.py):(\d+):", error)
        if match:
            file_path = match.group(1)
            if file_path not in file_errors:
                file_errors[file_path] = []
            file_errors[file_path].append(error)

    print(f"Found errors in {len(file_errors)} files")

    # Process each file
    for file_path_str, errors in file_errors.items():
        file_path = Path(file_path_str)
        if not file_path.exists():
            continue

        # Check error types
        has_decorator_errors = any("Untyped decorator" in e for e in errors)
        has_unused_ignores = any("Unused.*type: ignore" in e for e in errors)
        has_missing_returns = any("missing a return type" in e for e in errors)

        if has_unused_ignores:
            fix_unused_type_ignores(file_path)

        if has_missing_returns:
            add_missing_return_types(file_path)

        if has_decorator_errors:
            print(f"File has decorator errors: {file_path}")


if __name__ == "__main__":
    main()
