#!/usr/bin/env python3
"""Fix FastAPI decorator type errors."""

import re
from pathlib import Path


def fix_router_decorators(file_path: Path) -> None:
    """Add type: ignore to FastAPI router decorators."""
    content = file_path.read_text()
    lines = content.split("\n")

    modified = False
    new_lines = []

    for i, line in enumerate(lines):
        # Check if this is a FastAPI decorator without type: ignore
        if (
            re.match(r"^\s*@router\.(get|post|put|delete|patch)", line)
            and "# type: ignore" not in line
        ):
            # Add type: ignore[misc]
            line = line.rstrip() + "  # type: ignore[misc]"
            modified = True

        new_lines.append(line)

    if modified:
        file_path.write_text("\n".join(new_lines))
        print(f"Fixed router decorators in: {file_path}")


def main():
    """Fix all FastAPI files."""
    api_dir = Path("src/brain_go_brrr/api")

    # Fix routers
    for router_file in (api_dir / "routers").glob("*.py"):
        fix_router_decorators(router_file)

    # Fix main app file
    app_file = api_dir / "app.py"
    if app_file.exists():
        fix_router_decorators(app_file)


if __name__ == "__main__":
    main()
