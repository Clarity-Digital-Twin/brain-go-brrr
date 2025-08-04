#!/usr/bin/env python3
"""Fix Typer CLI decorator type errors."""

import re
from pathlib import Path


def fix_typer_decorators(file_path: Path) -> None:
    """Add type: ignore to Typer decorators."""
    content = file_path.read_text()
    lines = content.split("\n")

    modified = False
    new_lines = []

    for i, line in enumerate(lines):
        # Check if this is a Typer decorator without type: ignore
        if re.match(r"^\s*@app\.(command|callback)", line) and "# type: ignore" not in line:
            # Add type: ignore[misc]
            line = line.rstrip() + "  # type: ignore[misc]"
            modified = True

        new_lines.append(line)

    if modified:
        file_path.write_text("\n".join(new_lines))
        print(f"Fixed Typer decorators in: {file_path}")


def main():
    """Fix CLI file."""
    cli_file = Path("src/brain_go_brrr/cli.py")
    if cli_file.exists():
        fix_typer_decorators(cli_file)


if __name__ == "__main__":
    main()
