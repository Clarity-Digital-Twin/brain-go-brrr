#!/usr/bin/env python3
"""Add ARCHIVED banner to all files in docs/archive/."""

from pathlib import Path

BANNER = """
> ⚠️ **ARCHIVED DOCUMENT - DO NOT USE FOR CURRENT DEVELOPMENT**
> This document is preserved for historical reference only.
> For current documentation, see [docs/README.md](../../README.md)
> Archive date: September 2, 2025

---

"""


def add_banner_to_file(filepath: Path):
    """Add archive banner to a markdown file."""
    try:
        content = filepath.read_text()

        # Skip if banner already exists
        if "ARCHIVED DOCUMENT" in content:
            print(f"  ✓ Already has banner: {filepath}")
            return

        # Add banner after title if there is one
        lines = content.split('\n')
        if lines and lines[0].startswith('#'):
            # Insert after title
            lines.insert(1, BANNER)
            new_content = '\n'.join(lines)
        else:
            # Insert at beginning
            new_content = BANNER + content

        filepath.write_text(new_content)
        print(f"  + Added banner to: {filepath}")

    except Exception as e:
        print(f"  ✗ Error processing {filepath}: {e}")


def main():
    """Process all markdown files in docs/archive."""
    archive_dir = Path("docs/archive")

    if not archive_dir.exists():
        print("No docs/archive directory found!")
        return

    md_files = list(archive_dir.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in archive")

    for filepath in md_files:
        add_banner_to_file(filepath)

    print(f"\nProcessed {len(md_files)} files")


if __name__ == "__main__":
    main()
