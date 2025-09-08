#!/usr/bin/env python3
"""Prevent duplicate class definitions across src/.

Allows specific known duplicates (e.g., DTO vs Entity) via allowlist.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ALLOWED_DUPLICATES = {
    "JobData",  # API DTO vs Domain Entity pattern
    "NumpyEncoder",  # JSON encoder utility in multiple routers
    "ModelConfig",  # Pydantic config class in multiple configs
}


def find_duplicate_classes(root: str = "src") -> int:
    classes: dict[str, list[str]] = defaultdict(list)

    for py_file in Path(root).rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            continue
        for match in re.finditer(r"^class (\w+)", content, re.MULTILINE):
            class_name = match.group(1)
            if not class_name.startswith("_"):
                classes.setdefault(class_name, []).append(str(py_file))

    duplicates = {k: v for k, v in classes.items() if len(v) > 1}
    real_duplicates = {k: v for k, v in duplicates.items() if k not in ALLOWED_DUPLICATES}

    if real_duplicates:
        print("❌ Duplicate class definitions found!", file=sys.stderr)
        for class_name, files in real_duplicates.items():
            print(f"  {class_name}:", file=sys.stderr)
            for file in files:
                print(f"    - {file}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(find_duplicate_classes())

