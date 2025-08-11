#!/usr/bin/env python3
"""AST-based checker for torch.load safety."""

import ast
import pathlib
import sys
from typing import Any


def check_torch_load(node: ast.Call, source_lines: list[str]) -> str | None:
    """Check if a torch.load call has weights_only parameter."""
    # Check if this is torch.load
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    ):
        return None
    
    # Get the source for this call to check for nosec comment
    if node.lineno and node.end_lineno:
        call_lines = source_lines[node.lineno - 1 : node.end_lineno]
        call_source = " ".join(call_lines)
        
        # Allow escape hatch with nosec comment
        if "# nosec:weights_only" in call_source or "# nosec B614" in call_source:
            return None
    
    # Check if weights_only is in the keywords
    has_weights_only = any(
        keyword.arg == "weights_only" 
        for keyword in node.keywords
    )
    
    if not has_weights_only:
        return f"torch.load missing weights_only parameter (or # nosec:weights_only comment)"
    
    return None


def check_file(filepath: pathlib.Path) -> list[tuple[int, str]]:
    """Check a Python file for unsafe torch.load calls."""
    errors = []
    
    try:
        source = filepath.read_text()
        source_lines = source.splitlines()
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        # Skip files with syntax errors or encoding issues
        return errors
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            error = check_torch_load(node, source_lines)
            if error and hasattr(node, "lineno"):
                errors.append((node.lineno, error))
    
    return errors


def main(files: list[str]) -> int:
    """Check files and report errors."""
    exit_code = 0
    
    for file_path in files:
        path = pathlib.Path(file_path)
        
        # Skip test files and reference repos
        if any(part in path.parts for part in ["tests", "reference_repos", "__pycache__"]):
            continue
        
        # Only check Python files
        if path.suffix != ".py":
            continue
        
        errors = check_file(path)
        for lineno, message in errors:
            print(f"{path}:{lineno}: {message}")
            exit_code = 1
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))