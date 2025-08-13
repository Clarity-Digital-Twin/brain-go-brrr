#!/usr/bin/env python3
"""Fix imports after refactoring reorganization."""

import os
import re
from pathlib import Path

# Mapping of old imports to new imports
IMPORT_MAPPINGS = {
    # Config moved to application
    r"from brain_go_brrr\.config": "from brain_go_brrr.application.config",
    r"import brain_go_brrr\.config": "import brain_go_brrr.application.config",
    
    # Data moved to infra
    r"from brain_go_brrr\.data\.": "from brain_go_brrr.infra.data.",
    r"import brain_go_brrr\.data\.": "import brain_go_brrr.infra.data.",
    
    # Models moved to infra.ml_models
    r"from brain_go_brrr\.models\.": "from brain_go_brrr.infra.ml_models.",
    r"import brain_go_brrr\.models\.": "import brain_go_brrr.infra.ml_models.",
    
    # Constraints moved to domain
    r"from brain_go_brrr\.modules\.constraints": "from brain_go_brrr.domain.constraints",
    r"import brain_go_brrr\.modules\.constraints": "import brain_go_brrr.domain.constraints",
    
    # Preprocessing split
    r"from brain_go_brrr\.preprocessing\.basic": "from brain_go_brrr.domain.preprocessing.basic",
    r"from brain_go_brrr\.preprocessing\.window_extractor": "from brain_go_brrr.domain.preprocessing.window_extractor",
    r"from brain_go_brrr\.preprocessing\.sleep_preprocessor": "from brain_go_brrr.domain.preprocessing.sleep_preprocessor",
    r"from brain_go_brrr\.preprocessing\.autoreject_adapter": "from brain_go_brrr.infra.preprocessing.autoreject_adapter",
    r"from brain_go_brrr\.preprocessing\.chunked_autoreject": "from brain_go_brrr.infra.preprocessing.chunked_autoreject",
    r"from brain_go_brrr\.preprocessing\.eeg_preprocessor": "from brain_go_brrr.infra.preprocessing.eeg_preprocessor",
    r"from brain_go_brrr\.preprocessing\.flexible_preprocessor": "from brain_go_brrr.infra.preprocessing.flexible_preprocessor",
    
    # Features moved
    r"from brain_go_brrr\.preprocessing\.features": "from brain_go_brrr.domain.preprocessing.features",
    
    # Snippets moved
    r"from brain_go_brrr\.preprocessing\.snippets": "from brain_go_brrr.infra.preprocessing.snippets",
    
    # Visualization moved to presentation
    r"from brain_go_brrr\.visualization\.": "from brain_go_brrr.presentation.visualization.",
    r"import brain_go_brrr\.visualization\.": "import brain_go_brrr.presentation.visualization.",
}

def fix_imports_in_file(filepath: Path) -> bool:
    """Fix imports in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return False
    
    original_content = content
    
    # Apply all mappings
    for old_pattern, new_import in IMPORT_MAPPINGS.items():
        content = re.sub(old_pattern, new_import, content)
    
    # Write back if changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix all imports in the codebase."""
    # Find all Python files
    src_dir = Path("src/brain_go_brrr")
    test_dir = Path("tests")
    
    all_files = list(src_dir.rglob("*.py")) + list(test_dir.rglob("*.py"))
    
    fixed_count = 0
    for filepath in all_files:
        if fix_imports_in_file(filepath):
            fixed_count += 1
            print(f"Fixed: {filepath}")
    
    print(f"\nFixed imports in {fixed_count} files")

if __name__ == "__main__":
    main()