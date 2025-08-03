#!/usr/bin/env python
"""Fix imports in all test files after reorganization."""

import re
from pathlib import Path
from typing import List, Tuple

def fix_test_imports(test_file: Path) -> bool:
    """Fix imports in a single test file."""
    content = test_file.read_text()
    original = content
    
    # Calculate relative path depth
    relative_path = test_file.relative_to(Path('tests'))
    depth = len(relative_path.parts) - 1
    
    # Fix patterns
    fixes = []
    
    # 1. Fix conftest imports based on depth
    if depth > 0 and 'conftest' in content and 'conftest' not in test_file.name:
        rel_import = '../' * depth
        # Replace various conftest import patterns
        fixes.append((r'from conftest import', f'from {rel_import}conftest import'))
        fixes.append((r'import conftest', f'from {rel_import} import conftest'))
    
    # 2. Fix fixture imports
    if depth > 0:
        rel_fixtures = '../' * depth + 'fixtures'
        fixes.append((r'from tests\.fixtures import', f'from {rel_fixtures} import'))
        fixes.append((r'from fixtures import', f'from {rel_fixtures} import'))
    
    # 3. Fix mocks imports
    if depth > 0:
        rel_mocks = '../' * depth + '_mocks'
        fixes.append((r'from tests\._mocks import', f'from {rel_mocks} import'))
        fixes.append((r'from _mocks import', f'from {rel_mocks} import'))
    
    # 4. Fix relative imports that shouldn't be
    fixes.append((r'from \.\. import', 'from brain_go_brrr import'))
    fixes.append((r'from \.\.\. import', 'from brain_go_brrr import'))
    fixes.append((r'from \.\.brain_go_brrr', 'from brain_go_brrr'))
    
    # 5. Fix incorrect test imports
    fixes.append((r'from tests\.unit import', 'from .. import'))
    fixes.append((r'from tests\.integration import', 'from .. import'))
    fixes.append((r'from tests\.api import', 'from .. import'))
    
    # Apply fixes
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    # Special case: if importing from brain_go_brrr.api.test_api, fix it
    content = re.sub(r'from brain_go_brrr\.api\.test_api import', 'from brain_go_brrr.api import', content)
    
    # Check if file changed
    if content != original:
        test_file.write_text(content)
        return True
    return False


def check_missing_imports(test_file: Path) -> List[str]:
    """Check for potentially missing imports."""
    content = test_file.read_text()
    missing = []
    
    # Common patterns that might need imports
    patterns = {
        r'\bpytest\.': 'import pytest',
        r'\bunittest\.': 'import unittest',
        r'\bmock\.': 'from unittest import mock',
        r'\bMagicMock\b': 'from unittest.mock import MagicMock',
        r'\bpatch\b': 'from unittest.mock import patch',
        r'\bnp\.': 'import numpy as np',
        r'\btorch\.': 'import torch',
        r'\bPath\b': 'from pathlib import Path',
    }
    
    for pattern, import_stmt in patterns.items():
        if re.search(pattern, content) and import_stmt not in content:
            missing.append(import_stmt)
    
    return missing


def main():
    """Fix imports in all test files."""
    test_root = Path('tests')
    fixed_count = 0
    issues = []
    
    # Process all Python test files
    for test_file in test_root.rglob('*.py'):
        if '__pycache__' in str(test_file):
            continue
            
        # Skip __init__.py files
        if test_file.name == '__init__.py':
            continue
        
        # Fix imports
        if fix_test_imports(test_file):
            fixed_count += 1
            print(f"Fixed imports in: {test_file.relative_to(test_root)}")
        
        # Check for missing imports
        missing = check_missing_imports(test_file)
        if missing:
            issues.append((test_file, missing))
    
    print(f"\nFixed imports in {fixed_count} files")
    
    if issues:
        print("\nPotential missing imports:")
        print("-" * 80)
        for file, missing in issues[:10]:  # Show first 10
            print(f"\n{file.relative_to(test_root)}:")
            for imp in missing:
                print(f"  - {imp}")
    
    # Create import mapping file for reference
    mapping_file = test_root / 'import_mapping.md'
    with mapping_file.open('w') as f:
        f.write("# Test Import Guide\n\n")
        f.write("## From unit tests:\n")
        f.write("```python\n")
        f.write("from ..conftest import fixture_name\n")
        f.write("from ..fixtures import mock_data\n")
        f.write("from .._mocks import MockClass\n")
        f.write("```\n\n")
        f.write("## From integration tests:\n")
        f.write("```python\n")
        f.write("from ..conftest import fixture_name\n")
        f.write("from ..fixtures.mock_eeg_generator import generate_mock_eeg\n")
        f.write("```\n\n")
        f.write("## Common imports:\n")
        f.write("```python\n")
        f.write("import pytest\n")
        f.write("from unittest.mock import Mock, patch, MagicMock\n")
        f.write("from pathlib import Path\n")
        f.write("import numpy as np\n")
        f.write("from brain_go_brrr.module_name import ClassName\n")
        f.write("```\n")
    
    print(f"\nCreated import guide at: {mapping_file}")


if __name__ == "__main__":
    main()