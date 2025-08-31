#!/usr/bin/env python3
"""Pre-commit hook to prevent hardcoded dataset paths.

This prevents regression of the Sleep-EDF path centralization fix.
"""
import re
import sys
from pathlib import Path


def check_file(filepath: Path) -> list[str]:
    """Check a file for hardcoded paths."""
    if not filepath.exists():
        return []
    
    violations = []
    
    # Skip certain files that are allowed to have these strings
    # Use exact paths where possible to avoid accidental bypass
    allowed_exact_paths = {
        # Config file where dataset version defaults are defined
        "src/brain_go_brrr/application/config/base.py",
        # Scripts that legitimately need to verify dataset versions
        "scripts/data/verify_tuab_dataset.py",
        "scripts/data/download_datasets.py",
    }
    
    # Planning/documentation files can use basenames (less risky)
    allowed_doc_files = {
        "SLEEP_EDF_FIX_PLAN.md",
        "PATH_AUDIT.md", 
        "CLEANUP_STATUS.md",
        "DIVERGENCE_FIX_ACTION_PLAN.md",
        "TEST_DATA_DIVERGENCE_PLAN.md",
        "CACHE_PATH_ANALYSIS.md",
        "TUAB_TUEV_ALIGNMENT_PLAN.md",
        "FINAL_ALIGNMENT_ACTION_PLAN.md",
        "CURRENT_ALIGNMENT_STATUS.md",
        "FINAL_POLISH_CHECKLIST.md",
        "ALIGNMENT_STATUS.md",
        "FULL_ALIGNMENT_COMPLETE.md",
        "IMPLEMENTATION_COMPLETE.md",
    }
    
    # Check exact paths first
    if str(filepath).replace('\\', '/') in allowed_exact_paths:
        return []
    
    # Check documentation files by basename
    if filepath.suffix == '.md' and filepath.name in allowed_doc_files:
        return []
    
    # Patterns to detect
    patterns = [
        # Sleep-EDF patterns
        (r'SC4001E0-PSG\.edf', 'Hardcoded Sleep-EDF filename - use DataConfig.get_sleep_edf_psg_file()'),
        (r'sleep-edf-database-expanded-[\d.]+', 'Hardcoded Sleep-EDF version - use DataConfig.sleep_edf_version'),
        (r'/external/sleep-edf', 'Legacy Sleep-EDF path - use DataConfig.sleep_edf_root'),
        
        # TUAB patterns
        (r'/external/tuab', 'Legacy TUAB path - use DataConfig.tuab_root'),
        (r'data/datasets/tuab/v[\d.]+/edf', 'Hardcoded TUAB path - use DataConfig.tuab_root'),
        (r'v3\.0\.1', 'Hardcoded TUAB version - use DataConfig.tuab_version'),
        (r'01_tcp_ar', 'Hardcoded TUAB protocol - use DataConfig.get_tuab_sample_file()'),
        (r'tuh_eeg_abnormal', 'Legacy TUAB name - use DataConfig.tuab_root'),
        (r'/abnormal/01_tcp_ar', 'Hardcoded TUAB structure - use DataConfig.get_tuab_sample_file()'),
        (r'/normal/01_tcp_ar', 'Hardcoded TUAB structure - use DataConfig.get_tuab_sample_file()'),
        
        # TUEV patterns
        (r'/external/tuev', 'Legacy TUEV path - use DataConfig.tuev_root'),
        (r'v2\.0\.\d+', 'Hardcoded TUEV version - use DataConfig.tuev_version'),
        (r'tuh_eeg_events', 'Legacy TUEV name - use DataConfig.tuev_root'),
        (r'/bckg/', 'Hardcoded TUEV event type - use DataConfig.get_tuev_sample_file()'),
        (r'/gped/', 'Hardcoded TUEV event type - use DataConfig.get_tuev_sample_file()'),
        (r'/pled/', 'Hardcoded TUEV event type - use DataConfig.get_tuev_sample_file()'),
    ]
    
    content = filepath.read_text()
    
    for line_num, line in enumerate(content.splitlines(), 1):
        # Skip comments in Python/shell files
        if filepath.suffix in ['.py', '.sh'] and line.strip().startswith('#'):
            continue
            
        for pattern, message in patterns:
            if re.search(pattern, line):
                violations.append(f"{filepath}:{line_num}: {message}")
                violations.append(f"  Found: {line.strip()}")
    
    return violations


def main():
    """Check all staged files for hardcoded paths."""
    import subprocess
    
    # Get staged files
    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Failed to get staged files")
        return 1
    
    violations = []
    for filename in result.stdout.strip().split('\n'):
        if not filename:
            continue
        
        filepath = Path(filename)
        
        # Only check relevant files
        if filepath.suffix in ['.py', '.md', '.rst', '.txt', '.sh']:
            file_violations = check_file(filepath)
            violations.extend(file_violations)
    
    if violations:
        print("❌ Hardcoded dataset paths detected!")
        print("\nPlease use DataConfig methods instead:")
        print("  - DataConfig.get_sleep_edf_psg_file() for Sleep-EDF files")
        print("  - DataConfig.sleep_edf_root for Sleep-EDF directory")
        print("  - DataConfig.tuab_root for TUAB directory")
        print("\nViolations found:")
        for violation in violations:
            print(f"  {violation}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())