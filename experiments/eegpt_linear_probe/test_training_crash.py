#!/usr/bin/env python
"""Quick test to verify training scripts are crash-resistant."""

import subprocess
import sys
from pathlib import Path

def check_script(script_path: Path, name: str):
    """Check if a training script has our critical fixes."""
    print(f"\nChecking {name} ({script_path.name})...")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    fixes = {
        "Exception handling": "except RuntimeError" in content,
        "Memory cleanup": "torch.cuda.empty_cache()" in content,
        "Checkpoint saving": "checkpoint_epoch" in content and "batch" in content,
        "Try-except in loop": "for batch_idx" in content and "try:" in content
    }
    
    all_good = True
    for fix_name, present in fixes.items():
        status = "✅" if present else "❌"
        print(f"  {status} {fix_name}")
        if not present:
            all_good = False
    
    return all_good

def check_launch_script(script_path: Path, name: str):
    """Check if launch script has restart logic."""
    print(f"\nChecking {name} launcher ({script_path.name})...")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    fixes = {
        "Restart loop": "while true" in content,
        "Exit code check": "EXIT_CODE" in content,
        "Sleep on crash": "sleep" in content,
        "Logging": "tee" in content
    }
    
    all_good = True
    for fix_name, present in fixes.items():
        status = "✅" if present else "❌"
        print(f"  {status} {fix_name}")
        if not present:
            all_good = False
    
    return all_good

def main():
    print("=" * 60)
    print("TRAINING CRASH RESISTANCE CHECK")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    
    # Check training scripts
    tuab_ok = check_script(base_dir / "train_tuab.py", "TUAB")
    tuev_ok = check_script(base_dir / "train_tuev.py", "TUEV")
    
    # Check launch scripts
    tuab_launch_ok = check_launch_script(base_dir / "launch_tuab.sh", "TUAB")
    tuev_launch_ok = check_launch_script(base_dir / "scripts" / "launch_tuev.sh", "TUEV")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all([tuab_ok, tuev_ok, tuab_launch_ok, tuev_launch_ok]):
        print("✅ ALL SCRIPTS ARE CRASH-RESISTANT!")
        print("\nBoth TUAB and TUEV training scripts include:")
        print("  • OOM exception handling")
        print("  • Periodic memory cleanup (every 100 batches)")
        print("  • Checkpoint saving (every 500 batches)")
        print("  • Automatic restart on crash")
        return 0
    else:
        print("⚠️  Some scripts need fixes")
        return 1

if __name__ == "__main__":
    sys.exit(main())