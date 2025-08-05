#!/usr/bin/env python
"""Clean up and organize brain-go-brrr directory structure.

This script:
1. Removes macOS metadata files (._*)
2. Consolidates duplicate model directories
3. Ensures proper directory structure per DIRECTORY_STRUCTURE.md
"""

import shutil
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def remove_macos_metadata(root_dir: Path) -> int:
    """Remove all macOS metadata files (._*).
    
    Args:
        root_dir: Root directory to clean
        
    Returns:
        Number of files removed
    """
    count = 0
    for meta_file in root_dir.rglob("._*"):
        try:
            meta_file.unlink()
            print(f"Removed: {meta_file}")
            count += 1
        except Exception as e:
            print(f"Failed to remove {meta_file}: {e}")
    return count


def consolidate_models(project_root: Path) -> None:
    """Consolidate model files into data/models/ directory.
    
    Args:
        project_root: Project root directory
    """
    # Source directories
    old_models_dir = project_root / "models"
    target_models_dir = project_root / "data" / "models"
    
    # Ensure target exists
    target_models_dir.mkdir(parents=True, exist_ok=True)
    (target_models_dir / "pretrained").mkdir(exist_ok=True)
    (target_models_dir / "trained").mkdir(exist_ok=True)
    (target_models_dir / "trained" / "linear_probes").mkdir(exist_ok=True)
    (target_models_dir / "trained" / "checkpoints").mkdir(exist_ok=True)
    
    # Move files from old models directory if it exists
    if old_models_dir.exists():
        print(f"\nConsolidating models from {old_models_dir} to {target_models_dir}")
        
        # Move pretrained models
        old_pretrained = old_models_dir / "pretrained"
        if old_pretrained.exists():
            for model_file in old_pretrained.glob("*.ckpt"):
                target = target_models_dir / "pretrained" / model_file.name
                if not target.exists():
                    print(f"Moving: {model_file} -> {target}")
                    shutil.move(str(model_file), str(target))
                else:
                    print(f"Target exists, skipping: {model_file}")
        
        # Check if old directory is now empty and remove it
        if old_models_dir.exists() and not any(old_models_dir.rglob("*")):
            print(f"Removing empty directory: {old_models_dir}")
            shutil.rmtree(old_models_dir)
        elif old_models_dir.exists():
            print(f"WARNING: {old_models_dir} still contains files, manual review needed")


def organize_trained_models(project_root: Path) -> None:
    """Organize trained models into proper subdirectories.
    
    Args:
        project_root: Project root directory
    """
    models_dir = project_root / "data" / "models"
    
    # Find any misplaced model files
    for model_file in models_dir.glob("*.pt"):
        # Determine where it should go based on filename
        if "linear_probe" in model_file.name or "tuab" in model_file.name:
            target = models_dir / "trained" / "linear_probes" / model_file.name
        else:
            target = models_dir / "trained" / model_file.name
        
        if model_file != target:
            print(f"Moving: {model_file} -> {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(model_file), str(target))


def ensure_directory_structure(project_root: Path) -> None:
    """Ensure all required directories exist.
    
    Args:
        project_root: Project root directory
    """
    required_dirs = [
        "data/datasets/external/tuab",
        "data/datasets/external/sleep-edf",
        "data/cache",
        "data/cache/tuab_4s",  # For paper-aligned training
        "data/cache/tuab_8s",
        "data/models/pretrained",
        "data/models/trained/linear_probes",
        "data/models/trained/checkpoints",
        "experiments/eegpt_linear_probe/results",
        "logs",
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            print(f"Creating: {full_path}")
            full_path.mkdir(parents=True, exist_ok=True)


def main():
    """Run directory cleanup."""
    project_root = Path(__file__).parent.parent
    print(f"Cleaning up directory structure in: {project_root}")
    
    # 1. Remove macOS metadata files
    print("\n=== Removing macOS metadata files ===")
    num_removed = remove_macos_metadata(project_root)
    print(f"Removed {num_removed} macOS metadata files")
    
    # 2. Consolidate model directories
    print("\n=== Consolidating model directories ===")
    consolidate_models(project_root)
    
    # 3. Organize trained models
    print("\n=== Organizing trained models ===")
    organize_trained_models(project_root)
    
    # 4. Ensure directory structure
    print("\n=== Ensuring directory structure ===")
    ensure_directory_structure(project_root)
    
    print("\n✅ Directory cleanup complete!")
    print("\nNext steps:")
    print("1. Run: ./experiments/eegpt_linear_probe/build_4s_cache.sh")
    print("2. Train with paper-aligned config for 0.869 AUROC")


if __name__ == "__main__":
    main()