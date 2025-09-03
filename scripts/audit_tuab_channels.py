#!/usr/bin/env python3
"""Audit TUAB dataset to check channel presence, especially Oz."""

from pathlib import Path
import mne
from collections import Counter
import random

# Constants
TUAB_ROOT = Path("/data/datasets/tuab/v3.0.1/")
MODERN_CHANNELS = ["T7", "T8", "P7", "P8"]
OLD_CHANNELS = ["T3", "T4", "T5", "T6"]
KEY_CHANNELS = ["Fz", "Oz"]

def clean_channel_name(ch_name: str) -> str:
    """Clean channel name."""
    # Remove common EEG prefixes and suffixes
    clean = ch_name.upper()
    if clean.startswith("EEG "):
        clean = clean[4:]
    if clean.endswith("-REF"):
        clean = clean[:-4]
    # Map old to modern
    mapping = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    return mapping.get(clean, clean)

def audit_file(edf_path: Path) -> dict:
    """Audit a single EDF file."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        cleaned = [clean_channel_name(ch) for ch in raw.ch_names]
        return {
            "path": str(edf_path),
            "raw_count": len(raw.ch_names),
            "has_oz": "OZ" in cleaned,
            "has_fz": "FZ" in cleaned,
            "has_modern_naming": any(ch in cleaned for ch in MODERN_CHANNELS),
            "has_old_naming": any(ch in cleaned for ch in OLD_CHANNELS),
            "channels": cleaned
        }
    except Exception as e:
        return {"path": str(edf_path), "error": str(e)}

def main():
    """Run the audit."""
    # Find all EDF files
    edf_files = list(TUAB_ROOT.rglob("*.edf"))
    print(f"Found {len(edf_files)} EDF files in TUAB dataset")
    
    # Sample random files (or all if few)
    sample_size = min(100, len(edf_files))
    sampled = random.sample(edf_files, sample_size)
    print(f"Auditing {sample_size} random files...")
    
    # Audit each file
    results = []
    for i, edf_path in enumerate(sampled):
        if i % 10 == 0:
            print(f"  Progress: {i}/{sample_size}")
        result = audit_file(edf_path)
        if "error" not in result:
            results.append(result)
    
    # Analyze results
    print(f"\n=== AUDIT RESULTS ({len(results)} files) ===")
    
    # Channel counts
    counts = Counter(r["raw_count"] for r in results)
    print(f"\nChannel counts distribution:")
    for count, freq in sorted(counts.items()):
        print(f"  {count} channels: {freq} files ({freq/len(results)*100:.1f}%)")
    
    # Oz presence
    with_oz = sum(1 for r in results if r["has_oz"])
    print(f"\nOz presence:")
    print(f"  WITH Oz: {with_oz} files ({with_oz/len(results)*100:.1f}%)")
    print(f"  WITHOUT Oz: {len(results)-with_oz} files ({(len(results)-with_oz)/len(results)*100:.1f}%)")
    
    # Fz presence
    with_fz = sum(1 for r in results if r["has_fz"])
    print(f"\nFz presence:")
    print(f"  WITH Fz: {with_fz} files ({with_fz/len(results)*100:.1f}%)")
    print(f"  WITHOUT Fz: {len(results)-with_fz} files ({(len(results)-with_fz)/len(results)*100:.1f}%)")
    
    # Naming convention
    modern = sum(1 for r in results if r["has_modern_naming"])
    old = sum(1 for r in results if r["has_old_naming"])
    print(f"\nNaming convention:")
    print(f"  Modern (T7/T8/P7/P8): {modern} files ({modern/len(results)*100:.1f}%)")
    print(f"  Old (T3/T4/T5/T6): {old} files ({old/len(results)*100:.1f}%)")
    
    # Show a few examples
    print(f"\nExample files WITHOUT Oz:")
    no_oz = [r for r in results if not r["has_oz"]][:3]
    for r in no_oz:
        print(f"  {Path(r['path']).name}: {r['raw_count']} channels")
    
    print(f"\nExample files WITH Oz:")
    with_oz_files = [r for r in results if r["has_oz"]][:3]
    for r in with_oz_files:
        print(f"  {Path(r['path']).name}: {r['raw_count']} channels")

if __name__ == "__main__":
    main()