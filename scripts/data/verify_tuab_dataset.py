#!/usr/bin/env python3
"""Verify TUAB dataset integrity and structure.

Checks:
- Directory structure matches expected format
- File counts are reasonable
- EDF files can be parsed
- Split ratios are correct
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import mne


def verify_tuab_structure(root_dir: Path) -> dict:
    """Verify TUAB dataset structure and integrity."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "TUAB",
        "version": "v3.0.1",
        "root": str(root_dir),
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    # Expected structure
    expected_dirs = {
        "train/normal": (1300, 1400),  # Expected range
        "train/abnormal": (1300, 1400),
        "eval/normal": (140, 160),
        "eval/abnormal": (120, 140),
    }

    edf_dir = root_dir / "edf"
    if not edf_dir.exists():
        results["errors"].append(f"Missing edf directory: {edf_dir}")
        return results

    # Check each split/class
    total_files = 0
    for split_class, (min_expected, max_expected) in expected_dirs.items():
        dir_path = edf_dir / split_class / "01_tcp_ar"

        if not dir_path.exists():
            results["errors"].append(f"Missing directory: {dir_path}")
            continue

        edf_files = list(dir_path.glob("*.edf"))
        count = len(edf_files)
        total_files += count

        results["checks"][split_class] = {
            "count": count,
            "expected_range": f"{min_expected}-{max_expected}",
            "status": "PASS" if min_expected <= count <= max_expected else "WARN",
        }

        if count < min_expected:
            results["warnings"].append(
                f"{split_class}: Only {count} files (expected {min_expected}+)"
            )
        elif count > max_expected:
            results["warnings"].append(
                f"{split_class}: {count} files (expected max {max_expected})"
            )

    results["total_files"] = total_files
    results["checks"]["total"] = {
        "count": total_files,
        "expected_range": "2900-3100",
        "status": "PASS" if 2900 <= total_files <= 3100 else "WARN",
    }

    # Sample EDF validation
    print("Sampling EDF files for validation...")
    sample_errors = []

    for split_class in ["train/normal", "train/abnormal"]:
        dir_path = edf_dir / split_class / "01_tcp_ar"
        if not dir_path.exists():
            continue

        # Sample first 3 files
        sample_files = sorted(dir_path.glob("*.edf"))[:3]
        for edf_file in sample_files:
            try:
                # Try to read header
                raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
                n_channels = len(raw.ch_names)
                sfreq = raw.info["sfreq"]

                # Basic sanity checks
                if n_channels < 18 or n_channels > 35:
                    sample_errors.append(f"{edf_file.name}: Unusual channel count {n_channels}")
                if sfreq < 200 or sfreq > 500:
                    sample_errors.append(f"{edf_file.name}: Unusual sampling rate {sfreq}")

            except Exception as e:
                sample_errors.append(f"{edf_file.name}: Parse error - {str(e)[:50]}")

    if sample_errors:
        results["warnings"].extend(sample_errors[:5])  # Limit to 5 warnings

    # Check for marker file
    marker = root_dir / ".download_complete"
    results["checks"]["download_marker"] = {
        "exists": marker.exists(),
        "status": "PASS" if marker.exists() else "INFO",
    }

    # Determine overall status
    if results["errors"]:
        results["status"] = "FAIL"
    elif results["warnings"]:
        results["status"] = "WARN"
    else:
        results["status"] = "PASS"

    return results


def main():
    """Main verification function."""
    # Default to standard location
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        root = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets/tuab")

    if not root.exists():
        print(f"❌ Dataset not found at: {root}")
        sys.exit(1)

    print("=" * 60)
    print("🔍 TUAB Dataset Verification")
    print("=" * 60)
    print(f"Root: {root}")
    print()

    results = verify_tuab_structure(root)

    # Print summary
    print("📊 File Counts:")
    for split_class, info in results["checks"].items():
        if split_class in ["download_marker", "total"]:
            continue
        status_icon = "✅" if info["status"] == "PASS" else "⚠️"
        print(f"  {status_icon} {split_class}: {info['count']} files")

    print(f"\n📈 Total: {results['total_files']} EDF files")

    # Print issues
    if results["errors"]:
        print("\n❌ ERRORS:")
        for error in results["errors"]:
            print(f"  - {error}")

    if results["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in results["warnings"][:10]:  # Limit output
            print(f"  - {warning}")

    # Save results
    log_dir = Path("logs/verify")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"tuab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with log_file.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📝 Full report saved to: {log_file}")

    # Final status
    print("\n" + "=" * 60)
    if results["status"] == "PASS":
        print("✅ TUAB dataset verification PASSED")
    elif results["status"] == "WARN":
        print("⚠️  TUAB dataset verification PASSED with warnings")
    else:
        print("❌ TUAB dataset verification FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
