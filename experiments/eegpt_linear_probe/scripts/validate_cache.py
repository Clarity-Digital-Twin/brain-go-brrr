#!/usr/bin/env python3
"""
Cache validator - ensures cache integrity and correctness.
Run after cache build to verify all specs are met.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch


def validate_cache(cache_dir: Path) -> tuple[bool, list[str], dict]:
    """Validate cache against EEGPT requirements."""

    errors = []
    warnings = []
    stats = {
        "cache_dir": str(cache_dir),
        "manifests": [],
        "windows_checked": 0,
        "shape_errors": 0,
        "dtype_errors": 0,
    }

    # Check manifests
    for manifest_file in cache_dir.glob("manifest_*.json"):
        split = manifest_file.stem.replace("manifest_", "")

        try:
            with open(manifest_file) as f:
                manifest = json.load(f)

            stats["manifests"].append(
                {
                    "split": split,
                    "config_hash": manifest["config_hash"],
                    "n_windows": manifest["data_stats"]["n_windows"],
                    "n_files": manifest["data_stats"]["n_files"],
                }
            )

            # Validate critical specs
            prep = manifest["preprocessing"]

            # Check sample rate
            if prep["sample_rate_hz"] != 256:
                errors.append(f"{split}: Wrong sample rate {prep['sample_rate_hz']} (need 256)")

            # Check window size
            if prep["window_seconds"] != 4.0:
                errors.append(f"{split}: Wrong window size {prep['window_seconds']}s (need 4.0s)")

            if prep["window_samples"] != 1024:
                errors.append(f"{split}: Wrong samples {prep['window_samples']} (need 1024)")

            # Check channels
            corpus = manifest["corpus"]["name"]
            expected_channels = 19 if corpus == "TUAB" else 20
            if prep["channels"] != expected_channels:
                errors.append(
                    f"{split}: Wrong channels {prep['channels']} (need {expected_channels} for {corpus})"
                )

            # Check expected shape
            expected_shape = manifest["data_stats"]["expected_shape"]
            if expected_shape != [expected_channels, 1024]:
                errors.append(f"{split}: Wrong expected shape {expected_shape}")

            # Check feature dim
            if manifest["data_stats"]["feature_dim"] != 2048:
                warnings.append(
                    f"{split}: Feature dim {manifest['data_stats']['feature_dim']} != 2048"
                )

            # Check PHI safety
            if not manifest["integrity"]["phi_safe"]:
                errors.append(f"{split}: Not PHI safe!")

            if not manifest["integrity"]["atomic_writes"]:
                warnings.append(f"{split}: Not using atomic writes")

            if not manifest["integrity"]["paths_relative"]:
                warnings.append(f"{split}: Using absolute paths")

            print(
                f"✓ {split} manifest: {manifest['data_stats']['n_windows']} windows, hash {manifest['config_hash'][:8]}"
            )

        except Exception as e:
            errors.append(f"Failed to read {manifest_file}: {e}")

    # Sample actual cache files
    print("\nSampling cache files...")
    cache_files = list(cache_dir.glob("**/*.pt"))

    if not cache_files:
        warnings.append("No .pt cache files found")
    else:
        # Sample up to 100 files
        sample_size = min(100, len(cache_files))
        sample_files = np.random.choice(cache_files, sample_size, replace=False)

        for cache_file in sample_files:
            try:
                # Check for temp files
                if cache_file.suffix == '.tmp':
                    warnings.append(f"Temp file found: {cache_file.name}")
                    continue

                # Load and validate
                data = torch.load(cache_file, map_location='cpu')

                # Expected structure: dict with 'x' and 'y'
                if not isinstance(data, dict):
                    errors.append(f"{cache_file.name}: Not a dict")
                    continue

                if 'x' not in data or 'y' not in data:
                    errors.append(f"{cache_file.name}: Missing x or y")
                    continue

                x = data['x']
                y = data['y']

                # Check shapes
                if x.ndim != 2:
                    errors.append(f"{cache_file.name}: x has {x.ndim} dims (need 2)")
                    stats["shape_errors"] += 1

                if x.shape[1] != 1024:
                    errors.append(f"{cache_file.name}: x has {x.shape[1]} samples (need 1024)")
                    stats["shape_errors"] += 1

                # Check channels (19 for TUAB, 20 for TUEV)
                if x.shape[0] not in [19, 20]:
                    errors.append(f"{cache_file.name}: x has {x.shape[0]} channels (need 19 or 20)")
                    stats["shape_errors"] += 1

                # Check dtype
                if x.dtype not in [torch.float32, torch.float64]:
                    warnings.append(f"{cache_file.name}: x dtype {x.dtype} (prefer float32)")
                    stats["dtype_errors"] += 1

                # Check label
                if not isinstance(y, int | torch.Tensor):
                    errors.append(f"{cache_file.name}: y type {type(y)} invalid")

                y_val = y.item() if isinstance(y, torch.Tensor) else y

                if y_val not in [0, 1]:
                    errors.append(f"{cache_file.name}: y value {y_val} not in [0, 1]")

                stats["windows_checked"] += 1

            except Exception as e:
                errors.append(f"Failed to load {cache_file.name}: {e}")

        print(f"✓ Checked {stats['windows_checked']} windows")
        print(f"  Shape errors: {stats['shape_errors']}")
        print(f"  Dtype warnings: {stats['dtype_errors']}")

    # Summary
    print(f"\n{'=' * 60}")
    if errors:
        print(f"❌ VALIDATION FAILED: {len(errors)} errors")
        for e in errors[:10]:  # Show first 10
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print("✅ VALIDATION PASSED")

    if warnings:
        print(f"\n⚠️  {len(warnings)} warnings:")
        for w in warnings[:5]:
            print(f"  - {w}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more")

    print(f"{'=' * 60}")

    # Save validation report
    report = {
        "validated_at": str(Path.cwd()),
        "cache_dir": str(cache_dir),
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }

    report_path = cache_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📝 Report saved: {report_path}")

    return len(errors) == 0, errors, stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_cache.py <cache_dir>")
        sys.exit(1)

    cache_dir = Path(sys.argv[1])

    if not cache_dir.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        sys.exit(1)

    print(f"🔍 Validating cache: {cache_dir}")
    print(f"{'=' * 60}")

    passed, errors, stats = validate_cache(cache_dir)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
