#!/usr/bin/env python3
"""
Validate cache integrity and check for channel count issues.
Run this to verify your cache is correctly formatted.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch


def validate_cache(cache_dir: Path, verbose: bool = False) -> dict:
    """Validate a cache directory for channel consistency."""
    results = {
        'total_windows': 0,
        'channel_counts': defaultdict(int),
        'bad_files': [],
        'missing_files': [],
        'errors': [],
    }

    # Find index files
    index_files = list(cache_dir.glob("index_*.json"))
    if not index_files:
        results['errors'].append(f"No index files found in {cache_dir}")
        return results

    print(f"Found {len(index_files)} index file(s)")

    for index_file in index_files:
        print(f"\nValidating {index_file.name}...")

        try:
            with open(index_file) as f:
                index = json.load(f)
        except Exception as e:
            results['errors'].append(f"Failed to load {index_file}: {e}")
            continue

        # Check expected shape if present
        if 'expected_shape' in index:
            expected = tuple(index['expected_shape'])
            print(f"  Expected shape: {expected}")
        else:
            expected = None
            print("  ⚠️  No expected_shape in index (older cache version)")

        # Validate each window
        windows = index.get('windows', {})
        print(f"  Checking {len(windows)} windows...")

        for window_id, window_info in windows.items():
            results['total_windows'] += 1
            cache_file = cache_dir / window_info['cache_file']

            if not cache_file.exists():
                results['missing_files'].append(str(cache_file))
                continue

            try:
                data = torch.load(cache_file, weights_only=True)
                shape = tuple(data['x'].shape)
                results['channel_counts'][shape[0]] += 1

                # Check against expected shape
                if expected and shape != expected:
                    results['bad_files'].append(
                        {
                            'file': window_info['cache_file'],
                            'shape': shape,
                            'expected': expected,
                            'source': window_info.get('source', 'unknown'),
                            'window_id': window_id,
                        }
                    )
                    if verbose:
                        print(
                            f"    ❌ {window_info['cache_file']}: {shape} (from {window_info.get('source')})"
                        )

            except Exception as e:
                results['errors'].append(f"Failed to load {cache_file}: {e}")

    return results


def print_report(results: dict):
    """Print validation report."""
    print("\n" + "=" * 60)
    print("CACHE VALIDATION REPORT")
    print("=" * 60)

    print(f"\nTotal windows checked: {results['total_windows']}")

    print("\nChannel distribution:")
    for n_channels, count in sorted(results['channel_counts'].items()):
        pct = count / results['total_windows'] * 100 if results['total_windows'] > 0 else 0
        status = "✅" if n_channels == 19 else "❌"
        print(f"  {status} {n_channels} channels: {count} windows ({pct:.2f}%)")

    if results['bad_files']:
        print(f"\n❌ Found {len(results['bad_files'])} windows with incorrect shape:")

        # Group by source file
        by_source = defaultdict(list)
        for bad in results['bad_files']:
            by_source[bad['source']].append(bad)

        for source, windows in sorted(by_source.items())[:10]:  # Show first 10 sources
            print(f"\n  Source: {source}")
            print(f"    Affected windows: {len(windows)}")
            if len(windows) <= 3:
                for w in windows:
                    print(f"      - {w['file']}: {w['shape']}")
    else:
        print("\n✅ All windows have correct shape!")

    if results['missing_files']:
        print(f"\n⚠️  {len(results['missing_files'])} cache files are missing")

    if results['errors']:
        print(f"\n❌ {len(results['errors'])} errors occurred:")
        for error in results['errors'][:5]:
            print(f"  - {error}")

    # Summary
    print("\n" + "=" * 60)
    if results['bad_files'] or results['errors']:
        print("❌ VALIDATION FAILED - Cache has issues")
        print("\nRecommended actions:")
        print("1. For existing training: Keep using collate workaround")
        print("2. For new cache builds: Preprocessor will enforce 19 channels")
        print("3. To fix existing cache: See TECH_DEBT_CRITICAL.md")
    else:
        print("✅ VALIDATION PASSED - Cache is clean!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Validate EEGPT cache integrity')
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path('data/cache/tuab_mne_preprocessed'),
        help='Path to cache directory',
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', help='Show detailed output for each bad file'
    )

    args = parser.parse_args()

    if not args.cache_dir.exists():
        print(f"ERROR: Cache directory not found: {args.cache_dir}")
        sys.exit(1)

    print(f"Validating cache at: {args.cache_dir}")
    results = validate_cache(args.cache_dir, args.verbose)
    print_report(results)

    # Exit with error code if validation failed
    if results['bad_files'] or results['errors']:
        sys.exit(1)


if __name__ == '__main__':
    main()
