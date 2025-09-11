#!/usr/bin/env python3
"""Compare class distributions between TUEV v2.0.0 and v2.0.1.

This script analyzes the .rec annotation files to count event types
and compare class balance between versions.
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd


def count_events_in_rec_files(base_path):
    """Count event types from .rec annotation files."""
    event_counter = Counter()
    file_counter = Counter()

    # Event type mapping from rec files
    event_types = {'spsw': 0, 'gped': 0, 'pled': 0, 'eyem': 0, 'artf': 0, 'bckg': 0}

    # Search for .rec files
    rec_files = list(Path(base_path).glob("**/*.rec"))

    if not rec_files:
        print(f"No .rec files found in {base_path}")
        return None, None

    print(f"Found {len(rec_files)} .rec files")

    for rec_file in rec_files:
        try:
            with open(rec_file) as f:
                lines = f.readlines()

            # Parse rec file format (CSV-like)
            for line in lines[1:]:  # Skip header
                if not line.strip():
                    continue

                parts = line.strip().split(',')
                if len(parts) >= 4:
                    # Format: channel, start_time, end_time, event_type, probability
                    event_type = parts[3].strip().lower()
                    if event_type in event_types:
                        event_counter[event_type] += 1
                        file_counter[rec_file.parent.name] += 1

        except Exception as e:
            print(f"Error reading {rec_file}: {e}")

    return event_counter, file_counter


def analyze_cache_if_exists(cache_path):
    """Analyze our preprocessed cache if it exists."""
    cache_dir = Path(cache_path)

    if not cache_dir.exists():
        return None

    results = {}

    for split in ['train', 'eval']:
        index_file = cache_dir / split / 'index.json'
        if index_file.exists():
            with open(index_file) as f:
                data = json.load(f)

            labels = [seg['label'] for seg in data['segments']]
            counter = Counter(labels)
            results[split] = counter

    return results


def main():
    """Compare TUEV versions."""
    base_dir = "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets"

    print("=" * 80)
    print("TUEV VERSION COMPARISON - CLASS DISTRIBUTIONS")
    print("=" * 80)

    # Check v2.0.1 (current)
    print("\n📊 TUEV v2.0.1 (current version):")
    print("-" * 40)

    v201_path = Path(base_dir) / "tuev"
    if v201_path.exists():
        # Check our cache
        cache_results = analyze_cache_if_exists(v201_path / "cache" / "tuev_event_segments")
        if cache_results:
            print("\nFrom our preprocessed cache:")
            for split, counter in cache_results.items():
                total = sum(counter.values())
                print(f"\n{split.upper()} ({total} total):")
                for event, count in sorted(counter.items()):
                    pct = (count / total) * 100
                    print(f"  {event!s:6s}: {count:5d} ({pct:5.1f}%)")

            # Calculate imbalance ratio
            if 'train' in cache_results:
                counts = list(cache_results['train'].values())
                if counts:
                    ratio = max(counts) / min(counts)
                    print(f"\nClass imbalance ratio: {ratio:.1f}:1")

        # Also check raw .rec files
        print("\nChecking raw .rec files...")
        events_v201, files_v201 = count_events_in_rec_files(v201_path / "edf")
        if events_v201:
            print("\nRaw annotation counts:")
            total = sum(events_v201.values())
            for event, count in sorted(events_v201.items()):
                pct = (count / total) * 100 if total > 0 else 0
                print(f"  {event:6s}: {count:5d} ({pct:5.1f}%)")
    else:
        print("❌ v2.0.1 not found")

    # Check v2.0.0 (if downloaded)
    print("\n" + "=" * 40)
    print("\n📊 TUEV v2.0.0 (reference version):")
    print("-" * 40)

    v200_path = Path(base_dir) / "tuev_v200"
    if v200_path.exists():
        print("Checking raw .rec files...")
        events_v200, files_v200 = count_events_in_rec_files(v200_path / "edf")
        if events_v200:
            print("\nRaw annotation counts:")
            total = sum(events_v200.values())
            for event, count in sorted(events_v200.items()):
                pct = (count / total) * 100 if total > 0 else 0
                print(f"  {event:6s}: {count:5d} ({pct:5.1f}%)")

            # Calculate imbalance ratio
            counts = list(events_v200.values())
            if counts and min(counts) > 0:
                ratio = max(counts) / min(counts)
                print(f"\nClass imbalance ratio: {ratio:.1f}:1")

            # Compare with v2.0.1
            if events_v201:
                print("\n" + "=" * 40)
                print("\n🔍 VERSION COMPARISON:")
                print("-" * 40)

                comparison = []
                for event in ['spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg']:
                    v200_count = events_v200.get(event, 0)
                    v201_count = events_v201.get(event, 0)
                    diff = v200_count - v201_count
                    pct_change = ((diff / v201_count) * 100) if v201_count > 0 else 0

                    comparison.append(
                        {
                            'Event': event,
                            'v2.0.0': v200_count,
                            'v2.0.1': v201_count,
                            'Diff': diff,
                            'Change%': pct_change,
                        }
                    )

                df = pd.DataFrame(comparison)
                print(df.to_string(index=False))

                print("\n🎯 KEY FINDINGS:")
                # Check if rare events are better in v2.0.0
                rare_events = ['spsw', 'pled', 'eyem', 'artf']
                v200_rare = sum(events_v200.get(e, 0) for e in rare_events)
                v201_rare = sum(events_v201.get(e, 0) for e in rare_events)

                if v200_rare > v201_rare:
                    print(f"✅ v2.0.0 has {v200_rare - v201_rare} MORE rare events!")
                    print("   This could explain the 62% BAC in the paper!")
                elif v200_rare < v201_rare:
                    print(f"❌ v2.0.0 has {v201_rare - v200_rare} FEWER rare events")
                    print("   Version difference is NOT the explanation")
                else:
                    print("➖ Similar rare event counts between versions")
    else:
        print("❌ v2.0.0 not downloaded yet")
        print("\nTo download v2.0.0 for comparison:")
        print("  python scripts/data/download_tuev_v200.py")

    print("\n" + "=" * 80)
    print("\n📝 CONCLUSION:")
    print("-" * 40)
    print("If v2.0.0 has significantly more rare events (especially spsw),")
    print("this would explain why the reference achieves 62% BAC.")
    print("They train on v2.0.0 (better balance) but report on v2.0.1.")
    print("\nThis would be a form of data leakage/selection bias.")
    print("=" * 80)


if __name__ == "__main__":
    main()
