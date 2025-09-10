#!/usr/bin/env python
"""Validate rebuilt TUEV cache and subject separation.

Checks:
- Train and eval index.json exist and have n_segments > 0
- Subject grouping uses file-level 'subject' fields
- No overlap between train and eval subject sets

Usage:
  python scripts/validate_tuev_cache.py --data_dir data/datasets/tuev [--cache_dir data/datasets/tuev/cache/tuev_event_segments]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_index(cache_dir: Path, split: str) -> dict:
    index_path = cache_dir / split / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index: {index_path}")
    with index_path.open() as f:
        return json.load(f)


def gather_subjects(index: dict) -> set[str]:
    segs = index.get("segments", [])
    return {seg.get("subject", "") for seg in segs if "subject" in seg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Root TUEV data directory containing edf/")
    ap.add_argument(
        "--cache_dir",
        default=None,
        help="Cache directory (defaults to <data_dir>/cache/tuev_event_segments)",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir is not None
        else data_dir / "cache" / "tuev_event_segments"
    )

    train_index = load_index(cache_dir, "train")
    eval_index = load_index(cache_dir, "eval")

    n_train = train_index.get("n_segments", 0)
    n_eval = eval_index.get("n_segments", 0)

    print(f"Train index: {cache_dir / 'train' / 'index.json'}")
    print(f"Eval  index: {cache_dir / 'eval' / 'index.json'}")
    print(f"Train n_segments: {n_train}")
    print(f"Eval  n_segments: {n_eval}")

    if n_train <= 0 or n_eval <= 0:
        print("ERROR: One of the splits has zero segments.")
        return

    train_subjects = gather_subjects(train_index)
    eval_subjects = gather_subjects(eval_index)

    print(f"Train subjects: {len(train_subjects)}")
    print(f"Eval  subjects: {len(eval_subjects)}")

    overlap = train_subjects & eval_subjects
    print(f"Subject overlap count: {len(overlap)}")
    if overlap:
        print(f"Overlapping subjects (first 20): {sorted(list(overlap))[:20]}")
    else:
        print("OK: No subject overlap between train and eval.")


if __name__ == "__main__":
    main()
