"""TUSZ temporal detection dataset with sliding windows.

This dataset yields fixed-length windows and binary labels (seizure/background)
based on per-window seizure duration fraction.

Notes:
- Expects the standard TUSZ directory layout under `root_dir/<split>/**.edf` and
  corresponding annotation files (`.tse` or CSV) alongside recordings.
- Uses a single sampling-rate SSOT (default 256 Hz) and a fixed channel policy.
- Patient-wise splitting is assumed to be represented by the folder structure; do
  not mix train/dev/test patients.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import mne
except Exception:  # pragma: no cover - optional dependency
    mne = None

from brain_go_brrr.infra.data.channels import (
    CHANNEL_ALIASES,
    CHANNELS_TUAB_19,
    map_to_canonical,
)
from brain_go_brrr.infra.data.tusz_labels import is_seizure_label, merge_intervals

if TYPE_CHECKING:
    from collections.abc import Iterable


# Label policy is centralized in infra.data.tusz_labels.is_seizure_label


@dataclass(frozen=True)
class WindowConfig:
    fs: int = 256
    window_sec: float = 12.0
    stride_sec: float = 1.0
    positive_fraction: float = 0.2  # label positive if >= 20% of window is seizure


def _standardize_channel_name(name: str) -> str:
    return CHANNEL_ALIASES.get(name, name)


def _parse_tse(path: Path, include_pnes: bool = False) -> list[tuple[float, float]]:
    """Parse TSE file for seizure annotations ONLY.

    TSE format:
    - start_time end_time [label]
    - Recognizes TUSZ seizure codes (fnsz, gnsz, etc.) and generic 'seiz' labels
    - Background/artifact/other labels are ignored
    """
    events: list[tuple[float, float]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            # Check if this is a seizure annotation
            label = " ".join(parts[2:]) if len(parts) > 2 else ""
            if is_seizure_label(label, include_pnes=include_pnes):
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    if end > start:
                        events.append((start, end))
                except ValueError:
                    continue
    return merge_intervals(events)


def _parse_csv(path: Path, include_pnes: bool = False) -> list[tuple[float, float]]:
    """Parse CSV sidecar for seizure annotations ONLY.

    CSV format can be either:
    - channel,start_time,stop_time,label,confidence (TUSZ format)
    - start,end,label (simple format)

    Recognizes TUSZ seizure codes and generic 'seiz' labels.
    """
    events: list[tuple[float, float]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            # Skip header if present
            if line_num == 1 and ("start" in s.lower() or "channel" in s.lower()):
                continue

            # Split on comma first, then fallback to whitespace
            parts = [p for p in s.replace(",", " ").split() if p]

            # TUSZ CSV format: channel,start,stop,label,confidence
            # Simple format: start,end,label
            if len(parts) >= 3:
                try:
                    # Check if first field is a channel name (contains letters)
                    if any(c.isalpha() for c in parts[0]):
                        # TUSZ format - skip channel field
                        if len(parts) >= 4:
                            start = float(parts[1])
                            end = float(parts[2])
                            label = parts[3] if len(parts) > 3 else ""
                    else:
                        # Simple format
                        start = float(parts[0])
                        end = float(parts[1])
                        label = " ".join(parts[2:]) if len(parts) > 2 else ""

                    # Check if this is a seizure annotation
                    if is_seizure_label(label, include_pnes=include_pnes) and end > start:
                        events.append((start, end))
                except ValueError:
                    continue
    return merge_intervals(events)


def _events_to_mask(
    events: Iterable[tuple[float, float]], duration_sec: float, fs: int
) -> npt.NDArray[np.bool_]:
    n = round(duration_sec * fs)
    mask = np.zeros(n, dtype=bool)
    for s, e in events:
        i0 = max(0, round(s * fs))
        i1 = min(n, round(e * fs))
        if i1 > i0:
            mask[i0:i1] = True
    return mask


class TUSZDetectionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window temporal detection dataset for TUSZ.

    Yields: (window, label) where window is a `(C, T)` float32 tensor in Volts at `cfg.fs`,
    and label is a scalar int64 tensor (`1` for seizure, `0` for background).
    """

    def __init__(
        self,
        root_dir: Path | str,
        split: str = "train",
        cfg: WindowConfig | None = None,
        target_channels: list[str] | None = None,
        max_windows: int | None = None,
        preprocessor: SeizurePreprocessor
        | None = None,  # applied to full recording before windowing
        ensure_unipolar: bool = False,
    ) -> None:
        """Initialize the TUSZ detection dataset.

        Args:
            root_dir: Root directory containing TUSZ dataset splits.
            split: Dataset split ('train', 'dev', 'test').
            cfg: Window configuration for sliding window extraction.
            target_channels: List of target channel names to use.
            max_windows: Maximum number of windows to index (for memory efficiency).
        """
        if mne is None:  # pragma: no cover
            raise RuntimeError("mne is required to load TUSZ EDF files. Install with `uv add mne`.")

        self.root_dir = Path(root_dir)
        self.split = split
        self.cfg = cfg or WindowConfig()
        if self.cfg.fs != 256:
            raise ValueError(f"TUSZDetectionDataset expects cfg.fs == 256; got {self.cfg.fs}")
        # Default to TUAB 19-channel set as our SSOT; adjust if needed
        self.target_channels = target_channels or CHANNELS_TUAB_19
        self.max_windows = max_windows
        self.preprocessor = preprocessor
        self.ensure_unipolar = ensure_unipolar
        self.include_pnes = False

        self._records: list[dict[str, Any]] = []
        self._index: list[tuple[int, int]] = []  # (record_idx, window_start_sample@cfg.fs)

        self._discover_records()
        self._build_index()

    def _discover_records(self) -> None:
        split_dir = self.root_dir / self.split
        edfs = sorted(split_dir.rglob("*.edf"))
        for edf in edfs:
            tse = edf.with_suffix(".tse")
            csv = edf.with_suffix(".csv")
            events: list[tuple[float, float]] = []
            if tse.exists():
                events = _parse_tse(tse, include_pnes=self.include_pnes)
            elif csv.exists():
                events = _parse_csv(csv, include_pnes=self.include_pnes)
            self._records.append(
                {
                    "edf": edf,
                    "events": events,
                }
            )

    def _build_index(self) -> None:
        fs = self.cfg.fs
        win = round(self.cfg.window_sec * fs)
        stride = round(self.cfg.stride_sec * fs)
        print(f"Building index for {len(self._records)} recordings...")

        for ridx, rec in enumerate(tqdm(self._records, desc="Indexing recordings")):
            if self.max_windows and len(self._index) >= self.max_windows:
                warnings.warn(
                    f"Reached max_windows={self.max_windows}, stopping indexing", stacklevel=2
                )
                break

            # CRITICAL FIX: Don't preload, just get info
            try:
                raw = mne.io.read_raw_edf(str(rec["edf"]), preload=False, verbose="ERROR")
                duration_sec = float(raw.n_times) / float(raw.info["sfreq"])
                # Immediately delete raw to free memory
                del raw
            except Exception as e:
                warnings.warn(f"Skipping {rec['edf']}: {e}", stacklevel=2)
                continue

            n_target = round(duration_sec * fs)
            if n_target < win:
                continue
            starts = range(0, n_target - win + 1, stride)
            for s in starts:
                if self.max_windows and len(self._index) >= self.max_windows:
                    break
                self._index.append((ridx, s))

        print(f"Indexed {len(self._index)} windows from {len(self._records)} recordings")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ridx, s0_fs = self._index[idx]
        rec = self._records[ridx]
        edf: Path = rec["edf"]
        events: list[tuple[float, float]] = rec["events"]

        # Load raw
        raw = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")
        src_fs_f = float(raw.info["sfreq"])  # original Hz
        src_fs = int(round(src_fs_f))

        # Montage validation (heuristic)
        if self.ensure_unipolar:
            for ch_name in raw.ch_names:
                if "-" in ch_name and not ch_name.startswith("EEG"):
                    raise ValueError(f"Non-unipolar montage detected in {edf}")

        # Channel selection (alias normalization)
        raw.rename_channels(_standardize_channel_name)

        # Compute window bounds in target fs
        win = round(self.cfg.window_sec * self.cfg.fs)
        s1_fs = s0_fs + win

        # Preprocess entire recording if SSOT preprocessor is provided
        if self.preprocessor is not None:
            # Pull available target channels in original order
            available = [ch for ch in self.target_channels if ch in raw.ch_names]
            if available:
                full = raw.get_data(picks=available)  # (C, T_src)
            else:
                full = np.zeros((0, raw.n_times), dtype=np.float32)

            # Map to canonical order with zero-fill for missing (SSOT mapper BEFORE preprocessing)
            prepared, _ = map_to_canonical(full, available, self.target_channels)

            # Apply SSOT preprocessing (z-score → resample → bandpass → notch)
            full_proc = self.preprocessor.preprocess(prepared, fs_original=src_fs)

            # Slice window in target fs
            data = full_proc[:, s0_fs:s1_fs]
        else:
            # Fallback: operate with MNE resampling and per-window extraction
            if self.cfg.fs and round(src_fs_f) != self.cfg.fs:
                raw.resample(self.cfg.fs)

            # Prepare data in canonical order with zero-fill for missing channels
            n_expected_channels = len(self.target_channels)
            data = np.zeros((n_expected_channels, win), dtype=np.float32)
            for i, ch in enumerate(self.target_channels):
                if ch in raw.ch_names:
                    ch_idx = raw.ch_names.index(ch)
                    # Use get_data to respect MNE scaling and preload behavior
                    data[i] = raw.get_data(picks=[ch_idx], start=s0_fs, stop=s1_fs)[0]

        # Label window: fraction of seizure time >= threshold
        # Events are in seconds; compute recording duration in seconds (always in seconds)
        duration_sec = float(raw.n_times) / float(raw.info["sfreq"])  # seconds
        mask = _events_to_mask(events, duration_sec, self.cfg.fs)
        frac = float(mask[s0_fs:s1_fs].mean()) if s1_fs <= mask.shape[0] else 0.0
        y = 1 if frac >= self.cfg.positive_fraction else 0

        x = torch.from_numpy(data.astype(np.float32))
        label = torch.tensor(y, dtype=torch.int64)
        return x, label


__all__ = ["TUSZDetectionDataset", "WindowConfig"]
