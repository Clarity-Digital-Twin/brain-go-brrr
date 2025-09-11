"""TUSZ temporal detection dataset with sliding windows.

This dataset yields fixed-length windows and binary labels (seizure/background)
based on per-window seizure duration fraction.

Notes
- Expects the standard TUSZ directory layout under `root_dir/<split>/**.edf` and
  corresponding annotation files (`.tse` or CSV) alongside recordings.
- Uses a single sampling-rate SSOT (default 256 Hz) and a fixed channel policy.
- Patient-wise splitting is assumed to be represented by the folder structure; do
  not mix train/dev/test patients.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

try:
    import mne  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency
    mne = None  # type: ignore[assignment]

from brain_go_brrr.infra.data.channels import CHANNELS_TUAB_19, CHANNEL_ALIASES


@dataclass(frozen=True)
class WindowConfig:
    fs: int = 256
    window_sec: float = 12.0
    stride_sec: float = 1.0
    positive_fraction: float = 0.2  # label positive if >= 20% of window is seizure


def _standardize_channel_name(name: str) -> str:
    return CHANNEL_ALIASES.get(name, name)


def _parse_tse(path: Path) -> list[tuple[float, float]]:
    events: list[tuple[float, float]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            # Heuristic: keep any label with 'seiz' when present
            label = parts[2].lower() if len(parts) > 2 else ""
            if "seiz" in label or len(parts) >= 2:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    if end > start:
                        events.append((start, end))
                except ValueError:
                    continue
    return events


def _events_to_mask(
    events: Iterable[tuple[float, float]], duration_sec: float, fs: int
) -> npt.NDArray[np.bool_]:
    n = int(round(duration_sec * fs))
    mask = np.zeros(n, dtype=bool)
    for s, e in events:
        i0 = max(0, int(round(s * fs)))
        i1 = min(n, int(round(e * fs)))
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
    ) -> None:
        if mne is None:  # pragma: no cover
            raise RuntimeError(
                "mne is required to load TUSZ EDF files. Install with `uv add mne`."
            )

        self.root_dir = Path(root_dir)
        self.split = split
        self.cfg = cfg or WindowConfig()
        # Default to TUAB 19-channel set as our SSOT; adjust if needed
        self.target_channels = target_channels or CHANNELS_TUAB_19

        self._records: list[dict] = []
        self._index: list[tuple[int, int]] = []  # (record_idx, window_start_sample@cfg.fs)

        self._discover_records()
        self._build_index()

    def _discover_records(self) -> None:
        split_dir = self.root_dir / self.split
        edfs = sorted(split_dir.rglob("*.edf"))
        for edf in edfs:
            tse = edf.with_suffix(".tse")
            csv = edf.with_suffix(".csv")
            events = _parse_tse(tse) if tse.exists() else _parse_tse(csv)
            self._records.append({
                "edf": edf,
                "events": events,
            })

    def _build_index(self) -> None:
        fs = self.cfg.fs
        win = int(round(self.cfg.window_sec * fs))
        stride = int(round(self.cfg.stride_sec * fs))
        for ridx, rec in enumerate(self._records):
            raw = mne.io.read_raw_edf(str(rec["edf"]), preload=False, verbose="ERROR")
            duration_sec = float(raw.n_samples) / float(raw.info["sfreq"])  # type: ignore[index]
            n_target = int(round(duration_sec * fs))
            if n_target < win:
                continue
            starts = range(0, n_target - win + 1, stride)
            for s in starts:
                self._index.append((ridx, s))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ridx, s0_fs = self._index[idx]
        rec = self._records[ridx]
        edf: Path = rec["edf"]
        events: list[tuple[float, float]] = rec["events"]

        # Load raw (lazy) and resample if needed
        raw = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")
        src_fs = float(raw.info["sfreq"])  # type: ignore[index]

        # Channel selection (alias normalization)
        raw.rename_channels(_standardize_channel_name)  # type: ignore[attr-defined]
        available = [ch for ch in self.target_channels if ch in raw.ch_names]
        raw.pick(available, ordered=True)  # type: ignore[attr-defined]
        if self.cfg.fs and int(round(src_fs)) != self.cfg.fs:
            raw.resample(self.cfg.fs)  # type: ignore[attr-defined]

        # Compute window bounds in target fs
        win = int(round(self.cfg.window_sec * self.cfg.fs))
        s1_fs = s0_fs + win
        data = raw.get_data(start=s0_fs, stop=s1_fs)  # shape (C, T) in Volts

        # Label window: fraction of seizure time >= threshold
        duration_sec = float(raw.n_times) / float(self.cfg.fs)  # type: ignore[attr-defined]
        mask = _events_to_mask(events, duration_sec, self.cfg.fs)
        frac = float(mask[s0_fs:s1_fs].mean()) if s1_fs <= mask.shape[0] else 0.0
        y = 1 if frac >= self.cfg.positive_fraction else 0

        x = torch.from_numpy(data.astype(np.float32))
        label = torch.tensor(y, dtype=torch.int64)
        return x, label


__all__ = ["TUSZDetectionDataset", "WindowConfig"]

