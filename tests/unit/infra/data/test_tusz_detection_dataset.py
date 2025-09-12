from pathlib import Path
from typing import Any

import numpy as np
import pytest

from brain_go_brrr.infra.data.tusz_detection_dataset import (
    TUSZDetectionDataset,
    WindowConfig,
)


class _FakeRaw:
    def __init__(self, data: np.ndarray, sfreq: float, ch_names: list[str]):
        self._data = data
        self.info = {"sfreq": sfreq}
        self.n_times = data.shape[1]
        self.ch_names = list(ch_names)

    # MNE-like API subset
    def rename_channels(self, func: Any) -> None:
        self.ch_names = [func(ch) for ch in self.ch_names]

    def pick_channels(self, picks: list[str], ordered: bool = True) -> None:  # noqa: ARG002
        idx = [self.ch_names.index(ch) for ch in picks if ch in self.ch_names]
        self._data = self._data[idx, :]
        self.ch_names = [self.ch_names[i] for i in idx]

    def resample(self, new_fs: int) -> None:
        # Simple nearest-neighbor resample to keep test lightweight
        old_fs = int(self.info["sfreq"])  # type: ignore[index]
        if new_fs == old_fs:
            return
        ratio = new_fs / old_fs
        new_len = int(self._data.shape[1] * ratio)
        idx = (np.arange(new_len) / ratio).astype(int)
        idx = np.clip(idx, 0, self._data.shape[1] - 1)
        self._data = self._data[:, idx]
        self.info["sfreq"] = float(new_fs)  # type: ignore[index]
        self.n_times = self._data.shape[1]

    def get_data(self, start: int, stop: int) -> np.ndarray:
        return self._data[:, start:stop]


@pytest.mark.unit
@pytest.mark.synth
def test_tusz_detection_dataset_minimal_flow(tmp_path: Path, monkeypatch):
    # Create a fake TUSZ layout with one EDF and matching TSE
    root = tmp_path / "v2.0.1"
    split_dir = root / "train"
    split_dir.mkdir(parents=True)
    edf_path = split_dir / "patient1.edf"
    edf_path.write_text("placeholder")
    tse_path = split_dir / "patient1.tse"
    # Seizure from 0.0 to 2.0 seconds
    tse_path.write_text("0.0 2.0 seiz\n")

    # Fake mne.io.read_raw_edf
    def _fake_read_raw_edf(path: str, preload: bool = False, verbose: str = "ERROR") -> _FakeRaw:  # noqa: ARG001
        n_channels = 19
        fs = 256
        duration_sec = 10
        n_samples = fs * duration_sec
        data = np.zeros((n_channels, n_samples), dtype=np.float32)
        ch_names = [
            "Fp1",
            "Fp2",
            "F7",
            "F3",
            "F4",
            "F8",
            "T7",
            "C3",
            "Cz",
            "C4",
            "T8",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "O1",
            "Oz",
            "O2",
        ]
        return _FakeRaw(data=data, sfreq=float(fs), ch_names=ch_names)

    # Patch the dataset module's mne symbol to a minimal stub
    import types
    from brain_go_brrr.infra.data import tusz_detection_dataset as tdd

    fake_mne = types.SimpleNamespace(io=types.SimpleNamespace(read_raw_edf=_fake_read_raw_edf))
    monkeypatch.setattr(tdd, "mne", fake_mne, raising=True)

    cfg = WindowConfig(fs=256, window_sec=1.0, stride_sec=1.0, positive_fraction=0.2)
    ds = TUSZDetectionDataset(root_dir=root, split="train", cfg=cfg)
    assert len(ds) > 0

    x, y = ds[0]
    # x shape is (C, T)
    assert x.shape[0] == 19
    assert x.shape[1] == int(cfg.window_sec * cfg.fs)
    # First window overlaps fully with 0-2s seizure => positive label
    assert int(y.item()) == 1
