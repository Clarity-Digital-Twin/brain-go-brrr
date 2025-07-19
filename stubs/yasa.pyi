"""Type stubs for YASA (Yet Another Sleep Analyzer) library."""

from typing import Any

import mne
import numpy as np
import pandas as pd

def sleep_staging(
    eeg: np.ndarray | mne.io.BaseRaw,
    eog: np.ndarray | None = None,
    emg: np.ndarray | None = None,
    sf: float | None = None,
    metadata: dict[str, Any] | None = None,
    hypno: np.ndarray | None = None,
    include: list[str] | None = None,
    return_proba: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]: ...

class SleepStaging:
    """Automatic sleep staging using a pre-trained classifier."""

    def __init__(
        self,
        raw: mne.io.BaseRaw,
        eeg_name: str | list[str],
        eog_name: str | list[str] | None = None,
        emg_name: str | list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
    def fit(self) -> None: ...
    def predict(self, return_proba: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]: ...
    def predict_proba(self) -> np.ndarray: ...
    def get_features(self) -> pd.DataFrame: ...

def bandpower(
    data: np.ndarray,
    sf: float,
    ch_names: list[str] | None = None,
    hypno: np.ndarray | None = None,
    include: list[str] | None = None,
    win_sec: float = 4.0,
    relative: bool = True,
    bandpass: bool = True,
    bands: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame: ...
def sleep_statistics(hypno: np.ndarray, sf_hyp: float = 1 / 30.0) -> dict[str, float]: ...
def hypno_str_to_int(hypno: np.ndarray) -> np.ndarray: ...
def hypno_int_to_str(hypno: np.ndarray) -> np.ndarray: ...
def spindles_detect(
    data: np.ndarray,
    sf: float,
    ch_names: list[str] | None = None,
    hypno: np.ndarray | None = None,
    include: list[str] | None = None,
    freq_sp: tuple[float, float] = (12, 15),
    duration: tuple[float, float] = (0.5, 2),
    thresh: dict[str, float] | None = None,
) -> pd.DataFrame: ...
def sw_detect(
    data: np.ndarray,
    sf: float,
    ch_names: list[str] | None = None,
    hypno: np.ndarray | None = None,
    include: list[str] | None = None,
    freq_sw: tuple[float, float] = (0.3, 1.5),
    duration: tuple[float, float] = (0.8, 2),
    thresh_ptp: float | None = None,
) -> pd.DataFrame: ...
def rem_detect(
    eog: np.ndarray,
    sf: float,
    hypno: np.ndarray | None = None,
    include: int | None = None,
    amplitude: tuple[float, float] = (50, 325),
    duration: tuple[float, float] = (0.3, 1.2),
    freq_rem: tuple[float, float] = (0.5, 5),
    remove_outliers: bool = False,
) -> pd.DataFrame: ...
def plot_hypnogram(
    hypno: np.ndarray,
    sf_hypno: float = 1 / 30.0,
    lw: float = 1.5,
    fill_color: str | None = None,
    ax: Any | None = None,
) -> Any: ...

__version__: str
