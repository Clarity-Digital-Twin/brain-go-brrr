"""Type aliases and protocols for brain_go_brrr.

This module provides clean type annotations without requiring mypy to resolve
external libraries like MNE that don't have type stubs.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

# Common type aliases
FloatArray: TypeAlias = npt.NDArray[np.float64]
Float32Array: TypeAlias = npt.NDArray[np.float32]
StrArray: TypeAlias = npt.NDArray[np.str_]
IntArray: TypeAlias = npt.NDArray[np.int_]


class MNEInfo(Protocol):
    """Protocol for MNE Info objects."""

    def __getitem__(self, key: str) -> Any: ...
    def get(self, key: str, default: Any = None) -> Any: ...
    @property
    def ch_names(self) -> list[str]: ...
    @property
    def sfreq(self) -> float: ...


class MNERaw(Protocol):
    """Protocol for MNE Raw objects."""

    info: MNEInfo
    ch_names: list[str]

    def copy(self) -> MNERaw: ...
    def get_data(
        self,
        picks: list[str] | None = None,
        start: int | None = None,
        stop: int | None = None,
        reject_by_annotation: bool | str = "omit",
        return_times: bool = False,
        units: str | dict[str, str] | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        verbose: bool | str | int | None = None,
    ) -> FloatArray: ...
    def filter(
        self,
        l_freq: float | None,
        h_freq: float | None,
        picks: list[str] | None = None,
        filter_length: str | int = "auto",
        l_trans_bandwidth: str | float = "auto",
        h_trans_bandwidth: str | float = "auto",
        n_jobs: int | str = 1,
        method: str = "fir",
        iir_params: dict[str, Any] | None = None,
        phase: str = "zero",
        fir_window: str = "hamming",
        fir_design: str = "firwin",
        skip_by_annotation: str | list[str] | tuple[str, ...] = ("edge", "bad_acq_skip"),
        pad: str = "edge",
        verbose: bool | str | int | None = None,
    ) -> MNERaw: ...
    def resample(
        self,
        sfreq: float,
        npad: str | int = "auto",
        window: str | tuple[str, float] = "boxcar",
        stim_picks: list[str] | None = None,
        n_jobs: int | str = 1,
        events: IntArray | None = None,
        pad: str = "edge",
        verbose: bool | str | int | None = None,
    ) -> MNERaw: ...
    def pick(
        self,
        picks: list[str] | list[int] | str | None = None,
        exclude: list[str] | str = "bads",
        verbose: bool | str | int | None = None,
    ) -> MNERaw: ...
    def pick_channels(
        self, ch_names: list[str], ordered: bool = False, verbose: bool | str | int | None = None
    ) -> MNERaw: ...
    def rename_channels(
        self,
        mapping: dict[str, str] | Any,
        allow_duplicates: bool = False,
        verbose: bool | str | int | None = None,
    ) -> None: ...
    def set_channel_types(
        self, mapping: dict[str, str], verbose: bool | str | int | None = None
    ) -> MNERaw: ...
    def load_data(self, verbose: bool | str | int | None = None) -> MNERaw: ...
    def drop_channels(self, ch_names: list[str], on_missing: str = "raise") -> MNERaw: ...
    def set_eeg_reference(
        self,
        ref_channels: str | list[str] = "average",
        projection: bool = False,
        ch_type: str = "auto",
        forward: Any = None,
        verbose: bool | str | int | None = None,
    ) -> tuple[MNERaw, FloatArray]: ...
    def notch_filter(
        self,
        freqs: float | list[float],
        picks: list[str] | None = None,
        filter_length: str | int = "auto",
        notch_widths: float | list[float] | None = None,
        trans_bandwidth: float = 1.0,
        n_jobs: int | str = 1,
        method: str = "fir",
        iir_params: dict[str, Any] | None = None,
        mt_bandwidth: float | None = None,
        p_value: float = 0.05,
        phase: str = "zero",
        fir_window: str = "hamming",
        fir_design: str = "firwin",
        pad: str = "edge",
        verbose: bool | str | int | None = None,
    ) -> MNERaw: ...
    @property
    def times(self) -> FloatArray: ...
    @property
    def n_times(self) -> int: ...


class MNEEpochs(Protocol):
    """Protocol for MNE Epochs objects."""

    info: MNEInfo
    ch_names: list[str]
    events: IntArray
    event_id: dict[str, int] | None

    def copy(self) -> MNEEpochs: ...
    def get_data(
        self,
        picks: list[str] | None = None,
        item: int | slice | list[int] | None = None,
        units: str | dict[str, str] | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        verbose: bool | str | int | None = None,
    ) -> FloatArray: ...
    def drop_bad(
        self,
        reject: dict[str, float] | None = None,
        flat: dict[str, float] | None = None,
        verbose: bool | str | int | None = None,
    ) -> MNEEpochs: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> FloatArray: ...
    @property
    def times(self) -> FloatArray: ...
    @property
    def tmin(self) -> float: ...
    @property
    def tmax(self) -> float: ...
