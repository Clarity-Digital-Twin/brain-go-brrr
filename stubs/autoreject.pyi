"""Type stubs for autoreject library."""

from typing import Any

import mne

class AutoReject:
    """Automated rejection and repair of bad trials in MEG/EEG."""

    def __init__(
        self,
        n_interpolate: list[int] | None = None,
        consensus: float | None = None,
        n_jobs: int = 1,
        random_state: int | None = None,
        picks: Any | None = None,
        thresh_method: str = "bayesian_optimization",
        cv: int = 10,
        thresh_func: Any | None = None,
        verbose: bool = True,
    ) -> None: ...
    def fit(self, epochs: mne.Epochs) -> AutoReject: ...
    def transform(self, epochs: mne.Epochs) -> mne.Epochs: ...
    def fit_transform(self, epochs: mne.Epochs) -> mne.Epochs: ...
    def get_reject_log(self, epochs: mne.Epochs) -> Any: ...
