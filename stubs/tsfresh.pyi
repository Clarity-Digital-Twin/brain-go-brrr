"""Type stubs for tsfresh (time series feature extraction) library."""

from typing import Any

import pandas as pd

class utilities:
    class dataframe_functions:
        @staticmethod
        def impute(
            df: pd.DataFrame,
            col_to_max: dict[str, float] | None = None,
            col_to_min: dict[str, float] | None = None,
            col_to_median: dict[str, float] | None = None,
        ) -> pd.DataFrame: ...

def extract_features(
    timeseries_container: pd.DataFrame | dict[str, pd.DataFrame],
    default_fc_parameters: dict[str, Any] | None = None,
    kind_to_fc_parameters: dict[str, dict[str, Any]] | None = None,
    column_id: str | None = None,
    column_sort: str | None = None,
    column_kind: str | None = None,
    column_value: str | None = None,
    chunksize: int | None = None,
    n_jobs: int = 1,
    show_warnings: bool = True,
    disable_progressbar: bool = False,
    impute_function: Any | None = None,
    profile: bool = False,
    profiling_filename: str | None = None,
    profiling_sorting: str = "cumulative",
    distributor: Any | None = None,
) -> pd.DataFrame: ...
def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    test_for_binary_target_binary_feature: str = "fisher",
    test_for_binary_target_real_feature: str = "mann",
    test_for_real_target_binary_feature: str = "mann",
    test_for_real_target_real_feature: str = "kendall",
    fdr_level: float = 0.05,
    hypotheses_independent: bool = False,
    n_jobs: int = 1,
    show_warnings: bool = True,
    chunksize: int | None = None,
    ml_task: str = "auto",
    multiclass: bool = True,
    n_significant: int = 1,
) -> pd.DataFrame: ...

class EfficientFCParameters:
    """Efficient feature calculation parameters."""

    ...

class ComprehensiveFCParameters:
    """Comprehensive feature calculation parameters."""

    ...

class MinimalFCParameters:
    """Minimal feature calculation parameters."""

    ...
