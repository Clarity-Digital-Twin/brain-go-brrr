"""Shared test utilities for brain-go-brrr tests."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


class FakeAutoReject:
    """Mock AutoReject object for testing without the real dependency."""

    def __init__(
        self,
        thresholds: np.ndarray | None = None,
        consensus: list[float] | None = None,
        n_interpolate: list[int] | None = None,
        picks: list[int] | None = None,
    ):
        """Initialize fake AutoReject with parameters.

        Args:
            thresholds: Channel thresholds array
            consensus: Consensus values
            n_interpolate: Interpolation parameters
            picks: Channel picks
        """
        self.threshes_ = thresholds if thresholds is not None else np.random.rand(19, 10)
        self.consensus_ = consensus if consensus is not None else [0.1]
        self.n_interpolate_ = n_interpolate if n_interpolate is not None else [1, 4]
        self.picks_ = picks if picks is not None else list(range(19))

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "FakeAutoReject":
        """Create from parameter dictionary.

        Args:
            params: Dictionary with keys 'thresholds', 'consensus', etc.

        Returns:
            FakeAutoReject instance
        """
        return cls(
            thresholds=params.get("thresholds"),
            consensus=params.get("consensus"),
            n_interpolate=params.get("n_interpolate"),
            picks=params.get("picks"),
        )


# Metrics recording utilities (moved from test_accuracy_metrics.py)
def record_accuracy_metric(test_name: str, metric_name: str, value: float) -> None:
    """Record an accuracy metric for trend monitoring.
    
    Args:
        test_name: Name of the test
        metric_name: Name of the metric (e.g., "balanced_accuracy")
        value: Metric value
    """
    metrics_file = Path(__file__).parent / "test_accuracy_metrics.json"

    # Ensure file exists
    if not metrics_file.exists():
        metrics_file.write_text(json.dumps({"version": "1.0", "metrics": {}}, indent=2))

    # Load existing data
    data = json.loads(metrics_file.read_text())

    # Initialize structures if needed
    if test_name not in data["metrics"]:
        data["metrics"][test_name] = {}
    if metric_name not in data["metrics"][test_name]:
        data["metrics"][test_name][metric_name] = []

    # Record metric
    data["metrics"][test_name][metric_name].append({
        "value": value,
        "timestamp": datetime.now(UTC).isoformat()
    })

    # Save updated data
    metrics_file.write_text(json.dumps(data, indent=2))
