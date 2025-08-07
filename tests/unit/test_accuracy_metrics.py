"""Accuracy metrics recording for visibility without blocking.

This module provides utilities for recording model accuracy metrics
during test runs, enabling trend monitoring without hard failures.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest


class MetricsRecorder:
    """Records and tracks accuracy metrics across test runs."""

    def __init__(self, metrics_file: Path | None = None):
        """Initialize metrics recorder.

        Args:
            metrics_file: Path to metrics JSON file. Defaults to test_accuracy_metrics.json
        """
        if metrics_file is None:
            metrics_file = Path(__file__).parent.parent / "test_accuracy_metrics.json"
        self.metrics_file = metrics_file
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Ensure metrics file exists with proper structure."""
        if not self.metrics_file.exists():
            self.metrics_file.write_text(json.dumps({"version": "1.0", "metrics": {}}, indent=2))

    def record_metric(self, test_name: str, metric_name: str, value: float) -> None:
        """Record a metric value for a test.

        Args:
            test_name: Name of the test
            metric_name: Name of the metric (e.g., "balanced_accuracy")
            value: Metric value
        """
        data = json.loads(self.metrics_file.read_text())

        # Initialize test entry if needed
        if test_name not in data["metrics"]:
            data["metrics"][test_name] = {}

        # Initialize metric history if needed
        if metric_name not in data["metrics"][test_name]:
            data["metrics"][test_name][metric_name] = []

        # Append new measurement
        data["metrics"][test_name][metric_name].append(
            {
                "value": value,
                "timestamp": datetime.now(UTC).isoformat(),
                "commit": self._get_git_commit(),
            }
        )

        # Keep only last 100 measurements per metric
        data["metrics"][test_name][metric_name] = data["metrics"][test_name][metric_name][-100:]

        self.metrics_file.write_text(json.dumps(data, indent=2))

    def get_baseline(self, test_name: str, metric_name: str) -> float | None:
        """Get baseline (last passing) value for a metric.

        Args:
            test_name: Name of the test
            metric_name: Name of the metric

        Returns:
            Last recorded value or None if no history
        """
        data = json.loads(self.metrics_file.read_text())

        try:
            history = data["metrics"][test_name][metric_name]
            if history:
                return history[-1]["value"]
        except KeyError:
            pass

        return None

    def check_regression(
        self,
        test_name: str,
        metric_name: str,
        current_value: float,
        max_drop_pp: float = 3.0,
    ) -> bool:
        """Check if metric has regressed beyond threshold.

        Args:
            test_name: Name of the test
            metric_name: Name of the metric
            current_value: Current metric value
            max_drop_pp: Maximum allowed drop in percentage points

        Returns:
            True if regression detected, False otherwise
        """
        baseline = self.get_baseline(test_name, metric_name)

        if baseline is None:
            # No baseline, can't detect regression
            return False

        drop_pp = (baseline - current_value) * 100
        return drop_pp > max_drop_pp

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return None


# Global recorder instance
_recorder = MetricsRecorder()


def record_accuracy_metric(test_name: str, metric_name: str, value: float) -> None:
    """Record an accuracy metric (convenience function).

    Args:
        test_name: Name of the test
        metric_name: Name of the metric
        value: Metric value
    """
    _recorder.record_metric(test_name, metric_name, value)


@pytest.fixture
def metrics_recorder() -> MetricsRecorder:
    """Provide metrics recorder for tests."""
    return _recorder
