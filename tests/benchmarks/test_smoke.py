"""Smoke benchmark test to ensure benchmark JSON is never empty."""


def test_benchmark_smoke(benchmark):
    """Minimal benchmark that always runs to prevent empty JSON."""

    def noop():
        """No-op function for benchmarking."""
        return 1

    result = benchmark(noop)
    assert result == 1