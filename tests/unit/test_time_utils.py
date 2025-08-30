"""Tests for time utilities - REAL BEHAVIORAL TESTS, NO MOCKING."""

import re
from datetime import datetime, timedelta, timezone

from brain_go_brrr.utils.time import UTC, format_timestamp, timestamp_for_logging, utc_now


class TestTimeUtils:
    """Test time utility functions BEHAVIOR."""

    def test_utc_now_returns_aware_datetime(self):
        """Test utc_now returns timezone-aware UTC datetime."""
        now = utc_now()

        # Should be datetime
        assert isinstance(now, datetime)

        # Should be timezone-aware
        assert now.tzinfo is not None

        # Should be UTC
        assert now.tzinfo == UTC
        # Also check against standard UTC
        assert now.tzinfo == timezone.utc

    def test_utc_now_is_current_time(self):
        """Test utc_now returns current time (within reason)."""
        before = datetime.now(timezone.utc)
        now = utc_now()
        after = datetime.now(timezone.utc)

        # Should be between before and after (allowing small tolerance)
        assert before <= now <= after + timedelta(seconds=1)

    def test_format_timestamp_default(self):
        """Test format_timestamp with default (current time)."""
        timestamp = format_timestamp()

        # Should be ISO format
        assert "T" in timestamp  # ISO separator
        assert len(timestamp) > 20  # Should have date, time, timezone

        # Should be parseable back to datetime
        parsed = datetime.fromisoformat(timestamp)
        assert parsed.tzinfo is not None

    def test_format_timestamp_with_datetime(self):
        """Test format_timestamp with specific datetime."""
        dt = datetime(2024, 1, 15, 12, 30, 45, 123456, tzinfo=timezone.utc)
        timestamp = format_timestamp(dt)

        # Should format correctly
        assert timestamp == "2024-01-15T12:30:45.123456+00:00"

        # Should round-trip
        parsed = datetime.fromisoformat(timestamp)
        assert parsed == dt

    def test_format_timestamp_none_uses_current(self):
        """Test format_timestamp(None) uses current time."""
        before = utc_now()
        timestamp = format_timestamp(None)
        after = utc_now()

        # Parse back
        parsed = datetime.fromisoformat(timestamp)

        # Should be between before and after
        assert before <= parsed <= after + timedelta(seconds=1)

    def test_timestamp_for_logging_format(self):
        """Test timestamp_for_logging returns expected format."""
        timestamp = timestamp_for_logging()

        # Check format: YYYY-MM-DD HH:MM:SS UTC
        pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC$"
        assert re.match(pattern, timestamp), f"Unexpected format: {timestamp}"

        # Should be current time (within reason)
        # Extract datetime part
        dt_str = timestamp.replace(" UTC", "")
        parsed = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        parsed = parsed.replace(tzinfo=timezone.utc)

        now = utc_now()
        diff = abs((now - parsed).total_seconds())
        assert diff < 2, f"Time diff too large: {diff} seconds"

    def test_utc_constant_compatibility(self):
        """Test UTC constant is compatible with timezone.utc."""
        # UTC should be the same as timezone.utc
        assert timezone.utc == UTC

        # Should work in datetime operations
        dt1 = datetime.now(UTC)
        dt2 = datetime.now(timezone.utc)

        # Both should have same timezone
        assert dt1.tzinfo == dt2.tzinfo

    def test_timestamps_are_sortable(self):
        """Test that format_timestamp output is sortable chronologically."""
        # Create timestamps with small delays
        timestamps = []
        for _ in range(5):
            timestamps.append(format_timestamp())
            # Small sleep would go here in real test

        # ISO format should sort lexicographically = chronologically
        sorted_timestamps = sorted(timestamps)
        assert sorted_timestamps == timestamps  # Should already be in order

    def test_timestamp_precision(self):
        """Test timestamp includes microsecond precision."""
        dt = datetime(2024, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc)
        timestamp = format_timestamp(dt)

        # Should include microseconds
        assert ".123456" in timestamp

    def test_logging_timestamp_readable(self):
        """Test logging timestamp is human-readable."""
        timestamp = timestamp_for_logging()

        # Should be readable format
        parts = timestamp.split()
        assert len(parts) == 3  # date, time, UTC

        # Date should be YYYY-MM-DD
        assert len(parts[0]) == 10
        assert parts[0].count("-") == 2

        # Time should be HH:MM:SS
        assert len(parts[1]) == 8
        assert parts[1].count(":") == 2

        # Should end with UTC
        assert parts[2] == "UTC"


class TestTimeConsistency:
    """Test consistency and reliability of time functions."""

    def test_multiple_calls_increase_monotonically(self):
        """Test multiple utc_now calls return increasing times."""
        times = []
        for _ in range(10):
            times.append(utc_now())

        # Each should be >= previous (monotonic)
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1]

    def test_format_consistency(self):
        """Test format functions are consistent for same input."""
        dt = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)

        # Multiple calls should give same result
        result1 = format_timestamp(dt)
        result2 = format_timestamp(dt)
        assert result1 == result2

    def test_timezone_awareness_preserved(self):
        """Test timezone info is never lost."""
        # Start with aware datetime
        original = utc_now()
        assert original.tzinfo is not None

        # Format and parse back
        formatted = format_timestamp(original)
        parsed = datetime.fromisoformat(formatted)

        # Should still be aware
        assert parsed.tzinfo is not None
        # Should be equivalent (within microsecond precision)
        assert abs((original - parsed).total_seconds()) < 0.001

    def test_utc_everywhere(self):
        """Test all functions use UTC consistently."""
        # utc_now should be UTC
        now = utc_now()
        assert now.utcoffset() == timedelta(0)

        # format_timestamp should preserve UTC
        formatted = format_timestamp(now)
        assert "+00:00" in formatted or "Z" in formatted

        # timestamp_for_logging should indicate UTC
        log_ts = timestamp_for_logging()
        assert "UTC" in log_ts
