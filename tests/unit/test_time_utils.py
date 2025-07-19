"""Tests for time utilities."""

from datetime import UTC, datetime

from brain_go_brrr.utils.time import format_timestamp, timestamp_for_logging, utc_now


class TestTimeUtils:
    """Test time utility functions."""

    def test_utc_now_returns_utc_time(self):
        """Test that utc_now returns a UTC datetime."""
        now = utc_now()
        assert isinstance(now, datetime)
        assert now.tzinfo == UTC

    def test_utc_now_is_current_time(self):
        """Test that utc_now returns current time."""
        before = datetime.now(UTC)
        now = utc_now()
        after = datetime.now(UTC)

        # Should be between before and after
        assert before <= now <= after

    def test_format_timestamp_default(self):
        """Test format_timestamp with default (current time)."""
        timestamp = format_timestamp()

        # Should be ISO format with timezone
        assert isinstance(timestamp, str)
        assert "T" in timestamp  # ISO format separator
        assert "+" in timestamp or "Z" in timestamp  # Timezone indicator

    def test_format_timestamp_with_datetime(self):
        """Test format_timestamp with specific datetime."""
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        timestamp = format_timestamp(dt)

        assert timestamp == "2024-01-15T10:30:45+00:00"

    def test_format_timestamp_none_uses_current_time(self):
        """Test that None uses current time."""
        before = utc_now()
        timestamp = format_timestamp(None)
        after = utc_now()

        # Parse the timestamp back
        from datetime import datetime as dt

        parsed = dt.fromisoformat(timestamp)

        # Should be between before and after
        assert before <= parsed <= after

    def test_timestamp_for_logging_format(self):
        """Test timestamp_for_logging returns correct format."""
        timestamp = timestamp_for_logging()

        # Should match pattern: YYYY-MM-DD HH:MM:SS UTC
        assert isinstance(timestamp, str)
        assert " UTC" in timestamp
        assert len(timestamp) == 23  # "2024-01-15 10:30:45 UTC"

        # Verify format
        parts = timestamp.split()
        assert len(parts) == 3
        assert parts[2] == "UTC"

        # Date part
        date_parts = parts[0].split("-")
        assert len(date_parts) == 3
        assert len(date_parts[0]) == 4  # Year
        assert len(date_parts[1]) == 2  # Month
        assert len(date_parts[2]) == 2  # Day

        # Time part
        time_parts = parts[1].split(":")
        assert len(time_parts) == 3
        assert len(time_parts[0]) == 2  # Hour
        assert len(time_parts[1]) == 2  # Minute
        assert len(time_parts[2]) == 2  # Second

    def test_time_functions_consistency(self):
        """Test that all time functions use consistent timezone."""
        # Get timestamps from different functions
        now = utc_now()
        iso_timestamp = format_timestamp(now)
        _ = timestamp_for_logging()  # Used for validation that it runs without error

        # Parse ISO timestamp
        from datetime import datetime as dt

        parsed_iso = dt.fromisoformat(iso_timestamp)

        # Both should be UTC
        assert parsed_iso.tzinfo == UTC
        assert now.tzinfo == UTC

        # Should be the same time (allowing small difference)
        time_diff = abs((parsed_iso - now).total_seconds())
        assert time_diff < 0.01  # Less than 10ms difference
