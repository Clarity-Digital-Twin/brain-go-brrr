"""Time utilities for Brain-Go-Brrr.

Following Clean Code principles - single source of truth for time operations.
"""

import sys
from datetime import datetime, timezone

# Python 3.11+ has UTC constant; older versions use timezone.utc
if sys.version_info >= (3, 11):  # noqa: UP036
    from datetime import UTC
else:
    UTC = timezone.utc


def utc_now() -> datetime:
    """Get current UTC datetime.

    Single source of truth for timezone-aware current time.
    Replaces all instances of datetime.now(timezone.utc).

    Returns:
        Current UTC datetime with timezone info
    """
    return datetime.now(UTC)


def format_timestamp(dt: datetime | None = None) -> str:
    """Format datetime as ISO timestamp.

    Args:
        dt: Datetime to format, defaults to utc_now()

    Returns:
        ISO formatted timestamp string
    """
    if dt is None:
        dt = utc_now()
    return dt.isoformat()


def timestamp_for_logging() -> str:
    """Get timestamp formatted for logging.

    Returns:
        Timestamp string suitable for log entries
    """
    return utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")
