"""Time, for deadlines. One definition shared by planning and execution."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TypeAlias

Clock: TypeAlias = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


def seconds_left(deadline: datetime | None, now: datetime) -> float | None:
    """Seconds until `deadline` (0 once it has passed), or None when there is no deadline."""
    if deadline is None:
        return None
    return max(0.0, (deadline - now).total_seconds())
