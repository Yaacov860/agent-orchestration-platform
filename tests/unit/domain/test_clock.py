from datetime import UTC, datetime, timedelta

from orchestrator.domain.clock import seconds_left

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=UTC)


def test_no_deadline_means_no_limit() -> None:
    assert seconds_left(None, NOW) is None


def test_time_left_before_the_deadline() -> None:
    assert seconds_left(NOW + timedelta(seconds=90), NOW) == 90.0


def test_a_passed_deadline_leaves_zero_not_a_negative_number() -> None:
    assert seconds_left(NOW - timedelta(hours=1), NOW) == 0.0
