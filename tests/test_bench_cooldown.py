"""The cooldown gate (issue #2).

Its decision half is a pure function over one temperature reading, so the whole
policy - including "a laptop under sustained load may never get there" - is
testable without a GPU and without waiting minutes for one to cool.
"""

import pytest

from bench.cooldown import (
    CAPPED,
    DEFAULT_CAP_S,
    DEFAULT_THRESHOLD_C,
    REACHED,
    SKIPPED,
    WAITING,
    cooldown_verdict,
    skipped_cooldown,
    wait_for_cooldown,
)


def test_the_threshold_is_the_documented_62c():
    assert DEFAULT_THRESHOLD_C == 62.0
    assert DEFAULT_CAP_S > 0


def test_below_the_threshold_is_reached():
    assert cooldown_verdict(55.0, elapsed_s=0.0, threshold_c=62.0, cap_s=120.0) == REACHED
    assert cooldown_verdict(62.0, elapsed_s=0.0, threshold_c=62.0, cap_s=120.0) == REACHED


def test_above_the_threshold_and_inside_the_cap_keeps_waiting():
    assert cooldown_verdict(70.0, elapsed_s=10.0, threshold_c=62.0, cap_s=120.0) == WAITING


def test_the_cap_ends_the_wait_and_says_so():
    """A thermally-limited laptop may never cool down. Record that, do not block."""
    assert cooldown_verdict(70.0, elapsed_s=120.0, threshold_c=62.0, cap_s=120.0) == CAPPED
    assert cooldown_verdict(70.0, elapsed_s=999.0, threshold_c=62.0, cap_s=120.0) == CAPPED


def test_a_cold_gpu_at_the_cap_still_counts_as_reached():
    """Temperature wins over the clock: the point of the gate is the temperature."""
    assert cooldown_verdict(50.0, elapsed_s=999.0, threshold_c=62.0, cap_s=120.0) == REACHED


def test_an_unreadable_temperature_never_counts_as_reached():
    assert cooldown_verdict(None, elapsed_s=0.0, threshold_c=62.0, cap_s=120.0) == WAITING
    assert cooldown_verdict(None, elapsed_s=120.0, threshold_c=62.0, cap_s=120.0) == CAPPED


class FakeClock:
    """Monotonic time that only advances when someone sleeps."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_the_wait_loop_stops_as_soon_as_the_gpu_is_cold():
    clock = FakeClock()
    readings = iter([80.0, 70.0, 61.0, 10.0])
    record = wait_for_cooldown(
        lambda: next(readings), threshold_c=62.0, cap_s=120.0,
        poll_interval_s=5.0, sleep=clock.sleep, clock=clock,
    )
    assert record.outcome == REACHED
    assert record.final_temperature_c == 61.0
    assert record.waited_s == pytest.approx(10.0)
    assert len(record.samples) == 3


def test_the_wait_loop_gives_up_at_the_cap_and_records_the_hot_reading():
    clock = FakeClock()
    record = wait_for_cooldown(
        lambda: 84.0, threshold_c=62.0, cap_s=20.0,
        poll_interval_s=5.0, sleep=clock.sleep, clock=clock,
    )
    assert record.outcome == CAPPED
    assert record.final_temperature_c == 84.0
    assert record.waited_s == pytest.approx(20.0)
    assert record.reached is False


def test_skipping_the_cooldown_still_produces_a_record():
    """--no-cooldown must not turn into a missing field. The outcome is always recorded."""
    record = skipped_cooldown(threshold_c=62.0, cap_s=120.0)
    assert record.enabled is False
    assert record.outcome == SKIPPED
    assert record.reached is False
    assert record.to_dict()["outcome"] == SKIPPED


def test_the_record_serialises_to_plain_json_types():
    record = skipped_cooldown()
    data = record.to_dict()
    assert set(data) >= {"enabled", "outcome", "threshold_c", "cap_s", "waited_s",
                         "final_temperature_c", "samples", "reached"}
