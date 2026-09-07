"""The occupancy gate: was anything else drawing on the card? (issue #33)

The sibling of `tests/test_bench_cooldown.py`, and it exists because the same
class of mistake got past every door the harness already had.
"""

import pytest

from bench.contention import (
    BUSY,
    CLEAR,
    DEFAULT_BUSY_THRESHOLD_PCT,
    UNKNOWN,
    OccupancyRecord,
    measure_occupancy,
    occupancy_summary,
    occupancy_verdict,
)


def a_meter(*readings):
    """A utilization meter that returns each reading in turn, then repeats the last."""
    remaining = list(readings)

    def read():
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return read


# --- the verdict ------------------------------------------------------------


def test_an_idle_card_is_clear():
    assert occupancy_verdict(0.0) == CLEAR


def test_a_card_someone_else_is_drawing_on_is_busy():
    """The reading that would have refused issue #33's discarded run: a live
    real-time application held ~45% of the SMs while the bench measured 3.2x its
    own committed baseline."""
    assert occupancy_verdict(45.0) == BUSY


def test_the_threshold_is_the_boundary_and_it_is_inclusive():
    assert occupancy_verdict(DEFAULT_BUSY_THRESHOLD_PCT) == CLEAR
    assert occupancy_verdict(DEFAULT_BUSY_THRESHOLD_PCT + 0.1) == BUSY


def test_an_unreadable_meter_is_unknown_and_never_clear():
    """The rule `cooldown_verdict` applies to an unreadable temperature. A gate that
    cannot see the card has not seen an empty one."""
    assert occupancy_verdict(None) == UNKNOWN


# --- the sampled record -----------------------------------------------------


def test_the_verdict_is_taken_on_the_mean_so_one_spike_is_not_a_verdict():
    """A run lasts tens of seconds; a single frame of someone else's compositor is
    not what confounds it. A sustained load is."""
    record = measure_occupancy(a_meter(0.0, 0.0, 40.0, 0.0), sleep=lambda _: None)
    assert record.outcome == CLEAR
    assert record.clear


def test_a_steadily_occupied_card_is_busy_however_the_samples_fall():
    record = measure_occupancy(a_meter(42.0, 52.0, 57.0, 45.0), sleep=lambda _: None)
    assert record.outcome == BUSY
    assert not record.clear
    assert record.mean_utilization_pct == pytest.approx(49.0)


def test_every_reading_is_kept_so_a_run_that_only_just_passed_is_visible():
    record = measure_occupancy(a_meter(0.0, 9.0, 1.0, 8.0), sleep=lambda _: None)
    assert record.samples == [0.0, 9.0, 1.0, 8.0]


def test_a_meter_that_never_answers_yields_unknown_rather_than_an_empty_mean():
    record = measure_occupancy(a_meter(None), sleep=lambda _: None)
    assert record.outcome == UNKNOWN
    assert record.mean_utilization_pct is None


def test_readings_that_did_answer_decide_it_even_when_some_did_not():
    record = measure_occupancy(a_meter(None, 60.0, 60.0, None), sleep=lambda _: None)
    assert record.outcome == BUSY
    assert record.mean_utilization_pct == pytest.approx(60.0)


def test_the_record_always_carries_an_outcome_to_disk():
    """The cooldown record's rule: a comparison can always tell a clear run from a
    contended one from an unmeasured one."""
    payload = measure_occupancy(a_meter(0.0), samples=1,
                                sleep=lambda _: None).to_dict()
    assert payload["outcome"] == CLEAR
    assert payload["clear"] is True
    assert payload["threshold_pct"] == DEFAULT_BUSY_THRESHOLD_PCT
    assert payload["samples"] == [0.0]


def test_the_summary_says_what_was_seen_and_against_what():
    summary = occupancy_summary(
        OccupancyRecord(outcome=BUSY, mean_utilization_pct=49.0, samples=[49.0]))
    assert "busy" in summary and "49" in summary
    assert str(int(DEFAULT_BUSY_THRESHOLD_PCT)) in summary


def test_an_unknown_summary_does_not_quote_a_utilization_it_never_read():
    assert "%" not in occupancy_summary(OccupancyRecord(outcome=UNKNOWN))
