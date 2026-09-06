"""Clock regime and clock normalisation (issue #13).

The sweep in `bench/results/` is confounded: on a 120 W laptop the SM clock falls
from boost to the floor within about two seconds of a run starting, so a batch-1
cell catches boost while a batch-4 cell spends its whole run throttled. Comparing
those two milliseconds figures partly compares clocks.

Locking the clock is the fix, and this harness cannot do it - `nvidia-smi
--lock-gpu-clocks` needs elevation. So it detects the regime instead, and when the
clocks are not locked it reports an estimate beside the raw figure and says that is
what it is.

All of it is arithmetic over a sample trace and one string from `nvidia-smi`, so
the whole of it is covered here without a GPU, on synthetic traces.
"""

import pytest

from bench.clocks import (
    LOCKED,
    UNKNOWN,
    UNLOCKED,
    ClockLock,
    clock_normalization,
    interpret_event_reason,
    normalize_ms,
    regime_of,
    time_weighted_mean_clock,
)


def a_lock(**overrides) -> ClockLock:
    fields = dict(
        state=UNLOCKED,
        applied_clock_mhz=None,
        max_sm_clock_mhz=2100.0,
        current_sm_clock_mhz=210.0,
        evidence="clocks_event_reasons.applications_clocks_setting=Not Active",
    )
    fields.update(overrides)
    return ClockLock(**fields)


def trace(*pairs):
    """`[[elapsed_s, sm_clock_mhz, temperature_c], ...]`, the shape the runner records."""
    return [[elapsed, clock, 70.0] for elapsed, clock in pairs]


# --- lock state ------------------------------------------------------------

@pytest.mark.parametrize("value, state", [
    ("Active", LOCKED),
    ("active", LOCKED),
    ("Not Active", UNLOCKED),
    ("not active", UNLOCKED),
    # This laptop answers the *applications clock* query with a deprecation notice,
    # and a driver that does not support the field answers `[N/A]`. Neither is a
    # licence to call the GPU unlocked or locked.
    ("[N/A]", UNKNOWN),
    ("[Not Supported]", UNKNOWN),
    ("Requested functionality has been deprecated", UNKNOWN),
    ("", UNKNOWN),
    (None, UNKNOWN),
])
def test_the_event_reason_decides_the_regime(value, state):
    assert interpret_event_reason(value) == state


def test_only_a_detected_lock_counts_as_locked():
    """The trap: an undetectable lock is not a lock, and must not read as one."""
    assert a_lock(state=LOCKED).locked is True
    assert a_lock(state=UNLOCKED).locked is False
    assert a_lock(state=UNKNOWN).locked is False


def test_a_lock_serialises_with_its_evidence():
    data = a_lock().to_dict()
    assert data["state"] == UNLOCKED
    assert data["max_sm_clock_mhz"] == 2100.0
    assert "applications_clocks_setting" in data["evidence"]


def test_a_result_written_before_this_existed_reads_as_unlocked():
    """Issue #13's trap: the existing results stay as they are, unlocked by absence."""
    assert regime_of({"hardware": {"gpu_name": "RTX 3080"}}) == UNLOCKED
    assert regime_of({}) == UNLOCKED
    assert regime_of({"hardware": {"clock_lock": {"state": LOCKED}}}) == LOCKED
    assert regime_of({"hardware": {"clock_lock": {"state": UNKNOWN}}}) == UNKNOWN


# --- the clock trace -------------------------------------------------------

def test_evenly_spaced_samples_average_to_the_plain_mean():
    assert time_weighted_mean_clock(trace((0.0, 1600.0), (0.5, 1200.0), (1.0, 800.0))) == (
        pytest.approx(1200.0)
    )


def test_a_long_gap_weighs_more_than_a_short_one():
    """The gate a plain mean fails: a `nvidia-smi` call that took four seconds must
    not count the same as one taken half a second after the previous."""
    # Boost at the start, then the floor from 0.5 s to the end of a 5 s run.
    weighted = time_weighted_mean_clock(trace((0.0, 1665.0), (0.5, 1110.0), (5.0, 1110.0)))
    expected = (0.5 * (1665.0 + 1110.0) / 2 + 4.5 * 1110.0) / 5.0
    assert weighted == pytest.approx(expected)
    assert weighted < (1665.0 + 1110.0 + 1110.0) / 3.0 - 100.0, (
        "averaging the three readings equally would credit the run with boost it "
        "only held for half a second"
    )


def test_a_sample_with_no_clock_reading_is_dropped_not_guessed():
    assert time_weighted_mean_clock(trace((0.0, 1600.0), (0.5, None), (1.0, 800.0))) == (
        pytest.approx(1200.0)
    )


def test_a_trace_with_nothing_in_it_has_no_mean():
    assert time_weighted_mean_clock([]) is None
    assert time_weighted_mean_clock(trace((0.0, None))) is None


def test_a_single_sample_is_its_own_mean():
    assert time_weighted_mean_clock(trace((0.0, 1450.0))) == pytest.approx(1450.0)


# --- the normalisation arithmetic ------------------------------------------

def test_a_run_at_half_the_basis_clock_normalises_to_half_the_time():
    """Compute-bound work: ms x f / f_ref, first order and nothing more."""
    assert normalize_ms(80.0, clock_mhz=1050.0, basis_mhz=2100.0) == pytest.approx(40.0)
    assert normalize_ms(80.0, clock_mhz=2100.0, basis_mhz=2100.0) == pytest.approx(80.0)


def test_an_unlocked_run_reports_an_estimate_and_names_its_basis():
    normalisation = clock_normalization(
        a_lock(), samples=trace((0.0, 1050.0), (0.5, 1050.0)), raw_ms_per_frame=80.0
    )
    assert normalisation.regime == UNLOCKED
    assert normalisation.ms_per_frame == pytest.approx(40.0)
    assert normalisation.basis_mhz == 2100.0
    assert normalisation.basis_source == "clocks.max.sm"
    assert normalisation.sampled_mean_sm_clock_mhz == pytest.approx(1050.0)
    assert "estimate" in normalisation.note.lower(), (
        "the trap: a normalised figure presented as if it were measured"
    )
    assert "ms" in normalisation.method and "clock" in normalisation.method


def test_a_locked_run_reports_the_raw_figure_and_says_so():
    """The gate: with clocks locked there is nothing to correct for."""
    normalisation = clock_normalization(
        a_lock(state=LOCKED, applied_clock_mhz=1200.0),
        samples=trace((0.0, 1200.0), (0.5, 1200.0)), raw_ms_per_frame=80.0,
    )
    assert normalisation.regime == LOCKED
    assert normalisation.ms_per_frame is None
    assert "raw" in normalisation.note.lower()
    # The trace is still carried: it is the evidence that the lock actually held.
    assert normalisation.sampled_mean_sm_clock_mhz == pytest.approx(1200.0)


def test_an_undetectable_regime_is_normalised_but_not_called_unlocked():
    normalisation = clock_normalization(
        a_lock(state=UNKNOWN), samples=trace((0.0, 1050.0)), raw_ms_per_frame=80.0
    )
    assert normalisation.regime == UNKNOWN
    assert normalisation.ms_per_frame == pytest.approx(40.0)


def test_without_a_clock_trace_there_is_no_normalised_figure():
    normalisation = clock_normalization(a_lock(), samples=[], raw_ms_per_frame=80.0)
    assert normalisation.ms_per_frame is None
    assert "clock" in normalisation.note.lower()


def test_without_a_basis_clock_there_is_no_normalised_figure():
    normalisation = clock_normalization(
        a_lock(max_sm_clock_mhz=None), samples=trace((0.0, 1050.0)), raw_ms_per_frame=80.0
    )
    assert normalisation.ms_per_frame is None
    assert normalisation.basis_mhz is None


def test_a_normalisation_serialises_to_plain_types():
    data = clock_normalization(a_lock(), trace((0.0, 1050.0)), 80.0).to_dict()
    assert set(data) >= {"regime", "ms_per_frame", "basis_mhz", "basis_source",
                         "method", "sampled_mean_sm_clock_mhz", "note"}
    assert all(value is None or isinstance(value, (str, float, int))
               for value in data.values())


def test_the_confound_the_issue_describes_is_what_normalisation_removes():
    """Issue #13's context, as arithmetic, on the two committed cells it is about.

    `img2img-none-512x512-b1` measured 79.09 ms/frame at a mean 1290 MHz and
    `-b4` measured 72.80 ms/frame at 1141 MHz: batch 1 caught boost, batch 4 spent
    its run nearer the floor. Raw, the batch-4 gain is 8.6%. At one clock it is
    22.8% - a different answer to the question the sweep exists to settle. (The
    issue quotes ~26%, taken from the boost-to-floor range of the trace rather than
    from the two run means; either reading is far from 8.6%, which is the point.)
    """
    basis = 2100.0
    batch1 = clock_normalization(a_lock(max_sm_clock_mhz=basis),
                                 trace((0.0, 1290.0)), raw_ms_per_frame=79.09)
    batch4 = clock_normalization(a_lock(max_sm_clock_mhz=basis),
                                 trace((0.0, 1141.0)), raw_ms_per_frame=72.80)

    raw_gain = 79.09 / 72.80 - 1.0
    normalised_gain = batch1.ms_per_frame / batch4.ms_per_frame - 1.0
    assert raw_gain == pytest.approx(0.086, abs=0.005)
    assert normalised_gain == pytest.approx(0.228, abs=0.01)
