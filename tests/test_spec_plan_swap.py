"""Spec 8.9's block is the committed swaps - and issue #30's Gate on disk.

The same rule the other five generated blocks are held to: re-measure a swap and
the merge gate fails until 8.9 is regenerated with
`uv run python -m bench --swap-report`.

The Gate items are asserted here rather than only in the runner, because they are
properties of what is on disk rather than of a function: both swap kinds
committed, both criteria with a stated verdict and the machine that produced
them, no rebuild as a number, and bit-identity still holding across the swap.

GPU-free: `bench.plan_swap` reads JSON and formats it.
"""

import io
from pathlib import Path
from typing import List

import pytest
from sourceloader import ROOT

from bench.cli import report_swap
from bench.paths import SWAP_RESULTS_DIR
from bench.plan_swap import (
    CRITERION_1_BUDGET_MS,
    GUI_DEBOUNCE_MS,
    RUNTIME_SWAP,
    VOCABULARY_SWAP,
    latest_per_swap,
    load_swap_results,
    repeat_spread,
)
from bench.portability import is_deploy_gpu
from bench.results import gpu_of, load_records, measured_on, require_recordable

SPEC = Path(ROOT) / "docs/prompt-orchestrator-spec.md"
BEGIN = "<!-- BEGIN PLAN SWAP -->"
END = "<!-- END PLAN SWAP -->"


def spec_block() -> str:
    text = SPEC.read_text(encoding="utf-8")
    assert BEGIN in text and END in text, "8.9's block lost its anchors"
    return text.split(BEGIN, 1)[1].split(END, 1)[0].strip()


def computed_block() -> str:
    buffer = io.StringIO()
    report_swap(SWAP_RESULTS_DIR, out=buffer)
    return buffer.getvalue().strip()


def committed() -> List[dict]:
    return list(load_records(SWAP_RESULTS_DIR).values())


def rows() -> List[dict]:
    """One run per swap per machine - what the block's table is drawn from."""
    return sorted(latest_per_swap(load_swap_results(SWAP_RESULTS_DIR)).values(),
                  key=lambda result: (result["case"]["name"], gpu_of(result)))


def per_swap():
    return pytest.mark.parametrize(
        "result", rows(),
        ids=lambda result: f"{result['case']['name']}@{gpu_of(result)}")


def test_the_spec_block_is_what_the_committed_swaps_say():
    assert spec_block() == computed_block(), (
        "spec 8.9's block no longer matches bench/results/swaps/. "
        "Regenerate it with\n"
        "    uv run python scripts/regen_spec_blocks.py\n")


def test_every_committed_swap_could_have_reached_disk():
    swaps = committed()
    assert swaps, "issue #30's deliverable is a committed swap of each kind"
    for result in swaps:
        require_recordable(result)
        assert result["kind"] == "plan-swap"


def test_both_kinds_of_swap_were_measured():
    """One that re-encodes the detector's vocabulary and one that does not: they
    take different paths, and a single case would have answered for one of them."""
    assert {result["swap_kind"] for result in rows()} == {VOCABULARY_SWAP,
                                                          RUNTIME_SWAP}


def test_the_swaps_were_measured_on_deploy_hardware():
    """Criterion 1 is a latency claim and criterion 3 a frame-budget one, and
    §7.4 says those are deploy-hardware claims."""
    assert any(is_deploy_gpu(gpu_of(result)) for result in rows())


def test_every_swap_was_measured_twice():
    for gpu in {gpu_of(result) for result in rows()}:
        spread = repeat_spread(measured_on(committed(), gpu))
        assert spread.repeats >= 2, spread.statement


# --- the Gate ---------------------------------------------------------------


@per_swap()
def test_criterion_1_has_a_measured_verdict_with_the_debounce_in_it(result):
    """The issue's first trap: the user's clock starts at the keystroke."""
    criterion = result["gate"]["criterion_1"]
    timing = result["timing"]
    assert criterion["passed"], criterion["statement"]
    assert timing["debounce_ms"] == GUI_DEBOUNCE_MS
    assert criterion["keystroke_to_pixel_ms"] == pytest.approx(
        timing["debounce_ms"] + timing["worker_ms"], abs=0.01)
    assert criterion["keystroke_to_pixel_ms"] <= CRITERION_1_BUDGET_MS


@per_swap()
def test_criterion_1_is_reported_with_and_without_the_debounce(result):
    """Step 4 of the issue: the worker-side figure is the one an optimisation
    would move, so it has to be readable on its own."""
    timing = result["timing"]
    assert timing["worker_ms"] < timing["keystroke_to_pixel_ms"]
    assert timing["frames_to_pixel"] >= 1


@per_swap()
def test_criterion_3_is_judged_against_a_control_from_the_same_run(result):
    """The third trap. A verdict taken against 33.33 ms alone would call the
    steady state itself a stutter on a slower card."""
    criterion = result["gate"]["criterion_3"]
    assert criterion["passed"], criterion["statement"]
    assert criterion["control"]["frames"] > 0
    assert criterion["swap"]["frames"] > 0
    assert criterion["excess_ms"] <= criterion["allowance_ms"]


@per_swap()
def test_the_whole_interval_series_is_on_disk(result):
    """Step 2: the series, not a summary of it - the verdict is re-derivable."""
    intervals = result["intervals_ms"]
    assert len(intervals) == result["run"]["frames"]
    assert intervals[0] is None
    assert all(value > 0 for value in intervals[1:])


@per_swap()
def test_no_engine_was_rebuilt_and_it_is_a_number(result):
    rebuild = result["gate"]["rebuild"]
    assert rebuild["engine_rebuilds"] == 0
    assert rebuild["steps_before"] == rebuild["steps_after"] == 1
    assert rebuild["engine_id_before"] == rebuild["engine_id_after"]
    assert rebuild["schedule_moved"], "the swap never reached the engine at all"


@per_swap()
def test_the_background_stayed_bit_identical_across_the_swap(result):
    background = result["gate"]["background"]
    assert background["passed"], background["statement"]
    assert background["identical_frames"] == background["frames"] > 0
    assert background["worst_pixels_changed"] == 0


@per_swap()
def test_the_whole_gate_passed(result):
    assert result["gate"]["passed"], result["case"]["name"]


def test_the_vocabulary_swap_waited_on_a_detect_and_the_runtime_one_did_not():
    """The measurement's own sanity check: if the two kinds had cost the same,
    the run would not have exercised two paths."""
    by_kind = {result["swap_kind"]: result for result in rows()}
    assert by_kind[VOCABULARY_SWAP]["timing"]["detector_ticks_waited"] >= 1
    assert by_kind[RUNTIME_SWAP]["timing"]["detector_ticks_waited"] == 0
    assert (by_kind[VOCABULARY_SWAP]["timing"]["frames_to_pixel"]
            > by_kind[RUNTIME_SWAP]["timing"]["frames_to_pixel"])


def test_the_clip_a_human_watches_is_committed():
    for result in rows():
        assert result["comparison_clip"]
        assert (SWAP_RESULTS_DIR / result["comparison_clip"]).is_file()
        assert (SWAP_RESULTS_DIR / result["comparison_still"]).is_file()


def test_section_11_carries_the_verdict_on_both_criteria():
    """The issue's Verification: the criteria live in §11, and a reader looking
    for the answer looks there rather than in a bench directory."""
    section = SPEC.read_text(encoding="utf-8").split("## 11.", 1)[1]
    criteria = section.split("\n2. ", 1)[0], section.split("\n3. ", 1)[1]
    for text in criteria:
        assert "issue #30" in text
        assert "Met, measured" in text
