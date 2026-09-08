"""Spec 8.12's block is the committed sweep, and issue #46's Gate is on disk.

Held to a byte match the way the other thirteen blocks are: re-run an arm and the
merge gate fails until the spec is regenerated with
`uv run python scripts/regen_spec_blocks.py`.

The Gate items are asserted here rather than only in the runner, because each is a
property of what is on disk rather than of a function - the sweep has to cover both
routes, carry a millisecond figure and a quality figure per arm, hold bit-identity
at every step count, and carry a *measured* swap time, which is the one number the
window is allowed to quote.

GPU-free: everything here reads JSON and formats it.
"""

from pathlib import Path

from sourceloader import ROOT

from bench.cli import report_quality
from bench.paths import QUALITY_RESULTS_DIR
from bench.quality import (
    CASES,
    ROUTE_LADDER,
    ROUTE_UNBATCHED,
    STEP_LADDER,
    arms_of,
    contention_note,
    load_quality_results,
    on_route,
    recommend_route,
    swap_summary,
    timing_is_decisive,
)
from bench.results import gpu_of, latest_per, require_recordable

SPEC = Path(ROOT) / "docs/prompt-orchestrator-spec.md"
BLOCK = "STEP QUALITY"


def rendered() -> str:
    import io

    buffer = io.StringIO()
    report_quality(QUALITY_RESULTS_DIR, out=buffer)
    return buffer.getvalue().strip()


def committed() -> str:
    text = SPEC.read_text(encoding="utf-8")
    begin, end = f"<!-- BEGIN {BLOCK} -->", f"<!-- END {BLOCK} -->"
    assert begin in text and end in text, \
        "the STEP QUALITY anchors are gone from the spec"
    return text.split(begin, 1)[1].split(end, 1)[0].strip()


def results():
    found = load_quality_results(QUALITY_RESULTS_DIR)
    assert found, "no step-quality sweep is committed (issue #46)"
    return latest_per(found, lambda result: result["case"]["name"])


def test_the_spec_block_is_what_the_committed_results_say():
    assert rendered() == committed(), (
        "spec 8.12 has drifted from bench/results/quality/. Regenerate it with "
        "`uv run python scripts/regen_spec_blocks.py`.")


def test_both_routes_were_swept_over_the_same_rungs():
    """The Gate's first item. A sweep of one route decides nothing between two."""
    for result in results().values():
        arms = arms_of(result)
        for route in (ROUTE_LADDER, ROUTE_UNBATCHED):
            measured = [arm.steps for arm in on_route(arms, route) if arm.measured]
            assert measured, f"{route} produced no arms at all"
            assert set(measured) <= set(STEP_LADDER)
        assert 1 in [arm.steps for arm in on_route(arms, ROUTE_LADDER)], \
            "the shipped one-step configuration is not in the sweep"


def test_every_arm_carries_a_millisecond_figure_and_a_quality_one():
    """The Gate's first item spelled out: ms/frame *and* a quality figure per arm,
    so a route cannot be recommended on speed alone."""
    for result in results().values():
        for arm in arms_of(result):
            if not arm.measured:
                continue
            assert arm.ms_per_frame > 0.0, f"{arm.name} produced no timing"
            assert arm.adherence_frames == result["case"]["frames"]
            assert arm.net_change >= 0.0


def test_the_background_stayed_bit_identical_at_every_step_count():
    """The Gate's last item. Whatever a step count does inside the region, a pixel
    nobody asked about is the captured byte."""
    for result in results().values():
        for arm in arms_of(result):
            if not arm.measured:
                continue
            assert arm.background.passed, f"{arm.name}: {arm.background.statement}"
            assert arm.background.identical_frames == arm.background.frames


def test_a_cached_engine_swap_was_actually_measured():
    """The Gate's second item, and the issue's third trap: nothing about what a
    switch costs may be quoted until a run measured one."""
    for result in results().values():
        summary = swap_summary(arms_of(result))
        assert summary.measured, summary.statement
        assert summary.mean_seconds > 0.0
        assert summary.worst_seconds >= summary.mean_seconds


def test_the_block_carries_the_swap_figure():
    """Computed from the arms rather than written down beside them, and the one
    number the window is allowed to quote."""
    block = committed()
    for result in results().values():
        assert swap_summary(arms_of(result)).statement in block


def test_the_route_verdict_is_printed_only_when_the_card_was_this_run_s_own():
    """Issue #33's door on the one *absolute* millisecond judgement here. On a
    clear card the recommendation is in the block; on a shared one the block says
    so instead of quoting a verdict the contention decided."""
    block = committed()
    for result in results().values():
        if not timing_is_decisive(result):
            assert contention_note(result) in block
            continue
        recommendation = recommend_route(arms_of(result))
        assert recommendation is not None, \
            "neither route reached a rung inside the frame budget"
        assert recommendation.statement in block


def test_every_committed_run_names_the_machine_that_produced_it():
    """`require_recordable` is the door; this is the check that it was shut."""
    for result in results().values():
        require_recordable(result)
        assert gpu_of(result) != "unknown GPU"


def test_the_case_swept_is_the_one_the_registry_offers():
    swept = {result["case"]["name"] for result in results().values()}
    assert swept == set(CASES)
