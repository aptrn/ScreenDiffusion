"""Spec 8.11's block is the committed sweep, and issue #45's Gate is on disk.

Held to a byte match the way the other twelve blocks are: re-run an arm and the
merge gate fails until the spec is regenerated with
`uv run python scripts/regen_spec_blocks.py`.

The Gate items are asserted here rather than only in the runner, because each is a
property of what is on disk rather than of a function - the sweep has to have an
adherence number per arm, a millisecond figure per arm, an explicit engine list,
bit-identity at every arm, and a recommendation with the machine that produced it.

GPU-free: everything here reads JSON and formats it.
"""

from pathlib import Path

import pytest
from sourceloader import ROOT, load_symbols

from bench.cli import report_guidance
from bench.guidance import (
    CASES,
    CONTROL_ARM,
    arms_of,
    engine_statement,
    load_guidance_results,
    recommend_guidance,
)
from bench.paths import GUIDANCE_RESULTS_DIR
from bench.results import gpu_of, latest_per, require_recordable

SPEC = Path(ROOT) / "docs/prompt-orchestrator-spec.md"
BLOCK = "GUIDANCE"


def rendered() -> str:
    import io

    buffer = io.StringIO()
    report_guidance(GUIDANCE_RESULTS_DIR, out=buffer)
    return buffer.getvalue().strip()


def committed() -> str:
    text = SPEC.read_text(encoding="utf-8")
    begin, end = f"<!-- BEGIN {BLOCK} -->", f"<!-- END {BLOCK} -->"
    assert begin in text and end in text, "the GUIDANCE anchors are gone from the spec"
    return text.split(begin, 1)[1].split(end, 1)[0].strip()


def results():
    found = load_guidance_results(GUIDANCE_RESULTS_DIR)
    assert found, "no guidance sweep is committed (issue #45)"
    return latest_per(found, lambda result: result["case"]["name"])


def test_the_spec_block_is_what_the_committed_results_say():
    assert rendered() == committed(), (
        "spec 8.11 has drifted from bench/results/guidance/. Regenerate it with "
        "`uv run python scripts/regen_spec_blocks.py`.")


def test_both_base_models_were_swept():
    """Step 5 of the issue: whether the answer differs between SD-Turbo and SD 1.5
    is a question one sweep cannot answer."""
    swept = {result["case"]["name"] for result in results().values()}
    assert swept == set(CASES)


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_arm_carries_an_adherence_number_and_a_millisecond_one(case):
    """The Gate's first two items, and the sentence that makes them one: an
    adherence number per arm, not a screenshot."""
    for result in results().values():
        if result["case"]["name"] != case:
            continue
        arms = arms_of(result)
        assert len(arms) > 1
        for arm in arms:
            assert arm.measured, f"{arm.name} produced no numbers"
            assert arm.adherence_frames == result["case"]["frames"]
            assert arm.ms_per_frame > 0.0


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_background_stayed_bit_identical_at_every_arm(case):
    """The Gate's third item. 48/48 outside the rendered region, on every arm of
    every sweep - guidance changes what the diffusion call returns and must not
    change what reaches a pixel nobody asked about."""
    for result in results().values():
        if result["case"]["name"] != case:
            continue
        for arm in arms_of(result):
            assert arm.background.passed, f"{arm.name}: {arm.background.statement}"
            assert arm.background.identical_frames == arm.background.frames


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_sweep_carries_a_control_arm(case):
    """Without it there is no "against what", and `recommend_guidance` says so
    rather than picking the best of the arms it happens to have."""
    for result in results().values():
        if result["case"]["name"] != case:
            continue
        assert any(arm.name == CONTROL_ARM for arm in arms_of(result))


def test_every_committed_run_names_the_machine_that_produced_it():
    """The Gate's fourth item. `require_recordable` is the door; this is the
    check that it was actually shut on these records."""
    for result in results().values():
        require_recordable(result)
        assert gpu_of(result) != "unknown GPU"


def test_the_recommendation_and_the_engine_list_are_in_the_block():
    """The Gate asks for both by name, and both are computed from the arms rather
    than written down beside them."""
    block = committed()
    for result in results().values():
        arms = arms_of(result)
        assert recommend_guidance(arms).statement in block
        assert engine_statement(arms) in block


def test_the_shipped_default_is_the_one_the_sweep_recommends():
    """The app's own `DEFAULT_CFG_TYPE` against what the committed arms say, the
    way issue #33 holds `DEFAULT_DETECT_EVERY_N` to its sweep. A re-sweep that
    moved the recommendation fails here rather than leaving the app's default
    quietly behind it."""
    default = load_symbols("main_gpu_addon.py", ["DEFAULT_CFG_TYPE"],
                           {"CFG_NONE": CONTROL_ARM})["DEFAULT_CFG_TYPE"]
    for result in results().values():
        recommendation = recommend_guidance(arms_of(result))
        assert not recommendation.moves_default, (
            f"{result['case']['name']} now recommends "
            f"`{recommendation.arm}`; move DEFAULT_CFG_TYPE with it")
        assert default == recommendation.cfg_type
