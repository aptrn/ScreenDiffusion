"""Spec 8.8's cadence block is the committed sweep - and issue #23's Gate on disk.

The same rule the other four generated blocks are held to: re-run an arm and the
merge gate fails until 8.8 is regenerated with
`uv run python -m bench --cadence-report`.

The Gate items are asserted here rather than only in the runner. Two of them are
this issue's own - the background criterion has to pass at *every* configuration,
and every arm has to carry the amortised detector cost and the staleness it bought
- and both are properties of what is on disk rather than of a function.

GPU-free: `bench.cadence` reads JSON and formats it.
"""

import io
from pathlib import Path
from typing import List

import pytest
from sourceloader import ROOT

from bench.cadence import (
    cadence_of,
    comparability,
    latest_per_cadence,
    recommend_cadence,
    repeat_spread,
)
from bench.cli import report_cadence
from bench.paths import CADENCE_RESULTS_DIR, SELECTIVE_RESULTS_DIR
from bench.portability import is_deploy_gpu
from bench.results import gpu_of, load_records, measured_on, require_recordable
from bench.selective import load_selective_results

SPEC = Path(ROOT) / "docs/prompt-orchestrator-spec.md"
BEGIN = "<!-- BEGIN CADENCE SWEEP -->"
END = "<!-- END CADENCE SWEEP -->"

# The cadences issue #23 step 3 names. All four, or the sweep did not happen.
SWEPT = (2, 3, 5, 8)


def spec_block() -> str:
    text = SPEC.read_text(encoding="utf-8")
    assert BEGIN in text and END in text, "8.8's cadence block lost its anchors"
    return text.split(BEGIN, 1)[1].split(END, 1)[0].strip()


def computed_block() -> str:
    buffer = io.StringIO()
    report_cadence(CADENCE_RESULTS_DIR, out=buffer,
                   baseline_dir=SELECTIVE_RESULTS_DIR)
    return buffer.getvalue().strip()


def committed() -> List[dict]:
    return list(load_records(CADENCE_RESULTS_DIR).values())


def rows() -> List[dict]:
    """One arm per cadence per machine - what the block's table is drawn from."""
    return sorted(latest_per_cadence(load_records(CADENCE_RESULTS_DIR)).values(),
                  key=lambda result: (cadence_of(result), gpu_of(result)))


def per_arm():
    return pytest.mark.parametrize(
        "result", rows(),
        ids=lambda result: f"n{cadence_of(result)}@{gpu_of(result)}")


def test_the_spec_block_is_what_the_committed_sweep_says():
    assert spec_block() == computed_block(), (
        "spec 8.8's cadence block no longer matches bench/results/cadence/. "
        "Regenerate it with\n"
        "    uv run python scripts/regen_spec_blocks.py\n")


def test_every_committed_arm_could_have_reached_disk():
    arms = committed()
    assert arms, "issue #23's deliverable is a committed cadence sweep"
    for result in arms:
        require_recordable(result)
        assert result["kind"] == "selective"


def test_the_sweep_covers_the_cadences_the_issue_names():
    assert {cadence_of(result) for result in rows()} == set(SWEPT)


def test_the_sweep_was_measured_on_deploy_hardware():
    """A cadence recommendation is a frame-budget claim, and §7.4 says those are
    deploy-hardware claims."""
    assert any(is_deploy_gpu(gpu_of(result)) for result in rows())


def test_every_arm_was_measured_twice():
    """The recommendation turns on a 3.33 ms line, so one run per arm would not
    have shown which side of it an arm is on."""
    for gpu in {gpu_of(result) for result in rows()}:
        spread = repeat_spread(measured_on(committed(), gpu))
        assert spread.repeats >= len(SWEPT), spread.statement


# --- the Gate ---------------------------------------------------------------


@per_arm()
def test_the_background_stayed_bit_identical_at_every_configuration(result):
    """The issue's sharpest Gate item: a configuration that breaks bit-identity is
    disqualified, not the criterion. None of them did."""
    background = result["gate"]["background"]
    assert background["passed"], background["statement"]
    assert background["identical_frames"] == background["frames"] > 0
    assert background["worst_pixels_changed"] == 0


@per_arm()
def test_every_arm_carries_the_amortised_detector_cost(result):
    run = result["run"]
    assert run["detect"] is not None, "the arm ran without a detector"
    assert run["amortised_detect_ms"] > 0
    assert run["ms_per_frame_with_detection"] == pytest.approx(
        run["ms_per_frame"] + run["amortised_detect_ms"], abs=0.01)


@per_arm()
def test_every_arm_carries_how_stale_its_tracks_became(result):
    """The other half of the trade. A sweep recording only the milliseconds would
    recommend detecting once a second."""
    stale = result["staleness"]
    assert stale["detect_every_n"] == cadence_of(result)
    assert stale["ticks"] > 0 and stale["refreshes"] > 0
    assert stale["mean_age_frames"] > 0
    assert 0.0 <= stale["mean_refresh_iou"] <= 1.0


@per_arm()
def test_every_arm_reports_a_flicker_figure(result):
    assert result["flicker"]["mean_abs_diff"] is not None
    assert result["flicker"]["pairs_scored"] > 0


@per_arm()
def test_every_arm_passed_the_whole_selective_gate(result):
    assert result["gate"]["passed"], result["case"]["name"]


def test_a_staler_cadence_really_is_staler():
    """The measurement's own sanity check: if box age did not rise with the
    cadence, the sweep would not have swept anything."""
    ages = [(cadence_of(result), result["staleness"]["mean_age_frames"])
            for result in rows()]
    assert ages == sorted(ages), ages


def test_the_arms_rendered_the_same_amount_of_frame():
    """Region count drives cost, so a sweep across two of them would be measuring
    the selection rather than the cadence."""
    check = comparability(rows())
    assert check.comparable, check.statement


def test_the_recommendation_is_a_cadence_that_was_actually_measured():
    for gpu in {gpu_of(result) for result in rows()}:
        recommendation = recommend_cadence(measured_on(rows(), gpu))
        assert recommendation.detect_every_n in SWEPT
        assert recommendation.meets_budget, recommendation.statement


def test_the_clip_a_human_watches_at_the_recommended_cadence_is_committed():
    for gpu in {gpu_of(result) for result in rows()}:
        recommendation = recommend_cadence(measured_on(rows(), gpu))
        chosen = [result for result in measured_on(rows(), gpu)
                  if cadence_of(result) == recommendation.detect_every_n][0]
        assert chosen["comparison_clip"]
        assert (CADENCE_RESULTS_DIR / chosen["comparison_clip"]).is_file()
        assert (CADENCE_RESULTS_DIR / chosen["comparison_still"]).is_file()


def test_no_arm_landed_beside_the_baselines_spec_8_8_and_7_4_quote():
    """The routing rule, asserted on disk rather than in the CLI: a swept arm in
    `bench/results/selective/` would silently become the row those two sections
    are drawn from. `.get` because the baselines predate the field, and an absent
    override is the same thing as no override."""
    for result in load_selective_results(SELECTIVE_RESULTS_DIR).values():
        assert result["case"].get("detect_every_n") is None, result["case"]["name"]


def test_the_spec_says_no_lower_resolution_engine_was_built():
    """Issue #23's first Gate item and its first trap, in the section a reader
    would look in."""
    section = SPEC.read_text(encoding="utf-8").split("### 8.8", 1)[1].split("## 9.", 1)[0]
    assert "no lower-resolution engine was built" in section
    assert "issue #24" in section
