"""Spec 8.5's block is the committed sweep - and issue #32's Gate on disk.

The same rule the other seven generated blocks are held to: re-run an arm and the
merge gate fails until 8.5 is regenerated with
`uv run python -m bench --stability-report`.

The Gate items are asserted here rather than only in the runner, because each of
them is a property of what is on disk rather than of a function: a flicker figure
per setting, a visible-change figure net of a control per setting, background
bit-identity **still 48/48** at every setting, and a recommendation naming the
machine that produced it.

GPU-free: `bench.stability` reads JSON and formats it.
"""

import io
from pathlib import Path
from typing import List

import pytest
from sourceloader import ROOT

from bench.cli import report_stability
from bench.paths import SELECTIVE_RESULTS_DIR, STABILITY_RESULTS_DIR
from bench.portability import is_deploy_gpu
from bench.results import gpu_of, load_records, measured_on, require_recordable
from bench.selective import (
    VISIBLE_CHANGE,
    ema_suffix,
    load_selective_results,
)
from bench.stability import (
    SHIPPED_EMA,
    SHIPPED_POLICY,
    arm_label,
    arm_of,
    artefact_of,
    control_of,
    latest_per_arm,
    recommend_setting,
    repeat_spread,
    response_of,
)

SPEC = Path(ROOT) / "docs/prompt-orchestrator-spec.md"
BEGIN = "<!-- BEGIN STABILITY SWEEP -->"
END = "<!-- END STABILITY SWEEP -->"

# The settings issue #32's two steps name: all three seed policies, and an EMA
# sweep. Anything less and one of the levers was not measured.
SWEPT_POLICIES = ("fixed", "per_track", "random")
SWEPT_EMAS = (0.0, 0.25, 0.5, 0.75)


def spec_block() -> str:
    text = SPEC.read_text(encoding="utf-8")
    assert BEGIN in text and END in text, "8.5's block lost its anchors"
    return text.split(BEGIN, 1)[1].split(END, 1)[0].strip()


def computed_block() -> str:
    buffer = io.StringIO()
    report_stability(STABILITY_RESULTS_DIR, out=buffer,
                     baseline_dir=SELECTIVE_RESULTS_DIR)
    return buffer.getvalue().strip()


def committed() -> List[dict]:
    return list(load_records(STABILITY_RESULTS_DIR).values())


def rows() -> List[dict]:
    """One run per arm per machine - what the block's table is drawn from."""
    return sorted(latest_per_arm(load_records(STABILITY_RESULTS_DIR)).values(),
                  key=lambda result: (arm_of(result), gpu_of(result)))


def per_arm():
    return pytest.mark.parametrize(
        "result", rows(), ids=lambda result: f"{arm_label(result)}@{gpu_of(result)}")


def test_the_spec_block_is_what_the_committed_sweep_says():
    assert spec_block() == computed_block(), (
        "spec 8.5's block no longer matches bench/results/stability/. "
        "Regenerate it with\n"
        "    uv run python scripts/regen_spec_blocks.py\n")


def test_every_committed_arm_could_have_reached_disk():
    arms = committed()
    assert arms, "issue #32's deliverable is a committed stability sweep"
    for result in arms:
        require_recordable(result)
        assert result["kind"] == "selective"


def test_the_sweep_covers_every_seed_policy_the_plan_can_carry():
    """Including `random`, which does not ship: it is the control that says the
    metric can see the noise at all."""
    assert {arm_of(result)[0] for result in rows()} == set(SWEPT_POLICIES)


def test_the_sweep_covers_the_output_ema_range():
    assert {arm_of(result)[1] for result in rows()} >= set(SWEPT_EMAS)


def test_the_sweep_has_a_control_arm_at_the_shipped_default():
    """Every row is read against it, so a sweep without one measures nothing."""
    for gpu in {gpu_of(result) for result in rows()}:
        control = control_of(measured_on(rows(), gpu))
        assert control is not None
        assert arm_of(control) == (SHIPPED_POLICY, SHIPPED_EMA)


def test_the_sweep_was_measured_on_deploy_hardware():
    assert any(is_deploy_gpu(gpu_of(result)) for result in rows())


def test_every_arm_was_measured_twice():
    """The baseline is unusually steady, so what makes a flicker difference a
    difference is the run-to-run spread - which needs repeats to exist."""
    for gpu in {gpu_of(result) for result in rows()}:
        spread = repeat_spread(measured_on(committed(), gpu))
        assert spread.repeats >= len(measured_on(rows(), gpu)), spread.statement


# --- the Gate ---------------------------------------------------------------


@per_arm()
def test_the_background_stayed_bit_identical_at_every_setting(result):
    """The issue's first trap: an EMA reaching outside the mask would break §11's
    criterion 4, and that is not negotiable for a flicker win."""
    background = result["gate"]["background"]
    assert background["passed"], background["statement"]
    assert background["identical_frames"] == background["frames"] > 0
    assert background["worst_pixels_changed"] == 0


@per_arm()
def test_every_setting_reports_a_flicker_figure(result):
    assert result["flicker"]["mean_abs_diff"] is not None
    assert result["flicker"]["pairs_scored"] > 0


@per_arm()
def test_every_setting_reports_a_visible_change_net_of_a_control(result):
    """The Gate's second item. `capture_change` is the control the net figure
    subtracts - the capture's own uint8 round trip - so an arm cannot pass on the
    resampler's blur."""
    change = result["gate"]["change"]
    assert change["threshold"] == VISIBLE_CHANGE
    assert change["net_change"] == pytest.approx(
        change["region_change"] - change["capture_change"], abs=0.01)


@per_arm()
def test_every_setting_reports_what_it_did_where_the_source_moved(result):
    """Flicker's mirror. Without it a report could recommend an inert output."""
    assert response_of(result) is not None


@per_arm()
def test_every_arm_rendered_under_the_setting_its_name_claims(result):
    """The name is what was asked for and the plan is what was rendered; a clamp
    between them would put an arm in the wrong row."""
    policy, ema = arm_of(result)
    assert result["case"]["name"].endswith(f"-{policy}-{ema_suffix(ema)}")


def test_the_recommendation_names_a_setting_that_was_actually_measured():
    for gpu in {gpu_of(result) for result in rows()}:
        arms = measured_on(rows(), gpu)
        recommendation = recommend_setting(arms, measured_on(committed(), gpu))
        assert (recommendation.seed_policy, recommendation.output_ema) in {
            arm_of(result) for result in arms}
        assert recommendation.gpu == gpu


def test_the_recommendation_is_the_shipped_default_or_the_default_moved_with_it():
    """A recommendation nobody adopted and nobody explained is a loose end. Either
    it names what `render_plan` already defaults to, or spec 8.5 says why not."""
    from render_plan import DEFAULT_OUTPUT_EMA, DEFAULT_SEED_POLICY

    section = SPEC.read_text(encoding="utf-8").split("### 8.5", 1)[1] \
                  .split("### 8.6", 1)[0]
    for gpu in {gpu_of(result) for result in rows()}:
        recommendation = recommend_setting(measured_on(rows(), gpu),
                                           measured_on(committed(), gpu))
        shipped = (recommendation.seed_policy == DEFAULT_SEED_POLICY
                   and recommendation.output_ema == DEFAULT_OUTPUT_EMA)
        assert shipped or "not the shipped default" in section


@pytest.mark.parametrize("arm", sorted({arm_of(result) for result in rows()}),
                         ids=lambda arm: f"{arm[0]}-ema{arm[1]}")
def test_every_setting_has_a_clip_a_human_can_watch(arm):
    """The Gate's manual-verification half. Every arm, not only the recommended
    one: a reader who cannot see what `random` looks like cannot read the column."""
    written = artefact_of(committed(), arm)
    assert written is not None, f"no committed run of {arm} wrote clips"
    assert (STABILITY_RESULTS_DIR / written["comparison_clip"]).is_file()
    assert (STABILITY_RESULTS_DIR / written["comparison_still"]).is_file()


def test_no_arm_landed_beside_the_baselines_spec_8_8_and_7_4_quote():
    """The routing rule, asserted on disk rather than in the CLI: a swept arm in
    `bench/results/selective/` would silently become the row those two sections
    are drawn from. `.get` because the baselines predate the fields."""
    for result in load_selective_results(SELECTIVE_RESULTS_DIR).values():
        assert result["case"].get("seed_policy") is None, result["case"]["name"]
        assert result["case"].get("output_ema") is None, result["case"]["name"]


def test_the_spec_says_what_per_track_cannot_mean_under_the_shipped_primitive():
    """The issue's third trap: say it before building it, and write it into 8.5 if
    it turns out not to be expressible."""
    section = SPEC.read_text(encoding="utf-8").split("### 8.5", 1)[1] \
                  .split("### 8.6", 1)[0]
    assert "one diffusion call per frame" in section
    assert "canvas-pinned" in section
