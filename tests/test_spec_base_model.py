"""Issue #38's three blocks are the committed runs, and its Gate is on disk.

Spec 7.2's step-count sweep, 7.5's base-model comparison and 8.10's style LoRAs are
all generated from `bench/results/`, and held to a byte match the same way the other
nine are: re-run an arm and the merge gate fails until the spec is regenerated with
`uv run python scripts/regen_spec_blocks.py`.

The Gate items are asserted here rather than only in the runners, because each is a
property of what is on disk rather than of a function: the sweep has to have all
three step counts, both arms have to have kept the background bit-identical, and at
least two style LoRAs have to have loaded *and* changed the output.

GPU-free: every module here reads JSON and formats it.
"""

from pathlib import Path

import pytest
from sourceloader import ROOT

from bench.cli import report_models, report_steps, report_styles
from bench.models import (
    BASE_MODELS,
    DEFAULT_BASE,
    base_model_of,
    format_model_report,
)
from bench.paths import MODEL_RESULTS_DIR, STEPS_RESULTS_DIR, STYLE_RESULTS_DIR
from bench.results import load_records, require_recordable
from bench.selective import load_selective_results
from bench.steps import STEPS_SWEPT, latest_per_arm, steps_of, style_of
from bench.styles import BASE_ARM, arms_of, load_style_results, style_verdict

SPEC = Path(ROOT) / "docs/prompt-orchestrator-spec.md"

BLOCKS = {
    "STEP COUNT": (report_steps, STEPS_RESULTS_DIR),
    "BASE MODEL": (report_models, MODEL_RESULTS_DIR),
    "STYLE LORAS": (report_styles, STYLE_RESULTS_DIR),
}


def rendered(report, results_dir) -> str:
    import io

    buffer = io.StringIO()
    report(results_dir, out=buffer)
    return buffer.getvalue().strip()


def committed_between(name: str) -> str:
    text = SPEC.read_text(encoding="utf-8")
    begin, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
    assert begin in text and end in text, f"the {name} anchors are gone from the spec"
    return text.split(begin, 1)[1].split(end, 1)[0].strip()


@pytest.mark.parametrize("name", sorted(BLOCKS))
def test_the_spec_block_is_what_the_committed_results_say(name):
    report, results_dir = BLOCKS[name]
    assert committed_between(name) == rendered(report, results_dir), (
        f"spec's {name} block has drifted from {results_dir.name}/. Regenerate it "
        f"with `uv run python scripts/regen_spec_blocks.py`.")


# --- the step-count sweep (Gate item 1) --------------------------------------


def step_arms():
    return list(latest_per_arm(load_records(STEPS_RESULTS_DIR)).values())


def test_every_step_count_the_issue_names_was_measured():
    """1 / 2 / 4, or the estimate this sweep replaces has not been replaced."""
    shipped = [arm for arm in step_arms()
               if arm["scenario"]["model"] == BASE_MODELS[DEFAULT_BASE].model
               and arm["scenario"]["acceleration"] == "none"]
    assert {steps_of(arm) for arm in shipped} >= set(STEPS_SWEPT)


def test_every_arm_carries_the_per_module_split():
    """The Gate asks for UNet and VAE splits, not for a total with a note."""
    for arm in step_arms():
        split = arm["run"].get("per_module_ms") or {}
        assert set(split) == {"unet", "vae_encode", "vae_decode"}, arm["scenario"]["name"]


def test_the_style_arms_are_distinguishable_from_the_bare_ones():
    """A fused LoRA is a different engine at the same step count, so two rows that
    did not say so would look like one measured twice."""
    assert {style_of(arm) for arm in step_arms()} > {"-"}


def test_no_step_arm_landed_in_the_batch_curve():
    """`--marginal` reads every JSON in `bench/results/` as a (resolution, batch)
    cell, so an arm there would join spec 7.2's committed curve."""
    from bench.paths import RESULTS_DIR
    from bench.steps import is_step_arm

    beside = load_records(RESULTS_DIR)
    assert not [name for name, record in beside.items()
                if is_step_arm(record["scenario"]["name"])]


# --- the base-model arms (Gate items 2 and 5) --------------------------------


def model_arms():
    return list(load_selective_results(MODEL_RESULTS_DIR).values())


def test_both_base_models_were_run_through_the_shipped_path():
    assert {base_model_of(arm) for arm in model_arms()} == {
        base.model for base in BASE_MODELS.values()}


def test_every_base_model_arm_is_recordable():
    for arm in model_arms():
        require_recordable(arm)


def test_the_background_stayed_bit_identical_on_every_base_model():
    """The selective path's own criterion, and it must not be model-dependent."""
    for arm in model_arms():
        background = arm["gate"]["background"]
        assert background["passed"], base_model_of(arm)
        assert background["identical_frames"] == background["frames"]


def test_the_thirty_fps_verdict_names_the_region_count_and_the_machine():
    report = format_model_report(load_selective_results(MODEL_RESULTS_DIR))
    assert "regions/frame" in report
    assert "30 FPS MET" in report and "30 FPS NOT MET" in report


def test_no_base_model_arm_landed_beside_the_baselines():
    """`--selective-report` and `--portability-report` reduce that directory to the
    newest run per (case, GPU); an SD 1.5 arm there would become the row they
    quote for the shipped path."""
    from bench.paths import SELECTIVE_RESULTS_DIR

    for record in load_selective_results(SELECTIVE_RESULTS_DIR).values():
        assert not record["case"].get("base_model"), record["case"]["name"]


# --- the style LoRAs (Gate item 3) -------------------------------------------


def style_runs():
    return list(load_style_results(STYLE_RESULTS_DIR).values())


def test_at_least_two_style_loras_loaded_and_changed_the_output():
    """The Gate's number. A LoRA that loads and changes nothing is a failure."""
    for run in style_runs():
        verdicts = [style_verdict(arm) for arm in arms_of(run)]
        assert len([v for v in verdicts if v is not None and v.passed]) >= 2


def test_a_format_that_does_not_load_was_tried_and_recorded():
    """Issue #38's second trap: which formats fail is a measurement, not a note."""
    for run in style_runs():
        refused = [arm for arm in arms_of(run) if not arm.loaded]
        assert refused, "no LoCon arm - the format claim rests on a coincidence"
        for arm in refused:
            assert arm.error, arm.style


def test_the_control_arm_has_no_lora_and_is_not_judged_as_one():
    for run in style_runs():
        base = [arm for arm in arms_of(run) if arm.style == BASE_ARM]
        assert len(base) == 1
        assert style_verdict(base[0]) is None


def test_every_style_run_carries_the_artefact_a_human_judges_it_by():
    for run in style_runs():
        assert run["comparison_still"], "no still to look at"
        assert (STYLE_RESULTS_DIR / run["comparison_still"]).is_file()


def test_the_results_directory_is_not_swallowed_by_gitignore():
    """`.gitignore` carries a bare `models/` for the multi-GB downloads and it
    matches at any depth. Named the obvious way this directory would be silently
    untracked, which for one whose whole point is being committed is the worst
    kind of quiet."""
    import subprocess

    for directory in (MODEL_RESULTS_DIR, STEPS_RESULTS_DIR, STYLE_RESULTS_DIR):
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(directory / "README.md")],
            cwd=ROOT, capture_output=True)
        assert result.returncode != 0, f"{directory} is gitignored"
