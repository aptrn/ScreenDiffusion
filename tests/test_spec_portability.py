"""Spec 7.4's deploy-hardware block is the committed runs, not a transcription.

The same rule `tests/test_spec_measured_table.py` applies to 7.2 and
`tests/test_spec_selective_path.py` applies to 8.8: re-measure and the merge gate
fails until 7.4 is regenerated with `uv run python -m bench --portability-report`.

Issue #24's Gate is asserted here rather than only in the runner, for the reason
every other Gate is: a check that lives where the merge gate cannot see it is a
check that can quietly stop being made.

GPU-free: `bench.portability` reads JSON and formats it.
"""

import io
from pathlib import Path

from sourceloader import ROOT

from bench.cli import report_portability
from bench.paths import SELECTIVE_RESULTS_DIR
from bench.portability import (
    TARGET_FPS,
    criterion_verdict,
    fps_spread,
    is_deploy_gpu,
    portability_rows,
    split_by_role,
)
from bench.selective import load_selective_results

SPEC = Path(ROOT) / "docs/prompt-orchestrator-spec.md"
BEGIN = "<!-- BEGIN DEPLOY HARDWARE -->"
END = "<!-- END DEPLOY HARDWARE -->"


def spec_block() -> str:
    text = SPEC.read_text(encoding="utf-8")
    assert BEGIN in text and END in text, "7.4 lost its anchors"
    return text.split(BEGIN, 1)[1].split(END, 1)[0].strip()


def computed_block() -> str:
    buffer = io.StringIO()
    report_portability(SELECTIVE_RESULTS_DIR, out=buffer)
    return buffer.getvalue().strip()


def roles():
    return split_by_role(load_selective_results(SELECTIVE_RESULTS_DIR))


def test_the_spec_block_is_what_the_committed_runs_say():
    assert spec_block() == computed_block(), (
        "spec 7.4 no longer matches bench/results/selective/. Regenerate it with\n"
        "    uv run python -m bench --portability-report\n"
        "and paste the output between the DEPLOY HARDWARE anchors."
    )


# --- the Gate ----------------------------------------------------------------


def test_there_is_a_run_from_a_deploy_card():
    """Issue #24's first Gate item, read off the fingerprint rather than the prose."""
    _, deploy = roles()
    assert deploy, "no committed selective run from a 3090 Ti or 4090"
    assert is_deploy_gpu(deploy[0]["hardware"]["gpu_name"])


def test_there_is_still_a_dev_baseline_to_compare_it_against():
    """A portability table is a comparison. One row is a measurement, not an answer."""
    dev, _ = roles()
    assert dev, "the RTX 3080 laptop baseline was replaced rather than added to"


def test_the_verdict_names_the_region_count_it_was_measured_at():
    """The issue's Gate: 30 FPS at how many regions is the whole question."""
    _, deploy = roles()
    verdict = criterion_verdict(deploy[0])
    assert verdict.regions_per_frame > 1.0
    assert f"{verdict.regions_per_frame:.2f}" in verdict.statement
    assert verdict.statement in spec_block()


def test_the_two_runs_were_compared_at_the_same_region_count():
    """The issue's third trap. Two runs at different region counts are a comparison
    of how much frame was rendered, not of two cards."""
    dev, deploy = roles()
    assert deploy[0]["regions"]["regions_per_frame"] == (
        dev[0]["regions"]["regions_per_frame"])


def test_the_background_gate_still_passes_on_the_deploy_card():
    """The issue's third Gate item, on the record from the new hardware."""
    _, deploy = roles()
    background = deploy[0]["gate"]["background"]
    assert background["passed"], background["statement"]
    assert background["worst_pixels_changed"] == 0
    assert background["identical_frames"] == background["frames"] > 0


def test_the_verdict_is_outside_the_run_to_run_spread():
    """A verdict 2% from the threshold with a 5% spread is not a verdict. Every
    committed run on the card has to land on the same side of 30 FPS."""
    _, deploy = roles()
    spread = fps_spread(load_selective_results(SELECTIVE_RESULTS_DIR),
                        deploy[0]["hardware"]["gpu_name"])
    assert spread.runs > 1, "one run cannot say whether the verdict is noise"
    assert spread.decisive, spread.statement


def test_the_block_says_what_carried_and_what_did_not():
    """Both halves. A table on which everything carried would mean the comparison
    was not made, and one on which nothing did would mean it was made wrongly."""
    dev, deploy = roles()
    rows = portability_rows(dev[0], deploy[0])
    assert any(row.carried is True for row in rows)
    assert any(row.carried is False for row in rows)
    for row in rows:
        assert row.conclusion in spec_block(), row.conclusion


def test_the_absolute_milliseconds_are_the_thing_that_did_not_carry():
    """Spec 7.4 has always asserted this. It is now measured."""
    dev, deploy = roles()
    absolute = [row for row in portability_rows(dev[0], deploy[0])
                if "ms/frame" in row.conclusion]
    assert absolute and absolute[0].carried is False


def test_the_deploy_run_is_faster_than_the_laptop_it_replaced():
    """Not a tautology - it is the reason the criterion was re-measured at all."""
    dev, deploy = roles()
    assert (deploy[0]["run"]["ms_per_frame_with_detection"]
            < dev[0]["run"]["ms_per_frame_with_detection"])


def test_the_portability_table_in_7_4_still_names_the_criterion():
    """The hand-written half of 7.4: the conclusions table above the block."""
    section = SPEC.read_text(encoding="utf-8").split("### 7.4", 1)[1].split("\n---\n", 1)[0]
    assert f"{TARGET_FPS:.0f} FPS" in section
    assert "Engine build times" in section, "the table lost a row"
