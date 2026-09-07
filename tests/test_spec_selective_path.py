"""Spec 8.8 is the committed run, not a transcription - and issue #8's Gate on disk.

The same rule `tests/test_spec_measured_table.py` applies to 7.2 and
`tests/test_spec_primitive_decision.py` applies to 8.2: run the case again and the
merge gate fails until 8.8 is regenerated with
`uv run python -m bench --selective-report`.

The Gate items themselves are asserted here rather than only in the runner, because
a check that lives where the merge gate cannot see it is a check that can quietly
stop being made.

GPU-free: `bench.selective` reads JSON and formats it.
"""

import io
from pathlib import Path
from typing import List

import pytest
from sourceloader import ROOT

from bench.cli import report_selective
from bench.paths import SELECTIVE_RESULTS_DIR
from bench.portability import is_deploy_gpu
from bench.results import require_recordable
from bench.selective import (
    CASES,
    PRIORITY_CASE,
    latest_per_case_and_gpu,
    load_selective_results,
)

SPEC = Path(ROOT) / "docs/prompt-orchestrator-spec.md"
BEGIN = "<!-- BEGIN SELECTIVE PATH -->"
END = "<!-- END SELECTIVE PATH -->"


def spec_block() -> str:
    text = SPEC.read_text(encoding="utf-8")
    assert BEGIN in text and END in text, "8.8 lost its anchors"
    return text.split(BEGIN, 1)[1].split(END, 1)[0].strip()


def computed_block() -> str:
    buffer = io.StringIO()
    report_selective(SELECTIVE_RESULTS_DIR, out=buffer)
    return buffer.getvalue().strip()


def committed() -> List[dict]:
    """One run per case *and machine*, in a stable order (issue #24).

    Per machine because the same case now has a run on the deploy card as well as on
    the dev laptop, and the Gate is asserted on every one of them rather than on
    whichever happened to be newest.
    """
    latest = latest_per_case_and_gpu(load_selective_results(SELECTIVE_RESULTS_DIR))
    return sorted(latest.values(),
                  key=lambda result: (result["case"]["name"],
                                      result["hardware"]["gpu_name"]))


def runs_of(case: str = PRIORITY_CASE) -> List[dict]:
    """Every machine's newest run of `case`."""
    return [result for result in committed() if result["case"]["name"] == case]


def newest(case: str = PRIORITY_CASE) -> dict:
    """The most recent run of `case` on any machine - whose Gate lines 8.8 prints."""
    return max(runs_of(case), key=lambda result: result["run"]["finished_utc"])


def per_machine(case: str = PRIORITY_CASE):
    """Parametrise a Gate item over every machine that has a committed run."""
    return pytest.mark.parametrize(
        "result", runs_of(case), ids=lambda result: result["hardware"]["gpu_name"])


def test_the_spec_block_is_what_the_committed_run_says():
    assert spec_block() == computed_block(), (
        "spec 8.8 no longer matches bench/results/selective/. Regenerate it with\n"
        "    uv run python -m bench --selective-report\n"
        "and paste the output between the SELECTIVE PATH anchors."
    )


def test_every_committed_run_could_have_reached_disk():
    results = load_selective_results(SELECTIVE_RESULTS_DIR)
    assert results, "issue #8's deliverable is a committed end-to-end run"
    for filename, result in results.items():
        require_recordable(result)
        assert result["kind"] == "selective", filename


def test_the_priority_case_has_a_committed_run():
    assert {result["case"]["name"] for result in committed()} == set(CASES)


def test_the_case_was_measured_on_deploy_hardware_too():
    """Issue #24's Gate: a `selective-people` result from a 3090 Ti or a 4090."""
    assert any(is_deploy_gpu(result["hardware"]["gpu_name"])
               for result in runs_of()), (
        "spec 7.4's absolute rows need a run on the hardware this deploys to")


# --- the Gate ---------------------------------------------------------------


@per_machine()
def test_the_background_was_bit_identical_on_every_frame(result):
    """The issue's sharp criterion, asserted on what is on disk - and on every
    machine that has a run, which is issue #24's third Gate item."""
    background = result["gate"]["background"]
    assert background["passed"], background["statement"]
    assert background["identical_frames"] == background["frames"] > 0
    assert background["worst_pixels_changed"] == 0
    assert background["background_pixels"] > 0, "nothing was outside the regions"


@per_machine()
def test_the_region_was_visibly_restyled_net_of_the_control(result):
    change = result["gate"]["change"]
    assert change["passed"], change["statement"]
    assert change["net_change"] >= change["threshold"]


@per_machine()
def test_no_track_was_starved_by_the_round_robin(result):
    """The Gate's second item, exercised with more tracks than slots."""
    coverage = result["gate"]["coverage"]
    assert coverage["passed"], coverage["statement"]
    assert coverage["max_tracks"] > coverage["slots"], (
        "the probe did not force more tracks than slots, so it proved nothing")
    assert coverage["worst_gap_frames"] <= coverage["bound_frames"]


@per_machine()
def test_the_loop_produced_a_frame_for_every_frame_it_took(result):
    stall = result["gate"]["stall"]
    assert stall["passed"], stall["statement"]
    assert stall["worst_offer_ms"] <= stall["max_offer_ms"]


@per_machine()
def test_the_run_reports_an_output_fps_and_a_flicker_figure(result):
    """Step 5: both numbers, on the committed clip."""
    assert result["run"]["fps"] > 0
    assert result["flicker"]["mean_abs_diff"] is not None
    assert result["flicker"]["pairs_scored"] > 0


@per_machine()
def test_the_run_rendered_the_hardcoded_priority_case(result):
    plan = result["plan"]
    assert (plan["concept"], plan["region"], plan["mode"]) == (
        "person", "lower_half", "selective")


@per_machine()
def test_one_diffusion_call_per_frame_whatever_the_object_count(result):
    """Issue #5 chose the masked primitive; K is masked regions, not calls."""
    assert result["run"]["diffusion_calls"] <= result["run"]["frames"]
    assert result["regions"]["regions_per_frame"] > 1.0, (
        "nothing was selective about this run")


def test_how_many_regions_the_size_floor_skipped_is_recorded():
    """The issue's first trap: skipped is a number, not a silence."""
    assert "skipped_small_total" in newest()["regions"]


@per_machine()
def test_the_clip_a_human_has_to_watch_is_committed(result):
    """Manual verification is a Gate item; it needs a file to point at - one per
    machine, because a 4090's output frames are not the laptop's."""
    assert result["comparison_clip"]
    assert (SELECTIVE_RESULTS_DIR / result["comparison_clip"]).is_file()
    assert (SELECTIVE_RESULTS_DIR / result["comparison_still"]).is_file()


def test_the_spec_hands_the_frame_rate_verdict_to_7_4():
    """8.8 asks whether the path works, not whether it is fast enough. The frame
    rate is a deploy-hardware claim, and 7.4 is where issue #24 answered it."""
    section = SPEC.read_text(encoding="utf-8").split("### 8.8", 1)[1].split("## 9.", 1)[0]
    assert "30 FPS is not this section's claim" in section
    assert "7.4" in section


def test_the_selective_path_did_not_make_the_app_depend_on_the_harness():
    """The bench imports the shipped modules; the shipped modules must not import
    the bench. Checked over the imports, so a comment naming it does not fail."""
    import ast

    for filename in ("main_gpu_addon.py", "region_scheduler.py", "compositor.py",
                     "render_plan.py", "detection.py", "detector_worker.py"):
        tree = ast.parse((Path(ROOT) / filename).read_text(encoding="utf-8-sig"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "bench" not in imported, filename
