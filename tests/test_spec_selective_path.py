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

from sourceloader import ROOT

from bench.cli import report_selective
from bench.paths import SELECTIVE_RESULTS_DIR
from bench.results import require_recordable
from bench.selective import (
    CASES,
    PRIORITY_CASE,
    latest_per_case,
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


def committed() -> dict:
    latest = latest_per_case(load_selective_results(SELECTIVE_RESULTS_DIR))
    return {result["case"]["name"]: result for result in latest.values()}


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
    assert set(committed()) == set(CASES)


# --- the Gate ---------------------------------------------------------------


def test_the_background_was_bit_identical_on_every_frame():
    """The issue's sharp criterion, asserted on what is on disk."""
    background = committed()[PRIORITY_CASE]["gate"]["background"]
    assert background["passed"], background["statement"]
    assert background["identical_frames"] == background["frames"] > 0
    assert background["worst_pixels_changed"] == 0
    assert background["background_pixels"] > 0, "nothing was outside the regions"


def test_the_region_was_visibly_restyled_net_of_the_control():
    change = committed()[PRIORITY_CASE]["gate"]["change"]
    assert change["passed"], change["statement"]
    assert change["net_change"] >= change["threshold"]


def test_no_track_was_starved_by_the_round_robin():
    """The Gate's second item, exercised with more tracks than slots."""
    coverage = committed()[PRIORITY_CASE]["gate"]["coverage"]
    assert coverage["passed"], coverage["statement"]
    assert coverage["max_tracks"] > coverage["slots"], (
        "the probe did not force more tracks than slots, so it proved nothing")
    assert coverage["worst_gap_frames"] <= coverage["bound_frames"]


def test_the_loop_produced_a_frame_for_every_frame_it_took():
    stall = committed()[PRIORITY_CASE]["gate"]["stall"]
    assert stall["passed"], stall["statement"]
    assert stall["worst_offer_ms"] <= stall["max_offer_ms"]


def test_the_run_reports_an_output_fps_and_a_flicker_figure():
    """Step 5: both numbers, on the committed clip."""
    result = committed()[PRIORITY_CASE]
    assert result["run"]["fps"] > 0
    assert result["flicker"]["mean_abs_diff"] is not None
    assert result["flicker"]["pairs_scored"] > 0


def test_the_run_rendered_the_hardcoded_priority_case():
    plan = committed()[PRIORITY_CASE]["plan"]
    assert (plan["concept"], plan["region"], plan["mode"]) == (
        "person", "lower_half", "selective")


def test_one_diffusion_call_per_frame_whatever_the_object_count():
    """Issue #5 chose the masked primitive; K is masked regions, not calls."""
    run = committed()[PRIORITY_CASE]["run"]
    regions = committed()[PRIORITY_CASE]["regions"]
    assert run["diffusion_calls"] <= run["frames"]
    assert regions["regions_per_frame"] > 1.0, "nothing was selective about this run"


def test_how_many_regions_the_size_floor_skipped_is_recorded():
    """The issue's first trap: skipped is a number, not a silence."""
    assert "skipped_small_total" in committed()[PRIORITY_CASE]["regions"]


def test_the_clip_a_human_has_to_watch_is_committed():
    """Manual verification is a Gate item; it needs a file to point at."""
    result = committed()[PRIORITY_CASE]
    assert result["comparison_clip"]
    assert (SELECTIVE_RESULTS_DIR / result["comparison_clip"]).is_file()
    assert (SELECTIVE_RESULTS_DIR / result["comparison_still"]).is_file()


def test_the_spec_says_the_frame_rate_is_not_the_claim():
    """The issue's fourth trap: 30 FPS is M2's problem and cannot be judged here."""
    section = SPEC.read_text(encoding="utf-8").split("### 8.8", 1)[1].split("## 9.", 1)[0]
    assert "30 FPS is not claimed" in section
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
