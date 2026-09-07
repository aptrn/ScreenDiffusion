"""Spec 8.2's capture-geometry block is the committed runs. Issue #39, GPU-free.

The same rule the other eight generated blocks are held to: re-run a case and the
merge gate fails until 8.2 is regenerated with
`uv run python scripts/regen_spec_blocks.py`.

The Gate's items are asserted here rather than only in the runner, because they are
properties of what is on disk: non-target pixels bit-identical **at 1080p**, a
crop-against-masked comparison at K=1 at the new geometry, the cost of the larger
capture broken out per stage, a 30 FPS verdict with its region count and machine,
and the identity case re-run.
"""

import io
from pathlib import Path
from typing import List

import pytest
from sourceloader import ROOT

from bench.capture import (
    DOG_CASE,
    PEOPLE_CASE,
    arms_of,
    broke_background,
    latest_per_case,
)
from bench.cli import report_capture
from bench.paths import CAPTURE_RESULTS_DIR
from bench.portability import FRAME_BUDGET_MS, is_deploy_gpu
from bench.results import gpu_of, load_records, require_recordable
from render_plan import CROP, MASKED

SPEC = Path(ROOT) / "docs/prompt-orchestrator-spec.md"
BEGIN = "<!-- BEGIN CAPTURE GEOMETRY -->"
END = "<!-- END CAPTURE GEOMETRY -->"

# The geometry the Gate names by name.
DEPLOY_CAPTURE = (1920, 1080)


def spec_block() -> str:
    text = SPEC.read_text(encoding="utf-8")
    assert BEGIN in text and END in text, "8.2's capture block lost its anchors"
    return text.split(BEGIN, 1)[1].split(END, 1)[0].strip()


def computed_block() -> str:
    buffer = io.StringIO()
    report_capture(CAPTURE_RESULTS_DIR, out=buffer)
    return buffer.getvalue().strip()


def committed() -> List[dict]:
    return list(load_records(CAPTURE_RESULTS_DIR).values())


def rows() -> List[dict]:
    return sorted(latest_per_case(load_records(CAPTURE_RESULTS_DIR)).values(),
                  key=lambda result: (result["case"]["name"], gpu_of(result)))


def per_run():
    return pytest.mark.parametrize(
        "result", rows(),
        ids=lambda result: f"{result['case']['name']}@{gpu_of(result)}")


def per_arm():
    return pytest.mark.parametrize(
        "result,arm", [(result, arm) for result in rows() for arm in arms_of(result)],
        ids=lambda item: item["name"] if isinstance(item, dict) and "name" in item
        else "")


def test_the_spec_block_is_what_the_committed_runs_say():
    assert spec_block() == computed_block(), (
        "spec 8.2's capture block no longer matches bench/results/capture/. "
        "Regenerate it with\n"
        "    uv run python scripts/regen_spec_blocks.py\n")


def test_every_committed_run_could_have_reached_disk():
    runs = committed()
    assert runs, "issue #39's deliverable is a committed capture comparison"
    for result in runs:
        require_recordable(result)
        assert result["kind"] == "capture-geometry"


def test_both_cases_were_measured():
    """The priority case and the identity case - the issue's steps 3 and 5."""
    assert {result["case"]["name"] for result in rows()} == {PEOPLE_CASE, DOG_CASE}


def test_the_comparison_ran_on_deploy_hardware():
    """A frame-budget claim is a deploy-hardware claim (spec 7.4)."""
    assert any(is_deploy_gpu(gpu_of(result)) for result in rows())


# --- the Gate ---------------------------------------------------------------


@per_run()
def test_non_target_pixels_are_bit_identical_at_1080p(result):
    """The Gate's first item, and it does not get easier because the frame got
    bigger."""
    at_1080p = [arm for arm in arms_of(result)
                if (arm["capture_width"], arm["capture_height"]) == DEPLOY_CAPTURE]
    assert at_1080p, "no arm was measured at 1920x1080"
    for arm in at_1080p:
        background = arm["background"]
        assert background["passed"], f"{arm['name']}: {background['statement']}"
        assert background["worst_pixels_changed"] == 0


@per_run()
def test_no_arm_at_any_geometry_moved_a_non_target_pixel(result):
    assert not broke_background(result), [
        arm["name"] for arm in broke_background(result)]


@per_run()
def test_both_primitives_were_measured_at_every_geometry(result):
    """A comparison needs both arms at the same geometry or it is two runs."""
    seen = {(arm["primitive"], arm["capture_width"], arm["capture_height"])
            for arm in arms_of(result)}
    for width, height in {(arm["capture_width"], arm["capture_height"])
                          for arm in arms_of(result)}:
        assert (MASKED, width, height) in seen
        assert (CROP, width, height) in seen


@per_arm()
def test_every_arm_breaks_its_cost_out_per_stage(result, arm):
    """The Gate's third item: not folded into one number."""
    stages = arm["stages"]
    for stage in ("resize_in_ms", "diffuse_ms", "composite_ms", "host_copy_ms",
                  "ipc_put_ms", "ipc_roundtrip_ms", "preview_ms"):
        assert stages[stage] >= 0.0, stage
    assert stages["frame_path_ms"] == pytest.approx(
        stages["resize_in_ms"] + stages["diffuse_ms"] + stages["composite_ms"]
        + stages["ipc_put_ms"], abs=0.01)


@per_arm()
def test_every_arm_carries_its_30_fps_verdict_with_a_region_count(result, arm):
    """The Gate's fourth item, as a number a reader recomputes."""
    assert arm["meets_budget"] == (arm["ms_per_frame"] <= FRAME_BUDGET_MS)
    assert arm["regions_per_frame"] > 0
    assert gpu_of(result)


@per_arm()
def test_every_arm_was_measured_at_the_strength_it_needed(result, arm):
    """Spec 8.2 measured that the two primitives need different strengths for the
    same job, so timing both at one of them would measure the wrong primitive."""
    assert arm["denoise_points"], "the ladder was not swept for this arm"
    assert arm["denoise"]["t_index"] in [point["t_index"]
                                         for point in arm["denoise_points"]]


@per_arm()
def test_crop_gave_the_object_the_whole_canvas(result, arm):
    if arm["primitive"] != CROP:
        return
    assert arm["crop_frames"] == arm["diffusion_calls"] > 0, (
        "an arm asking for crop fell back to masked on some frame")
    assert arm["detail"]["canvas_px"] == result["run"]["canvas"]


@per_arm()
def test_every_arm_costs_exactly_one_diffusion_call_per_rendered_frame(result, arm):
    """K=1 is what makes crop affordable; two regions would be two calls."""
    assert arm["diffusion_calls"] <= arm["frames"]
    assert arm["regions_per_frame"] <= 1.0


def test_the_identity_case_states_its_result_either_way():
    """The issue's step 5: crop lost `identity-dog` at every rung on the old
    geometry, and the finding at the new one is a finding either way."""
    dog = [result for result in rows() if result["case"]["name"] == DOG_CASE]
    assert dog, "the identity case was not re-run"
    for arm in arms_of(dog[0]):
        assert arm["identity"] is not None, (
            f"{arm['name']} has no identity verdict, so the case was not judged")
        assert arm["identity"]["statement"]


def test_the_clip_a_human_watches_is_committed():
    for result in rows():
        assert result["comparison_clip"]
        assert (CAPTURE_RESULTS_DIR / result["comparison_clip"]).is_file()
        assert (CAPTURE_RESULTS_DIR / result["comparison_still"]).is_file()


def test_the_section_says_what_the_comparison_changed_about_its_own_decision():
    """The Gate's last item: 8.2 either revises the primitive decision or records
    why it stands."""
    section = SPEC.read_text(encoding="utf-8").split("### 8.2", 1)[1].split(
        "### 8.3", 1)[0]
    assert "issue #39" in section
    assert "K=1" in section
