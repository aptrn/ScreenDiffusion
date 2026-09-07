"""The capture-geometry comparison's arithmetic. Issue #39, spec 8.2, GPU-free.

`bench.capture` decides what a run *means*: how much canvas an object actually got,
whether an arm expressed the case, whether the background survived a bigger frame,
and which arm the section should recommend. The milliseconds come from
`bench.capture_runner`, which touches a GPU; everything asserted here is the rule
the merge gate can hold.
"""

from __future__ import annotations

import json

import pytest

from bench.capture import (
    CANVAS,
    CASES,
    DOG_CASE,
    GEOMETRIES,
    PEOPLE_CASE,
    PRIMITIVES,
    arm_name,
    background_statement,
    compare_primitives,
    detail_summary,
    format_capture_report,
    latest_per_case,
    recommendation,
    scaling_statement,
)
from bench.portability import FRAME_BUDGET_MS
from render_plan import CROP, MASKED


# --- the shipped vocabulary, pinned ------------------------------------------


def test_the_canvas_matches_the_apps_own_constant():
    from sourceloader import load_symbols

    assert CANVAS == load_symbols("main_gpu_addon.py", ["DIFFUSION_CANVAS"])[
        "DIFFUSION_CANVAS"]


def test_the_preview_dimension_matches_the_guis_own():
    """The preview stage is timed against the panel the GUI actually letterboxes
    into, so a change to one has to be a change to both."""
    import ast
    from pathlib import Path

    from bench.capture_runner import PREVIEW_DIM

    source = (Path(__file__).resolve().parent.parent / "main_gpu_addon.py").read_text(
        encoding="utf-8-sig")
    assigned = [node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Assign)
                and any(getattr(target, "attr", None) == "preview_dim"
                        for target in node.targets)]
    assert assigned and assigned[0].value.value == PREVIEW_DIM


def test_the_primitives_are_the_plans_own():
    assert PRIMITIVES == (MASKED, CROP)


def test_every_case_plans_through_the_validator_at_one_slot():
    for case in CASES.values():
        for primitive in case.primitives:
            plan = case.plan(primitive)
            assert plan.settings.primitive == primitive
            assert plan.honoured_target.max_instances == 1, (
                "K is not 1, so `crop` would fall back to `masked` on every frame")


def test_the_geometries_start_at_the_canvas_and_grow_past_it():
    """512x512 is the control - the app as it shipped and what every committed
    measurement here was taken at."""
    assert GEOMETRIES[0] == (CANVAS, CANVAS)
    assert any(width > CANVAS for width, _ in GEOMETRIES)


def test_a_1080p_geometry_is_measured():
    """The Gate asks for a 1080p verdict by name."""
    assert (1920, 1080) in GEOMETRIES


def test_the_identity_case_is_re_run_at_the_new_geometry():
    """The issue's step 5: crop lost `identity-dog` on 45-293 px crops, and this
    changes that input."""
    assert CASES[DOG_CASE].becomes == "cat"
    assert CASES[DOG_CASE].geometries == GEOMETRIES


# --- what a primitive spends on an object ------------------------------------


def test_crop_gives_the_object_the_whole_canvas():
    assert detail_summary(CROP, 300, 1920).canvas_px == CANVAS


def test_masked_gives_the_object_its_share_of_the_canvas():
    """The sentence spec 8.2 wrote down, as arithmetic: a region occupying a
    fraction of the frame is diffused at that fraction of 512 px."""
    assert detail_summary(MASKED, 480, 1920).canvas_px == pytest.approx(128.0)


def test_a_bigger_capture_costs_the_masked_primitive_detail():
    """The issue's first trap: raising the capture alone is a regression."""
    small = detail_summary(MASKED, 128, 512).canvas_px
    large = detail_summary(MASKED, 480, 1920).canvas_px
    assert large == pytest.approx(small)
    assert detail_summary(MASKED, 128, 1920).canvas_px < small


def test_the_detail_gain_is_the_ratio_the_claim_rests_on():
    assert detail_summary(CROP, 256, 1920).gain == pytest.approx(2.0)


def test_a_region_of_no_width_has_no_gain_rather_than_a_division():
    assert detail_summary(CROP, 0, 1920).gain == 0.0


# --- the record's own rules ---------------------------------------------------


def an_arm(primitive=MASKED, width=1920, height=1080, frame_path_ms=20.0,
           object_px=100.0, net_change=11.0, background_ok=True, identity=None,
           strength=0.49):
    stages = {"resize_in_ms": 0.4, "diffuse_ms": 15.0, "composite_ms": 1.2,
              "host_copy_ms": 0.9, "ipc_put_ms": 0.3, "ipc_roundtrip_ms": 6.0,
              "preview_ms": 2.0, "frame_path_ms": frame_path_ms}
    return {
        "name": arm_name(primitive, width, height), "primitive": primitive,
        "capture_width": width, "capture_height": height, "frames": 48,
        "diffusion_calls": 48, "crop_frames": 48 if primitive == CROP else 0,
        "regions_per_frame": 1.0, "stages": stages,
        "latency": {"mean_ms": frame_path_ms},
        "ms_per_frame": frame_path_ms,
        "fps": round(1000.0 / frame_path_ms, 4),
        "denoise": {"met": True, "t_index": 40, "timestep": 199,
                    "strength": strength, "kind": "restyle",
                    "statement": f"t_index 40, strength {strength}"},
        "denoise_points": [],
        "detail": {"region_px": object_px, "canvas_px": object_px, "gain": 1.0,
                   "statement": "a region"},
        "region_change": net_change + 1.0, "resample_change": 1.0,
        "net_region_change": net_change, "visible": net_change >= 8.0,
        "meets_budget": frame_path_ms <= FRAME_BUDGET_MS,
        "flicker": {"mean_abs_diff": 1.5},
        "background": {"passed": background_ok, "worst_pixels_changed": 0 if background_ok
                       else 4200, "statement": ""},
        "identity": identity,
    }


def a_result(arms, name=PEOPLE_CASE, gpu="NVIDIA GeForce RTX 4090"):
    return {
        "schema_version": 1, "kind": "capture-geometry",
        "case": {"name": name, "kind": "restyle", "clip": "people.mp4"},
        "clip": {"name": "people.mp4", "width": 1280, "height": 720,
                 "frames_used": 48},
        "arms": arms,
        "run": {"started_utc": "2026-09-07T17:00:00Z",
                "finished_utc": "2026-09-07T17:05:00Z",
                "engine_scenario": "img2img-tensorrt-512x512-b1", "canvas": CANVAS},
        "hardware": {"gpu_name": gpu},
        "comparison_clip": f"{name}-triptych.mp4",
    }


def test_a_broken_background_is_named_rather_than_explained():
    statement = background_statement([a_result([an_arm(background_ok=False)])])
    assert "NOT bit-identical" in statement
    assert "4200" in statement


def test_an_intact_background_says_which_geometries_it_held_at():
    statement = background_statement([
        a_result([an_arm(width=512, height=512), an_arm(width=1920, height=1080)])])
    assert "bit-identical" in statement and "NOT" not in statement
    assert "1920x1080" in statement


def test_the_comparison_reports_both_arms_at_one_geometry():
    result = a_result([an_arm(MASKED, object_px=69.0),
                       an_arm(CROP, object_px=512.0, frame_path_ms=21.0)])
    line = compare_primitives(result, 1920, 1080)
    assert "512 px" in line and "69" in line
    assert "21.00 ms" in line and "20.00 ms" in line


def test_the_comparison_is_none_when_only_one_primitive_ran_there():
    result = a_result([an_arm(MASKED)])
    assert compare_primitives(result, 1920, 1080) is None


def test_an_arm_that_broke_the_background_cannot_be_recommended():
    """The Gate's first item is a disqualifier, not a footnote."""
    result = a_result([an_arm(CROP, object_px=512.0, background_ok=False),
                       an_arm(MASKED, object_px=69.0)])
    assert "masked" in recommendation(result)


def test_an_arm_that_missed_the_budget_cannot_be_recommended():
    result = a_result([an_arm(CROP, object_px=512.0, frame_path_ms=90.0),
                       an_arm(MASKED, object_px=69.0)])
    assert "masked" in recommendation(result)


def test_an_arm_that_did_not_express_the_case_cannot_be_recommended():
    """Issue #5's first trap, repeated: the cheapest primitive that cannot do the
    job is not the winner, however fast it was."""
    result = a_result([an_arm(CROP, object_px=512.0, net_change=2.0),
                       an_arm(MASKED, object_px=69.0)])
    assert "masked" in recommendation(result)


def test_the_recommendation_is_the_most_canvas_that_fits():
    result = a_result([an_arm(MASKED, object_px=69.0),
                       an_arm(CROP, object_px=512.0, frame_path_ms=21.0)])
    assert "crop-1920x1080" in recommendation(result)


def test_nothing_qualifying_is_said_out_loud():
    result = a_result([an_arm(MASKED, background_ok=False)])
    assert "No arm" in recommendation(result)


def test_an_identity_arm_is_judged_by_the_detector_and_not_by_the_change():
    """A frame can change a great deal and still be a dog."""
    changed = an_arm(CROP, object_px=512.0, net_change=30.0,
                     identity={"achieved": False, "became": 1, "frames_probed": 48})
    kept = an_arm(MASKED, object_px=69.0, net_change=30.0,
                  identity={"achieved": True, "became": 40, "frames_probed": 48})
    assert "masked" in recommendation(a_result([changed, kept], name=DOG_CASE))


# --- the per-stage breakdown --------------------------------------------------


def test_the_scaling_statement_names_every_stage():
    small = an_arm(MASKED, width=512, height=512, frame_path_ms=16.0)
    small["stages"].update(resize_in_ms=0.1, host_copy_ms=0.2, ipc_put_ms=0.05,
                           ipc_roundtrip_ms=1.0, preview_ms=0.5, composite_ms=0.6)
    statement = scaling_statement(a_result([small, an_arm(MASKED)]))
    for stage in ("resize in", "diffuse", "composite", "host copy", "ipc put",
                  "ipc roundtrip", "preview"):
        assert stage in statement


def test_the_scaling_statement_says_the_canvas_cannot_move():
    statement = scaling_statement(a_result([
        an_arm(MASKED, width=512, height=512), an_arm(MASKED)]))
    assert "512x512" in statement and "9.58 ms" in statement


def test_one_geometry_alone_has_no_scaling_to_report():
    assert scaling_statement(a_result([an_arm(MASKED)])) is None


# --- the block ----------------------------------------------------------------


def test_no_results_says_so_rather_than_drawing_an_empty_table():
    assert "no capture geometry measured" in format_capture_report({})


def test_the_block_carries_a_row_per_arm():
    report = format_capture_report({"a.json": a_result([
        an_arm(MASKED, width=512, height=512), an_arm(CROP, object_px=512.0)])})
    assert report.count("| capture-people |") == 2
    assert "30 FPS on the frame path" in report


def test_the_block_grows_a_gpu_column_only_past_one_machine():
    one = format_capture_report({"a.json": a_result([an_arm()])})
    assert "| GPU |" not in one
    two = format_capture_report({
        "a.json": a_result([an_arm()]),
        "b.json": a_result([an_arm()], name=DOG_CASE, gpu="RTX 3080 Laptop GPU")})
    assert "| GPU |" in two


def test_one_run_per_case_and_gpu_survives_a_second_machine():
    """The rule every block here follows: a 4090 run adds a row beside the 3080's
    instead of erasing it (issue #25)."""
    laptop = a_result([an_arm()], gpu="NVIDIA GeForce RTX 3080 Laptop GPU")
    laptop["run"]["finished_utc"] = "2026-09-06T10:00:00Z"
    kept = latest_per_case({"a.json": a_result([an_arm()]), "b.json": laptop})
    assert len(kept) == 2


def test_the_record_round_trips_through_json():
    case = CASES[PEOPLE_CASE]
    assert json.loads(json.dumps(case.to_dict()))["geometries"] == [
        list(pair) for pair in case.geometries]


def test_a_case_can_be_shortened_for_development():
    assert CASES[PEOPLE_CASE].replace(frames=4).frames == 4
