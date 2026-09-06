"""The rendering-primitive comparison's vocabulary, cost model and decision rule.

Issue #5, spec 8.2. GPU-free: `bench.primitives` decides what the milliseconds
*mean*, and the milliseconds themselves come from `bench.primitive_runner`.
"""

import json

import pytest

from bench.primitives import (
    CASES,
    CROP,
    DENOISE_LADDER,
    FULL_BOX,
    IDENTITY,
    IDENTITY_CASE,
    LOWER_HALF,
    MASKED,
    PRIMITIVES,
    REGIONS,
    RESTYLE,
    RESTYLE_CASE,
    SMALL_OBJECT_PX,
    Box,
    DenoisePoint,
    Measurement,
    calls_per_frame,
    clamp_box,
    clip_path,
    decide,
    denoise_strength,
    iou,
    one_step_finding,
    load_track,
    region_box,
    required_denoise,
    small_object_summary,
    smooth_track,
    track_from_dict,
    track_path,
)


# --- the region vocabulary --------------------------------------------------------

def test_the_region_vocabulary_is_the_one_the_issue_fixes():
    """The issue's fifth trap: use these names, do not invent others."""
    assert set(REGIONS) == {"full_box", "upper_third", "upper_half", "center",
                            "lower_half", "lower_third"}


def test_full_box_is_the_box():
    assert region_box(Box(10, 20, 30, 80), FULL_BOX) == Box(10, 20, 30, 80)


@pytest.mark.parametrize("region,expected", [
    ("upper_third", (0, 20)),
    ("upper_half", (0, 30)),
    ("center", (20, 40)),
    ("lower_half", (30, 60)),
    ("lower_third", (40, 60)),
])
def test_each_region_is_the_band_of_the_box_its_name_says(region, expected):
    band = region_box(Box(0, 0, 10, 60), region)
    assert (band.y0, band.y1) == expected
    assert (band.x0, band.x1) == (0, 10), "a region is a horizontal band, not a crop"


def test_center_is_the_middle_third_the_two_thirds_leave_over():
    upper = region_box(Box(0, 0, 10, 30), "upper_third")
    center = region_box(Box(0, 0, 10, 30), "center")
    lower = region_box(Box(0, 0, 10, 30), "lower_third")
    assert (upper.y1, center.y0, center.y1, lower.y0) == (10, 10, 20, 20)


def test_a_region_never_rounds_away_to_nothing():
    """A region of zero height silently drops an object from the render."""
    for region in REGIONS:
        assert region_box(Box(5, 5, 6, 6), region).height >= 1, region


def test_an_invented_region_name_is_refused():
    with pytest.raises(ValueError, match="region vocabulary"):
        region_box(Box(0, 0, 10, 10), "torso")


def test_a_box_is_clamped_into_the_frame_and_keeps_a_pixel():
    assert clamp_box(Box(-20, -5, 40, 12), 30, 10) == Box(0, 0, 30, 10)
    assert clamp_box(Box(29, 9, 29, 9), 30, 10) == Box(29, 9, 30, 10)


# --- small objects ------------------------------------------------------------------

def test_the_small_object_summary_counts_both_readings_of_small():
    """A 45x345 region is narrow, not small - the gap issue #17 recorded."""
    summary = small_object_summary([Box(0, 0, 45, 345), Box(0, 0, 300, 400)])
    assert summary.min_side_under == 1
    assert summary.both_sides_under == 0
    assert summary.min_side_px == 45
    assert "still open" in summary.statement


def test_a_genuinely_small_region_counts_under_both_readings():
    summary = small_object_summary([Box(0, 0, 40, 50)])
    assert (summary.min_side_under, summary.both_sides_under) == (1, 1)
    assert "still open" not in summary.statement


def test_the_small_object_threshold_is_the_gate_s_96_px():
    assert SMALL_OBJECT_PX == 96


def test_no_region_measured_says_so_rather_than_reporting_zeros():
    summary = small_object_summary([])
    assert summary.min_side_px is None and "nothing was measured" in summary.statement


# --- the cost model -------------------------------------------------------------------

def test_the_crop_primitive_costs_one_call_per_object():
    assert calls_per_frame(CROP, 4) == 4


def test_the_masked_primitive_costs_one_call_whatever_the_object_count():
    assert calls_per_frame(MASKED, 1) == calls_per_frame(MASKED, 9) == 1


def test_a_frame_with_no_object_costs_nothing_under_either_primitive():
    """There is no region to composite into, so a full-frame pass is work thrown away."""
    assert calls_per_frame(CROP, 0) == calls_per_frame(MASKED, 0) == 0


def test_an_unknown_primitive_is_refused():
    with pytest.raises(ValueError):
        calls_per_frame("controlnet", 1)


def test_both_implemented_primitives_state_what_they_cannot_express():
    """The Gate asks for it per primitive, and no benchmark produces it."""
    assert set(PRIMITIVES) == {CROP, MASKED}
    assert "one prompt and one denoise per frame" in \
        PRIMITIVES[MASKED].cannot_express.lower()
    assert "small-crop quality floor" in PRIMITIVES[CROP].cannot_express.lower()
    assert PRIMITIVES[CROP].spec_option == "A"
    assert PRIMITIVES[MASKED].spec_option == "B"


# --- the cases ---------------------------------------------------------------------

def test_the_two_cases_are_the_two_the_issue_names():
    assert set(CASES) == {RESTYLE_CASE, IDENTITY_CASE}
    assert CASES[RESTYLE_CASE].kind == RESTYLE
    assert CASES[IDENTITY_CASE].kind == IDENTITY


def test_the_priority_case_is_the_sub_region_restyle():
    restyle = CASES[RESTYLE_CASE]
    assert restyle.priority is True
    assert restyle.region == LOWER_HALF
    assert CASES[IDENTITY_CASE].priority is False


def test_the_identity_case_says_what_the_subject_should_become():
    identity = CASES[IDENTITY_CASE]
    assert (identity.target, identity.becomes) == ("dog", "cat")
    assert identity.region == FULL_BOX


def test_both_cases_point_at_a_committed_clip():
    """The issue's second trap: live capture makes the runs non-reproducible."""
    for case in CASES.values():
        assert clip_path(case.clip).is_file(), case.clip


def test_the_case_frames_are_consecutive_enough_to_measure_flicker():
    for case in CASES.values():
        assert case.frames >= 2, "a flicker metric needs consecutive pairs"


def test_the_track_path_sits_beside_its_clip():
    assert track_path("people.mp4").name == "people.track.json"
    assert track_path("people.mp4").parent == clip_path("people.mp4").parent


def test_a_missing_track_says_how_to_regenerate_it():
    with pytest.raises(FileNotFoundError, match="--write-track"):
        load_track("nothing-here.mp4")


def test_a_track_round_trips_through_its_dict_form():
    track = track_from_dict({
        "clip": "people.mp4", "target": "person", "detector": "yolo-world-s-640",
        "conf": 0.25, "width": 1280, "height": 720, "fps": 30.0, "frame_count": 450,
        "generated_utc": "2026-09-06T00:00:00Z",
        "frames": [{"index": 0, "boxes": [[1, 2, 3, 4]]}, {"index": 1, "boxes": []}],
    })
    assert track.at(0) == [Box(1, 2, 3, 4)]
    assert track.at(1) == [] and track.at(99) == []
    assert track.at(0, limit=0) == []
    assert json.loads(json.dumps(track.to_dict()))["frames"][0]["boxes"] == [[1, 2, 3, 4]]


# --- denoise strength -----------------------------------------------------------------

def test_denoise_strength_is_the_noise_amplitude_of_the_timestep():
    assert denoise_strength(1.0) == 0.0  # no noise at all
    assert denoise_strength(0.0) == 1.0  # pure noise
    assert denoise_strength(0.75) == pytest.approx(0.5)


def test_an_impossible_alpha_is_refused():
    with pytest.raises(ValueError):
        denoise_strength(1.5)


def test_the_ladder_is_ordered_and_lands_inside_the_schedulers_range():
    """`set_t_index_list` clamps to 1..49; a rung outside it would be silently moved."""
    assert list(DENOISE_LADDER) == sorted(DENOISE_LADDER)
    assert all(1 <= rung <= 49 for rung in DENOISE_LADDER)


def point(t_index, region_change, hits=None, frames=4):
    return DenoisePoint(t_index=t_index, timestep=1000 - 20 * t_index,
                        strength=1.0 - t_index / 50, region_change=region_change,
                        outside_change=0.0, frames=frames,
                        identity_hits=hits,
                        identity_frames=None if hits is None else frames)


def test_a_restyle_needs_the_least_denoise_that_is_still_visible():
    """Higher t_index is less denoise, so "least that worked" is the largest index."""
    requirement = required_denoise(
        [point(20, 60.0), point(35, 20.0), point(40, 12.0), point(45, 3.0)], RESTYLE)
    assert requirement.met is True
    assert requirement.t_index == 40
    assert "least denoise" in requirement.rule


def test_a_restyle_that_never_became_visible_is_recorded_as_unmet():
    requirement = required_denoise([point(40, 1.0), point(45, 0.5)], RESTYLE)
    assert requirement.met is False
    assert requirement.t_index == 40, "the strongest denoise tried is the fallback"
    assert "not achieved" in requirement.statement


def test_an_identity_change_is_judged_by_the_detector_not_by_how_much_moved():
    """A frame can change enormously and still be a dog."""
    requirement = required_denoise(
        [point(20, 90.0, hits=4), point(30, 70.0, hits=3), point(40, 60.0, hits=0)],
        IDENTITY)
    assert requirement.t_index == 30, "60/255 of change with no cat is not a cat"
    assert "new identity" in requirement.rule


def test_an_identity_change_nothing_achieved_is_a_finding_with_a_fallback_setting():
    requirement = required_denoise(
        [point(20, 90.0, hits=0), point(35, 40.0, hits=1, frames=8)], IDENTITY)
    assert requirement.met is False
    assert requirement.t_index == 20
    assert "was read as the new identity in 0/4 frames" in requirement.statement


def test_a_sweep_with_no_rungs_is_an_error_rather_than_an_answer():
    with pytest.raises(ValueError):
        required_denoise([], RESTYLE)


# --- the decision -----------------------------------------------------------------------

def measurement(primitive, ms, expresses=True, priority=True, flicker=1.0,
                case=RESTYLE_CASE, kind=RESTYLE):
    return Measurement(case=case, kind=kind, priority=priority, primitive=primitive,
                       ms_per_frame=ms, flicker=flicker, objects_per_frame=4.0,
                       expresses=expresses, t_index=40)


def test_the_cheapest_primitive_that_expressed_the_priority_case_wins():
    decision = decide([measurement(MASKED, 30.0), measurement(CROP, 90.0)])
    assert decision.primitive == MASKED
    assert decision.priority_case == RESTYLE_CASE
    assert "3.00x" in decision.statement


def test_the_cheapest_primitive_that_cannot_express_the_case_does_not_win():
    """The issue's first trap, and the only reason this function exists."""
    decision = decide([measurement(MASKED, 30.0, expresses=False),
                       measurement(CROP, 90.0)])
    assert decision.primitive == CROP
    assert "did not express it" in decision.statement


def test_a_decision_always_states_what_the_winner_cannot_express():
    decision = decide([measurement(MASKED, 30.0)])
    assert decision.cannot_express == PRIMITIVES[MASKED].cannot_express
    assert "cannot express" in decision.statement


def test_the_eventual_case_is_reported_beside_the_decision_not_folded_into_it():
    """A primitive that wins v1 and cannot grow into the identity change is a
    decision someone takes knowingly."""
    decision = decide([
        measurement(MASKED, 30.0),
        measurement(MASKED, 32.0, expresses=False, priority=False,
                    case=IDENTITY_CASE, kind=IDENTITY),
    ])
    assert decision.primitive == MASKED
    assert "did not hold up" in decision.statement


def test_nothing_expressing_the_priority_case_is_recorded_rather_than_forced():
    decision = decide([measurement(MASKED, 30.0, expresses=False),
                       measurement(CROP, 90.0, expresses=False)])
    assert decision.primitive is None
    assert "C and D" in decision.statement


def test_a_decision_needs_a_measurement_on_the_priority_case():
    with pytest.raises(ValueError, match="priority case"):
        decide([measurement(MASKED, 30.0, priority=False, case=IDENTITY_CASE)])


# --- the fixed box track ----------------------------------------------------------------

def test_two_identical_boxes_overlap_completely():
    assert iou(Box(0, 0, 10, 10), Box(0, 0, 10, 10)) == 1.0


def test_boxes_that_do_not_touch_do_not_overlap():
    assert iou(Box(0, 0, 10, 10), Box(20, 20, 30, 30)) == 0.0


def test_the_overlap_is_intersection_over_union():
    assert iou(Box(0, 0, 10, 10), Box(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_a_jittering_detection_is_smoothed_towards_where_it_was():
    """Spec 8.5's box-smoothing lever, applied once so both primitives see it."""
    smoothed = smooth_track([[Box(0, 0, 100, 100)], [Box(10, 0, 110, 100)]],
                            smoothing=0.4)
    assert smoothed[0] == [Box(0, 0, 100, 100)]
    assert smoothed[1] == [Box(4, 0, 104, 100)], "40% of the way to the new box"


def test_a_detection_that_moved_too_far_starts_its_own_track():
    smoothed = smooth_track([[Box(0, 0, 10, 10)], [Box(500, 500, 510, 510)]])
    assert smoothed[1] == [Box(500, 500, 510, 510)], "not blended with a stranger"


def test_a_track_nobody_claimed_is_dropped_rather_than_carried():
    """A missing detection means no region to render, not a region from a stale box."""
    smoothed = smooth_track([[Box(0, 0, 10, 10)], [], [Box(0, 0, 10, 10)]])
    assert smoothed == [[Box(0, 0, 10, 10)], [], [Box(0, 0, 10, 10)]]


def test_two_objects_keep_their_own_tracks_and_their_detection_order():
    frames = [[Box(0, 0, 20, 20), Box(100, 100, 140, 140)],
              [Box(2, 0, 22, 20), Box(104, 100, 144, 140)]]
    smoothed = smooth_track(frames, smoothing=0.5)
    assert smoothed[1] == [Box(1, 0, 21, 20), Box(102, 100, 142, 140)]


def test_one_detection_cannot_claim_a_track_another_already_took():
    """Two overlapping detections of different objects must not collapse into one."""
    frames = [[Box(0, 0, 100, 100)],
              [Box(0, 0, 100, 100), Box(5, 5, 105, 105)]]
    smoothed = smooth_track(frames, smoothing=1.0)
    assert smoothed[1] == [Box(0, 0, 100, 100), Box(5, 5, 105, 105)]


def test_the_one_step_finding_only_fires_when_the_case_failed():
    requirement = required_denoise([point(20, 90.0, hits=0)], IDENTITY)
    assert one_step_finding(CASES[IDENTITY_CASE], requirement, achieved=True) is None

    finding = one_step_finding(CASES[IDENTITY_CASE], requirement, achieved=False)
    assert "did not perform the identity change" in finding
    assert "different TensorRT engine" in finding, "the implication, not just the fact"
    assert "t_index_list" in finding


# --- the committed tracks -------------------------------------------------------------

def rendered_regions(case):
    """Every region the case would render, from its committed track."""
    track = load_track(case.clip)
    return [clamp_box(region_box(box, case.region), track.width, track.height)
            for index in range(case.frames)
            for box in track.at(case.start_frame + index, limit=case.max_objects)]


def test_every_case_has_a_committed_track_covering_the_frames_it_renders():
    """Reproducibility: the boxes are read, not detected, so two runs match."""
    for case in CASES.values():
        track = load_track(case.clip)
        assert track.target == case.target
        covered = [index for index in range(case.start_frame,
                                            case.start_frame + case.frames)
                   if index in track.boxes]
        assert len(covered) == case.frames, case.name


def test_the_priority_case_renders_a_small_object():
    """The Gate: at least one source region under 96 px. It is the *smallest*
    subject and also the lowest-confidence one, which is why the object cap is six."""
    summary = small_object_summary(rendered_regions(CASES[RESTYLE_CASE]))
    assert summary.min_side_under > 0, summary.statement
    assert summary.min_side_px < SMALL_OBJECT_PX


def test_the_object_cap_is_what_keeps_the_small_subject_in():
    """A regression guard on the cap: at four objects the 45 px region drops out."""
    capped = CASES[RESTYLE_CASE].replace(max_objects=4)
    assert small_object_summary(rendered_regions(capped)).min_side_under == 0


def test_a_restyle_cannot_pass_the_criterion_on_its_own_resampling_blur():
    """The masked primitive squeezes the frame onto a 512x512 canvas and back, which
    changes the region before anything is diffused. The criterion is net of it."""
    blurry = [DenoisePoint(t_index=45, timestep=99, strength=0.32,
                           region_change=10.4, outside_change=0.0, frames=4,
                           resample_change=9.0),
              DenoisePoint(t_index=30, timestep=399, strength=0.76,
                           region_change=19.5, outside_change=0.0, frames=4,
                           resample_change=9.0)]
    requirement = required_denoise(blurry, RESTYLE)
    assert requirement.t_index == 30, "1.4/255 of actual change is not a restyle"
    assert "net of the resize control" in requirement.rule
    assert "the resize alone costs" in requirement.statement


def test_the_net_change_never_goes_negative():
    point = DenoisePoint(t_index=45, timestep=99, strength=0.32, region_change=2.0,
                         outside_change=0.0, frames=4, resample_change=9.0)
    assert point.net_region_change == 0.0
    assert point.to_dict()["net_region_change"] == 0.0
