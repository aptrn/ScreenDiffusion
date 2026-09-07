"""The selective render record and its four Gate checks (issue #8, bench half).

GPU-free: `bench.selective` is the arithmetic and the record shape, and everything
the Gate asserts is decided here rather than in the runner - so a threshold cannot
quietly move in a module the merge gate never executes.

The coverage probe is the one worth reading twice. It replays a run's own per-frame
`Tracks` through a real `RegionScheduler` with K forced down, so what it measures is
the shipped rotation policy and not an imitation of it.
"""

import copy

import numpy as np
import pytest

from bench.clocks import clock_normalization
from bench.cooldown import REACHED, CooldownRecord
from bench.detector_results import LatencySummary
from bench.flicker import flicker_score
from bench.primitive_results import ClipRecord
from bench.selective import (
    CASES,
    MAX_OFFER_MS,
    PRIORITY_CASE,
    SELECTIVE_README_HEADER,
    SELECTIVE_README_SEPARATOR,
    VISIBLE_CHANGE,
    background_check,
    change_check,
    coverage_check,
    format_selective_report,
    latest_per_case,
    plan_record,
    selective_readme_row,
    staleness_summary,
    RegionSummary,
    SelectiveResult,
    SelectiveRunMetrics,
    stall_check,
    with_slots,
)
from detection import Box, Track, Tracks
from render_plan import priority_case_plan
from test_bench_results import a_fingerprint

CANVAS = 512


def tracks_of(count, start_id=0):
    return Tracks(tracks=tuple(
        Track(track_id=start_id + index, box=Box(5 + index * 40, 100,
                                                 35 + index * 40, 300),
              concept="person", confidence=0.9)
        for index in range(count)), ticks=1)


# --- the case ---------------------------------------------------------------


def test_the_case_renders_the_plan_the_worker_starts_on():
    """Not a copy of it: a case that could describe a plan the app cannot be put
    into would measure something the app does not do."""
    plan = CASES[PRIORITY_CASE].plan()
    assert plan.to_dict() == priority_case_plan().to_dict()


def test_the_plan_record_says_what_was_rendered_and_at_what_strength():
    record = plan_record(priority_case_plan())
    assert record["concept"] == "person"
    assert record["region"] == "lower_half"
    assert record["t_index"] == 40
    assert record["prompt"]


# --- background -------------------------------------------------------------


def test_a_run_that_changed_no_background_pixel_passes():
    check = background_check([0] * 48, background_pixels=200_000)
    assert check.passed
    assert check.identical_frames == 48


def test_one_changed_pixel_on_one_frame_fails_the_whole_run():
    """The criterion is bit-identity, so an average would bury exactly the failure
    it exists to catch."""
    check = background_check([0] * 47 + [1], background_pixels=200_000)
    assert not check.passed
    assert check.worst_pixels_changed == 1
    assert "1 of 200000" in check.statement


def test_a_run_with_no_frames_does_not_pass_by_vacuum():
    assert not background_check([], background_pixels=0).passed


# --- the visible change -----------------------------------------------------


def test_a_change_above_the_threshold_net_of_the_control_passes():
    check = change_check(region_change=13.0, capture_change=0.0)
    assert check.passed
    assert check.net_change == 13.0


def test_the_capture_round_trip_is_subtracted_before_the_verdict():
    check = change_check(region_change=VISIBLE_CHANGE + 1.0,
                         capture_change=2.0)
    assert not check.passed, "the control was not subtracted"
    assert check.net_change == pytest.approx(VISIBLE_CHANGE - 1.0)


def test_a_negative_net_change_is_zero_rather_than_negative():
    assert change_check(region_change=1.0, capture_change=3.0).net_change == 0.0


# --- the round robin --------------------------------------------------------


def test_the_probe_forces_the_slot_count_and_changes_nothing_else():
    plan = priority_case_plan()
    probed = with_slots(plan, 2)
    assert probed.targets[0].max_instances == 2
    assert probed.targets[0].concept == plan.targets[0].concept
    assert probed.effective_prompt == plan.effective_prompt


def test_every_track_is_rendered_inside_the_bound():
    """Five tracks over two slots: the bound is three frames and nobody waits
    longer than that."""
    snapshots = [tracks_of(5)] * 12
    check = coverage_check(snapshots, priority_case_plan(), CANVAS, CANVAS, slots=2)
    assert check.max_tracks == 5
    assert check.bound_frames == 3
    assert check.worst_gap_frames <= 3
    assert check.passed


def test_a_track_that_arrives_late_is_not_counted_as_starved_from_the_start():
    snapshots = [tracks_of(2)] * 4 + [tracks_of(4)] * 6
    check = coverage_check(snapshots, priority_case_plan(), CANVAS, CANVAS, slots=2)
    assert check.passed, check.statement


def test_a_run_with_no_track_has_nothing_to_starve():
    check = coverage_check([Tracks()] * 5, priority_case_plan(), CANVAS, CANVAS,
                           slots=2)
    assert check.max_tracks == 0
    assert check.passed


# --- stalling ---------------------------------------------------------------


def test_a_loop_that_produced_every_frame_and_never_waited_passes():
    check = stall_check(frames_in=48, frames_out=48, worst_offer_ms=0.03,
                        passthrough_frames=2)
    assert check.passed


def test_a_dropped_output_frame_fails():
    assert not stall_check(48, 47, 0.03, 0).passed


def test_an_offer_that_blocked_the_frame_path_fails():
    """The whole point of the detector's thread: an offer returns, it does not
    wait for a detect."""
    assert not stall_check(48, 48, MAX_OFFER_MS + 1.0, 0).passed


# --- the record and the report ----------------------------------------------


def a_selective_result(**overrides) -> SelectiveResult:
    """A whole record, built out of the record's own dataclasses.

    Not a dict written out by hand: a fixture that spelt the shape itself would
    keep passing after a field was renamed, and the README row and the report are
    exactly what read those field names.

    A function as well as a fixture because the cross-machine report tests
    (issue #25) need two of these under two different fingerprints.
    """
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    plan = priority_case_plan()
    fields = dict(
        case=CASES[PRIORITY_CASE],
        plan=plan_record(plan),
        clip=ClipRecord(name="people.mp4", sha256="abc", width=1280, height=720,
                        fps=25.0, total_frames=378, start_frame=0, frames_used=48),
        run=SelectiveRunMetrics(
            started_utc="2026-09-06T20:00:00Z", finished_utc="2026-09-06T20:00:10Z",
            warmup_frames=3, engine_scenario="img2img-tensorrt-512x512-b1",
            detector="yolo-world-s-640", frames=48, diffusion_calls=46,
            render=LatencySummary.from_samples([88.0, 90.0, 86.0]),
            composite=LatencySummary.from_samples([0.4, 0.5]),
            detect=LatencySummary.from_samples([18.0, 19.0]),
            detect_every_n=3, detector_ticks=16, amortised_detect_ms=6.17,
            ms_per_frame=88.0, ms_per_frame_with_detection=94.17, fps=10.6,
            mean_sm_clock_mhz=1575.0, max_temperature_c=68.0,
            peak_vram_bytes=6 * 1024 ** 3),
        regions=RegionSummary(
            slots=6, regions_rendered=250, regions_per_frame=5.2,
            tracks_per_frame=5.2, deferred_total=0, skipped_small_total=0,
            min_region_px=16, feather_px=6, min_side_px=22, max_side_px=180),
        flicker=flicker_score(frames, frames),
        background=background_check([0] * 48, 200_000),
        change=change_check(13.0, 0.0),
        coverage=coverage_check([tracks_of(5)] * 6, plan, CANVAS, CANVAS, 2),
        stall=stall_check(48, 48, 0.02, 2),
        staleness=staleness_summary(
            [tracks_of(5)] * 6, detect_every_n=3, ms_per_frame=88.0),
        cooldown=CooldownRecord(enabled=True, outcome=REACHED, threshold_c=62.0,
                                cap_s=120.0, waited_s=12.0,
                                final_temperature_c=60.0, samples=[[0.0, 70.0]]),
        hardware=a_fingerprint(),
        clock_normalization=clock_normalization(
            a_fingerprint().clock_lock, [[0.0, 1575.0, 60.0]],
            raw_ms_per_frame=88.0),
        comparison_clip="selective-people-x-comparison.mp4",
        comparison_still="selective-people-x-comparison.jpg",
    )
    fields.update(overrides)
    return SelectiveResult(**fields)


@pytest.fixture
def record() -> dict:
    return a_selective_result().to_dict()


def test_a_record_carries_a_machine_and_a_clock_regime(record):
    """Both door rules, on the shape this module actually writes."""
    from bench.results import require_recordable

    require_recordable(record)


def test_the_gate_passes_only_when_every_check_does(record):
    assert record["gate"]["passed"] is True


def test_the_readme_row_fits_the_header_it_is_appended_under(record):
    """A row is appended under whatever header the file already has, so the two
    have to be written together."""
    row = selective_readme_row(record, "selective-people-x.json")
    assert row.count("|") == SELECTIVE_README_HEADER.count("|")
    assert row.count("|") == SELECTIVE_README_SEPARATOR.count("|")


def test_the_row_names_the_background_verdict_in_words(record):
    assert "identical" in selective_readme_row(record, "f.json")


def test_the_report_carries_the_table_and_every_gate_line(record):
    report = format_selective_report({"f.json": record})
    assert PRIORITY_CASE in report
    for phrase in ("Non-target pixels bit-identical", "visibly restyled",
                   "starved", "never stalls"):
        assert phrase in report
    assert "Manual verification artefact" in report


def test_an_empty_results_directory_reports_that_rather_than_a_blank_table():
    assert format_selective_report({}) == "no selective render run committed yet"


# --- one row per machine (issue #24) ----------------------------------------


DEPLOY_GPU = "NVIDIA GeForce RTX 4090"


def on_gpu(record, gpu_name, finished):
    """`record` as the same case measured on another card."""
    other = copy.deepcopy(record)
    other["hardware"]["gpu_name"] = gpu_name
    other["run"]["finished_utc"] = finished
    return other


def test_a_second_machine_adds_a_row_rather_than_replacing_one(record):
    """The same case on two GPUs is two answers, not a newer version of one.

    Keyed by case alone, the deploy run (issue #24) would silently delete the
    laptop baseline from the spec the moment it was committed.
    """
    results = {"laptop.json": record,
               "deploy.json": on_gpu(record, DEPLOY_GPU, "2026-09-07T10:00:00Z")}
    assert set(latest_per_case(results)) == {"laptop.json", "deploy.json"}


def test_two_runs_on_one_machine_are_still_one_row(record):
    results = {"old.json": on_gpu(record, DEPLOY_GPU, "2026-09-07T09:00:00Z"),
               "new.json": on_gpu(record, DEPLOY_GPU, "2026-09-07T10:00:00Z")}
    assert set(latest_per_case(results)) == {"new.json"}


def test_the_report_names_the_gpu_in_every_row(record):
    report = format_selective_report({
        "laptop.json": record,
        "deploy.json": on_gpu(record, DEPLOY_GPU, "2026-09-07T10:00:00Z")})
    assert "| GPU |" in report
    assert report.count("| selective-people |") == 2
    assert DEPLOY_GPU in report


def test_the_gate_lines_belong_to_the_newest_run_and_say_which_machine(record):
    """A Gate line is one run's, so the block has to name whose it is."""
    report = format_selective_report({
        "laptop.json": record,
        "deploy.json": on_gpu(record, DEPLOY_GPU, "2026-09-07T10:00:00Z")})
    assert f"The Gate, measured on {DEPLOY_GPU}:" in report


def test_each_machine_states_what_its_cadence_cost_in_box_age(record):
    """Issue #33's Gate: the cadence's price is box age, and it is quoted per
    machine because the frames are portable and the milliseconds are not."""
    report = format_selective_report({
        "laptop.json": record,
        "deploy.json": on_gpu(record, DEPLOY_GPU, "2026-09-07T10:00:00Z")})
    assert report.count("What the cadence cost") == 2
    assert "frames old" in report


def test_a_run_predating_the_staleness_block_says_so_rather_than_nothing(record):
    """The 3080 baseline was written before issue #23 added the measurement. An
    absent block is a record that is older than the question, not a zero."""
    older = copy.deepcopy(record)
    older["staleness"] = None
    assert "not recorded" in format_selective_report({"laptop.json": older})


# --- how stale the boxes a frame renders are (issue #23) --------------------


def snapshot(boxes, frame_index, tick):
    """One detector tick as the frame loop reads it: `boxes` is {track_id: Box}."""
    return Tracks(
        tracks=tuple(Track(track_id=track_id, box=box, concept="person",
                           confidence=0.9)
                     for track_id, box in boxes.items()),
        frame_index=frame_index, ticks=tick)


def test_the_age_of_the_boxes_a_frame_renders_is_what_a_cadence_costs():
    """One detect on frame 0, read by six frames: the sixth renders boxes five
    frames old, and that is the whole of what raising `detect_every_n` buys."""
    boxes = {1: Box(0, 0, 40, 40)}
    summary = staleness_summary([snapshot(boxes, 0, 1)] * 6, detect_every_n=6,
                                ms_per_frame=20.0)
    assert summary.mean_age_frames == pytest.approx(2.5)
    assert summary.worst_age_frames == 5
    assert summary.mean_age_ms == pytest.approx(50.0)


def test_a_frame_rendered_before_the_first_detect_is_counted_not_aged():
    """`EMPTY_TRACKS` has no frame to be old relative to, and averaging it in as
    age zero would report the staleness of a frame that had no boxes at all."""
    snapshots = [Tracks(), Tracks(), snapshot({1: Box(0, 0, 8, 8)}, 2, 1)]
    summary = staleness_summary(snapshots, detect_every_n=2)
    assert summary.frames_without_tracks == 2
    assert summary.mean_age_frames == 0.0
    assert summary.mean_age_ms is None, "no ms/frame was given to convert with"


def test_how_far_an_object_moves_between_two_refreshes_is_measured():
    """The other half of staleness: not how old the box is, but how wrong. A
    40 px box that moved 40 px shares no pixels with where it turns out to be."""
    first = snapshot({1: Box(0, 0, 40, 40)}, 0, 1)
    second = snapshot({1: Box(40, 0, 80, 40)}, 6, 2)
    summary = staleness_summary([first] * 6 + [second] * 6, detect_every_n=6)
    assert summary.refreshes == 1
    assert summary.mean_refresh_iou == 0.0
    assert summary.mean_refresh_shift_px == pytest.approx(40.0)


def test_a_box_that_did_not_move_between_refreshes_is_perfectly_fresh():
    boxes = {1: Box(10, 10, 50, 50)}
    summary = staleness_summary([snapshot(boxes, 0, 1), snapshot(boxes, 3, 2)],
                                detect_every_n=3)
    assert summary.mean_refresh_iou == 1.0
    assert summary.worst_refresh_shift_px == 0.0


def test_an_identity_the_tracker_lost_shows_up_as_a_second_id():
    """Spec 8.5 pins seeds to track ids, so an object that comes back under a new
    id has paid the cadence in identity rather than in milliseconds."""
    summary = staleness_summary(
        [snapshot({1: Box(0, 0, 40, 40)}, 0, 1),
         snapshot({2: Box(300, 0, 340, 40)}, 8, 2)], detect_every_n=8)
    assert summary.distinct_track_ids == 2
    assert summary.max_concurrent_tracks == 1
    assert summary.refreshes == 0, "no track survived the refresh to be compared"


def test_a_run_with_no_detector_says_so_rather_than_reporting_zero_staleness():
    summary = staleness_summary([Tracks()] * 4, detect_every_n=3)
    assert summary.ticks == 0
    assert "no detect" in summary.statement


def test_the_staleness_summary_names_the_cadence_it_belongs_to():
    summary = staleness_summary([snapshot({1: Box(0, 0, 40, 40)}, 0, 1)] * 3,
                                detect_every_n=5, ms_per_frame=25.0)
    assert summary.detect_every_n == 5
    assert "5" in summary.statement and "frames old" in summary.statement


def test_the_record_carries_the_staleness_the_cadence_bought(record):
    """Issue #23's Gate asks for staleness per setting, so it is a field of the
    record rather than something a sweep recomputes from clips it did not keep."""
    assert record["staleness"]["detect_every_n"] == 3
    assert record["staleness"]["statement"]


def test_a_record_written_before_the_field_existed_still_reads():
    """The 3080 and 4090 baselines predate issue #23 and are not retro-edited."""
    older = a_selective_result(staleness=None).to_dict()
    assert older["staleness"] is None


# --- the cadence a case is measured at (issue #23) ---------------------------


def test_a_case_renders_the_plan_at_the_cadence_it_was_asked_for():
    """The sweep changes one plan field and nothing else, and it changes it
    through the validator rather than by reaching into a frozen dataclass."""
    case = CASES[PRIORITY_CASE].replace(detect_every_n=8)
    plan = case.plan()
    assert plan.settings.detect_every_n == 8
    assert plan.effective_prompt == priority_case_plan().effective_prompt
    assert plan.targets[0].to_dict() == priority_case_plan().targets[0].to_dict()


def test_a_case_with_no_cadence_override_is_the_shipped_plan_untouched():
    assert CASES[PRIORITY_CASE].detect_every_n is None
    assert CASES[PRIORITY_CASE].plan().to_dict() == priority_case_plan().to_dict()


def test_a_cadence_the_validator_clamps_is_the_clamped_one_that_is_recorded():
    """`detect_every_n` is validated into 1..30. A sweep that asked for 99 must
    record what the plan actually ran at, not what it typed."""
    plan = CASES[PRIORITY_CASE].replace(detect_every_n=99).plan()
    assert plan.settings.detect_every_n == 30


def test_a_record_says_where_its_blend_ran():
    """A composite measured on the host and one measured on the device are two
    designs as much as two numbers, and spec 7.4 compares them across machines
    (issue #31). Records written before it ran on the host, and say so."""
    from bench.selective import HOST_COMPOSITE

    assert a_selective_result().to_dict()["run"]["composite_path"] == HOST_COMPOSITE


def test_the_two_places_a_blend_can_run_are_the_shipped_module_s_own_names():
    """`bench.selective` spells them itself, because the shipped modules are
    imported inside its functions. One pair of values, or a record would name a
    path nothing reads back."""
    import device_compositor
    from bench.selective import DEVICE_COMPOSITE, HOST_COMPOSITE

    assert (HOST_COMPOSITE, DEVICE_COMPOSITE) == (device_compositor.HOST,
                                                  device_compositor.DEVICE)


def test_the_wire_key_the_cadence_override_writes_is_the_plan_s_own():
    """`bench.selective` spells `global` itself rather than importing it, because
    the shipped modules are imported inside its functions. One value, or a cadence
    override would land in a key the validator drops."""
    import render_plan
    from bench.selective import GLOBAL_KEY

    assert GLOBAL_KEY == render_plan.GLOBAL_KEY


# --- was anything else on the card while this ran? (issue #33) ---------------


def test_a_run_records_whether_the_card_was_its_own():
    """The door issue #33's discarded run walked through. Every other field said the
    run was sound; this is the one that would not have."""
    from bench.contention import CLEAR, OccupancyRecord

    occupied = a_selective_result(occupancy=OccupancyRecord(
        outcome=CLEAR, mean_utilization_pct=1.0, samples=[1.0])).to_dict()
    assert occupied["occupancy"]["outcome"] == CLEAR
    assert occupied["occupancy"]["clear"] is True


def test_a_record_predating_the_occupancy_gate_carries_none_not_a_pass(record):
    """The fourteen committed records were written before the gate existed. An
    absent block is a run that never answered the question, not one that passed."""
    assert record["occupancy"] is None


def test_a_run_measured_beside_something_else_says_so_where_it_is_quoted():
    """A door that only fires at run time is not a door: `--require-idle-gpu` is
    opt-in, so a contended run can still reach `bench/results/`. The block that
    quotes it has to say what it was measured beside."""
    from bench.contention import BUSY, OccupancyRecord

    busy = a_selective_result(occupancy=OccupancyRecord(
        outcome=BUSY, mean_utilization_pct=45.0, samples=[45.0])).to_dict()
    report = format_selective_report({"busy.json": busy})
    assert "45%" in report
    assert "busy" in report


def test_a_clear_run_adds_no_line_at_all(record):
    """The rule `GpuColumn` follows: a caveat that is always there is not read. A
    clean run renders exactly the block it always did, so no byte-match churns."""
    from bench.contention import CLEAR, OccupancyRecord

    clear = a_selective_result(occupancy=OccupancyRecord(
        outcome=CLEAR, mean_utilization_pct=1.0, samples=[1.0])).to_dict()
    assert (format_selective_report({"a.json": clear})
            == format_selective_report({"a.json": record}))
