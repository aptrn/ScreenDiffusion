"""The selective render record and its four Gate checks (issue #8, bench half).

GPU-free: `bench.selective` is the arithmetic and the record shape, and everything
the Gate asserts is decided here rather than in the runner - so a threshold cannot
quietly move in a module the merge gate never executes.

The coverage probe is the one worth reading twice. It replays a run's own per-frame
`Tracks` through a real `RegionScheduler` with K forced down, so what it measures is
the shipped rotation policy and not an imitation of it.
"""

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
    latest_per_case_and_gpu,
    plan_record,
    selective_readme_row,
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


@pytest.fixture
def record() -> dict:
    """A whole record, built out of the record's own dataclasses.

    Not a dict written out by hand: a fixture that spelt the shape itself would
    keep passing after a field was renamed, and the README row and the report are
    exactly what read those field names.
    """
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    plan = priority_case_plan()
    result = SelectiveResult(
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
    return result.to_dict()


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


def on_gpu(record, gpu_name, finished):
    """`record` as the same case measured on another card."""
    import copy

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
               "deploy.json": on_gpu(record, "NVIDIA GeForce RTX 4090",
                                     "2026-09-07T10:00:00Z")}
    assert set(latest_per_case_and_gpu(results)) == {"laptop.json", "deploy.json"}


def test_two_runs_on_one_machine_are_still_one_row(record):
    results = {"old.json": on_gpu(record, "NVIDIA GeForce RTX 4090",
                                  "2026-09-07T09:00:00Z"),
               "new.json": on_gpu(record, "NVIDIA GeForce RTX 4090",
                                  "2026-09-07T10:00:00Z")}
    assert set(latest_per_case_and_gpu(results)) == {"new.json"}


def test_the_report_names_the_gpu_in_every_row(record):
    report = format_selective_report({
        "laptop.json": record,
        "deploy.json": on_gpu(record, "NVIDIA GeForce RTX 4090",
                              "2026-09-07T10:00:00Z")})
    assert "| GPU |" in report
    assert report.count("| selective-people |") == 2
    assert "NVIDIA GeForce RTX 4090" in report


def test_the_gate_lines_belong_to_the_newest_run_and_say_which_machine(record):
    """A Gate line is one run's, so the block has to name whose it is."""
    report = format_selective_report({
        "laptop.json": record,
        "deploy.json": on_gpu(record, "NVIDIA GeForce RTX 4090",
                              "2026-09-07T10:00:00Z")})
    assert "The Gate, measured on NVIDIA GeForce RTX 4090:" in report
