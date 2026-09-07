"""The plan-swap record and its four checks (issue #30, bench half).

Acceptance criteria 1 and 3 (spec 11) are the two that were never measured, and
this is where their arithmetic is decided - not in the runner, so a threshold
cannot quietly move in a module the merge gate never executes.

Two of these tests are the issue's traps made executable. Criterion 1's clock
starts at the *keystroke*, so the GUI's debounce is part of the figure and a check
that reported the worker-side half alone would pass a swap the user waits half a
second longer for. And criterion 3 is judged against a steady-state control from
the same run, so a 40 ms frame on a card rendering 32 ms frames is not
automatically a stutter and a 34 ms one is not automatically fine.

GPU-free: `bench.plan_swap` reads JSON and formats it, like every other `bench.*`
results module.
"""

import dataclasses

import pytest

from bench.cooldown import REACHED, CooldownRecord
from bench.detector_results import LatencySummary
from bench.plan_swap import (
    CASES,
    repeat_spread,
    CRITERION_1_BUDGET_MS,
    GUI_DEBOUNCE_MS,
    RUNTIME_SWAP,
    STYLE_CASE,
    TARGET_CASE,
    VOCABULARY_SWAP,
    SwapResult,
    SwapRunMetrics,
    SwapTiming,
    first_pixel_frame,
    format_swap_report,
    intervals_of,
    latency_check,
    latest_per_swap,
    plan_swap_readme_row,
    rebuild_check,
    stutter_check,
    swap_kind,
    swap_timing,
)
from bench.primitive_results import ClipRecord
from bench.selective import background_check
from test_bench_results import a_fingerprint

DEPLOY_GPU = "NVIDIA GeForce RTX 4090"
LAPTOP_GPU = "NVIDIA GeForce RTX 3080 Laptop GPU"
BUDGET_MS = 1000.0 / 30.0


# --- the cases --------------------------------------------------------------


def test_the_before_plan_is_the_plan_the_worker_starts_on():
    """Both swaps begin from the committed baseline, so the frames before the swap
    are the steady state issues #24 and #23 measured."""
    from render_plan import priority_case_plan

    for case in CASES.values():
        before, _ = case.plans()
        assert before.to_dict() == priority_case_plan().to_dict()


def test_the_two_cases_are_the_two_kinds_of_swap_the_issue_names():
    """One that re-encodes the detector's vocabulary, one that only moves runtime
    state. They exercise different paths, which is why one case is not enough."""
    target_before, target_after = CASES[TARGET_CASE].plans()
    style_before, style_after = CASES[STYLE_CASE].plans()
    assert swap_kind(target_before, target_after) == VOCABULARY_SWAP
    assert swap_kind(style_before, style_after) == RUNTIME_SWAP


def test_the_cheap_swap_still_moves_the_style_and_the_strength():
    """Otherwise it would be a swap that changes nothing and stutters for free."""
    before, after = CASES[STYLE_CASE].plans()
    assert after.effective_prompt != before.effective_prompt
    assert after.effective_denoise != before.effective_denoise


def test_a_plan_the_gui_would_refuse_never_becomes_a_case():
    """Every case's two plans go through `plan_from_fields`, the shipped producer,
    so a case cannot describe an instruction a user could not type."""
    for case in CASES.values():
        for plan in case.plans():
            assert plan.plan_version >= 0


# --- when the pixels arrive -------------------------------------------------


def a_frame(index, plan_version=1, concepts=("person",), diffuses=True):
    return {"index": index, "plan_version": plan_version,
            "tracks_concepts": list(concepts), "diffuses": diffuses,
            "regions": 5 if diffuses else 0}


def test_the_first_frame_showing_a_new_target_is_the_one_that_rendered_its_boxes():
    """A frame that bound the new plan but had no tracks for it yet renders the
    capture untouched - which is the old instruction still on screen."""
    frames = [
        a_frame(0, plan_version=0),
        a_frame(1, plan_version=1, concepts=(), diffuses=False),
        a_frame(2, plan_version=1, concepts=("person",), diffuses=False),
        a_frame(3, plan_version=1, concepts=("shoes",), diffuses=True),
    ]
    assert first_pixel_frame(frames, plan_version=1, concepts=("shoes",)) == 3


def test_a_swap_that_changed_no_concept_shows_on_the_first_frame_that_diffuses():
    """The tracks in force already serve the new plan, so nothing waits for a
    detect - one rule for both kinds rather than a branch on the kind."""
    frames = [a_frame(0, plan_version=0), a_frame(1, plan_version=1)]
    assert first_pixel_frame(frames, plan_version=1, concepts=("person",)) == 1


def test_a_swap_whose_pixels_never_arrived_is_not_reported_as_instant():
    assert first_pixel_frame([a_frame(0, plan_version=0)], plan_version=1,
                             concepts=("person",)) is None


# --- criterion 1 ------------------------------------------------------------


def a_timing(**overrides) -> SwapTiming:
    fields = dict(
        swap_frame=48, plan_version_before=1, plan_version_after=2,
        kind=VOCABULARY_SWAP, validate_ms=1.5, debounce_ms=GUI_DEBOUNCE_MS,
        accepted_to_applied_ms=26.0, accepted_to_pixel_ms=210.0,
        frames_to_applied=1, frames_to_pixel=8, unrestyled_frames=7,
        detector_ticks_waited=1,
    )
    fields.update(overrides)
    return swap_timing(**fields)


def test_the_users_clock_includes_the_debounce_the_gui_waits_out():
    """The issue's first trap. Criterion 1 starts at the keystroke, and the 400 ms
    the GUI spends deciding the user stopped typing is part of the answer."""
    timing = a_timing()
    assert timing.keystroke_to_pixel_ms == pytest.approx(
        GUI_DEBOUNCE_MS + timing.validate_ms + timing.accepted_to_pixel_ms)
    assert timing.worker_ms == pytest.approx(
        timing.validate_ms + timing.accepted_to_pixel_ms)


def test_criterion_1_is_judged_on_the_keystroke_figure_and_reports_both():
    check = latency_check(a_timing())
    assert check.passed
    assert f"{GUI_DEBOUNCE_MS:.0f} ms" in check.statement
    assert "3.00 s" in check.statement


def test_a_swap_that_took_longer_than_the_criterion_allows_fails_it():
    check = latency_check(a_timing(accepted_to_pixel_ms=CRITERION_1_BUDGET_MS))
    assert not check.passed


def test_a_swap_whose_pixels_never_arrived_fails_criterion_1():
    """`None` is not a fast swap; it is a swap that did not happen."""
    check = latency_check(a_timing(accepted_to_pixel_ms=None, frames_to_pixel=None))
    assert not check.passed
    assert "never" in check.statement


# --- criterion 3 ------------------------------------------------------------


def a_series(steady=24.0, swap=None, frames=60):
    """A run's interval series: `None` for the first frame, steady state after."""
    intervals = [None] + [steady] * (frames - 1)
    if swap is not None:
        intervals[48] = swap
    return intervals


def test_a_swap_no_worse_than_the_steady_state_it_interrupted_is_no_stutter():
    check = stutter_check(a_series(swap=25.0), swap_index=48, pixel_index=50,
                          budget_ms=BUDGET_MS)
    assert check.passed
    assert check.swap.worst_ms == 25.0
    assert check.control.worst_ms == 24.0


def test_a_frame_over_the_budget_is_not_a_stutter_when_steady_state_is_too():
    """The issue's third trap: 33.33 ms alone cannot tell you whether a swap cost
    anything, and a card whose steady state is already over budget would report
    every swap as a stutter."""
    check = stutter_check(a_series(steady=40.0, swap=41.0), swap_index=48,
                          pixel_index=50, budget_ms=BUDGET_MS)
    assert check.passed
    assert check.swap.over_budget == 3
    assert check.control.over_budget > 0


def test_the_issues_own_example_is_not_a_stutter():
    """"A swap that costs one 40 ms frame on a card rendering 32 ms frames has not
    stuttered" - the trap, pinned. Which is why the bar is an allowance in
    milliseconds and not a percentage of steady state."""
    check = stutter_check(a_series(steady=32.0, swap=40.0), swap_index=48,
                          pixel_index=50, budget_ms=BUDGET_MS)
    assert check.passed
    assert check.excess_ms == pytest.approx(8.0)


def test_a_swap_that_cost_the_stream_a_whole_frame_is_a_stutter():
    """Past one frame budget of extra work the output has lost a frame to the
    swap, which is the smallest thing a viewer could call a stutter."""
    check = stutter_check(a_series(swap=90.0), swap_index=48, pixel_index=50,
                          budget_ms=BUDGET_MS)
    assert not check.passed
    assert check.excess_ms == pytest.approx(66.0)
    assert check.worst_ratio == pytest.approx(90.0 / 24.0)


def test_the_control_is_the_worse_of_the_two_steady_states_around_the_swap():
    """Both are from the same run, and the bar has to be the one a stutter would
    have to beat - not whichever half flatters the verdict."""
    intervals = a_series(steady=24.0)
    intervals[55] = 31.0  # the settled state after the swap is the dearer one
    check = stutter_check(intervals, swap_index=48, pixel_index=50,
                          budget_ms=BUDGET_MS)
    assert check.control.worst_ms == 31.0
    assert check.after.worst_ms == 31.0 and check.before.worst_ms == 24.0


def test_the_window_across_the_swap_runs_from_the_swap_to_the_pixels():
    check = stutter_check(a_series(), swap_index=48, pixel_index=52,
                          budget_ms=BUDGET_MS)
    assert check.swap.frames == 5


def test_a_run_with_no_steady_state_to_compare_against_does_not_pass_by_vacuum():
    check = stutter_check([None, 24.0, 24.0], swap_index=1, pixel_index=2,
                          budget_ms=BUDGET_MS)
    assert not check.passed
    assert "no steady state" in check.statement


def test_the_interval_series_is_read_off_the_frame_completion_times():
    """One reading of what an inter-frame interval is, so the runner cannot invent
    a second one."""
    assert intervals_of([1.0, 1.024, 1.050]) == [None, 24.0, 26.0]


# --- no rebuild -------------------------------------------------------------


def test_no_rebuild_is_a_checked_number_rather_than_an_assumption():
    check = rebuild_check(engine_id_before="StreamDiffusionWrapper@1",
                          engine_id_after="StreamDiffusionWrapper@1",
                          unet_id_before="UNet2DConditionModel@2",
                          unet_id_after="UNet2DConditionModel@2",
                          t_index_before=[40], t_index_after=[33])
    assert check.passed
    assert check.engine_rebuilds == 0
    assert check.steps_before == check.steps_after == 1
    assert "40" in check.statement and "33" in check.statement
    assert "step count" in check.statement


def test_a_step_count_that_moved_is_a_rebuild_and_is_reported_as_one():
    """The step *count* is what rebuilds. A plan change moves schedule values only,
    and a run where it moved the count measured a different thing."""
    check = rebuild_check(engine_id_before="w@1", engine_id_after="w@1",
                          unet_id_before="u@2", unet_id_after="u@2",
                          t_index_before=[40], t_index_after=[30, 40])
    assert not check.passed
    assert check.engine_rebuilds == 1


def test_a_replaced_engine_object_is_a_rebuild_however_the_schedule_looks():
    check = rebuild_check(engine_id_before="w@1", engine_id_after="w@9",
                          unet_id_before="u@2", unet_id_after="u@8",
                          t_index_before=[40], t_index_after=[40])
    assert not check.passed


# --- the record -------------------------------------------------------------


def a_swap_result(**overrides) -> SwapResult:
    """A whole record, out of the record's own dataclasses.

    Not a dict spelt by hand: the README row and the report read these field
    names, and a hand-written fixture would keep passing after one was renamed.
    """
    case = CASES[TARGET_CASE]
    before, after = case.plans()
    intervals = a_series(swap=25.0)
    fields = dict(
        case=case,
        plan_before=_plan_record(before), plan_after=_plan_record(after),
        clip=ClipRecord(name="people.mp4", sha256="abc", width=1280, height=720,
                        fps=25.0, total_frames=377, start_frame=0, frames_used=60),
        run=SwapRunMetrics(
            started_utc="2026-09-07T20:00:00Z", finished_utc="2026-09-07T20:00:10Z",
            warmup_frames=3, engine_scenario="img2img-tensorrt-512x512-b1",
            detector="yolo-world-s-640", frames=60, diffusion_calls=52,
            render=LatencySummary.from_samples([24.0, 25.0]),
            detect=LatencySummary.from_samples([22.0, 23.0]),
            detect_every_n=3, detector_ticks=20,
            regions_per_frame_before=5.04, regions_per_frame_after=1.3,
            ms_per_frame=24.0, mean_sm_clock_mhz=2700.0, max_temperature_c=48.0,
            peak_vram_bytes=6 * 1024 ** 3),
        timing=a_timing(),
        intervals_ms=intervals,
        latency=latency_check(a_timing()),
        stutter=stutter_check(intervals, swap_index=48, pixel_index=50,
                              budget_ms=BUDGET_MS),
        rebuild=rebuild_check("w@1", "w@1", "u@2", "u@2", [40], [33]),
        background=background_check([0] * 60, 200_000),
        cooldown=CooldownRecord(enabled=True, outcome=REACHED, threshold_c=62.0,
                                cap_s=120.0, waited_s=3.0,
                                final_temperature_c=45.0, samples=[[0.0, 46.0]]),
        hardware=a_fingerprint(gpu_name=DEPLOY_GPU),
        comparison_clip="swap-target-x-comparison.mp4",
        comparison_still="swap-target-x-comparison.jpg",
    )
    fields.update(overrides)
    return SwapResult(**fields)


def _plan_record(plan) -> dict:
    from bench.selective import plan_record

    return plan_record(plan)


@pytest.fixture
def record() -> dict:
    return a_swap_result().to_dict()


def test_a_record_carries_a_machine_and_a_clock_regime(record):
    from bench.results import require_recordable

    require_recordable(record)


def test_the_record_carries_the_whole_interval_series_across_the_swap(record):
    """Step 2 of the issue: the series, not a summary of it - a reader has to be
    able to re-derive the verdict."""
    assert record["intervals_ms"][48] == 25.0
    assert len(record["intervals_ms"]) == record["run"]["frames"]


def test_the_gate_is_every_check_a_machine_can_answer(record):
    assert record["gate"]["passed"]
    assert set(record["gate"]) == {"passed", "criterion_1", "criterion_3",
                                   "rebuild", "background"}


def test_one_failed_check_fails_the_gate():
    result = a_swap_result(background=background_check([0] * 59 + [3], 200_000))
    assert not result.gate_passed


def test_the_readme_row_names_the_swap_the_machine_and_both_verdicts(record):
    row = plan_swap_readme_row(record, "swap-target-x.json")
    assert DEPLOY_GPU in row
    assert "swap-target" in row
    assert row.count("|") == 17


# --- the report -------------------------------------------------------------


def a_pair(**overrides):
    """One committed run of each swap kind, which is what the block reports."""
    style = CASES[STYLE_CASE]
    before, after = style.plans()
    return {
        "swap-target-1.json": a_swap_result(**overrides).to_dict(),
        "swap-style-1.json": a_swap_result(
            case=style, plan_before=_plan_record(before),
            plan_after=_plan_record(after),
            timing=a_timing(kind=RUNTIME_SWAP, accepted_to_pixel_ms=25.0,
                            frames_to_pixel=1, unrestyled_frames=0,
                            detector_ticks_waited=0),
            latency=latency_check(a_timing(kind=RUNTIME_SWAP,
                                           accepted_to_pixel_ms=25.0,
                                           frames_to_pixel=1)),
            **overrides).to_dict(),
    }


def test_the_report_states_a_verdict_on_both_criteria():
    report = format_swap_report(a_pair())
    assert "criterion 1" in report.lower()
    assert "criterion 3" in report.lower()
    assert "MET" in report


def test_the_report_names_the_machine_and_the_clock_regime():
    report = format_swap_report(a_pair())
    assert DEPLOY_GPU in report
    assert "unlocked" in report


def test_the_report_says_no_rebuild_happened_on_either_swap():
    report = format_swap_report(a_pair())
    assert "0 TensorRT rebuild" in report or "no TensorRT rebuild" in report


def test_a_report_with_nothing_committed_says_so_rather_than_drawing_a_table():
    assert "no plan swap" in format_swap_report({})


def test_one_row_per_swap_per_machine():
    """The same reduction the other reports use: a second machine adds a row
    rather than deleting the first one's (issue #25)."""
    runs = {
        "a.json": a_swap_result(
            run=dataclasses.replace(a_swap_result().run,
                                    finished_utc="2026-09-07T20:00:00Z")).to_dict(),
        "b.json": a_swap_result(
            run=dataclasses.replace(a_swap_result().run,
                                    finished_utc="2026-09-07T21:00:00Z")).to_dict(),
        "c.json": a_swap_result(hardware=a_fingerprint(gpu_name=LAPTOP_GPU)).to_dict(),
    }
    assert set(latest_per_swap(runs)) == {"b.json", "c.json"}


def test_the_gpu_column_appears_only_once_the_rows_span_two_machines():
    one = format_swap_report(a_pair())
    assert "| GPU |" not in one
    two = dict(a_pair())
    two["swap-target-laptop.json"] = a_swap_result(
        hardware=a_fingerprint(gpu_name=LAPTOP_GPU)).to_dict()
    assert "| GPU |" in format_swap_report(two)


def test_the_debounce_the_record_carries_is_the_gui_s_own():
    """Spelt in two places, held to one value - `bench.plan_swap` cannot import
    `main_gpu_addon`, which primes the DLL search path and imports Tk. If the GUI's
    debounce moves, criterion 1's figure moves with it."""
    from sourceloader import load_symbols

    gui = load_symbols("main_gpu_addon.py", ["PLAN_DEBOUNCE_MS"])
    assert gui["PLAN_DEBOUNCE_MS"] == GUI_DEBOUNCE_MS


def test_the_report_says_what_a_swap_cost_when_it_cost_no_milliseconds():
    """The issue's fourth trap: a vocabulary swap can be free on the frame path
    and cost the output several frames of unstyled capture."""
    report = format_swap_report(a_pair())
    assert "showed the capture untouched for 7 of the 8 frames" in report


# --- is the rule deciding, or is the noise? ---------------------------------


def test_two_runs_of_one_swap_are_what_makes_a_verdict_a_verdict():
    """Issue #24 asked this of its 1 ms margin and #23 of its 3.33 ms one. Here
    the margins are seconds and tens of milliseconds, and the repeat is what says
    so rather than a reader assuming it."""
    runs = list(a_pair().values()) + list(a_pair().values())
    spread = repeat_spread(runs)
    assert spread.repeats == 2
    assert spread.decisive, spread.statement
    assert "twice" in spread.statement


def test_one_run_per_swap_says_there_is_no_spread_rather_than_none():
    spread = repeat_spread(list(a_pair().values()))
    assert spread.repeats == 0
    assert not spread.decisive
    assert "once" in spread.statement


def test_a_verdict_inside_the_run_to_run_spread_is_labelled_one():
    slow = a_swap_result(timing=a_timing(accepted_to_pixel_ms=2500.0))
    runs = [a_swap_result().to_dict(), slow.to_dict()]
    spread = repeat_spread(runs)
    assert not spread.decisive
    assert "inside" in spread.statement


def test_the_report_says_how_repeatable_the_two_verdicts_are():
    runs = dict(a_pair())
    runs["swap-target-2.json"] = a_swap_result().to_dict()
    assert "measured twice" in format_swap_report(runs)
