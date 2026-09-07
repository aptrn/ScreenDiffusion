"""What survives the move from the dev laptop to deploy hardware (issue #24).

Spec 7.4 says which conclusions are portable and which are not, and until now it
said it from first principles: every measurement in the repo came from one RTX 3080
laptop. This module is that table computed from two committed runs instead - one per
GPU - so "the selection carried, the milliseconds did not" is arithmetic a reviewer
can recompute rather than prose they have to trust.

GPU-free, like every other `bench.*` results module: it reads JSON and formats it.

The one worth reading twice is `comparability`. Region count drives cost, so two
selective runs at different regions/frame are not a hardware comparison at all -
the issue's third trap. It is a gate on the report, not a footnote inside it.
"""

import pytest

from bench.portability import (
    DEPLOY_GPU_MARKERS,
    FRAME_BUDGET_MS,
    REGIONS_TOLERANCE,
    TARGET_FPS,
    comparability,
    criterion_verdict,
    format_portability_report,
    fps_spread,
    is_deploy_gpu,
    latest_per_gpu,
    portability_rows,
    split_by_role,
)
from test_bench_results import a_fingerprint

LAPTOP = "NVIDIA GeForce RTX 3080 Laptop GPU"
DEPLOY = "NVIDIA GeForce RTX 4090"


def a_selective_result(gpu_name=LAPTOP, finished="2026-09-06T20:25:15Z",
                       ms_per_frame=55.06, with_detection=74.73,
                       regions_per_frame=5.04, detect_ms=59.03,
                       composite_ms=4.03, flicker=1.49, peak_vram_bytes=3923605504,
                       background_passed=True, coverage_bound=3, coverage_worst=2,
                       clock_state="unlocked"):
    """One selective record, cut down to the fields the portability report reads."""
    fingerprint = a_fingerprint().to_dict()
    fingerprint["gpu_name"] = gpu_name
    fingerprint["clock_lock"] = dict(fingerprint["clock_lock"], state=clock_state)
    return {
        "schema_version": 2,
        "kind": "selective",
        "case": {"name": "selective-people", "canvas": 512, "frames": 48},
        "clip": {"name": "people.mp4", "width": 1280, "height": 720,
                 "frames_used": 48},
        "plan": {"concept": "person", "region": "lower_half", "t_index": 40,
                 "detect_every_n": 3, "max_instances": 6},
        "run": {
            "finished_utc": finished, "frames": 48, "diffusion_calls": 48,
            "engine_scenario": "img2img-tensorrt-512x512-b1",
            "ms_per_frame": ms_per_frame,
            "ms_per_frame_with_detection": with_detection,
            "fps": round(1000.0 / with_detection, 4),
            "detect": {"mean_ms": detect_ms},
            "composite": {"mean_ms": composite_ms},
            "peak_vram_bytes": peak_vram_bytes,
            "mean_sm_clock_mhz": 1667.5,
        },
        "regions": {"regions_per_frame": regions_per_frame, "slots": 6},
        "flicker": {"mean_abs_diff": flicker},
        "gate": {
            "passed": background_passed,
            "background": {"passed": background_passed, "frames": 48,
                           "identical_frames": 48 if background_passed else 40,
                           "worst_pixels_changed": 0 if background_passed else 7},
            "change": {"passed": True, "net_change": 11.76, "threshold": 8.0},
            "coverage": {"passed": coverage_worst <= coverage_bound, "slots": 2,
                         "max_tracks": 6, "bound_frames": coverage_bound,
                         "worst_gap_frames": coverage_worst},
            "stall": {"passed": True, "frames_in": 48, "frames_out": 48,
                      "worst_offer_ms": 0.168},
        },
        "hardware": fingerprint,
        "clock_normalization": None,
    }


# --- which machine is which -------------------------------------------------


@pytest.mark.parametrize("name", [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4090 Laptop GPU",
    "NVIDIA GeForce RTX 3090 Ti",
    "nvidia geforce rtx 3090 ti",
])
def test_the_deploy_cards_are_recognised(name):
    assert is_deploy_gpu(name)


@pytest.mark.parametrize("name", [
    "NVIDIA GeForce RTX 3080 Laptop GPU",
    "NVIDIA GeForce RTX 3090",  # not a Ti: a different card, not this target
    "NVIDIA RTX A4000",
])
def test_a_card_that_is_not_a_deploy_target_is_not_one(name):
    assert not is_deploy_gpu(name)


def test_the_markers_are_the_two_cards_spec_7_4_names():
    assert DEPLOY_GPU_MARKERS == ("3090 ti", "4090")


def test_one_run_per_gpu_and_it_is_the_newest():
    results = {
        "old.json": a_selective_result(DEPLOY, "2026-09-07T09:00:00Z",
                                       ms_per_frame=30.0),
        "new.json": a_selective_result(DEPLOY, "2026-09-07T10:00:00Z",
                                       ms_per_frame=20.0),
        "laptop.json": a_selective_result(LAPTOP),
    }
    assert set(latest_per_gpu(results)) == {"new.json", "laptop.json"}


def test_the_two_roles_are_split_by_the_card_in_the_fingerprint():
    results = {"laptop.json": a_selective_result(LAPTOP),
               "deploy.json": a_selective_result(DEPLOY)}
    dev, deploy = split_by_role(results)
    assert [result["hardware"]["gpu_name"] for result in dev] == [LAPTOP]
    assert [result["hardware"]["gpu_name"] for result in deploy] == [DEPLOY]


# --- the third trap: same regions/frame, or no comparison -------------------


def test_two_runs_at_the_same_region_count_are_comparable():
    check = comparability(a_selective_result(LAPTOP, regions_per_frame=5.04),
                          a_selective_result(DEPLOY, regions_per_frame=5.04))
    assert check.comparable
    assert "5.04" in check.statement


def test_two_runs_at_different_region_counts_are_not():
    """Region count drives cost, so this pair says nothing about the hardware."""
    check = comparability(a_selective_result(LAPTOP, regions_per_frame=5.04),
                          a_selective_result(DEPLOY, regions_per_frame=2.10))
    assert not check.comparable
    assert "2.10" in check.statement and "5.04" in check.statement


def test_the_tolerance_is_a_fraction_of_the_larger_count():
    assert 0.0 < REGIONS_TOLERANCE < 0.5
    inside = comparability(
        a_selective_result(LAPTOP, regions_per_frame=5.0),
        a_selective_result(DEPLOY,
                           regions_per_frame=5.0 * (1 - REGIONS_TOLERANCE / 2)))
    assert inside.comparable


# --- acceptance criterion 2 -------------------------------------------------


def test_the_budget_is_the_frame_time_30_fps_allows():
    assert TARGET_FPS == 30.0
    assert FRAME_BUDGET_MS == pytest.approx(1000.0 / 30.0)


def test_a_run_inside_the_budget_meets_the_criterion():
    verdict = criterion_verdict(a_selective_result(DEPLOY, with_detection=25.0,
                                                   regions_per_frame=5.04))
    assert verdict.met
    assert verdict.regions_per_frame == 5.04
    assert "5.04" in verdict.statement and "30" in verdict.statement


def test_a_run_outside_the_budget_does_not():
    verdict = criterion_verdict(a_selective_result(LAPTOP, with_detection=74.73))
    assert not verdict.met
    assert verdict.budget_ms == pytest.approx(FRAME_BUDGET_MS)
    assert verdict.over_budget_x == pytest.approx(74.73 / FRAME_BUDGET_MS, rel=1e-3)


def test_the_verdict_is_judged_on_the_cost_with_detection_running():
    """The frame path alone is not what a frame costs: the detector runs beside it."""
    verdict = criterion_verdict(a_selective_result(DEPLOY, ms_per_frame=20.0,
                                                   with_detection=40.0))
    assert not verdict.met, "judged on 20 ms rather than the 40 a frame costs"


def test_the_verdict_names_the_region_count_it_was_measured_at():
    """The issue's Gate: a verdict with no region count beside it is not one."""
    assert "3.50" in criterion_verdict(
        a_selective_result(DEPLOY, regions_per_frame=3.5)).statement


def test_the_verdict_carries_the_clock_regime_it_was_measured_under():
    verdict = criterion_verdict(a_selective_result(DEPLOY, clock_state="unlocked"))
    assert verdict.clock_regime == "unlocked"


# --- and whether the verdict is inside the run-to-run spread ---------------


def deploy_runs(*with_detection):
    return {f"run-{index}.json": a_selective_result(
        DEPLOY, f"2026-09-07T1{index}:00:00Z", with_detection=cost)
        for index, cost in enumerate(with_detection)}


def test_a_single_run_is_its_own_spread():
    spread = fps_spread(deploy_runs(24.0), DEPLOY)
    assert spread.runs == 1 and spread.decisive
    assert spread.lowest_fps == spread.highest_fps


def test_runs_all_short_of_the_target_are_a_decisive_verdict():
    spread = fps_spread(deploy_runs(34.0, 34.5, 35.0), DEPLOY)
    assert spread.runs == 3 and spread.decisive
    assert "short of" in spread.statement


def test_runs_either_side_of_the_target_are_not():
    """A verdict 2% short and a verdict 2% over are a spread, not an answer."""
    spread = fps_spread(deploy_runs(32.0, 34.0), DEPLOY)
    assert not spread.decisive
    assert "spread" in spread.statement


def test_the_spread_reads_every_run_on_the_card_not_the_newest():
    results = dict(deploy_runs(34.0, 40.0), **{"dev.json": a_selective_result(LAPTOP)})
    assert fps_spread(results, DEPLOY).runs == 2


def test_a_verdict_inside_the_spread_says_so_in_the_report():
    results = dict(deploy_runs(32.0, 34.0),
                   **{"dev.json": a_selective_result(LAPTOP)})
    assert "not decisively" in format_portability_report(results)


# --- what carried, and what did not ----------------------------------------


def rows_matching(baseline, deploy, needle):
    return [row for row in portability_rows(baseline, deploy)
            if needle in row.conclusion]


def test_bit_identity_carries_when_both_runs_kept_the_background():
    rows = rows_matching(a_selective_result(LAPTOP), a_selective_result(DEPLOY),
                         "bit-identical")
    assert rows and rows[0].carried is True


def test_bit_identity_does_not_carry_when_the_new_card_broke_it():
    """The Gate's third item. A card that changes a background pixel is the whole
    reason this is measured rather than assumed."""
    rows = rows_matching(a_selective_result(LAPTOP),
                         a_selective_result(DEPLOY, background_passed=False),
                         "bit-identical")
    assert rows and rows[0].carried is False


def test_the_absolute_millisecond_figure_does_not_carry():
    rows = rows_matching(a_selective_result(LAPTOP, ms_per_frame=55.06),
                         a_selective_result(DEPLOY, ms_per_frame=18.0),
                         "ms/frame")
    assert rows and rows[0].carried is False
    assert "55.06" in rows[0].evidence and "18.00" in rows[0].evidence


def test_an_identical_millisecond_figure_would_carry():
    """The row is computed from the two numbers, not asserted from spec 7.4."""
    rows = rows_matching(a_selective_result(LAPTOP, ms_per_frame=55.06),
                         a_selective_result(DEPLOY, ms_per_frame=55.06),
                         "ms/frame")
    assert rows and rows[0].carried is True


def test_what_the_scheduler_chose_carries_because_it_is_not_hardware():
    rows = rows_matching(a_selective_result(LAPTOP), a_selective_result(DEPLOY),
                         "regions")
    assert rows and rows[0].carried is True


def test_every_row_carries_its_evidence():
    for row in portability_rows(a_selective_result(LAPTOP),
                                a_selective_result(DEPLOY, ms_per_frame=18.0)):
        assert row.evidence.strip(), row.conclusion


# --- the block spec 7.4 carries --------------------------------------------


def test_with_no_deploy_run_the_report_says_so_rather_than_inventing_one():
    assert "no deploy-hardware" in format_portability_report(
        {"laptop.json": a_selective_result(LAPTOP)})


def test_with_no_baseline_run_the_report_says_so():
    assert "no dev-hardware" in format_portability_report(
        {"deploy.json": a_selective_result(DEPLOY)})


def test_the_report_states_the_criterion_verdict_and_both_machines():
    report = format_portability_report({
        "laptop.json": a_selective_result(LAPTOP),
        "deploy.json": a_selective_result(DEPLOY, "2026-09-07T12:00:00Z",
                                          ms_per_frame=18.0, with_detection=24.0),
    })
    assert LAPTOP in report and DEPLOY in report
    assert "Acceptance criterion 2" in report
    assert "MET" in report
    assert "5.04" in report


def test_an_incomparable_pair_is_refused_in_the_report_itself():
    """Not a footnote: a table comparing 5 regions against 2 is a wrong answer."""
    report = format_portability_report({
        "laptop.json": a_selective_result(LAPTOP, regions_per_frame=5.04),
        "deploy.json": a_selective_result(DEPLOY, "2026-09-07T12:00:00Z",
                                          regions_per_frame=2.10),
    })
    assert "not comparable" in report
    assert "| measure |" not in report, "an incomparable pair still printed a table"
