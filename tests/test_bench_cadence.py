"""The `detect_every_n` sweep: what raising the cadence buys, and what it spends.

Issue #23 step 3. The sweep is the one lever on the frame budget that costs no
image quality - it trades *freshness* - so the report has to put the milliseconds
it buys and the staleness it spends in the same table, and the recommendation has
to be arithmetic over those two rather than a preference.

GPU-free: `bench.cadence` reads committed JSON and formats it, like every other
`bench.*` results module.
"""

import dataclasses

import pytest

from bench.cadence import (
    HEADROOM_FRACTION,
    cadence_of,
    format_cadence_report,
    latest_per_cadence,
    recommend_cadence,
    repeat_spread,
)
from bench.detector_results import LatencySummary
from bench.portability import FRAME_BUDGET_MS
from bench.selective import (
    CASES,
    PRIORITY_CASE,
    RegionSummary,
    background_check,
    plan_record,
    staleness_summary,
)
from test_bench_results import a_fingerprint
from test_bench_selective import a_selective_result, tracks_of

DEPLOY_GPU = "NVIDIA GeForce RTX 4090"
LAPTOP_GPU = "NVIDIA GeForce RTX 3080 Laptop GPU"


def an_arm(detect_every_n, ms_per_frame=24.5, detect_ms=23.0, gpu=DEPLOY_GPU,
           finished="2026-09-07T12:00:00Z", background_ok=True, **overrides):
    """One arm of the sweep: the shipped record, at one cadence.

    Built out of the selective record's own dataclasses for the reason
    `a_selective_result` is - the report reads field names, and a dict spelt by
    hand here would keep passing after one was renamed.
    """
    case = CASES[PRIORITY_CASE].replace(detect_every_n=detect_every_n)
    amortised = round(detect_ms / detect_every_n, 4)
    with_detection = round(ms_per_frame + amortised, 4)
    fields = dict(
        case=case, plan=plan_record(case.plan()),
        run=dataclasses.replace(
            a_selective_result().run,
            render=LatencySummary.from_samples([ms_per_frame]),
            composite=LatencySummary.from_samples([2.7]),
            detect=LatencySummary.from_samples([detect_ms]),
            finished_utc=finished,
            detect_every_n=detect_every_n,
            amortised_detect_ms=amortised,
            ms_per_frame=ms_per_frame,
            ms_per_frame_with_detection=with_detection,
            fps=round(1000.0 / with_detection, 4)),
        staleness=staleness_summary([tracks_of(5)] * 6,
                                    detect_every_n=detect_every_n,
                                    ms_per_frame=ms_per_frame),
        hardware=a_fingerprint(gpu_name=gpu),
    )
    if not background_ok:
        fields["background"] = background_check([0] * 47 + [12], 200_000)
    fields.update(overrides)
    return a_selective_result(**fields).to_dict()


def a_sweep(*cadences, **kwargs):
    return {f"selective-people-n{n}-2026090{index}.json": an_arm(n, **kwargs)
            for index, n in enumerate(cadences)}


# --- reading one arm --------------------------------------------------------


def test_the_cadence_an_arm_was_measured_at_is_read_off_the_plan_it_rendered():
    """Not off the case name: the validator clamps, and what the arm ran at is
    what the plan says it ran at."""
    assert cadence_of(an_arm(5)) == 5


def test_one_row_per_cadence_per_machine():
    """The same reduction the other reports use: a second machine adds a row
    rather than deleting the first one's (issue #25)."""
    arms = {"a.json": an_arm(3, finished="2026-09-07T12:00:00Z"),
            "b.json": an_arm(3, finished="2026-09-07T13:00:00Z"),
            "c.json": an_arm(3, gpu=LAPTOP_GPU)}
    assert set(latest_per_cadence(arms)) == {"b.json", "c.json"}


# --- the recommendation -----------------------------------------------------


def test_the_freshest_cadence_that_clears_the_budget_with_headroom_wins():
    """Staleness is the only thing a cadence costs, so among the settings that
    fit, the lowest one is the answer."""
    arms = [an_arm(2), an_arm(3), an_arm(5), an_arm(8)]
    recommendation = recommend_cadence(arms)
    assert recommendation.detect_every_n == 5
    assert recommendation.meets_budget and recommendation.has_headroom


def test_a_setting_that_only_just_fits_is_named_as_only_just_fitting():
    """The 4090 baseline clears 30 FPS by about a millisecond. A recommendation
    that reported that as headroom would be recommending the margin away."""
    recommendation = recommend_cadence([an_arm(3, ms_per_frame=31.0, detect_ms=4.0)])
    assert recommendation.meets_budget and not recommendation.has_headroom
    assert "without the" in recommendation.statement


def test_no_setting_that_fits_is_said_plainly_rather_than_recommended_anyway():
    recommendation = recommend_cadence([an_arm(2, ms_per_frame=60.0),
                                        an_arm(8, ms_per_frame=60.0)])
    assert not recommendation.meets_budget
    assert recommendation.detect_every_n == 8, "the cheapest measured, named as such"
    assert "no cadence measured" in recommendation.statement


def test_a_cadence_that_broke_the_background_criterion_is_disqualified():
    """The Gate: a configuration that breaks bit-identity is disqualified, not the
    criterion. So it cannot be recommended however fast it is."""
    recommendation = recommend_cadence([an_arm(2, ms_per_frame=10.0,
                                               background_ok=False), an_arm(5)])
    assert recommendation.detect_every_n == 5
    assert recommendation.disqualified == (2,)
    assert "bit-identical" in recommendation.statement


def test_the_headroom_is_a_fraction_of_the_frame_budget_and_says_so():
    assert 0.0 < HEADROOM_FRACTION < 1.0
    recommendation = recommend_cadence([an_arm(5)])
    assert recommendation.budget_ms == pytest.approx(FRAME_BUDGET_MS, abs=1e-3)
    assert f"{FRAME_BUDGET_MS:.2f} ms" in recommendation.statement


def test_a_sweep_of_one_arm_says_it_was_one_arm():
    assert recommend_cadence([an_arm(3, ms_per_frame=20.0)]).arms == 1


# --- the block the spec carries ---------------------------------------------


def test_the_report_names_every_cadence_it_swept():
    report = format_cadence_report(a_sweep(2, 3, 5, 8))
    for cadence in (2, 3, 5, 8):
        assert f"| {cadence} |" in report


def test_the_report_puts_the_milliseconds_bought_beside_the_staleness_spent():
    report = format_cadence_report(a_sweep(2, 3, 5, 8))
    assert "amortised" in report and "box age" in report
    assert "refresh IoU" in report


def test_the_report_states_the_baseline_gap_from_the_committed_run_it_was_given():
    """Step 1 of the issue: the gap is taken from #24's baseline, not re-derived
    here. Given no baseline, the report says that rather than inventing one."""
    baseline = {"b.json": a_selective_result(
        hardware=a_fingerprint(gpu_name=DEPLOY_GPU)).to_dict()}
    report = format_cadence_report(a_sweep(3, 5), baseline=baseline)
    assert "issue #24" in report
    assert "FPS at" in report

    assert "no deploy-hardware baseline" in format_cadence_report(a_sweep(3))


def test_a_sweep_whose_arms_rendered_different_amounts_of_frame_is_not_a_sweep():
    """Region count drives cost - the same trap spec 7.4's report answers. Two
    arms that rendered different regions are not a cadence comparison."""
    regions = a_selective_result().regions
    odd = an_arm(8, regions=RegionSummary(**{**regions.to_dict(),
                                             "regions_per_frame": 1.0}))
    report = format_cadence_report({"a.json": an_arm(3), "b.json": odd})
    assert "not comparable" in report


def test_an_empty_directory_reports_that_nothing_was_swept():
    assert "no cadence sweep" in format_cadence_report({})


def test_the_report_takes_its_recommendation_once_per_machine():
    laptop = {f"l-{name}": arm
              for name, arm in a_sweep(3, 5, gpu=LAPTOP_GPU).items()}
    report = format_cadence_report({**laptop, **a_sweep(3, 5)})
    assert report.count("Recommended") == 2
    assert DEPLOY_GPU in report and LAPTOP_GPU in report


# --- is the recommendation outside the run-to-run spread? --------------------


def test_a_cadence_measured_twice_reports_how_far_apart_the_two_runs_were():
    """The recommendation turns on a 3.33 ms threshold, so how repeatable an arm
    is decides whether the rule decided anything - the same question #24 asked of
    its 1 ms margin."""
    arms = {"a.json": an_arm(3, ms_per_frame=24.0, finished="2026-09-07T11:00:00Z"),
            "b.json": an_arm(3, ms_per_frame=24.5, finished="2026-09-07T12:00:00Z")}
    spread = repeat_spread(arms.values())
    assert spread.repeats == 1
    assert spread.worst_ms == pytest.approx(0.5)


def test_a_verdict_a_repeat_would_overturn_is_labelled_as_one():
    """An arm 0.1 ms from the headroom line, measured twice 2 ms apart, has not
    been shown to be on either side of it."""
    close = FRAME_BUDGET_MS * (1 - HEADROOM_FRACTION) - 0.1
    arms = [an_arm(3, ms_per_frame=close, detect_ms=0.0,
                   finished="2026-09-07T11:00:00Z"),
            an_arm(3, ms_per_frame=close + 2.0, detect_ms=0.0,
                   finished="2026-09-07T12:00:00Z")]
    assert not repeat_spread(arms).decisive


def test_a_verdict_no_repeat_comes_near_is_decisive():
    arms = [an_arm(5, ms_per_frame=10.0, finished="2026-09-07T11:00:00Z"),
            an_arm(5, ms_per_frame=10.2, finished="2026-09-07T12:00:00Z")]
    assert repeat_spread(arms).decisive


def test_a_cadence_measured_once_has_no_spread_to_report():
    spread = repeat_spread([an_arm(3)])
    assert spread.repeats == 0
    assert "once" in spread.statement


def test_the_report_carries_the_spread_and_the_clip_to_watch():
    report = format_cadence_report(a_sweep(3, 5))
    assert "measured once" in report or "repeat" in report
    assert "comparison.mp4" in report
