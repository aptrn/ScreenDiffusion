"""The temporal-stability sweep: what the two §8.5 levers buy and what they spend.

Issue #32. The report's whole job is to keep three rules executable rather than
argued - an arm that renders less is disqualified, an arm that breaks bit-identity
is disqualified, and a flicker win that costs the restyle's response to motion has
to say so - so each of them is a test here rather than a sentence in the block.

GPU-free: `bench.stability` reads committed JSON and formats it, like every other
`bench.*` results module.
"""

import dataclasses

import numpy as np

from bench.flicker import flicker_score, response_score
from bench.selective import (
    CASES,
    PRIORITY_CASE,
    change_check,
    ema_suffix,
    plan_record,
)
from bench.stability import (
    RESPONSE_RETENTION,
    SHIPPED_EMA,
    SHIPPED_POLICY,
    arm_of,
    control_of,
    disqualifications,
    format_stability_report,
    latest_per_arm,
    output_ema_of,
    recommend_setting,
    repeat_spread,
    response_of,
    seed_policy_of,
)
from test_bench_results import a_fingerprint
from test_bench_selective import a_selective_result

DEPLOY_GPU = "NVIDIA GeForce RTX 4090"
LAPTOP_GPU = "NVIDIA GeForce RTX 3080 Laptop GPU"


def _scores(flicker: float, response: float, frames: int = 4):
    """A flicker and a response figure, built through the metrics themselves.

    Two flat sequences: one where the source stood still and the output moved by
    `flicker`, one where the source moved and the output followed by `response`.
    Going through the real functions means the record carries the real shape, and a
    renamed field fails here rather than in a report six months later.
    """
    still = [np.zeros((4, 4, 3)) for _ in range(frames)]
    boiling = [np.full((4, 4, 3), index * flicker) for index in range(frames)]
    moving = [np.full((4, 4, 3), index * 255.0) for index in range(frames)]
    following = [np.full((4, 4, 3), index * response) for index in range(frames)]
    return flicker_score(still, boiling), response_score(moving, following)


def an_arm(seed_policy=SHIPPED_POLICY, output_ema=SHIPPED_EMA, flicker=1.49,
           response=6.0, net_change=11.8, gpu=DEPLOY_GPU,
           finished="2026-09-07T12:00:00Z", background_ok=True, **overrides):
    """One arm of the sweep: the shipped record, at one setting.

    Built out of the selective record's own dataclasses for the reason
    `a_selective_result` is - the report reads field names.
    """
    case = CASES[PRIORITY_CASE].replace(
        name=f"{PRIORITY_CASE}-{seed_policy}-{ema_suffix(output_ema)}",
        seed_policy=seed_policy, output_ema=output_ema)
    flicker_score_, response_score_ = _scores(flicker, response)
    fields = dict(
        case=case, plan=plan_record(case.plan()),
        run=dataclasses.replace(a_selective_result().run, finished_utc=finished),
        flicker=flicker_score_, response=response_score_,
        change=change_check(net_change, 0.0),
        hardware=a_fingerprint(gpu_name=gpu),
    )
    if not background_ok:
        from bench.selective import background_check

        fields["background"] = background_check([0] * 47 + [12], 200_000)
    fields.update(overrides)
    return a_selective_result(**fields).to_dict()


def a_sweep(*arms):
    """`(policy, ema, flicker, response)` tuples as a directory of records."""
    return {f"arm-{index}.json": an_arm(policy, ema, flicker=flicker,
                                        response=response)
            for index, (policy, ema, flicker, response) in enumerate(arms)}


# --- reading one arm ---------------------------------------------------------


def test_the_setting_an_arm_ran_at_is_read_off_the_plan_it_rendered():
    """Not off the case name: the validator clamps, and what the arm ran at is
    what the plan says it ran at."""
    arm = an_arm("per_track", 0.5)
    assert arm_of(arm) == ("per_track", 0.5)


def test_a_record_from_before_this_issue_reads_as_the_shipped_default():
    """Every committed baseline was rendered on the field the engine prepared and
    with no EMA, which is exactly what these two names mean - so it is an answer
    rather than a gap."""
    old = a_selective_result().to_dict()
    old["plan"].pop("seed_policy")
    old["plan"].pop("output_ema")
    assert seed_policy_of(old) == SHIPPED_POLICY
    assert output_ema_of(old) == SHIPPED_EMA


def test_a_record_from_before_this_issue_has_no_responsiveness_figure():
    """`None`, not zero: zero is what a frozen output scores."""
    old = a_selective_result().to_dict()
    old["response"] = None
    assert response_of(old) is None


def test_one_row_per_arm_per_machine():
    """The same reduction the other reports use (issue #25)."""
    arms = {"a.json": an_arm(finished="2026-09-07T12:00:00Z"),
            "b.json": an_arm(finished="2026-09-07T13:00:00Z"),
            "c.json": an_arm(gpu=LAPTOP_GPU)}
    assert set(latest_per_arm(arms)) == {"b.json", "c.json"}


def test_the_control_is_the_arm_at_the_shipped_default():
    arms = [an_arm("per_track", 0.5), an_arm(), an_arm("random", 0.0)]
    assert arm_of(control_of(arms)) == (SHIPPED_POLICY, SHIPPED_EMA)


def test_a_sweep_with_no_control_arm_has_none():
    assert control_of([an_arm("per_track", 0.5)]) is None


# --- disqualification --------------------------------------------------------


def test_an_arm_that_rendered_less_than_visibly_is_disqualified():
    """The issue's Gate, stated as arithmetic: an EMA that lowers flicker by
    suppressing the render cannot be recommended however steady it looks."""
    refused, = disqualifications([an_arm("fixed", 0.9, flicker=0.1, net_change=3.0)])
    assert "lowered flicker by rendering less" in refused.reason


def test_an_arm_that_broke_bit_identity_is_disqualified():
    """Criterion 4 is not negotiable for a flicker win - the issue's first trap."""
    refused, = disqualifications([an_arm("random", 0.0, background_ok=False)])
    assert "bit-identical" in refused.reason


def test_a_disqualified_arm_is_never_recommended():
    arms = [an_arm(), an_arm("fixed", 0.75, flicker=0.2, net_change=2.0)]
    assert recommend_setting(arms).output_ema == SHIPPED_EMA


def test_an_arm_that_passed_both_is_not_disqualified():
    assert disqualifications([an_arm(), an_arm("per_track", 0.0)]) == ()


# --- the recommendation ------------------------------------------------------


def test_the_steadiest_qualifying_arm_wins():
    arms = [an_arm(), an_arm("fixed", 0.25, flicker=1.2),
            an_arm("fixed", 0.5, flicker=0.9)]
    recommendation = recommend_setting(arms)
    assert (recommendation.seed_policy, recommendation.output_ema) == ("fixed", 0.5)
    assert recommendation.flicker_delta < 0


def test_an_arm_that_lost_its_response_to_motion_is_not_recommended():
    """The issue's second trap: an EMA strong enough to kill boiling also kills the
    restyle's response, and a rule that did not price that would always recommend
    the strongest setting measured."""
    arms = [an_arm(response=6.0),
            an_arm("fixed", 0.75, flicker=0.3, response=6.0 * RESPONSE_RETENTION / 2)]
    assert recommend_setting(arms).is_shipped_default


def test_an_arm_that_kept_its_response_is_recommended():
    arms = [an_arm(response=6.0),
            an_arm("fixed", 0.5, flicker=0.9, response=6.0 * RESPONSE_RETENTION)]
    assert recommend_setting(arms).output_ema == 0.5


def test_a_flicker_win_inside_the_run_to_run_spread_is_not_a_win():
    """The baseline is unusually steady, so a lever that moves it by less than two
    runs of one arm move it has not moved it."""
    sweep = {"a.json": an_arm(flicker=1.49, finished="2026-09-07T12:00:00Z"),
             "b.json": an_arm(flicker=1.20, finished="2026-09-07T13:00:00Z"),
             "c.json": an_arm("per_track", 0.0, flicker=1.05)}
    every_run = list(sweep.values())
    rows = list(latest_per_arm(sweep).values())
    assert recommend_setting(rows, every_run).is_shipped_default
    # Without the repeats there is no spread, and the same 0.15 reads as a win.
    assert not recommend_setting(rows).is_shipped_default


def test_the_spread_the_rule_uses_is_over_every_committed_run():
    """A spread computed over the reduced rows is zero by construction, and a rule
    turning on zero is not a rule."""
    sweep = {"a.json": an_arm(flicker=1.49, finished="2026-09-07T12:00:00Z"),
             "b.json": an_arm(flicker=1.20, finished="2026-09-07T13:00:00Z"),
             "c.json": an_arm("per_track", 0.0, flicker=1.05)}
    assert "**Recommended: `seed_policy: fixed`" in format_stability_report(sweep)


def test_no_lever_paying_is_a_result_rather_than_a_gap():
    arms = [an_arm(), an_arm("per_track", 0.0, flicker=1.51)]
    recommendation = recommend_setting(arms)
    assert recommendation.is_shipped_default
    assert "neither lever pays for itself" in recommendation.statement


def test_the_recommendation_says_what_it_traded_away():
    arms = [an_arm(response=6.0), an_arm("fixed", 0.5, flicker=0.9, response=5.4)]
    assert "response to motion" in recommend_setting(arms).cost


def test_a_sweep_with_no_control_arm_recommends_nothing():
    assert recommend_setting([an_arm("per_track", 0.5)]) is None


# --- the repeat spread -------------------------------------------------------


def test_two_runs_of_one_arm_are_the_spread():
    sweep = {"a.json": an_arm(flicker=1.49), "b.json": an_arm(flicker=1.53)}
    spread = repeat_spread(list(sweep.values()))
    assert spread.repeats == 1
    assert spread.worst == 0.04


def test_one_run_per_arm_says_there_is_no_spread_to_judge_against():
    spread = repeat_spread([an_arm(), an_arm("per_track", 0.0)])
    assert spread.repeats == 0
    assert "no run-to-run spread" in spread.statement


# --- the block the spec carries ----------------------------------------------


def test_an_empty_directory_says_so_rather_than_drawing_an_empty_table():
    assert "no temporal-stability sweep" in format_stability_report({})


def test_the_block_carries_a_row_per_arm_and_the_recommendation():
    report = format_stability_report(
        a_sweep(("fixed", 0.0, 1.49, 6.0), ("per_track", 0.0, 1.44, 6.0),
                ("fixed", 0.5, 0.90, 5.5)))
    assert "| seed policy | output EMA | flicker |" in report
    assert "per_track" in report and "0.50" in report
    assert "**Recommended:" in report


def test_the_control_row_is_labelled_rather_than_given_a_delta_of_zero():
    report = format_stability_report(a_sweep(("fixed", 0.0, 1.49, 6.0),
                                             ("per_track", 0.0, 1.44, 6.0)))
    assert "| control |" in report


def test_the_block_states_the_shipped_paths_own_flicker_when_it_has_one():
    baseline = {"base.json": a_selective_result().to_dict()}
    report = format_stability_report(a_sweep(("fixed", 0.0, 1.49, 6.0)),
                                     baseline=baseline)
    assert "spec 8.8" in report


def test_the_block_says_when_the_control_does_not_reproduce_the_baseline():
    """A control that measured something else would make every row a comparison
    with something else."""
    baseline = {"base.json": an_arm(flicker=1.49)}
    report = format_stability_report(a_sweep(("fixed", 0.0, 9.90, 6.0)),
                                     baseline=baseline)
    assert "is *not* that figure" in report


def test_a_shuffled_directory_renders_the_same_bytes():
    sweep = a_sweep(("fixed", 0.0, 1.49, 6.0), ("per_track", 0.0, 1.44, 6.0),
                    ("random", 0.0, 2.10, 6.0))
    reversed_sweep = dict(reversed(list(sweep.items())))
    assert format_stability_report(sweep) == format_stability_report(reversed_sweep)


def test_two_machines_grow_a_gpu_column_and_take_a_verdict_each():
    sweep = {"a.json": an_arm(), "b.json": an_arm(gpu=LAPTOP_GPU),
             "c.json": an_arm("fixed", 0.5, flicker=0.9, response=5.9),
             "d.json": an_arm("fixed", 0.5, flicker=0.9, response=5.9,
                              gpu=LAPTOP_GPU)}
    report = format_stability_report(sweep)
    assert "| GPU |" in report
    assert report.count("**Recommended:") == 2


def test_one_machine_draws_no_gpu_column():
    assert "| GPU |" not in format_stability_report(
        a_sweep(("fixed", 0.0, 1.49, 6.0), ("per_track", 0.0, 1.44, 6.0)))
