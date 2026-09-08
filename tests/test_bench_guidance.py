"""Classifier-free guidance: the arms, the adherence rule, and the recommendation.

Issue #45. The app runs at `cfg_type: none`, `guidance_scale: 0.0` and has never
measured anything else, so "the prompt is followed only weakly" has never been shown
to be the checkpoint rather than a switch that is off.

Three rules live here and each is one of the issue's traps made executable.

- **Adherence is scored, not eyeballed.** §8.2's `identity-dog` shape: the
  open-vocabulary detector is asked whether the rendered region reads back as the
  thing the prompt asked for. An arm's number is that fraction, with the detector's
  mean confidence beside it because a fraction over 48 frames saturates.
- **A stronger pull is not free.** The same axis that makes the prompt land makes
  the output drift from the captured frame, so an arm carries both and the
  recommendation prices the trade rather than reporting half of it.
- **`delta` is meaningless for some cfg types.** The pipeline reads it only under
  `self` and `initialize`; sweeping it under `full` and reporting the flat line
  would be reporting an argument that is never read.

GPU-free: this is the arithmetic and the record shape. `bench.guidance_runner` is
the half that touches a GPU.
"""

from __future__ import annotations

import pytest

from bench.guidance import (
    ADHERENCE_TIE,
    CASES,
    CONTROL_ARM,
    GUIDANCE_LADDER,
    MAX_COST_RATIO,
    MIN_ADHERENCE_GAIN,
    MIN_ADHERENCE_GAIN_FOR_BUILD,
    ArmSpec,
    GuidanceArm,
    arm_name,
    engine_keying,
    equivalence_note,
    format_guidance_report,
    recommend_guidance,
    showcase_specs,
    uses_delta,
)
from bench.selective import BackgroundCheck
from engine_cache import CFG_FULL, CFG_INITIALIZE, CFG_NONE, CFG_SELF


def a_background(passed=True, frames=48):
    return BackgroundCheck(
        frames=frames, identical_frames=frames if passed else frames - 1,
        worst_pixels_changed=0 if passed else 12, background_pixels=200000,
        passed=passed,
        statement=(f"{frames if passed else frames - 1}/{frames} frames left every "
                   f"pixel outside the rendered regions as captured"))


def an_arm(cfg_type=CFG_SELF, guidance_scale=1.4, delta=1.0, hits=40, frames=48,
           conf=0.61, retained=6, region_change=30.0, control_change=4.0,
           ms=27.0, unet_batch=1, background=None, loaded=True, error=None,
           flicker=1.5):
    return GuidanceArm(
        cfg_type=cfg_type, guidance_scale=guidance_scale, delta=delta,
        delta_applies=uses_delta(cfg_type), unet_batch=unet_batch,
        engine_dir=f"eng--max_batch-{unet_batch}", keys_new_engine=unet_batch != 1,
        engine_cached=None, ms_per_frame=ms, adherence_hits=hits,
        adherence_frames=frames, adherence_conf=conf, retained_hits=retained,
        region_change=region_change, control_change=control_change,
        flicker=flicker, background=background or a_background(frames=frames),
        frames=frames, loaded=loaded, error=error,
    )


def a_control(hits=8, **changes):
    return an_arm(cfg_type=CFG_NONE, guidance_scale=1.0, delta=1.0, hits=hits,
                  **changes)


def a_result(arms=None, gpu="NVIDIA GeForce RTX 4090",
             finished="2026-09-08T12:00:00Z", name="cfg-dog"):
    arms = arms if arms is not None else [a_control(), an_arm()]
    return {
        "case": {"name": name, "clip": "dog.mp4", "frames": 48, "canvas": 512,
                 "base_scenario": "img2img-none-512x512-b1", "base_model": "sd-turbo",
                 "steps": 1, "concept": "dog", "reads_back_as": "cat",
                 "region": "full_box", "denoise": 0.72,
                 "prompt": "a cat, feline face, whiskers, pointed ears, photograph"},
        "clip": {"name": "dog.mp4", "sha256": "abc", "width": 960, "height": 540,
                 "fps": 25.0, "total_frames": 120, "start_frame": 0,
                 "frames_used": 48},
        "arms": [arm.to_dict() for arm in arms],
        "run": {"started_utc": finished, "finished_utc": finished},
        "cooldown": {"outcome": "reached"},
        "hardware": {"gpu_name": gpu},
        "clock_normalization": {"regime": "unlocked"},
        "comparison_still": "cfg-dog-arms.jpg",
        "comparison_clip": "cfg-dog-arms.mp4",
    }


# --- the arm vocabulary ------------------------------------------------------


def test_delta_is_only_read_where_the_pipeline_reads_it():
    """`noise_pred_uncond = self.stock_noise * self.delta` sits under
    `cfg_type in ("self", "initialize")` and nowhere else, so an arm that swept it
    under `full` would be reporting a flat line from an unread argument."""
    assert uses_delta(CFG_SELF)
    assert uses_delta(CFG_INITIALIZE)
    assert not uses_delta(CFG_FULL)
    assert not uses_delta(CFG_NONE)


def test_an_arm_is_named_after_every_setting_that_moved_it():
    assert arm_name(ArmSpec(CFG_NONE, 1.0, 1.0)) == CONTROL_ARM
    assert arm_name(ArmSpec(CFG_SELF, 1.4, 1.0)) == "self-g14-d10"
    assert arm_name(ArmSpec(CFG_SELF, 1.4, 0.5)) == "self-g14-d05"
    assert arm_name(ArmSpec(CFG_FULL, 2.0, 1.0)) == "full-g20"


def test_the_control_arm_is_the_setting_the_app_ships():
    assert CONTROL_ARM == CFG_NONE


# --- the two numbers an arm carries ------------------------------------------


def test_adherence_is_the_fraction_of_frames_that_read_back_as_the_prompt():
    assert an_arm(hits=24, frames=48).adherence == pytest.approx(0.5)
    assert an_arm(hits=0, frames=48).adherence == 0.0


def test_an_arm_that_never_ran_scores_no_adherence_rather_than_zero_percent():
    """Zero would rank it as a measured failure. It is not measured at all."""
    assert an_arm(loaded=False, hits=0, frames=0).adherence == 0.0
    assert not an_arm(loaded=False, frames=0).measured


def test_the_drift_subtracts_the_control_the_capture_round_trip_costs():
    assert an_arm(region_change=30.0, control_change=4.0).net_change == \
        pytest.approx(26.0)


# --- the recommendation ------------------------------------------------------


def test_guidance_that_buys_nothing_leaves_the_shipped_default_alone():
    """The whole point of a control arm: an axis that does not move adherence is
    an axis whose default should not move either."""
    recommendation = recommend_guidance([a_control(hits=20), an_arm(hits=22)])
    assert recommendation.cfg_type == CFG_NONE
    assert not recommendation.moves_default
    assert f"{MIN_ADHERENCE_GAIN:.0%}" in recommendation.statement


def test_guidance_that_lands_the_prompt_moves_the_default_and_says_by_how_much():
    recommendation = recommend_guidance([a_control(hits=8), an_arm(hits=40)])
    assert recommendation.moves_default
    assert recommendation.cfg_type == CFG_SELF
    assert recommendation.guidance_scale == 1.4


def test_an_arm_that_costs_a_build_has_to_be_plainly_worth_it_not_marginally():
    """A gain that would move the default for free does not move it for ~5 GB and
    a third of the frame path. Two bars, because two arms buying the same
    adherence at wildly different prices are not the same recommendation."""
    gain = (MIN_ADHERENCE_GAIN + MIN_ADHERENCE_GAIN_FOR_BUILD) / 2
    control = a_control(hits=0, frames=100)
    marginal = an_arm(cfg_type=CFG_FULL, hits=int(gain * 100), frames=100,
                      unet_batch=2, ms=31.0)
    assert not recommend_guidance([control, marginal]).moves_default
    free = an_arm(cfg_type=CFG_SELF, hits=int(gain * 100), frames=100,
                  unet_batch=1)
    assert recommend_guidance([control, free]).moves_default


def test_an_arm_the_frame_budget_cannot_afford_is_not_a_default():
    """However well it reads back. The default has to run at 30 FPS, and an arm
    that lands the prompt at half the frame rate is a setting, not a default."""
    control = a_control(hits=10, frames=100, ms=47.5)
    dear = an_arm(cfg_type=CFG_FULL, hits=90, frames=100, unet_batch=2,
                  ms=47.5 * MAX_COST_RATIO + 0.1)
    recommendation = recommend_guidance([control, dear])
    assert not recommendation.moves_default
    assert "out on cost rather than on quality" in recommendation.statement


def test_the_same_arm_just_inside_the_budget_is_a_default():
    control = a_control(hits=10, frames=100, ms=47.5)
    affordable_arm = an_arm(cfg_type=CFG_FULL, hits=90, frames=100, unet_batch=2,
                            ms=47.5 * MAX_COST_RATIO)
    assert recommend_guidance([control, affordable_arm]).moves_default


def test_a_build_arm_that_is_plainly_worth_it_is_still_recommended():
    control = a_control(hits=10, frames=100)
    strong = an_arm(cfg_type=CFG_FULL, hits=60, frames=100, unet_batch=2, ms=31.0)
    recommendation = recommend_guidance([control, strong])
    assert recommendation.moves_default
    assert recommendation.keys_new_engine


def test_a_free_arm_that_clears_its_bar_wins_over_a_stronger_build_arm():
    """The free arms are asked first and on their own, so a build cannot take a
    recommendation a runtime setting had already earned."""
    arms = [a_control(hits=10, frames=100),
            an_arm(cfg_type=CFG_SELF, hits=30, frames=100, unet_batch=1),
            an_arm(cfg_type=CFG_FULL, hits=90, frames=100, unet_batch=2, ms=31.0)]
    assert recommend_guidance(arms).cfg_type == CFG_SELF


def test_the_refusal_prices_the_build_it_refused():
    arms = [a_control(hits=10, frames=100),
            an_arm(cfg_type=CFG_FULL, hits=25, frames=100, unet_batch=2, ms=31.0)]
    statement = recommend_guidance(arms).statement
    assert "31.00" in statement and "UNet batch 2" in statement
    assert f"{MIN_ADHERENCE_GAIN_FOR_BUILD:.0%}" in statement


def test_between_arms_that_tie_on_adherence_the_cheapest_engine_wins():
    """`self` keys the engine the app already has and `full` keys a ~5 GB build,
    so a tie has to break towards the one that costs nothing to ship."""
    arms = [a_control(hits=8),
            an_arm(cfg_type=CFG_SELF, guidance_scale=1.4, hits=40, unet_batch=1),
            an_arm(cfg_type=CFG_FULL, guidance_scale=1.4, hits=41, unet_batch=2,
                   ms=44.0)]
    recommendation = recommend_guidance(arms)
    assert recommendation.cfg_type == CFG_SELF
    assert ADHERENCE_TIE > 0


def test_an_arm_that_broke_bit_identity_cannot_be_recommended_however_well_it_read():
    arms = [a_control(hits=8),
            an_arm(hits=48, background=a_background(passed=False))]
    recommendation = recommend_guidance(arms)
    assert not recommendation.moves_default
    assert "bit-identical" in recommendation.statement


def test_an_arm_that_did_not_load_is_not_a_candidate():
    arms = [a_control(hits=8),
            an_arm(hits=0, frames=0, loaded=False, error="boom")]
    recommendation = recommend_guidance(arms)
    assert not recommendation.moves_default


def test_the_recommendation_prices_the_drift_it_costs():
    """The first trap: the axis that makes the prompt land makes the output drift
    from the capture, and reporting one side of that is half an answer."""
    arms = [a_control(hits=8, region_change=14.0),
            an_arm(hits=40, region_change=42.0)]
    statement = recommend_guidance(arms).statement
    assert "38.0" in statement and "10.0" in statement


def test_a_run_with_no_control_arm_cannot_recommend_anything():
    assert recommend_guidance([an_arm()]).moves_default is False


# --- the block ---------------------------------------------------------------


def test_the_report_names_every_arm_that_keys_a_new_engine():
    """The Gate asks for that list explicitly, because two of the four cfg types
    run extra latents through the UNet and therefore are a ~5 GB build."""
    arms = [a_control(),
            an_arm(cfg_type=CFG_SELF, unet_batch=1),
            an_arm(cfg_type=CFG_INITIALIZE, unet_batch=2),
            an_arm(cfg_type=CFG_FULL, unet_batch=2)]
    report = format_guidance_report({"a.json": a_result(arms)})
    assert "initialize" in report and "full" in report
    assert "max_batch-2" in report


def test_the_report_says_when_no_arm_keys_a_new_engine():
    report = format_guidance_report({"a.json": a_result([a_control(), an_arm()])})
    assert "key a new engine: none" in report.lower()


def test_the_report_states_the_background_gate_over_every_arm():
    arms = [a_control(), an_arm(background=a_background(passed=False))]
    report = format_guidance_report({"a.json": a_result(arms)})
    assert "changed a pixel outside" in report


def test_an_empty_directory_says_so():
    assert "no guidance" in format_guidance_report({}).lower()


def test_two_machines_grow_a_gpu_column_and_one_does_not():
    one = format_guidance_report({"a.json": a_result()})
    assert "| GPU |" not in one
    two = format_guidance_report({
        "a.json": a_result(),
        "b.json": a_result(gpu="NVIDIA GeForce RTX 3080 Laptop GPU",
                           finished="2026-09-08T13:00:00Z", name="cfg-people"),
    })
    assert "| GPU |" in two


# --- which arms are a build rather than a setting ----------------------------


def test_the_shipped_cfg_types_key_the_engine_the_app_already_has():
    case = CASES["cfg-dog"]
    for cfg_type in (CFG_NONE, CFG_SELF):
        keying = engine_keying(case, ArmSpec(cfg_type, 1.4, 1.0))
        assert not keying.keys_new_engine
        assert keying.unet_batch == 1


def test_initialize_and_full_key_a_build_and_the_directory_says_which():
    case = CASES["cfg-dog"]
    for cfg_type in (CFG_INITIALIZE, CFG_FULL):
        keying = engine_keying(case, ArmSpec(cfg_type, 1.4, 1.0))
        assert keying.keys_new_engine
        assert keying.unet_batch == 2
        assert "max_batch-2" in keying.directory


def test_the_engine_a_cfg_arm_needs_follows_the_step_count_too():
    """At four steps `initialize` is a batch-5 engine and `full` a batch-8 one, so
    a hard-coded list of cfg types would be right once and wrong after."""
    case = CASES["cfg-dog-sd15"]
    assert engine_keying(case, ArmSpec(CFG_SELF, 1.4, 1.0)).unet_batch == 4
    assert engine_keying(case, ArmSpec(CFG_INITIALIZE, 1.4, 1.0)).unet_batch == 5
    assert engine_keying(case, ArmSpec(CFG_FULL, 1.4, 1.0)).unet_batch == 8


def test_whether_that_engine_is_on_disk_is_unanswered_rather_than_guessed(tmp_path):
    case = CASES["cfg-dog"]
    spec = ArmSpec(CFG_FULL, 1.4, 1.0)
    assert engine_keying(case, spec).cached is None
    assert engine_keying(case, spec, engines_root=tmp_path).cached is False


# --- the artefact ------------------------------------------------------------


def test_the_comparison_clip_shows_one_panel_per_cfg_type_at_one_rung():
    """Twenty panels is a strip nobody can read; one per cfg type at a shared rung
    is the comparison a human is actually making."""
    shown = showcase_specs(CASES["cfg-dog"].specs())
    assert [spec.cfg_type for spec in shown[:4]] == [CFG_NONE, CFG_SELF,
                                                    CFG_INITIALIZE, CFG_FULL]
    assert {spec.guidance_scale for spec in shown[1:4]} == {GUIDANCE_LADDER[0]}


def test_the_comparison_clip_also_shows_the_top_of_the_ladder():
    """A strip taken only where guidance works is a picture of the good news, and
    the useful range on a short LCM schedule is narrow."""
    shown = showcase_specs(CASES["cfg-dog"].specs())
    assert len(shown) == 5
    assert shown[-1].guidance_scale == GUIDANCE_LADDER[-1]


def test_two_cfg_types_that_computed_the_same_thing_are_reported_as_one():
    """Read off the run rather than argued: the render is deterministic, so two
    arms agreeing to four figures agree because they did the same arithmetic."""
    arms = [a_control(),
            an_arm(cfg_type=CFG_INITIALIZE, guidance_scale=1.4, hits=16,
                   unet_batch=2),
            an_arm(cfg_type=CFG_FULL, guidance_scale=1.4, hits=16, unet_batch=2)]
    note = equivalence_note(arms)
    assert note is not None
    assert "initialize" in note and "full" in note


def test_arms_that_differ_get_no_equivalence_note():
    assert equivalence_note([a_control(), an_arm(hits=16), an_arm(hits=40)]) is None


def test_both_base_models_are_swept_by_the_same_ladder():
    """Step 5 of the issue - and the two cases have to differ in the model and the
    step count only, or the comparison is two sweeps rather than one."""
    turbo, sd15 = CASES["cfg-dog"], CASES["cfg-dog-sd15"]
    assert turbo.specs() == sd15.specs()
    assert (turbo.denoise, turbo.prompt, turbo.clip, turbo.region) ==         (sd15.denoise, sd15.prompt, sd15.clip, sd15.region)
    assert (turbo.steps, turbo.base_model) != (sd15.steps, sd15.base_model)
