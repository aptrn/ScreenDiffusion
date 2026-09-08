"""The step-quality sweep's arithmetic (issue #46). GPU-free.

What is tested here is the part that decides something: which engine each arm of
each route needs, what a cached-engine swap figure is allowed to be read from, and
the rule that picks between the two ways of paying for a runtime step count.
"""

import dataclasses

import pytest

from bench.quality import (
    ADHERENCE_CONF,
    CASES,
    DETECTION_ALLOWANCE_MS,
    FRAME_BUDGET_MS,
    MIN_ADHERENCE_GAIN,
    QUALITY_CASE,
    ROUTE_LADDER,
    ROUTE_UNBATCHED,
    STEP_LADDER,
    QualityArm,
    StepSpec,
    arm_name,
    engine_keying,
    ladder,
    on_route,
    qualified,
    recommend_route,
    rungs_to_build,
    step_gain,
    swap_summary,
)
from bench.selective import background_check


def arm(steps=1, batched=True, ms=20.0, adherence=0.5, cached=True,
        swap=6.0, identical=48, frames=48, loaded=True) -> QualityArm:
    keying = engine_keying(CASES[QUALITY_CASE], StepSpec(steps, batched))
    return QualityArm(
        steps=steps, use_denoising_batch=batched, t_index_list=[30] * steps,
        unet_batch=keying.unet_batch, engine_dir=keying.directory,
        engine_cached=cached, keys_new_engine=keying.keys_new_engine,
        swap_seconds=swap, ms_per_frame=ms,
        adherence_hits=int(round(adherence * frames)), adherence_frames=frames,
        adherence_conf=0.4, retained_hits=0, region_change=20.0,
        control_change=1.0, flicker=1.5, response=4.0,
        background=background_check([0] * identical + [7] * (frames - identical),
                                    100000),
        frames=frames, loaded=loaded)


# --- the two routes and the engines they key ---------------------------------

def test_the_unbatched_route_keys_no_new_engine_at_any_rung():
    """The whole of route B in one assertion: with `use_denoising_batch` off the
    UNet batch is the frame buffer whatever the step count, so every rung runs on
    the engine the app already ships."""
    case = CASES[QUALITY_CASE]
    shipped = engine_keying(case, StepSpec(1, True))
    for steps in STEP_LADDER:
        keying = engine_keying(case, StepSpec(steps, False))
        assert keying.unet_batch == shipped.unet_batch
        assert keying.directory == shipped.directory
        assert keying.keys_new_engine is False


@pytest.mark.parametrize("steps", [s for s in STEP_LADDER if s > 1])
def test_every_deeper_batched_rung_is_its_own_engine(steps):
    """And the whole of route A: each rung is a distinct ~5 GB build."""
    case = CASES[QUALITY_CASE]
    keying = engine_keying(case, StepSpec(steps, True))
    assert keying.unet_batch == steps
    assert keying.keys_new_engine is True
    assert keying.directory != engine_keying(case, StepSpec(1, True)).directory


def test_the_one_step_batched_arm_is_the_shipped_engine():
    """The control has to be the configuration every committed figure belongs to,
    or the sweep is measuring against something the app does not run."""
    keying = engine_keying(CASES[QUALITY_CASE], StepSpec(1, True))
    assert keying.keys_new_engine is False
    assert "max_batch-1" in keying.directory


def test_an_arm_is_named_after_its_count_and_its_route():
    assert arm_name(StepSpec(4, True)) == "s4"
    assert arm_name(StepSpec(4, False)) == "s4-nobatch"


def test_the_ladder_sweeps_both_routes_over_the_same_rungs():
    specs = ladder()
    assert len(specs) == 2 * len(STEP_LADDER)
    for route in (True, False):
        assert [spec.steps for spec in specs
                if spec.use_denoising_batch is route] == list(STEP_LADDER)


# --- the cached-engine swap (the Gate's second item) -------------------------

def test_a_swap_figure_is_read_only_from_arms_whose_engine_already_existed():
    """An arm that compiled its engine measured a *build*, which is the other
    number - averaging the two would produce a figure that is neither."""
    summary = swap_summary([arm(steps=1, cached=True, swap=5.0),
                            arm(steps=2, cached=False, swap=300.0)])
    assert summary.arms == 1
    assert summary.mean_seconds == pytest.approx(5.0)
    assert summary.uncached_arms == 1
    assert "5.0 s" in summary.statement


def test_a_run_where_nothing_was_cached_quotes_no_swap_figure():
    """The issue's third trap: nothing goes in the window until it was measured."""
    summary = swap_summary([arm(cached=False, swap=300.0)])
    assert summary.measured is False
    assert summary.mean_seconds == 0.0
    assert "not measured" in summary.statement


def test_the_worst_swap_is_carried_beside_the_mean():
    summary = swap_summary([arm(steps=1, swap=4.0), arm(steps=2, swap=9.0)])
    assert summary.worst_seconds == pytest.approx(9.0)


# --- what more steps bought ---------------------------------------------------

def test_a_deeper_rung_that_reads_back_no_better_is_not_worth_the_frame_rate():
    arms = [arm(steps=1, adherence=0.5), arm(steps=4, adherence=0.52)]
    gain = step_gain(arms, ROUTE_LADDER)
    assert gain.worthwhile is False
    assert "not been shown" in gain.statement


def test_a_deeper_rung_that_clears_the_bar_is():
    arms = [arm(steps=1, adherence=0.35),
            arm(steps=4, adherence=0.35 + MIN_ADHERENCE_GAIN)]
    gain = step_gain(arms, ROUTE_LADDER)
    assert gain.worthwhile is True
    assert gain.best_steps == 4


def test_a_route_with_no_one_step_control_scores_no_gain():
    """A gain measured against nothing is a number invented rather than measured."""
    assert step_gain([arm(steps=4)], ROUTE_LADDER) is None


def test_an_arm_that_painted_outside_the_region_is_disqualified_before_anything():
    """The Gate's last item, as a disqualifier rather than a caveat."""
    arms = [arm(steps=1), arm(steps=4, identical=47)]
    assert [one.steps for one in qualified(arms)] == [1]
    assert step_gain(arms, ROUTE_LADDER) is None


# --- the recommendation (step 4) ---------------------------------------------

def test_the_route_that_reaches_the_deeper_affordable_rung_wins():
    arms = [arm(steps=1, batched=True, ms=18.0), arm(steps=4, batched=True, ms=25.0),
            arm(steps=1, batched=False, ms=18.0), arm(steps=4, batched=False, ms=70.0)]
    recommendation = recommend_route(arms)
    assert recommendation.route == ROUTE_LADDER
    assert recommendation.deepest_affordable_steps == 4
    assert recommendation.other_deepest_steps == 1


def test_a_tie_breaks_towards_the_route_that_compiles_nothing():
    """Two routes that reach the same rung are not worth ~5 GB apiece."""
    arms = [arm(steps=1, batched=True, ms=18.0), arm(steps=4, batched=True, ms=90.0),
            arm(steps=1, batched=False, ms=18.0), arm(steps=4, batched=False, ms=90.0)]
    recommendation = recommend_route(arms)
    assert recommendation.route == ROUTE_UNBATCHED
    assert recommendation.engines_to_build == 0
    assert recommendation.rungs_shipped == []


def test_the_budget_leaves_room_for_the_detector():
    """An arm that fills the whole 33.33 ms has not been shown to fit the path the
    app runs - spec 8.8 amortises detection on top of it."""
    assert arm(ms=FRAME_BUDGET_MS - DETECTION_ALLOWANCE_MS).fits_budget is True
    assert arm(ms=FRAME_BUDGET_MS - DETECTION_ALLOWANCE_MS + 0.1).fits_budget is False


def test_a_run_missing_one_route_recommends_nothing():
    assert recommend_route([arm(steps=1, batched=True)]) is None


def test_the_rungs_a_release_would_build_are_read_off_the_arms():
    """Step 5 of the issue: which rungs get built, answered from the sweep rather
    than chosen."""
    arms = [arm(steps=steps, batched=True) for steps in STEP_LADDER]
    assert rungs_to_build(arms, 4) == [1, 2, 4]
    assert rungs_to_build(arms, 1) == [1]


# --- the case ------------------------------------------------------------------

def test_the_case_renders_through_the_shipped_tensorrt_cell():
    """The engine question is a TensorRT question; an arm on `none` would price a
    path no release ships."""
    from bench.scenarios import SCENARIOS

    scenario = SCENARIOS[CASES[QUALITY_CASE].base_scenario]
    assert scenario.acceleration == "tensorrt"
    assert scenario.batch_size == 1


def test_the_case_plan_validates_through_the_shipped_producer():
    plan = CASES[QUALITY_CASE].plan()
    assert plan.honoured_target.concept == CASES[QUALITY_CASE].concept
    assert plan.effective_denoise == CASES[QUALITY_CASE].denoise
    assert plan.honoured_target.max_instances == 1


def test_the_denoise_is_the_app_s_own_default():
    """The issue's fifth trap: the arms differ in the step count and in nothing
    else, and the strength they are all held at is the one the app renders at."""
    from render_plan import DEFAULT_DENOISE

    assert CASES[QUALITY_CASE].denoise == DEFAULT_DENOISE


def test_the_identity_probe_uses_the_same_confidence_as_the_other_two_records():
    from bench.guidance import ADHERENCE_CONF as GUIDANCE_CONF

    assert ADHERENCE_CONF == GUIDANCE_CONF


def test_an_arm_can_be_round_tripped_through_its_record():
    original = arm(steps=4, batched=False)
    assert QualityArm.from_dict(original.to_dict()) == original


def test_a_case_is_serialisable_and_carries_its_arms():
    data = CASES[QUALITY_CASE].to_dict()
    assert [spec["steps"] for spec in data["arms"]] == list(STEP_LADDER) * 2
    assert dataclasses.is_dataclass(CASES[QUALITY_CASE])


def test_arms_are_grouped_by_route_in_rung_order():
    arms = [arm(steps=4, batched=True), arm(steps=1, batched=True),
            arm(steps=2, batched=False)]
    assert [one.steps for one in on_route(arms, ROUTE_LADDER)] == [1, 4]
    assert [one.steps for one in on_route(arms, ROUTE_UNBATCHED)] == [2]
