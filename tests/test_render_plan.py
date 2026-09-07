"""The Render Plan schema and its validator (issue #6, spec section 6).

Everything here is GPU-free and import-free of the application modules: the plan is
a stdlib-only contract precisely so both processes and the merge gate can hold it.
"""

from __future__ import annotations

import pickle

import pytest

from render_plan import (
    ACTIVE_DETECTOR,
    BOX_SCALE_RANGE,
    CONFIDENCE_RANGE,
    DEFAULT_BOX_SCALE,
    DEFAULT_CONFIDENCE,
    DEFAULT_DENOISE,
    DEFAULT_DETECT_EVERY_N,
    DEFAULT_FPS_TARGET,
    DEFAULT_MAX_INSTANCES,
    DEFAULT_OUTPUT_EMA,
    DEFAULT_PRIORITY,
    DEFAULT_REGION,
    DEFAULT_SEED_POLICY,
    DENOISE_RANGE,
    DETECT_EVERY_N_RANGE,
    DETECTOR_VOCABULARIES,
    FPS_TARGET_RANGE,
    GLOBAL,
    INVERSE,
    MAX_CONCEPT_CHARS,
    MAX_INSTANCES_RANGE,
    MODES,
    OUTPUT_EMA_RANGE,
    PASSTHROUGH,
    PRIORITY_RANGE,
    REGIONS,
    SCHEDULE_T_INDEX_RANGE,
    SELECTIVE,
    STYLIZE,
    ActivePlan,
    DetectorVocabulary,
    RenderPlan,
    global_plan,
    noise_amplitude,
    plan_from_fields,
    t_index_ladder,
    validate_plan,
)

OPEN_VOCAB = DETECTOR_VOCABULARIES[ACTIVE_DETECTOR]
CLOSED_VOCAB = DETECTOR_VOCABULARIES["yolov8n-640"]


def ok(raw, **kwargs) -> RenderPlan:
    """Validate `raw` and insist it was accepted, so a test reads as its subject."""
    result = validate_plan(raw, **kwargs)
    assert result.ok, f"expected a valid plan, got {result.errors}"
    return result.plan


# --- defaults ---------------------------------------------------------------


def test_a_minimal_plan_gets_a_default_for_every_field():
    """Spec 6: "a plan carrying just targets[0].concept and .prompt must render
    something sensible"."""
    plan = ok({"targets": [{"concept": "person", "prompt": "wearing a red hat"}]})

    assert plan.mode == SELECTIVE
    assert plan.source_prompt == ""
    assert plan.confidence == DEFAULT_CONFIDENCE
    assert plan.notes == ""
    assert plan.background.action == PASSTHROUGH
    assert plan.settings.fps_target == DEFAULT_FPS_TARGET
    assert plan.settings.detect_every_n == DEFAULT_DETECT_EVERY_N
    assert plan.settings.output_ema == DEFAULT_OUTPUT_EMA

    target = plan.targets[0]
    assert target.id == "t0"
    assert target.detector_class is None
    assert target.region == DEFAULT_REGION
    assert target.box_scale == DEFAULT_BOX_SCALE
    assert target.negative_prompt == ""
    assert target.denoise == DEFAULT_DENOISE
    assert target.seed_policy == DEFAULT_SEED_POLICY
    assert target.max_instances == DEFAULT_MAX_INSTANCES
    assert target.priority == DEFAULT_PRIORITY


def test_the_empty_plan_is_valid_and_renders_the_whole_frame():
    plan = ok({})
    assert plan.mode == GLOBAL
    assert plan.targets == ()


def test_mode_defaults_to_what_the_plan_can_actually_do():
    """No target to select means nothing selective to do; a target means there is."""
    assert ok({}).mode == GLOBAL
    assert ok({"targets": [{"concept": "dog"}]}).mode == SELECTIVE


@pytest.mark.parametrize("mode", MODES)
def test_an_explicit_mode_is_honoured(mode):
    targets = [{"concept": "dog"}] if mode != GLOBAL else []
    assert ok({"mode": mode, "targets": targets}).mode == mode


def test_ids_are_assigned_by_position_when_the_producer_omits_them():
    plan = ok({"targets": [{"concept": "person"}, {"concept": "dog"}]})
    assert [t.id for t in plan.targets] == ["t0", "t1"]


def test_an_explicit_id_survives():
    plan = ok({"targets": [{"concept": "person", "id": "hat"}]})
    assert plan.targets[0].id == "hat"


# --- versioning -------------------------------------------------------------


def test_the_validator_assigns_a_monotonically_increasing_version():
    first = ok({}, previous_version=0)
    second = ok({}, previous_version=first.plan_version)
    third = ok({}, previous_version=second.plan_version)
    assert [first.plan_version, second.plan_version, third.plan_version] == [1, 2, 3]


def test_a_producer_cannot_choose_its_own_plan_version():
    """The version orders plans in the worker; a producer that set it could go backwards."""
    plan = ok({"plan_version": 99}, previous_version=4)
    assert plan.plan_version == 5


# --- clamping ---------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("box_scale", 0.1, BOX_SCALE_RANGE[0]),
        ("box_scale", 9.0, BOX_SCALE_RANGE[1]),
        ("denoise", -1.0, DENOISE_RANGE[0]),
        ("denoise", 4.0, DENOISE_RANGE[1]),
        ("max_instances", 0, MAX_INSTANCES_RANGE[0]),
        ("max_instances", 999, MAX_INSTANCES_RANGE[1]),
        ("priority", -7, PRIORITY_RANGE[0]),
        ("priority", 10_000, PRIORITY_RANGE[1]),
    ],
)
def test_out_of_range_target_numbers_are_clamped(field, value, expected):
    plan = ok({"targets": [{"concept": "person", field: value}]})
    assert getattr(plan.targets[0], field) == expected


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("fps_target", 0, FPS_TARGET_RANGE[0]),
        ("fps_target", 5000, FPS_TARGET_RANGE[1]),
        ("detect_every_n", 0, DETECT_EVERY_N_RANGE[0]),
        ("detect_every_n", 900, DETECT_EVERY_N_RANGE[1]),
        ("output_ema", -0.5, OUTPUT_EMA_RANGE[0]),
        ("output_ema", 1.0, OUTPUT_EMA_RANGE[1]),
    ],
)
def test_out_of_range_global_settings_are_clamped(field, value, expected):
    plan = ok({"global": {field: value}})
    assert getattr(plan.settings, field) == expected


@pytest.mark.parametrize("value, expected", [(-0.5, CONFIDENCE_RANGE[0]), (7, CONFIDENCE_RANGE[1])])
def test_confidence_is_clamped(value, expected):
    assert ok({"confidence": value}).confidence == expected


def test_a_clamp_is_recorded_rather_than_applied_silently():
    result = validate_plan({"targets": [{"concept": "person", "denoise": 4.0}]})
    assert any("denoise" in note for note in result.notes)


def test_whole_numbers_arrive_as_ints_and_fractions_as_floats():
    plan = ok({"targets": [{"concept": "person", "max_instances": 3.7, "priority": "2"}]})
    assert plan.targets[0].max_instances == 3 and isinstance(plan.targets[0].max_instances, int)
    assert plan.targets[0].priority == 2 and isinstance(plan.targets[0].priority, int)


def test_a_numeric_string_is_accepted_the_way_set_t_index_list_accepts_one():
    assert ok({"targets": [{"concept": "person", "denoise": "0.25"}]}).targets[0].denoise == 0.25


# --- unknown fields ---------------------------------------------------------


def test_unknown_plan_fields_are_dropped_and_noted():
    result = validate_plan({"llm_model": "mistral-7b", "targets": []})
    assert result.ok
    assert not hasattr(result.plan, "llm_model")
    assert any("llm_model" in note for note in result.notes)


def test_unknown_target_fields_are_dropped_and_noted():
    result = validate_plan({"targets": [{"concept": "person", "lora": "anime.safetensors"}]})
    assert result.ok
    assert any("lora" in note for note in result.notes)


# --- rejection --------------------------------------------------------------


def test_the_region_vocabulary_is_exactly_the_six_the_issue_fixes():
    assert REGIONS == (
        "full_box",
        "upper_third",
        "upper_half",
        "center",
        "lower_half",
        "lower_third",
    )


@pytest.mark.parametrize("region", REGIONS)
def test_every_named_region_is_accepted(region):
    assert ok({"targets": [{"concept": "person", "region": region}]}).targets[0].region == region


@pytest.mark.parametrize("region", ["head", "UPPER_THIRD", "", 3])
def test_a_region_outside_the_vocabulary_is_rejected_with_the_vocabulary(region):
    result = validate_plan({"targets": [{"concept": "person", "region": region}]})
    assert not result.ok
    assert "region" in result.reason
    assert "upper_third" in result.reason


def test_a_null_field_means_unspecified_rather_than_invalid():
    """JSON's way of saying "I have no opinion". Every field defaults on None."""
    plan = ok({"targets": [{"concept": "person", "region": None, "denoise": None}]})
    assert plan.targets[0].region == DEFAULT_REGION
    assert plan.targets[0].denoise == DEFAULT_DENOISE


@pytest.mark.parametrize("raw", [None, "a plan", 7, ["targets"]])
def test_a_plan_that_is_not_a_mapping_is_rejected_with_a_reason(raw):
    result = validate_plan(raw)
    assert not result.ok and result.plan is None
    assert result.reason


@pytest.mark.parametrize("targets", ["person", 3, {"concept": "person"}])
def test_targets_must_be_a_list(targets):
    result = validate_plan({"targets": targets})
    assert not result.ok
    assert "targets" in result.reason


def test_a_target_that_is_not_a_mapping_is_rejected():
    result = validate_plan({"targets": ["person"]})
    assert not result.ok
    assert result.reason


@pytest.mark.parametrize("mode", ["everything", "Global", "", 2])
def test_an_unknown_mode_is_rejected(mode):
    result = validate_plan({"mode": mode, "targets": [{"concept": "dog"}]})
    assert not result.ok
    assert "mode" in result.reason


@pytest.mark.parametrize("mode", [SELECTIVE, INVERSE])
def test_a_selecting_mode_with_nothing_to_select_is_rejected(mode):
    result = validate_plan({"mode": mode, "targets": []})
    assert not result.ok
    assert mode in result.reason


def test_an_unparseable_number_is_a_malformed_plan_not_a_default():
    """A producer that emits "quite a lot" for a denoise is broken; rendering a
    default would hide that behind a plausible frame."""
    result = validate_plan({"targets": [{"concept": "person", "denoise": "quite a lot"}]})
    assert not result.ok
    assert "denoise" in result.reason


def test_duplicate_target_ids_are_rejected():
    result = validate_plan(
        {"targets": [{"concept": "person", "id": "t0"}, {"concept": "dog", "id": "t0"}]}
    )
    assert not result.ok
    assert "t0" in result.reason


def test_an_unknown_seed_policy_is_rejected():
    result = validate_plan({"targets": [{"concept": "person", "seed_policy": "per_frame"}]})
    assert not result.ok
    assert "seed_policy" in result.reason


def test_an_unknown_background_action_is_rejected():
    result = validate_plan({"background": {"action": "delete"}})
    assert not result.ok
    assert "background" in result.reason


def test_a_stylized_background_is_accepted():
    plan = ok({"background": {"action": STYLIZE, "prompt": "an oil painting"}})
    assert plan.background.action == STYLIZE
    assert plan.background.prompt == "an oil painting"


# --- what the detector can serve --------------------------------------------


def test_an_open_vocabulary_detector_refuses_no_concept():
    assert OPEN_VOCAB.refusal("a chipped enamel mug") is None


def test_a_closed_vocabulary_detector_states_why_it_cannot_serve_a_concept():
    reason = CLOSED_VOCAB.refusal("red mug")
    assert reason and "red mug" in reason and CLOSED_VOCAB.name in reason
    assert CLOSED_VOCAB.refusal("person") is None


def test_a_concept_the_detector_cannot_serve_drops_its_target_with_a_reason():
    result = validate_plan(
        {"targets": [{"concept": "person"}, {"concept": "red mug"}]}, detector=CLOSED_VOCAB
    )
    assert result.ok
    assert [t.concept for t in result.plan.targets] == ["person"]
    assert any("red mug" in note for note in result.notes)


def test_a_plan_whose_every_concept_is_unservable_is_rejected():
    result = validate_plan({"targets": [{"concept": "red mug"}]}, detector=CLOSED_VOCAB)
    assert not result.ok
    assert "red mug" in result.reason


@pytest.mark.parametrize("concept", ["", "   ", None, 4])
def test_a_target_with_no_concept_is_not_a_target(concept):
    result = validate_plan({"targets": [{"concept": concept}]})
    assert not result.ok
    assert "concept" in result.reason


def test_a_concept_longer_than_the_text_encoder_can_hold_is_rejected():
    result = validate_plan({"targets": [{"concept": "x" * (MAX_CONCEPT_CHARS + 1)}]})
    assert not result.ok
    assert "concept" in result.reason


def test_the_detector_registry_mirrors_the_benchmarked_one():
    """`bench.detectors` measures them and this decides what they can be asked for.
    Two registries, one set of names - a detector in one and not the other is a bug."""
    from bench.detectors import DETECTORS

    assert set(DETECTOR_VOCABULARIES) == set(DETECTORS)
    for name, vocabulary in DETECTOR_VOCABULARIES.items():
        assert vocabulary.open_vocabulary == DETECTORS[name].open_vocabulary
    assert ACTIVE_DETECTOR in DETECTOR_VOCABULARIES


def test_the_active_detector_is_the_one_the_evaluation_chose():
    from bench.detectors import PRIMARY_DETECTOR

    assert ACTIVE_DETECTOR == PRIMARY_DETECTOR


# --- what the engine actually honours ---------------------------------------


def test_the_effective_prompt_is_the_first_targets():
    plan = ok({"source_prompt": "make them fancy", "targets": [{"concept": "person", "prompt": "a red hat"}]})
    assert plan.effective_prompt == "a red hat"


def test_a_target_without_a_prompt_falls_back_to_the_source_prompt():
    plan = ok({"source_prompt": "an oil painting", "targets": [{"concept": "person"}]})
    assert plan.effective_prompt == "an oil painting"


def test_a_plan_with_no_targets_renders_its_source_prompt():
    plan = ok({"source_prompt": "an oil painting"})
    assert plan.effective_prompt == "an oil painting"
    assert plan.honoured_target is None


def test_only_the_first_targets_prompt_and_denoise_reach_the_engine():
    """Issue #5 chose primitive B: one masked full-frame pass, one prompt embedding.
    The other targets' values are carried, not applied - and the plan says so."""
    result = validate_plan(
        {
            "targets": [
                {"concept": "person", "prompt": "a red hat", "denoise": 0.4},
                {"concept": "dog", "prompt": "a blue hat", "denoise": 0.9},
            ]
        }
    )
    assert result.ok
    plan = result.plan
    assert plan.effective_prompt == "a red hat"
    assert plan.effective_denoise == 0.4
    assert plan.honoured_target.id == "t0"
    assert len(plan.targets) == 2, "the unhonoured target is kept in the schema"
    assert any("first target" in note for note in result.notes)


def test_targets_that_agree_with_the_first_raise_no_note():
    result = validate_plan(
        {
            "targets": [
                {"concept": "person", "prompt": "a red hat", "denoise": 0.4},
                {"concept": "dog", "prompt": "a red hat", "denoise": 0.4},
            ]
        }
    )
    assert not any("first target" in note for note in result.notes)


def test_the_negative_prompt_follows_the_same_rule():
    plan = ok({"targets": [{"concept": "person", "negative_prompt": "blurry"}]})
    assert plan.effective_negative_prompt == "blurry"


# --- crossing the process boundary ------------------------------------------


def test_a_plan_survives_a_pickle_round_trip():
    """Plans cross a `multiprocessing.Queue`, so every field must be picklable."""
    plan = ok({"targets": [{"concept": "person", "prompt": "a red hat"}], "notes": "assumed head"})
    assert pickle.loads(pickle.dumps(plan)) == plan


def test_a_plan_dict_survives_a_pickle_round_trip():
    plan = ok({"targets": [{"concept": "person"}]})
    assert pickle.loads(pickle.dumps(plan.to_dict())) == plan.to_dict()


def test_a_plan_round_trips_through_its_own_dict():
    """to_dict is the wire form: what it emits must validate back to the same plan."""
    plan = ok(
        {
            "source_prompt": "find all people and give them a red hat",
            "mode": SELECTIVE,
            "targets": [
                {
                    "concept": "person",
                    "region": "upper_third",
                    "box_scale": 1.2,
                    "prompt": "wearing a vibrant red hat",
                    "negative_prompt": "blurry",
                    "denoise": 0.45,
                    "seed_policy": "fixed",
                    "max_instances": 4,
                    "priority": 2,
                    "detector_class": 0,
                }
            ],
            "background": {"action": STYLIZE, "prompt": "a watercolour"},
            "global": {"fps_target": 24, "detect_every_n": 5},
            "confidence": 0.8,
            "notes": "assumed head region only",
        }
    )
    again = ok(plan.to_dict(), previous_version=plan.plan_version)
    assert again.to_dict() == dict(plan.to_dict(), plan_version=plan.plan_version + 1)


def test_the_wire_form_uses_the_spec_key_for_the_global_block():
    """`global` is a Python keyword, so the field is `settings` - the wire key is not."""
    data = ok({}).to_dict()
    assert "global" in data and "settings" not in data


# --- the GUI as producer ----------------------------------------------------


def test_the_gui_builds_a_selective_plan_from_its_two_fields():
    result = plan_from_fields(target="every person", style="wearing a red hat")
    assert result.ok
    plan = result.plan
    assert plan.mode == SELECTIVE
    assert plan.targets[0].concept == "every person"
    assert plan.targets[0].prompt == "wearing a red hat"
    assert plan.effective_prompt == "wearing a red hat"


def test_an_empty_target_field_means_restyle_everything():
    """The style field alone is today's app: one prompt over the whole frame."""
    result = plan_from_fields(target="  ", style="an oil painting")
    assert result.ok
    assert result.plan.mode == GLOBAL
    assert result.plan.targets == ()
    assert result.plan.effective_prompt == "an oil painting"


def test_the_producer_passes_its_fields_through_the_validator():
    result = plan_from_fields(target="person", style="a red hat", region="nose")
    assert not result.ok and "region" in result.reason


def test_the_producer_counts_up_from_the_plan_it_is_replacing():
    first = plan_from_fields(target="person", style="a red hat").plan
    second = plan_from_fields(target="dog", style="a cat", previous_version=first.plan_version).plan
    assert second.plan_version == first.plan_version + 1


def test_a_global_plan_reproduces_the_prompt_path():
    plan = global_plan("a photograph of a city street", "blurry, deformed")
    assert plan.mode == GLOBAL
    assert plan.effective_prompt == "a photograph of a city street"
    assert plan.effective_negative_prompt == "blurry, deformed"


def test_an_unservable_target_is_refused_to_the_producer_not_silently_widened():
    result = plan_from_fields(target="red mug", style="a red hat", detector=CLOSED_VOCAB)
    assert not result.ok
    assert "red mug" in result.reason


# --- the active plan --------------------------------------------------------


def one_plan(version: int) -> RenderPlan:
    return global_plan(f"prompt {version}", previous_version=version - 1)


def test_the_frame_sees_the_plan_that_was_active_when_it_started():
    active = ActivePlan(one_plan(1))
    frame = active.begin_frame()
    active.submit(one_plan(2))
    assert active.frame_plan is frame.plan
    assert active.frame_plan.plan_version == 1


def test_a_submitted_plan_takes_effect_at_the_next_frame_boundary():
    active = ActivePlan(one_plan(1))
    active.begin_frame()
    active.submit(one_plan(2))
    assert active.begin_frame().plan.plan_version == 2


def test_the_first_frame_of_a_new_plan_is_the_one_that_changed():
    active = ActivePlan(one_plan(1))
    assert active.begin_frame().changed is False, "the startup plan is already applied"
    active.submit(one_plan(2))
    assert active.begin_frame().changed is True
    assert active.begin_frame().changed is False


def test_only_the_newest_of_several_submissions_is_ever_rendered():
    """Between two frames the drain can take a burst; the frame renders one plan."""
    active = ActivePlan(one_plan(1))
    active.begin_frame()
    for version in (2, 3, 4):
        active.submit(one_plan(version))
    frame = active.begin_frame()
    assert frame.plan.plan_version == 4 and frame.changed


def test_the_latest_submitted_version_is_what_the_next_plan_counts_from():
    active = ActivePlan(one_plan(1))
    active.submit(one_plan(2))
    assert active.latest.plan_version == 2


# --- the output EMA (issue #32) ---------------------------------------------


def test_the_output_ema_is_off_by_default():
    """A stability lever nobody measured yet is not something a plan opts out of."""
    assert DEFAULT_OUTPUT_EMA == 0.0
    assert ok({}).settings.output_ema == 0.0


def test_the_output_ema_is_clamped_below_one():
    """At 1.0 the output is the previous output for ever - a frozen frame, not a
    steadier one - so the range stops short of it and the clamp is recorded."""
    assert OUTPUT_EMA_RANGE[1] < 1.0
    result = validate_plan({"global": {"output_ema": 1.0}})
    assert result.plan.settings.output_ema == OUTPUT_EMA_RANGE[1]
    assert any("output_ema" in note for note in result.notes)


def test_an_output_ema_that_is_not_a_number_is_refused():
    result = validate_plan({"global": {"output_ema": "smooth"}})
    assert not result.ok
    assert "output_ema" in result.reason


# --- the rendering primitive (issue #39, spec 8.2) --------------------------
#
# `global.primitive` is the lever that gives one object the whole 512x512 canvas
# instead of the fraction of it the frame's own downscale leaves. It is a plan
# field rather than a harness flag for the reason `output_ema` is: the worker has
# to be able to be put into it from the producer the GUI already sends.


def test_the_shipped_primitive_is_masked():
    """What issue #5 chose and every committed baseline was measured under."""
    from render_plan import DEFAULT_PRIMITIVE, MASKED

    assert DEFAULT_PRIMITIVE == MASKED
    assert ok({}).settings.primitive == MASKED


def test_a_plan_can_ask_for_crop():
    from render_plan import CROP

    assert ok({"global": {"primitive": "crop"}}).settings.primitive == CROP


def test_a_primitive_outside_the_vocabulary_is_refused():
    result = validate_plan({"global": {"primitive": "inpaint"}})
    assert result.plan is None
    assert "primitive" in result.reason


def test_crop_with_more_than_one_slot_is_noted_rather_than_refused():
    """Crop is one diffusion call per region, so a frame that selects six of them
    costs six. The plan is still renderable - the compositor falls back per frame -
    but a producer that asked for it is told."""
    result = validate_plan({
        "global": {"primitive": "crop"},
        "targets": [{"id": "t0", "concept": "person", "max_instances": 6}],
    })
    assert result.plan is not None
    assert any("crop" in note and "max_instances" in note for note in result.notes)


def test_crop_at_one_slot_is_noted_about_nothing():
    result = validate_plan({
        "global": {"primitive": "crop"},
        "targets": [{"id": "t0", "concept": "person", "max_instances": 1}],
    })
    assert result.plan is not None
    assert not any("max_instances" in note for note in result.notes)


# --- the step ladder a multi-step base model needs (issue #38) ---------------


def test_one_step_ladder_is_the_index_the_denoise_chose():
    """The shipped path. One step is `[first]` and nothing else."""
    assert t_index_ladder(30, 1) == [30]


def test_a_four_step_ladder_opens_where_the_denoise_asked():
    """`denoise` still picks the opening index; the extra steps are spent after it."""
    ladder = t_index_ladder(30, 4)
    assert ladder[0] == 30
    assert len(ladder) == 4


def test_a_ladder_descends_in_noise_towards_the_end_of_the_schedule():
    """A higher index is *less* noise, so a denoising trajectory ascends in index."""
    ladder = t_index_ladder(20, 4)
    assert ladder == sorted(ladder)
    assert ladder[-1] == SCHEDULE_T_INDEX_RANGE[1]
    assert all(noise_amplitude(later) <= noise_amplitude(earlier)
               for earlier, later in zip(ladder, ladder[1:]))


def test_a_ladder_stays_inside_the_schedule_range():
    for first in (1, 25, 49):
        for steps in (1, 2, 4, 8):
            ladder = t_index_ladder(first, steps)
            assert len(ladder) == steps
            low, high = SCHEDULE_T_INDEX_RANGE
            assert all(low <= index <= high for index in ladder)


def test_a_ladder_with_no_room_left_repeats_rather_than_dropping_a_step():
    """The step *count* keys a TensorRT engine, so it is not the ladder's to change.

    A plan asking for four steps at a denoise that lands on the last index has
    nowhere to spend three of them; repeating the index is the honest answer, and
    silently returning two would build a different engine than the one asked for.
    """
    ladder = t_index_ladder(SCHEDULE_T_INDEX_RANGE[1], 4)
    assert ladder == [SCHEDULE_T_INDEX_RANGE[1]] * 4
