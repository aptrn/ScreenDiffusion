"""The GUI's target and style fields as a Render Plan (issue #22, steps 2-4).

Tk is never instantiated here, which is the Gate's own condition. The mapping from
what the user typed to the control message the worker receives is a pure top-level
function, so it is executed straight out of `main_gpu_addon.py` and exercised on
plain strings - the widgets that hold those strings are the wiring test's business.

`extra_globals` stands in for the `render_plan` import the real module makes at the
top of the file.
"""

import json
from typing import Any, Dict, NamedTuple, Optional, Tuple

import pytest
from render_plan import (
    CROP,
    DEFAULT_MAX_INSTANCES,
    DEFAULT_REGION,
    MASKED,
    GLOBAL,
    MAX_CONCEPT_CHARS,
    SELECTIVE,
    plan_from_fields,
    validate_plan,
)

from sourceloader import load_symbols

_symbols = load_symbols(
    "main_gpu_addon.py",
    ["PlanUpdate", "_plan_status_line", "_plan_update_from_fields",
     "DETAIL_PRESETS", "DETAIL_ALL_OBJECTS", "DETAIL_ONE_OBJECT", "detail_plan"],
    extra_globals={"plan_from_fields": plan_from_fields, "NamedTuple": NamedTuple,
                   "Optional": Optional, "Dict": Dict, "Any": Any, "Tuple": Tuple,
                   "PLAN_MASKED": MASKED, "PLAN_CROP": CROP,
                   "DEFAULT_MAX_INSTANCES": DEFAULT_MAX_INSTANCES},
)
_plan_status_line = _symbols["_plan_status_line"]
_plan_update_from_fields = _symbols["_plan_update_from_fields"]
DETAIL_PRESETS = _symbols["DETAIL_PRESETS"]
DETAIL_ALL_OBJECTS = _symbols["DETAIL_ALL_OBJECTS"]
DETAIL_ONE_OBJECT = _symbols["DETAIL_ONE_OBJECT"]

PROMPT = "flip book animation, black and white rough sketch"
NEGATIVE = "low quality, blurry"


def _plan_of(update):
    """The plan the worker would end up rendering, through the same door it uses."""
    assert update.message is not None, update.reason
    result = validate_plan(update.message["plan"])
    assert result.plan is not None, result.reason
    return result.plan


# --- the mapping -------------------------------------------------------------


def test_a_target_and_a_style_make_a_selective_plan():
    plan = _plan_of(_plan_update_from_fields("person", "wet denim", PROMPT, NEGATIVE))
    assert plan.mode == SELECTIVE
    assert [t.concept for t in plan.targets] == ["person"]
    assert plan.targets[0].region == DEFAULT_REGION
    assert plan.effective_prompt == "wet denim"


def test_an_empty_target_is_todays_global_behaviour():
    """Step 3: clearing the target is not "restyle nothing", it is "restyle all"."""
    plan = _plan_of(_plan_update_from_fields("", "", PROMPT, NEGATIVE))
    assert plan.mode == GLOBAL
    assert plan.targets == ()
    assert plan.effective_prompt == PROMPT
    assert plan.effective_negative_prompt == NEGATIVE


@pytest.mark.parametrize("typed", ["", "   ", "\n", "\t "])
def test_whitespace_is_not_a_target(typed):
    assert _plan_of(_plan_update_from_fields(typed, "wet denim", PROMPT, "")).mode == GLOBAL


def test_a_blank_style_falls_back_to_the_prompt_box():
    """The style box is what goes to StreamDiffusion; empty, the prompt box still is.

    Without the fallback, naming a target and leaving the style blank would send an
    empty embedding to the engine - the prompt control regressed by a plan that
    means "restyle people with nothing".
    """
    plan = _plan_of(_plan_update_from_fields("person", "", PROMPT, NEGATIVE))
    assert plan.effective_prompt == PROMPT
    assert plan.targets[0].prompt == PROMPT


def test_the_negative_prompt_box_travels_with_the_plan():
    plan = _plan_of(_plan_update_from_fields("person", "wet denim", PROMPT, NEGATIVE))
    assert plan.effective_negative_prompt == NEGATIVE


def test_the_fields_are_stripped_before_they_reach_the_detector():
    plan = _plan_of(_plan_update_from_fields("  person \n", " wet denim ", PROMPT, ""))
    assert plan.targets[0].concept == "person"
    assert plan.effective_prompt == "wet denim"


# --- what crosses the queue --------------------------------------------------


def test_the_message_is_the_workers_set_plan():
    update = _plan_update_from_fields("person", "wet denim", PROMPT, NEGATIVE)
    assert update.message["type"] == "set_plan"
    assert isinstance(update.message["plan"], dict)


def test_the_plan_crosses_as_a_plain_dict():
    """The trap: the GUI process holds no RenderPlan the worker has to unpickle."""
    update = _plan_update_from_fields("person", "wet denim", PROMPT, NEGATIVE)
    assert json.loads(json.dumps(update.message)) == update.message


def test_the_worker_counts_the_version_up_from_its_own_plan():
    """Whatever version the GUI validated at, the worker's is the one that orders."""
    update = _plan_update_from_fields("person", "wet denim", PROMPT, "")
    assert validate_plan(update.message["plan"], previous_version=7).plan.plan_version == 8


# --- refusal -----------------------------------------------------------------


def test_a_refused_target_sends_nothing_and_says_why():
    """Step 4: a plan the validator will not have fails visibly, in the GUI."""
    update = _plan_update_from_fields("a " * MAX_CONCEPT_CHARS, "wet denim", PROMPT, "")
    assert update.message is None, "the GUI sent a plan its own validator refused"
    assert update.reason
    assert update.reason in update.status
    assert str(MAX_CONCEPT_CHARS) in update.status


# --- the status line ---------------------------------------------------------


def test_the_status_names_the_concept_and_the_region():
    update = _plan_update_from_fields("person", "wet denim", PROMPT, "")
    assert "person" in update.status
    assert DEFAULT_REGION in update.status


def test_the_status_says_when_nothing_is_targeted():
    status = _plan_update_from_fields("", "", PROMPT, "").status
    assert "person" not in status
    assert "whole frame" in status


def test_the_validators_notes_are_surfaced():
    """Step 4's other half: what the validator changed on the way through."""
    plan = plan_from_fields("person", "wet denim").plan
    line = _plan_status_line(plan, ("box_scale clamped to 2.0",))
    assert "box_scale clamped to 2.0" in line


# --- the detail control (issue #39, spec 8.2) --------------------------------
#
# The third field: which rendering primitive the plan asks for. `crop` gives one
# object the engine's whole 512x512 canvas, which only makes sense at K=1, so the
# two travel together and the GUI offers them as one choice rather than as two
# fields a user could set inconsistently.


def test_the_default_detail_is_what_the_app_has_always_done():
    from render_plan import DEFAULT_MAX_INSTANCES, MASKED

    plan = _plan_of(_plan_update_from_fields("person", "wet denim", PROMPT, NEGATIVE))
    assert plan.settings.primitive == MASKED
    assert plan.targets[0].max_instances == DEFAULT_MAX_INSTANCES


def test_asking_for_one_object_at_full_canvas_sends_crop_at_k_one():
    from render_plan import CROP

    plan = _plan_of(_plan_update_from_fields(
        "person", "wet denim", PROMPT, NEGATIVE, detail=DETAIL_ONE_OBJECT))
    assert plan.settings.primitive == CROP
    assert plan.targets[0].max_instances == 1, (
        "crop above one slot falls back to masked on every frame")


def test_asking_for_all_objects_is_the_masked_primitive():
    from render_plan import DEFAULT_MAX_INSTANCES, MASKED

    plan = _plan_of(_plan_update_from_fields(
        "person", "wet denim", PROMPT, NEGATIVE, detail=DETAIL_ALL_OBJECTS))
    assert plan.settings.primitive == MASKED
    assert plan.targets[0].max_instances == DEFAULT_MAX_INSTANCES


def test_an_unknown_detail_label_falls_back_rather_than_refusing():
    """The label comes from a widget a stale preference could have set; the plan
    the app has always rendered is a better answer than no plan."""
    from render_plan import MASKED

    plan = _plan_of(_plan_update_from_fields(
        "person", "wet denim", PROMPT, NEGATIVE, detail="every third object"))
    assert plan.settings.primitive == MASKED


def test_the_status_line_says_which_object_is_being_restyled():
    """"every person" and "one person at a time" are different promises, and the
    round-robin makes the second one true rather than "the first person"."""
    one = _plan_update_from_fields("person", "wet denim", PROMPT, NEGATIVE,
                                   detail=DETAIL_ONE_OBJECT)
    every = _plan_update_from_fields("person", "wet denim", PROMPT, NEGATIVE)
    assert "one person" in one.status
    assert "every person" in every.status


def test_the_detail_choice_is_ignored_by_a_global_plan():
    """`crop` names one region to spend the canvas on; a blank target names none."""
    from render_plan import GLOBAL as GLOBAL_MODE

    plan = _plan_of(_plan_update_from_fields("", "", PROMPT, NEGATIVE,
                                             detail=DETAIL_ONE_OBJECT))
    assert plan.mode == GLOBAL_MODE
