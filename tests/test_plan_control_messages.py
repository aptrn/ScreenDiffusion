"""`set_plan` on the worker's control_queue (issue #6, steps 4 and 5).

The plan reaches the worker as a plain dict - a control message like any other - and
`_control_transition` turns it into a validated `RenderPlan` or into the reason there
is not one. The GPU-side effects, including the swap itself, stay in the frame loop.

`_control_transition` is executed straight out of `main_gpu_addon.py`, so its module
scope has to be supplied: `extra_globals` stands in for the render_plan import the
real module makes at the top of the file.
"""

import pytest
from render_plan import (
    GLOBAL,
    INITIAL_PLAN_VERSION,
    SELECTIVE,
    RenderPlan,
    global_plan,
    validate_plan,
)

from sourceloader import load_symbols

_symbols = load_symbols(
    "main_gpu_addon.py",
    ["T_INDEX_MIN", "T_INDEX_MAX", "_clamp_t_index", "_control_transition"],
    extra_globals={"validate_plan": validate_plan, "INITIAL_PLAN_VERSION": INITIAL_PLAN_VERSION},
)
_control_transition = _symbols["_control_transition"]

CURRENT = [10, 20, 30]
ACTIVE = global_plan("a photograph of a city street")


def test_a_valid_plan_arrives_as_a_render_plan():
    update = _control_transition(
        {"type": "set_plan", "plan": {"targets": [{"concept": "person", "prompt": "a red hat"}]}},
        CURRENT,
        ACTIVE,
    )
    assert isinstance(update["plan"], RenderPlan)
    assert update["plan"].mode == SELECTIVE
    assert update["plan"].effective_prompt == "a red hat"


def test_the_new_plan_counts_up_from_the_active_one():
    update = _control_transition({"type": "set_plan", "plan": {}}, CURRENT, ACTIVE)
    assert update["plan"].plan_version == ACTIVE.plan_version + 1


def test_the_first_plan_counts_from_the_initial_version():
    """No active plan yet - the worker has not built one. Still monotonic."""
    update = _control_transition({"type": "set_plan", "plan": {}}, CURRENT)
    assert update["plan"].plan_version == INITIAL_PLAN_VERSION + 1


def test_a_rejected_plan_comes_back_as_a_reason_and_no_plan():
    update = _control_transition(
        {"type": "set_plan", "plan": {"targets": [{"concept": "person", "region": "head"}]}},
        CURRENT,
        ACTIVE,
    )
    assert "plan" not in update
    assert "region" in update["plan_error"]


@pytest.mark.parametrize("plan", [None, "a plan", 7])
def test_a_plan_that_is_not_a_plan_is_refused_rather_than_ignored(plan):
    update = _control_transition({"type": "set_plan", "plan": plan}, CURRENT, ACTIVE)
    assert update["plan_error"]


def test_a_set_plan_message_carrying_no_plan_changes_nothing():
    assert _control_transition({"type": "set_plan"}, CURRENT, ACTIVE) == {}


def test_the_validators_notes_travel_with_the_plan_under_their_own_key():
    """Not "notes": `RenderPlan.notes` is the plan's own field and a different thing."""
    update = _control_transition(
        {"type": "set_plan", "plan": {"targets": [{"concept": "person", "denoise": 9}]}},
        CURRENT,
        ACTIVE,
    )
    assert any("denoise" in note for note in update["plan_notes"])


def test_a_global_plan_carries_exactly_what_set_prompt_carries():
    """Step 5: `mode: "global"` with one prompt is today's behaviour, unchanged.

    The GPU test asserts the pixels; this asserts the two paths agree on the string,
    which is the only thing either of them hands the engine.
    """
    prompt = "a photograph of a city street, cinematic lighting"
    by_prompt = _control_transition({"type": "set_prompt", "prompt": prompt}, CURRENT)
    by_plan = _control_transition(
        {"type": "set_plan", "plan": {"mode": GLOBAL, "source_prompt": prompt}},
        CURRENT,
        ACTIVE,
    )
    assert by_plan["plan"].effective_prompt == by_prompt["prompt"]


def test_set_plan_never_touches_the_step_schedule():
    """A plan swap must not look like an engine rebuild - that costs minutes."""
    update = _control_transition({"type": "set_plan", "plan": {}}, CURRENT, ACTIVE)
    assert "t_index_list" not in update and "engine_swap" not in update


@pytest.mark.parametrize(
    "msg, expected",
    [
        ({"type": "set_prompt", "prompt": "a cat"}, {"prompt": "a cat"}),
        ({"type": "set_negative_prompt", "negative_prompt": "blurry"},
         {"negative_prompt": "blurry"}),
        ({"type": "set_t_index_list", "t_index_list": [5, 6, 7]},
         {"t_index_list": [5, 6, 7], "engine_swap": False}),
        ({"type": "set_region", "region": {"left": 0, "top": 0, "width": 8, "height": 8}},
         {"region": {"left": 0, "top": 0, "width": 8, "height": 8}}),
    ],
)
def test_the_existing_messages_are_unchanged_by_the_new_argument(msg, expected):
    assert _control_transition(msg, CURRENT, ACTIVE) == expected
    assert _control_transition(msg, CURRENT) == expected
