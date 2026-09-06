"""`_control_transition` (main_gpu_addon.py) - the pure core of the control_queue drain.

The worker's frame loop drains `control_queue` and turns each message into new
state. The decision half of that is pure: message + current t_index_list in,
state delta out. The GPU-side effects stay in the loop.
"""

import pytest

from sourceloader import load_symbols

# _clamp_t_index is not referenced directly below, but _control_transition resolves
# it out of this namespace at call time - drop it from the list and the tests break.
_symbols = load_symbols(
    "main_gpu_addon.py",
    ["T_INDEX_MIN", "T_INDEX_MAX", "_clamp_t_index", "_control_transition"],
)
T_INDEX_MIN = _symbols["T_INDEX_MIN"]
T_INDEX_MAX = _symbols["T_INDEX_MAX"]
_control_transition = _symbols["_control_transition"]

CURRENT = [10, 20, 30]


@pytest.mark.parametrize(
    "msg",
    [
        None,
        "set_prompt",
        ["set_prompt"],
        {},
        {"type": "not_a_real_message"},
    ],
)
def test_junk_messages_change_nothing(msg):
    assert _control_transition(msg, CURRENT) == {}


def test_set_region_normalises_to_ints():
    delta = _control_transition(
        {"type": "set_region", "region": {"left": "10", "top": 20.7, "width": 512, "height": 512}},
        CURRENT,
    )
    assert delta == {"region": {"left": 10, "top": 20, "width": 512, "height": 512}}


@pytest.mark.parametrize(
    "region",
    [None, "512x512", {"left": 0, "top": 0, "width": 512}],
)
def test_an_incomplete_region_is_ignored(region):
    assert _control_transition({"type": "set_region", "region": region}, CURRENT) == {}


def test_set_t_at_replaces_one_step_without_an_engine_swap():
    delta = _control_transition({"type": "set_t_at", "index": 1, "value": 25}, CURRENT)
    assert delta == {"t_index_list": [10, 25, 30], "engine_swap": False}


def test_set_t_at_clamps_to_the_scheduler_range():
    assert _control_transition({"type": "set_t_at", "index": 0, "value": 999}, CURRENT)[
        "t_index_list"
    ] == [T_INDEX_MAX, 20, 30]
    assert _control_transition({"type": "set_t_at", "index": 0, "value": -5}, CURRENT)[
        "t_index_list"
    ] == [T_INDEX_MIN, 20, 30]


@pytest.mark.parametrize("index", [-1, 3, 99])
def test_set_t_at_out_of_range_is_ignored(index):
    assert _control_transition({"type": "set_t_at", "index": index, "value": 25}, CURRENT) == {}


def test_set_t_at_never_mutates_the_caller_list():
    current = list(CURRENT)
    _control_transition({"type": "set_t_at", "index": 0, "value": 5}, current)
    assert current == CURRENT


def test_same_length_t_index_list_is_a_live_update():
    delta = _control_transition({"type": "set_t_index_list", "t_index_list": [5, 6, 7]}, CURRENT)
    assert delta == {"t_index_list": [5, 6, 7], "engine_swap": False}


def test_a_different_step_count_demands_an_engine_swap():
    # Step *count* keys a distinct TensorRT engine, so this one costs minutes.
    delta = _control_transition({"type": "set_t_index_list", "t_index_list": [5, 6]}, CURRENT)
    assert delta == {"t_index_list": [5, 6], "engine_swap": True}


def test_t_index_list_values_are_clamped():
    delta = _control_transition(
        {"type": "set_t_index_list", "t_index_list": [0, 50.9, "30"]}, CURRENT
    )
    assert delta["t_index_list"] == [T_INDEX_MIN, T_INDEX_MAX, 30]


@pytest.mark.parametrize("t_index_list", [None, [], (), "30", 30])
def test_an_empty_or_malformed_t_index_list_is_ignored(t_index_list):
    msg = {"type": "set_t_index_list", "t_index_list": t_index_list}
    assert _control_transition(msg, CURRENT) == {}


def test_set_prompt_and_set_negative_prompt():
    assert _control_transition({"type": "set_prompt", "prompt": "a cat"}, CURRENT) == {
        "prompt": "a cat"
    }
    assert _control_transition(
        {"type": "set_negative_prompt", "negative_prompt": "blurry"}, CURRENT
    ) == {"negative_prompt": "blurry"}


def test_a_prompt_message_with_no_payload_leaves_the_prompt_alone():
    assert _control_transition({"type": "set_prompt"}, CURRENT) == {}
    assert _control_transition({"type": "set_negative_prompt"}, CURRENT) == {}


def test_prompts_are_coerced_to_str():
    assert _control_transition({"type": "set_prompt", "prompt": 42}, CURRENT) == {"prompt": "42"}
