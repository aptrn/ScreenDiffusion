"""The GUI's end of the Render Plan (issue #22), checked structurally.

`StreamGUI` cannot be instantiated in this tier - importing `main_gpu_addon.py` primes
the Windows DLL search path and pulls in the whole customtkinter stack - so these
assertions read the source. What they hold up is the part the unit tests cannot see:
that an edit is debounced rather than sent per keystroke, that only a plan the
producer validated crosses the queue, that a headless `SD_DEMO_PLAN` run is not
overwritten by a blank field, and that the controls that were already there still
send what they always sent.
"""

import ast
from pathlib import Path

from sourceloader import load_symbols

SOURCE = Path(__file__).resolve().parent.parent / "main_gpu_addon.py"
TEXT = SOURCE.read_text(encoding="utf-8-sig")
TREE = ast.parse(TEXT, filename=str(SOURCE))

_DELAYS = load_symbols("main_gpu_addon.py", ["PLAN_DEBOUNCE_MS", "PROMPT_DEBOUNCE_MS"])
PLAN_DEBOUNCE_MS = _DELAYS["PLAN_DEBOUNCE_MS"]
PROMPT_DEBOUNCE_MS = _DELAYS["PROMPT_DEBOUNCE_MS"]


def _class(name: str) -> ast.ClassDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"main_gpu_addon.py defines no class {name}")


GUI = _class("StreamGUI")


def _method(name: str) -> ast.FunctionDef:
    for node in GUI.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"StreamGUI defines no {name}")


def _calls_named(node: ast.AST, name: str) -> list:
    return [call for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and getattr(call.func, "attr", getattr(call.func, "id", None)) == name]


def _mentions(node: ast.AST, name: str) -> bool:
    return any(getattr(n, "attr", getattr(n, "id", None)) == name for n in ast.walk(node))


# --- the fields --------------------------------------------------------------


def test_the_two_fields_have_variables_the_gui_holds():
    init = _method("__init__")
    assigned = {node.attr for node in ast.walk(init)
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)}
    assert {"target_var", "style_var"} <= assigned


def test_both_fields_are_built_and_bound_to_the_same_debounced_handler():
    build = _method("_build_ui")
    binds = [call for call in _calls_named(build, "bind")
             if _mentions(call, "_on_plan_field_changed")]
    assert len(binds) == 2, "target and style are not both wired to the plan handler"


def test_the_fields_stay_editable_while_generation_runs():
    """A locked target field would mean stopping the run to change what is restyled."""
    build = _method("_build_ui")
    for call in _calls_named(build, "_register_lockables"):
        for widget in ("_w_target_entry", "_w_style_entry"):
            assert not _mentions(call, widget), f"{widget} is disabled while running"


# --- the debounce ------------------------------------------------------------


def test_an_edit_is_debounced_rather_than_sent_per_keystroke():
    handler = _method("_on_plan_field_changed")
    assert _calls_named(handler, "after_cancel"), "a pending plan send is never cancelled"
    after, = _calls_named(handler, "after")
    assert _mentions(after, "PLAN_DEBOUNCE_MS")
    assert _mentions(after, "_push_plan_runtime")


def test_the_plan_waits_longer_than_a_prompt_edit_does():
    """The trap: a vocabulary change costs a ~108 ms throwaway detect (spec 8.1).

    Firing per keystroke spends three frames' budget per character, so the field
    that reloads the detector's vocabulary cannot debounce like a prompt box.
    """
    assert PLAN_DEBOUNCE_MS >= 2 * PROMPT_DEBOUNCE_MS


def test_the_prompt_boxes_still_debounce_at_their_own_delay():
    """Not regressed: the plan's longer wait is not imposed on the prompt controls."""
    for name in ("_on_prompt_changed", "_on_neg_prompt_changed"):
        after, = _calls_named(_method(name), "after")
        assert _mentions(after, "PROMPT_DEBOUNCE_MS")


# --- what is sent ------------------------------------------------------------


def test_the_send_goes_through_the_pure_mapping():
    push = _method("_push_plan_runtime")
    assert _calls_named(push, "_plan_update_from_fields"), \
        "the GUI builds its plan somewhere other than the tested mapping"


def test_only_a_validated_plan_reaches_the_queue():
    """A refusal is a status line, not a message: the worker never sees it."""
    push = _method("_push_plan_runtime")
    puts = _calls_named(push, "put_nowait")
    assert puts, "the plan never reaches control_q"
    guards = [node for node in ast.walk(push)
              if isinstance(node, ast.If) and any(put in ast.walk(node) for put in puts)]
    assert any("message" in ast.dump(node.test) for node in guards), \
        "the GUI puts something on the queue without checking there is a plan"


def test_the_status_area_carries_the_outcome():
    """Step 4: notes and rejections are read where the fields were typed."""
    push = _method("_push_plan_runtime")
    assert any(_mentions(call, "status_var") for call in _calls_named(push, "set"))


def test_a_blank_target_does_not_overwrite_a_headless_demo_plan():
    """Step 5. At start the GUI speaks only if the user asked for something.

    `SD_DEMO_PLAN` submits the priority case inside the worker; a start that sent an
    unasked-for global plan would replace it a few frames later.
    """
    start = _method("_on_start")
    pushes = _calls_named(start, "_push_plan_runtime")
    assert pushes, "a target typed before Start never reaches the worker"
    guards = [node for node in ast.walk(start)
              if isinstance(node, ast.If) and any(p in ast.walk(node) for p in pushes)]
    assert any(_mentions(node.test, "target_var") for node in guards), \
        "the GUI sends a plan at start whether or not a target was typed"


# --- the controls that were already there ------------------------------------


def test_the_existing_runtime_controls_still_send_their_messages():
    sent = {node.value for node in ast.walk(GUI)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert {"set_prompt", "set_negative_prompt", "set_region", "set_t_index_list"} <= sent
