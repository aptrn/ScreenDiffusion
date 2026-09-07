"""What the window says about the plan, and what it stops shouting about.

Issue #40, steps 2, 3 and 5. The object-aware selective path is the active project
and the app gave it one blank field below the preview, among the engine knobs, with
no sign anywhere of whether detection was running or on what. The two fields are
the product's interface (spec 6, issue #22 - there is no LLM producer in v1), so
they lead, the engine knobs move behind a collapsed "Advanced" section, and the
plan's state is a sentence rather than the tail of the FPS line.

The sentences are pure functions and are executed; where they reach the widgets is
read off the source through `guisource`, because `StreamGUI` cannot be instantiated
in this tier.
"""

import ast
from typing import Any, Dict, NamedTuple, Optional, Sequence

import pytest
from render_plan import plan_from_fields

from guisource import assignment_to, calls_named, gui_method, mentions
from sourceloader import load_symbols

CUSTOM_COLORS = {"success": "#10B981", "error": "#EF4444", "surface": "#374151"}

_symbols = load_symbols(
    "main_gpu_addon.py",
    ["ADVANCED", "GLOBAL_STATE", "PLAN_NOTE_COLOR", "PlanUpdate", "SHOW", "_plan_note",
     "_plan_state_line", "_plan_status_line", "_plan_update_from_fields"],
    extra_globals={"plan_from_fields": plan_from_fields, "NamedTuple": NamedTuple,
                   "Optional": Optional, "Dict": Dict, "Any": Any,
                   "Sequence": Sequence, "CUSTOM_COLORS": CUSTOM_COLORS},
)
ADVANCED = _symbols["ADVANCED"]
PLAN_NOTE_COLOR = _symbols["PLAN_NOTE_COLOR"]
SHOW = _symbols["SHOW"]
_plan_note = _symbols["_plan_note"]
_plan_state_line = _symbols["_plan_state_line"]
_plan_update_from_fields = _symbols["_plan_update_from_fields"]

PROMPT = "flip book animation, black and white rough sketch"


def _first_arg(call: ast.AST) -> str:
    """A Tk widget's parent is its first positional argument, by name."""
    assert isinstance(call, ast.Call) and call.args, ast.dump(call)
    return getattr(call.args[0], "id", getattr(call.args[0], "attr", ""))


def _parent_of(method: ast.FunctionDef, attribute: str) -> str:
    """The name of the frame `self.<attribute>` was built into."""
    return _first_arg(assignment_to(method, attribute).value)


# The local name `_build_ui` gives the collapsible frame the engine knobs live in.
ADVANCED_BODY = "adv_body"


def _is_inside(method: ast.FunctionDef, frame: str, ancestor: str) -> bool:
    """Is the local frame `frame` that frame, or built (transitively) inside it?"""
    parents = {}
    for node in ast.walk(method):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call) and node.value.args):
            parents[node.targets[0].id] = _first_arg(node.value)
    seen = set()
    while frame and frame not in seen:
        if frame == ancestor:
            return True
        seen.add(frame)
        frame = parents.get(frame, "")
    return False


def _detecting(**overrides) -> Dict[str, Any]:
    """An fps payload from a run whose detector has ticked at least once."""
    payload = {"fps": 30, "detections": 3, "detector_ms": 21.4, "detect_every_n": 5,
               "amortised_ms": 4.3, "concepts": ("person",), "ticks": 12}
    payload.update(overrides)
    return payload


# --- step 3: is detection on, and on what ------------------------------------


def test_a_blank_target_says_the_whole_frame_and_that_detection_is_off():
    """The defect: nothing in the window said the selective path was idle, or why."""
    line = _plan_state_line("", running=True)
    assert "global" in line
    assert "whole frame" in line
    assert "off" in line


def test_a_typed_target_before_start_says_detection_has_not_begun():
    line = _plan_state_line("person", running=False)
    assert "person" in line
    assert "on," not in line


def test_a_typed_target_with_no_detector_tick_yet_says_so():
    """A vocabulary change drops the tracks, so this is a state a run passes through."""
    line = _plan_state_line("person", running=True, payload={"fps": 30})
    assert "person" in line
    assert "starting" in line


def test_a_running_detector_names_the_concept_the_count_and_the_cadence():
    line = _plan_state_line("person", running=True, payload=_detecting())
    assert "person" in line
    assert "3 object" in line
    assert "5" in line, "the cadence is not readable, so a lagging mask has no reason"


def test_the_concept_reported_is_the_one_the_detector_is_actually_holding():
    """The field may be mid-edit and the plan debounced; the payload is the truth."""
    line = _plan_state_line("dog", running=True, payload=_detecting(concepts=("person",)))
    assert "person" in line
    assert "dog" not in line


def test_one_object_is_not_pluralised():
    assert "1 object held" in _plan_state_line("person", True, _detecting(detections=1))


def test_holding_nothing_is_not_the_same_as_not_looking():
    line = _plan_state_line("person", running=True, payload=_detecting(detections=0))
    assert "0 object" in line
    assert "on," in line


def test_a_headless_demo_plan_is_reported_even_though_no_field_was_typed():
    """`SD_DEMO_PLAN` drives the worker directly; the window should still say so."""
    line = _plan_state_line("", running=True, payload=_detecting())
    assert "person" in line
    assert "whole frame" not in line


def test_a_bare_number_on_the_fps_queue_does_not_take_the_line_down():
    """The fps channel carried an int for as long as this app has existed."""
    assert _plan_state_line("person", running=True, payload=30)
    assert _plan_state_line("", running=False, payload=None)


def test_the_line_is_refreshed_from_the_newest_payload():
    poll = gui_method("_poll_queues")
    assert calls_named(poll, "_refresh_plan_state"), \
        "the plan line is never redrawn, so it can only be as old as the last edit"
    refresh = gui_method("_refresh_plan_state")
    assert calls_named(refresh, "_plan_state_line")


def test_typing_updates_the_line_before_the_debounced_send():
    """Reading what the plan will be must not wait 400 ms behind the queue write."""
    handler = gui_method("_on_plan_field_changed")
    refreshes = calls_named(handler, "_refresh_plan_state")
    assert refreshes, "the line only moves once the plan has been sent"
    guards = [node for node in ast.walk(handler)
              if isinstance(node, (ast.If, ast.Return))]
    first_guard = min((node.lineno for node in guards), default=0)
    assert refreshes[0].lineno < first_guard, \
        "the refresh sits behind the `not running` guard, so an idle window never updates"


def test_starting_and_stopping_redraw_it():
    for name in ("_on_start", "_on_stop"):
        assert calls_named(gui_method(name), "_refresh_plan_state"), \
            f"{name} leaves the detection state saying whatever it said before"


# --- step 5: the validator's own words, beside the field ---------------------


def test_a_refusal_is_shown_in_the_error_colour():
    update = _plan_update_from_fields("a " * 200, "wet denim", PROMPT, "")
    note, colour = _plan_note(update)
    assert update.reason in note
    assert colour == CUSTOM_COLORS["error"]


def test_an_accepted_plan_with_nothing_to_report_says_nothing():
    note, colour = _plan_note(_plan_update_from_fields("person", "wet denim", PROMPT, ""))
    assert note == ""
    assert colour == PLAN_NOTE_COLOR


def test_the_validators_notes_reach_the_field_they_came_from():
    update = _plan_update_from_fields("person", "wet denim", PROMPT, "")
    noted = update._replace(notes=("box_scale clamped to 2.0",))
    note, colour = _plan_note(noted)
    assert "box_scale clamped to 2.0" in note
    assert colour == PLAN_NOTE_COLOR


def test_the_update_carries_the_notes_rather_than_only_a_rendered_line():
    """`status` interleaves them with the plan; the note area wants them apart."""
    update = _plan_update_from_fields("person", "wet denim", PROMPT, "")
    assert isinstance(update.notes, (tuple, list))


def test_the_push_writes_the_note_where_the_fields_are():
    push = gui_method("_push_plan_runtime")
    assert calls_named(push, "_plan_note")
    assert calls_named(push, "_show_plan_note"), \
        "the refusal is only in the shared status bar"
    assert mentions(push, "status_var"), "the status bar stopped carrying the outcome"
    assert mentions(gui_method("_show_plan_note"), "plan_note_var")


def test_the_note_row_is_not_there_when_there_is_nothing_to_say():
    show = gui_method("_show_plan_note")
    assert calls_named(show, "grid_remove"), \
        "an empty note leaves a hole under the two fields on every clean plan"
    assert calls_named(show, "grid")


# --- step 2: what leads, and what is demoted ---------------------------------


def test_every_advanced_control_is_one_show_knows_about():
    assert set(ADVANCED) <= set(SHOW), f"unknown: {sorted(set(ADVANCED) - set(SHOW))}"


def test_the_engine_knobs_the_issue_named_are_all_demoted():
    assert {"seed", "acceleration", "frame_buffer_size", "use_lcm_lora",
            "step_count"} <= set(ADVANCED)


def test_the_step_count_is_off_by_default_and_the_sliders_are_not():
    """Changing the count rebuilds the engine; moving a value is a runtime update."""
    assert SHOW["step_count"] is False


def test_the_two_fields_are_built_before_any_engine_knob():
    build = gui_method("_build_ui")
    target = assignment_to(build, "_w_target_entry").lineno
    for knob in ("_w_model_entry", "_w_seed_entry", "_w_accel_combo"):
        assert target < assignment_to(build, knob).lineno, \
            f"{knob} is still built above the field that says what to restyle"


# The widget each name in `ADVANCED` is built as. Without this the constant could
# only be checked against `SHOW`, which is another list of names - the point is
# where the window actually puts the control.
ADVANCED_WIDGETS = {
    "seed": "_w_seed_entry",
    "frame_buffer_size": "_w_buffer_entry",
    "acceleration": "_w_accel_combo",
    "use_lcm_lora": "_w_lcm_switch",
    "use_denoising_batch": "_w_denoise_switch",
    "step_count": "_w_step_add",
}


@pytest.mark.parametrize("control", ADVANCED)
def test_the_engine_knobs_are_built_inside_the_advanced_section(control):
    knob = ADVANCED_WIDGETS.get(control)
    assert knob, f"`ADVANCED` demotes {control}, which no widget here maps to"
    build = gui_method("_build_ui")
    holder = _parent_of(build, knob)
    assert _is_inside(build, holder, ADVANCED_BODY), \
        f"{knob} sits in `{holder}`, which is not inside the advanced section"


def test_the_advanced_section_starts_collapsed():
    build = gui_method("_build_ui")
    hides = [call for call in calls_named(build, "grid_remove")
             if mentions(call, "_advanced_body")]
    assert hides, "the engine knobs are visible at start, which is the demotion undone"


def test_the_advanced_section_can_be_opened_again():
    toggle = gui_method("_toggle_advanced")
    assert calls_named(toggle, "grid"), "nothing shows the knobs once they are hidden"
    assert mentions(toggle, "advanced_var")


def test_the_advanced_section_says_what_a_rebuild_costs():
    """Step 4's standing half: the warning is there before anything is touched."""
    build = gui_method("_build_ui")
    assert mentions(build, "ENGINE_REBUILD_HINT")


def test_the_fields_still_stay_editable_while_generation_runs():
    """Unchanged from issue #22, and the promotion must not have re-locked them."""
    build = gui_method("_build_ui")
    for call in calls_named(build, "_register_lockables"):
        for widget in ("_w_target_entry", "_w_style_entry"):
            assert not mentions(call, widget), f"{widget} is disabled while running"
