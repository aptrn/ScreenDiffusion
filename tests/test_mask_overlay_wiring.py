"""Both ends of the mask overlay, and the three states the window now tells apart.

Issue #47. Two things were wrong at once. The preview said nothing about *where*
the mask was, so a selective run and a global one looked identical - which under
the `masked` primitive they very nearly are, since the whole frame is diffused and
only the composite is selective. And the words underneath collapsed three
different situations into one reading: no target set, a target with the detector
holding nothing, and a target with N objects being restyled.

The sentences are pure and are executed out of the source; where the switch is
built and what guards the drawing are read off the source through `guisource`,
because `StreamGUI` cannot be instantiated in this tier.
"""

import ast
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

from mask_overlay import OVERLAY_SWITCH_TEXT

from guisource import assignment_to, calls_named, gui_method, mentions
from sourceloader import load_symbols
from test_selective_render_wiring import WORKER

CUSTOM_COLORS = {"success": "#10B981", "error": "#EF4444", "surface": "#374151"}

_symbols = load_symbols(
    "main_gpu_addon.py",
    ["GLOBAL_STATE", "NOTHING_FOUND", "_plan_state_line", "_restyling_phrase"],
    extra_globals={"NamedTuple": NamedTuple, "Optional": Optional, "Dict": Dict,
                   "Any": Any, "Tuple": Tuple, "Sequence": Sequence,
                   "CUSTOM_COLORS": CUSTOM_COLORS},
)
GLOBAL_STATE = _symbols["GLOBAL_STATE"]
NOTHING_FOUND = _symbols["NOTHING_FOUND"]
_plan_state_line = _symbols["_plan_state_line"]


def _detecting(**overrides) -> Dict[str, Any]:
    """An fps payload from a run whose detector has ticked at least once."""
    payload = {"fps": 30, "detections": 3, "detector_ms": 21.4, "detect_every_n": 5,
               "amortised_ms": 4.3, "concepts": ("person",), "ticks": 12,
               "regions": 3, "slots": 6, "deferred": 0, "skipped_small": 0}
    payload.update(overrides)
    return payload


# --- step 4: three states, three readings ------------------------------------


def test_no_target_set_reads_differently_from_a_detector_that_found_nothing():
    """The Gate's second item. Both leave the frame unrestyled in the target's
    sense and they are not the same thing."""
    nothing_set = _plan_state_line("", running=True)
    found_nothing = _plan_state_line(
        "person", running=True, payload=_detecting(detections=0, regions=0))
    assert nothing_set != found_nothing
    assert "no target" in nothing_set
    assert "person" in found_nothing


def test_a_detector_holding_nothing_says_the_preview_is_the_capture():
    """A selective frame with no regions costs no diffusion call at all, so what
    is on screen is the capture - the one state where nothing is happening."""
    line = _plan_state_line("person", running=True,
                            payload=_detecting(detections=0, regions=0))
    assert NOTHING_FOUND in line
    assert "0 object" in line, "issue #40's reading of the count must survive"


def test_a_detector_holding_objects_says_how_many_regions_are_being_restyled():
    line = _plan_state_line("person", running=True, payload=_detecting())
    assert "3 region" in line
    assert NOTHING_FOUND not in line


def test_one_region_is_not_pluralised():
    line = _plan_state_line("person", running=True,
                            payload=_detecting(detections=1, regions=1))
    assert "1 region;" in line
    assert "1 regions" not in line


def test_the_line_says_the_rest_of_the_frame_is_untouched():
    """The Context's other half: under `masked` the whole frame is diffused and
    the mask decides what is kept, which is why the two paths look alike."""
    line = _plan_state_line("person", running=True, payload=_detecting())
    assert "untouched" in line


def test_objects_held_and_regions_rendered_are_two_numbers():
    """K slots, N objects: with more objects than slots the round-robin renders
    some of them this frame and the rest next frame."""
    line = _plan_state_line("person", running=True,
                            payload=_detecting(detections=9, regions=6, slots=6))
    assert "9 object" in line
    assert "6 region" in line


def test_a_payload_from_before_the_overlay_still_renders():
    """The fps channel's older shapes: a bare number, and a dict with counts but
    no `regions` key at all."""
    assert _plan_state_line("person", running=True, payload=30)
    payload = _detecting()
    payload.pop("regions")
    assert _plan_state_line("person", running=True, payload=payload)


def test_the_global_state_still_says_the_whole_frame():
    assert "whole frame" in GLOBAL_STATE
    assert "off" in GLOBAL_STATE


# --- steps 1 and 2: the overlay, off by default ------------------------------


def test_the_overlay_is_off_by_default():
    """The Gate's third item. The preview is a view of the output; a decoration
    on it by default makes the one honest picture into one to be discounted."""
    init = gui_method("__init__")
    assign = assignment_to(init, "overlay_var")
    assert "BooleanVar" in ast.unparse(assign.value)
    assert ast.unparse(assign.value.keywords[0].value) == "False"


def test_the_switch_is_built_beside_the_preview_and_bound_to_that_variable():
    build = gui_method("_build_ui")
    switch = assignment_to(build, "_w_overlay_switch")
    assert "CTkSwitch" in ast.unparse(switch.value)
    assert mentions(switch.value, "overlay_var")
    assert mentions(switch.value, "OVERLAY_SWITCH_TEXT")


def test_the_switch_says_what_it_shows():
    assert "mask" in OVERLAY_SWITCH_TEXT.lower()


def test_the_switch_stays_usable_while_generation_runs():
    """Judging what is being restyled is exactly what a live run is for."""
    build = gui_method("_build_ui")
    for call in calls_named(build, "_register_lockables"):
        assert not mentions(call, "_w_overlay_switch")


def test_the_preview_draws_no_overlay_unless_the_switch_is_on():
    """With it off the preview is byte for byte what it always was, because the
    drawing is not reached at all."""
    update = gui_method("_update_preview")
    draws = calls_named(update, "draw_overlay")
    assert len(draws) == 1, "the overlay is drawn unconditionally or not at all"
    guards = [node for node in ast.walk(update)
              if isinstance(node, ast.If) and mentions(node.test, "overlay_var")]
    assert guards, "nothing tests the switch before drawing"
    guard = guards[0]
    assert any(draw is call for call in calls_named(guard, "draw_overlay")
               for draw in draws), "the draw sits outside the switch's own branch"


def test_the_overlay_is_drawn_after_the_frame_has_been_scaled_down():
    """The issue's third trap: the preview already costs 9.80 ms at 1080p, so the
    overlay goes on the panel-sized canvas and not on the full capture."""
    update = gui_method("_update_preview")
    paste, = calls_named(update, "paste")
    draw, = calls_named(update, "draw_overlay")
    assert paste.lineno < draw.lineno, \
        "the overlay is drawn before the frame is resized, so it costs with capture size"


def test_the_overlay_is_told_the_scaled_size_and_the_letterbox_offset():
    """The issue's fourth trap: boxes are capture pixels and the preview is
    scaled and centred, so the transform needs both."""
    draw, = calls_named(gui_method("_update_preview"), "draw_overlay")
    arguments = {ast.unparse(argument) for argument in draw.args}
    arguments |= {ast.unparse(keyword.value) for keyword in draw.keywords}
    assert "img.width" in arguments and "img.height" in arguments, \
        "the overlay is scaled against the panel rather than the drawn image"
    assert any("x, y" in argument for argument in arguments), \
        "the overlay is not offset by where the image was pasted"


def test_the_overlay_reads_the_payload_the_status_line_reads():
    """Step 3 again, from the other end: one channel, one payload, no second
    queue for the boxes."""
    update = gui_method("_update_preview")
    assert mentions(update, "_fps_payload")


def test_the_newest_payload_is_taken_before_the_frame_it_describes_is_drawn():
    """Both queues are drained in one poll. Drawing first would outline the
    previous frame's regions on this frame's picture."""
    poll = gui_method("_poll_queues")
    fps = min(node.lineno for node in ast.walk(poll) if mentions(node, "fps_q"))
    preview = min(call.lineno for call in calls_named(poll, "_update_preview"))
    assert fps < preview


def test_toggling_the_switch_redraws_what_is_already_on_screen():
    """A stopped run holds its last frame; a switch that only took effect on the
    next one would look broken."""
    toggle = gui_method("_on_overlay_toggled")
    assert calls_named(toggle, "_update_preview")
    assert mentions(gui_method("_update_preview"), "_last_preview")


# --- step 3: the worker's end, on the queue that was already there -----------


def test_the_gui_draws_the_overlay_rather_than_the_worker():
    """The issue's first trap: an overlay drawn before the composite would reach
    the output frames and break bit-identity outright."""
    assert not calls_named(WORKER, "draw_overlay"), \
        "the worker draws the overlay, so it reaches the composited frame"


def test_the_worker_puts_the_boxes_on_the_payload_it_already_sends():
    """The Gate's fourth item: no new IPC channel. One `overlay_status` call,
    folded into the dict that goes on the fps queue."""
    status, = calls_named(WORKER, "overlay_status")
    updates = [call for call in calls_named(WORKER, "update")
               if calls_named(call, "overlay_status")]
    assert updates, "the overlay never reaches the payload"
    assert mentions(updates[0].func, "payload")
    assert len(calls_named(WORKER, "Queue")) == 0, "the worker built a queue of its own"
    arguments = [ast.unparse(argument) for argument in status.args]
    assert arguments[:2] == ["selection", "render"], \
        "the overlay is built from something other than this frame's own decision"


def test_the_overlay_is_measured_against_the_capture_the_boxes_are_in():
    """The issue's fourth trap from the sending end: the boxes are capture
    pixels, so the geometry sent with them has to be the capture's."""
    status, = calls_named(WORKER, "overlay_status")
    arguments = [ast.unparse(argument) for argument in status.args]
    assert arguments[2:] == ["width", "height"], \
        "the canvas was sent as the frame geometry, so every box would be scaled wrong"


def test_the_frame_path_gains_no_array_work():
    """The Gate's fourth item, second half. `overlay_status` reads boxes the
    selection already holds; nothing here allocates per frame."""
    import inspect

    import mask_overlay

    source = inspect.getsource(mask_overlay)
    assert "import numpy" not in source
    assert "import torch" not in source
