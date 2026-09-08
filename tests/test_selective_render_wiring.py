"""The worker's end of the selective render path (issue #8), checked structurally.

`image_generation_process` owns the GPU and a capture thread, so it cannot be called
here. What these assertions hold up is the wiring the unit tests cannot see: that the
scheduler is asked once per frame off the frame's own plan, that the compositor
decides whether the frame costs a diffusion call at all, that the capture is what the
render is composited onto, and that a plan's denoise reaches the engine as a schedule
value rather than as an engine rebuild.

The pure helpers - the demo-plan switch and the status line - are executed out of the
source file rather than described.
"""

import ast
from pathlib import Path

from sourceloader import load_symbols

SOURCE = Path(__file__).resolve().parent.parent / "main_gpu_addon.py"
TEXT = SOURCE.read_text(encoding="utf-8-sig")
TREE = ast.parse(TEXT, filename=str(SOURCE))


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"main_gpu_addon.py defines no {name}")


def _calls_named(node: ast.AST, name: str) -> list:
    return [call for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and getattr(call.func, "attr", getattr(call.func, "id", None)) == name]


WORKER = _function("image_generation_process")
WORKER_TEXT = ast.get_source_segment(TEXT, WORKER)

_symbols = load_symbols("main_gpu_addon.py",
                        ["DEMO_PLAN_ENV", "DEMO_PLAN_VALUES", "demo_plan_requested",
                         "_format_fps"])
demo_plan_requested = _symbols["demo_plan_requested"]
DEMO_PLAN_ENV = _symbols["DEMO_PLAN_ENV"]
_format_fps = _symbols["_format_fps"]


# --- the wiring --------------------------------------------------------------


def test_the_selective_modules_are_imported_at_module_scope():
    """Both are stdlib-or-numpy; neither drags torch into the GUI process."""
    imported = {alias.name for node in ast.walk(TREE)
                if isinstance(node, ast.ImportFrom)
                and node.module in ("region_scheduler", "compositor",
                                    "device_compositor", "detector_worker",
                                    "render_plan")
                for alias in node.names}
    assert "RegionScheduler" in imported
    assert {"DeviceCompositor", "MASKED"} <= imported
    assert "frame_to_array" in imported, "the capture never becomes an array to blend"
    assert {"priority_case_plan", "t_index_for_denoise"} <= imported


def test_the_scheduler_is_asked_once_per_frame_for_the_frames_own_plan():
    """A second selection inside one frame would mask against boxes the render
    was not made from."""
    select, = _calls_named(WORKER, "select")
    arguments = {ast.unparse(argument) for argument in select.args}
    assert "tracks" in arguments
    assert "frame_plan.plan" in arguments, (
        "the selection is not made from the plan this frame bound")


def test_the_compositor_decides_what_the_frame_costs():
    assert len(_calls_named(WORKER, "frame")) == 1
    assert len(_calls_named(WORKER, "blend_device")) == 1


def test_a_frame_with_nothing_to_restyle_costs_no_diffusion_call():
    """The compositor's `diffuses` guards the engine call, so a selective plan
    that found nothing passes the capture through instead of restyling it.

    Every engine call, not one of them: the masked frame and the full-frame one
    ask for different output types (issue #31) and both are the same decision."""
    calls = _calls_named(WORKER, "img2img")
    assert calls
    for img2img in calls:
        guards = [node for node in ast.walk(WORKER)
                  if isinstance(node, ast.If)
                  and "diffuses" in ast.unparse(node.test)
                  and img2img in list(ast.walk(node))]
        assert guards, "a diffusion call is not behind the compositor's verdict"


def test_the_render_is_composited_onto_the_capture_itself():
    """Not onto the previous output: the gate is bit-identity with the capture.

    The capture is now the tensor the engine was given rather than an array made
    from it - the blend runs on the device, so the frame it is composited onto is
    the one that never left."""
    blend, = _calls_named(WORKER, "blend_device")
    assert ast.unparse(blend.args[0]) == "batch"


def test_the_masked_render_never_comes_home_before_the_blend():
    """Issue #31's step 2: the engine hands back a device tensor, the composite
    runs where it is, and the frame pays one host copy afterwards. Asking for the
    PIL image here would be the round trip the change exists to remove."""
    blend, = _calls_named(WORKER, "blend_device")
    rendered = ast.unparse(blend.args[1])
    masked = [call for call in _calls_named(WORKER, "img2img")
              if any(keyword.arg == "output_type"
                     and keyword.value.value == "pt" for keyword in call.keywords)]
    assert len(masked) == 1, "the masked branch does not ask the engine for a tensor"
    bound = [ast.unparse(node.targets[0]) for node in ast.walk(WORKER)
             if isinstance(node, ast.Assign) and masked[0] in list(ast.walk(node))]
    assert bound == [rendered], (
        f"the blend takes `{rendered}`, not what the engine left on the device")


def test_the_plans_denoise_reaches_the_engine_without_an_engine_rebuild():
    """Changing t_index *values* is a runtime update; changing the step count is
    a rebuild. A plan carries one strength, so it may only move a value."""
    assert len(_calls_named(WORKER, "t_index_for_denoise")) == 1
    assert "set_t_index_list" in WORKER_TEXT
    assert "_build_stream(current_t_index_list)" in WORKER_TEXT
    denoise_index = WORKER_TEXT.index("t_index_for_denoise")
    rebuild_index = WORKER_TEXT.index("_build_stream(current_t_index_list)")
    assert denoise_index > rebuild_index, (
        "the denoise update sits in the engine-swap branch")


def test_the_plans_denoise_reaches_the_engine_at_any_step_count():
    """Issue #46 made the count a runtime choice, so the update that carries the
    plan's strength has to survive one: it spends the count it already has through
    `t_index_ladder` rather than only firing when the count is 1."""
    assert len(_calls_named(WORKER, "t_index_ladder")) == 1
    ladder = WORKER_TEXT.index("t_index_ladder")
    assert "len(current_t_index_list) == 1" not in WORKER_TEXT, (
        "the plan's denoise is dropped at every step count but one")
    assert WORKER_TEXT.index("t_index_for_denoise") > ladder, (
        "the ladder is built from something other than the plan's own strength")


def test_the_hardcoded_demo_plan_is_behind_the_switch():
    """Step 4 drives the path from a hardcoded plan; the app as shipped is
    unchanged until something wires the GUI up, which is a later issue."""
    assert len(_calls_named(WORKER, "priority_case_plan")) == 1
    assert len(_calls_named(WORKER, "demo_plan_requested")) == 1


def test_the_selection_is_reported_on_the_existing_fps_channel():
    """One payload per frame: the detector's fields and the scheduler's own
    account of the selection, on the queue the GUI already reads."""
    assert len(_calls_named(WORKER, "status")) == 1
    assert len(_calls_named(WORKER, "fps_payload")) == 1


# --- the switch --------------------------------------------------------------


def test_the_demo_plan_is_off_unless_it_is_asked_for():
    assert demo_plan_requested({}) is False
    assert demo_plan_requested({DEMO_PLAN_ENV: ""}) is False
    assert demo_plan_requested({DEMO_PLAN_ENV: "0"}) is False
    assert demo_plan_requested({DEMO_PLAN_ENV: "no"}) is False


def test_the_demo_plan_switch_takes_the_words_people_type():
    for value in ("1", "true", "TRUE", "yes", "on", " 1 "):
        assert demo_plan_requested({DEMO_PLAN_ENV: value}) is True


# --- the readout -------------------------------------------------------------


def test_a_payload_without_a_selection_says_nothing_about_regions():
    assert _format_fps({"fps": 30}) == "FPS: 30"


def test_a_payload_with_a_selection_shows_what_was_rendered_and_what_waited():
    line = _format_fps({"fps": 24, "regions": 4, "slots": 6, "deferred": 2,
                        "skipped_small": 1})
    assert line.startswith("FPS: 24")
    assert "4/6" in line, f"the slot occupancy is not readable: {line}"
    assert "2" in line and "1" in line
