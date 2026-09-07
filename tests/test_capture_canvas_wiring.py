"""Capture geometry and diffusion canvas are two things (issue #39, spec 8.2).

The engine is 512x512 whatever an engine directory's name claims (spec 7.2), and
until this issue the capture was the same 512x512 - so a screen region big enough
to hold a person could not be captured, and every object was diffused at whatever
fraction of 512 px it occupied. What is checked here is the split: the capture
thread and the capture window take the *capture* size, the wrapper takes the
*canvas*, and the frame loop resizes between them rather than handing the engine a
frame it never sized for.

`image_generation_process` owns a GPU and a capture thread and `StreamGUI` owns a
Tk root, so neither is called here. The structural half is read out of the source
with `ast`; the pure helpers are executed out of it with `sourceloader`.
"""

import ast
from pathlib import Path

import pytest

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
                        ["DIFFUSION_CANVAS", "CAPTURE_PRESETS", "DEFAULT_CAPTURE",
                         "capture_size"])
DIFFUSION_CANVAS = _symbols["DIFFUSION_CANVAS"]
CAPTURE_PRESETS = _symbols["CAPTURE_PRESETS"]
DEFAULT_CAPTURE = _symbols["DEFAULT_CAPTURE"]
capture_size = _symbols["capture_size"]


# --- the capture presets ----------------------------------------------------


def test_the_canvas_is_the_only_size_tensorrt_actually_builds():
    """Spec 7.2: `EngineBuilder.build` defaults to 512 and the wrapper never
    forwards a resolution, so a canvas that is not 512 is a directory name."""
    assert DIFFUSION_CANVAS == 512


def test_the_default_capture_is_the_canvas():
    """What the app has always done, so nothing about this change moves a
    committed measurement until somebody asks it to."""
    assert capture_size(DEFAULT_CAPTURE) == (DIFFUSION_CANVAS, DIFFUSION_CANVAS)


def test_every_preset_is_a_pair_of_positive_sides():
    assert CAPTURE_PRESETS
    for label, (width, height) in CAPTURE_PRESETS.items():
        assert width > 0 and height > 0, label


def test_a_preset_larger_than_the_canvas_is_offered():
    """The point of the issue: a 512x512 capture window cannot get a whole object
    into frame."""
    assert any(width > DIFFUSION_CANVAS or height > DIFFUSION_CANVAS
               for width, height in CAPTURE_PRESETS.values())


def test_an_unknown_capture_label_falls_back_to_the_canvas():
    """A stale preference must not start the worker on a size nothing built."""
    assert capture_size("4096 x 4096") == (DIFFUSION_CANVAS, DIFFUSION_CANVAS)


# --- the worker's two geometries --------------------------------------------


def test_the_worker_takes_a_canvas_beside_its_capture_size():
    names = {argument.arg for argument in WORKER.args.args}
    assert {"width", "height"} <= names, "the capture size is still the worker's"
    assert {"canvas_width", "canvas_height"} <= names, (
        "the worker has no canvas of its own, so the capture is still the engine's")


def test_the_engine_is_built_at_the_canvas_and_never_at_the_capture():
    build, = [call for call in _calls_named(WORKER, "StreamDiffusionWrapper")]
    sized = {keyword.arg: ast.unparse(keyword.value) for keyword in build.keywords}
    assert sized["width"] == "canvas_width"
    assert sized["height"] == "canvas_height"


def test_the_capture_thread_takes_the_capture_size_and_not_the_canvas():
    thread, = [call for call in _calls_named(WORKER, "Thread")
               if any(keyword.arg == "target"
                      and "capture" in ast.unparse(keyword.value)
                      for keyword in thread_keywords(call))]
    args = ast.unparse([keyword.value for keyword in thread.keywords
                        if keyword.arg == "args"][0])
    assert "height, width" in args
    assert "canvas_" not in args


def thread_keywords(call):
    return call.keywords


def test_the_scheduler_and_the_compositor_work_in_captured_pixels():
    """The mask is composited onto the capture, so its coordinates are the
    capture's - the canvas only ever exists between the resize and the engine."""
    select, = _calls_named(WORKER, "select")
    assert [ast.unparse(argument) for argument in select.args][-2:] == ["width", "height"]
    frame, = _calls_named(WORKER, "frame")
    assert [ast.unparse(argument) for argument in frame.args][1:3] == ["width", "height"]


def test_the_frame_loop_asks_the_compositor_for_the_plans_primitive():
    frame, = _calls_named(WORKER, "frame")
    assert "primitive" in ast.unparse(frame), (
        "the compositor is never told which primitive the plan asked for")


def test_every_engine_call_is_handed_a_canvas_and_never_the_capture():
    """The whole point of the split: `img2img` may not see a 1080p frame."""
    for call in _calls_named(WORKER, "img2img"):
        argument = ast.unparse(call.args[0])
        assert argument != "batch", (
            f"img2img({argument}) hands the engine the capture, not the canvas")


def test_the_crop_primitive_reaches_the_engine_and_the_blend():
    assert len(_calls_named(WORKER, "crop_to_canvas")) == 1
    blend, = _calls_named(WORKER, "blend_device")
    assert "render.crop" in ast.unparse(blend), (
        "the blend is never told where the crop patch belongs")


def test_the_noise_field_is_told_which_canvas_the_boxes_land_on():
    """`per_track` pins noise to a track's latent cells, and at a capture that is
    not the canvas those are two coordinate systems (issue #39)."""
    apply, = _calls_named(WORKER, "apply")
    assert len(apply.args) == 3, "noise.apply is still reading boxes as canvas pixels"
    geometry, = _calls_named(WORKER, "CanvasGeometry")
    assert ast.unparse(apply.args[2]) in {"geometry", ast.unparse(geometry)}
    assert [ast.unparse(argument) for argument in geometry.args] == [
        "width", "height", "canvas_width", "canvas_height", "render.crop"]


@pytest.mark.parametrize("name", ["to_canvas", "crop_to_canvas", "CanvasGeometry",
                                  "CROP"])
def test_the_pieces_of_the_split_are_imported_at_module_scope(name):
    """All three modules are numpy-or-stdlib on import, so none of them drags
    torch into the GUI process."""
    imported = {alias.name for node in ast.walk(TREE)
                if isinstance(node, ast.ImportFrom)
                and node.module in ("compositor", "device_compositor", "seeding")
                for alias in node.names}
    assert name in imported


# --- the GUI's end ----------------------------------------------------------


GUI_TEXT = ast.get_source_segment(TEXT, _function("_on_start"))


def test_the_gui_starts_the_worker_on_a_capture_size_it_chose():
    assert "capture_size(" in GUI_TEXT, (
        "the GUI still hands the worker one hardcoded pair of numbers")


def test_the_capture_window_is_the_capture_size_rather_than_the_preview():
    window, = _calls_named(_function("_on_start"), "FloatingCaptureWindow")
    sized = {keyword.arg: ast.unparse(keyword.value) for keyword in window.keywords}
    assert "preview_dim" not in ast.unparse(window), (
        "the capture window is still sized by the preview panel")
    assert {"inner_w", "inner_h"} <= set(sized)
