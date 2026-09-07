"""The worker's end of detection (issue #7), checked structurally, plus the readout.

`image_generation_process` owns the GPU and a capture thread, so it cannot be
called here; these assertions read the source instead. What they hold up is the
part the unit tests cannot: that the detector is built after the diffusion engine,
that the frame loop *offers* frames rather than detecting on them, that the offer
is behind the plan's cadence, and that the detector is stopped when the loop ends.

The GUI half - a status line the fps channel can carry - is a pure function, and is
executed out of the source file rather than described.
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

_format_fps = load_symbols("main_gpu_addon.py", ["_format_fps"])["_format_fps"]


# --- the worker --------------------------------------------------------------


def test_the_detection_modules_are_imported_at_module_scope():
    """Both are stdlib-and-numpy at import time; ultralytics is loaded on the thread."""
    imported = {alias.name for node in ast.walk(TREE)
                if isinstance(node, ast.ImportFrom)
                and node.module in ("detection", "detector_worker")
                for alias in node.names}
    assert {"is_detect_frame", "fps_payload"} <= imported
    assert {"BackgroundDetector", "UltralyticsDetector"} <= imported


def test_the_detector_is_built_after_the_diffusion_engine():
    """Step 1's ordering: the engine is the tenant that must get its VRAM first."""
    engine = min(call.lineno for call in _calls_named(WORKER, "_build_stream"))
    detector, = _calls_named(WORKER, "BackgroundDetector")
    assert detector.lineno > engine


def test_the_detector_thread_is_started_and_stopped():
    """Named on the detector itself: the worker already starts a capture thread."""
    called = {call.func.attr for call in ast.walk(WORKER)
              if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
              and getattr(call.func.value, "id", None) == "detection"}
    assert "start" in called, "the detector never leaves the frame path"
    assert "stop" in called, "the detector thread would outlive the loop"


def test_the_frame_loop_never_detects_on_the_frame_path():
    """The issue's first trap. `offer` returns; `detect` would not."""
    assert _calls_named(WORKER, "detect") == []
    assert len(_calls_named(WORKER, "offer")) == 1


def test_the_offer_is_behind_the_plans_cadence():
    """Step 2: `global.detect_every_n` decides, and it comes off the frame's plan."""
    assert len(_calls_named(WORKER, "is_detect_frame")) == 1
    assert "detect_every_n" in ast.get_source_segment(TEXT, WORKER)


def test_a_new_plan_reaches_the_detector_exactly_once():
    """`follow` is the cold path: it re-encodes a vocabulary and re-warms.

    Counted on the receiver rather than on the bare name: the noise field follows
    a plan change too (issue #32), and it is not this one.
    """
    followed = [ast.unparse(call.func) for call in _calls_named(WORKER, "follow")]
    assert followed.count("detection.follow") == 1


def test_the_detection_readout_goes_on_the_existing_fps_channel():
    """Step 5, and on the channel the GUI already polls rather than a new one."""
    assert len(_calls_named(WORKER, "fps_payload")) == 1


# --- the GUI -----------------------------------------------------------------


def test_the_gui_formats_whatever_the_worker_put_on_the_fps_queue():
    poll = _function("_poll_queues")
    assert _calls_named(poll, "_format_fps"), "the GUI still parses the payload by hand"


def test_a_bare_number_still_reads_as_fps():
    """The fps channel carried an int for as long as this app has existed."""
    assert _format_fps(30) == "FPS: 30"
    assert _format_fps(29.6) == "FPS: 30"


def test_a_payload_without_detection_says_nothing_about_detection():
    assert _format_fps({"fps": 30}) == "FPS: 30"


def test_a_payload_with_detection_shows_the_count_and_the_cost():
    line = _format_fps({"fps": 28, "detections": 4, "detector_ms": 14.3,
                        "detect_every_n": 3, "amortised_ms": 4.77})
    assert line.startswith("FPS: 28")
    assert "4" in line and "14.3" in line and "4.8" in line


def test_nonsense_on_the_fps_queue_does_not_take_the_gui_down():
    assert _format_fps(None) == "FPS: --"
    assert _format_fps({}) == "FPS: --"
