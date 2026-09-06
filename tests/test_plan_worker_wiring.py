"""The worker's end of the Render Plan (issue #6, step 4), checked structurally.

`image_generation_process` cannot be called here - it imports torch, spawns a capture
thread and owns the GPU - so these assertions read the source instead. What they hold
up is the part the GPU-free unit tests cannot: that the plan is held across the loop,
that the frame reads it exactly once, and that the capture thread's shedding is
untouched by any of it.
"""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "main_gpu_addon.py"
TEXT = SOURCE.read_text(encoding="utf-8-sig")
TREE = ast.parse(TEXT, filename=str(SOURCE))


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"main_gpu_addon.py defines no {name}")


def _calls_named(node: ast.AST, name: str) -> list[ast.Call]:
    return [call for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and getattr(call.func, "attr", getattr(call.func, "id", None)) == name]


WORKER = _function("image_generation_process")


def test_the_plan_module_is_imported_at_module_scope():
    """It is stdlib-only, so it costs the GUI process nothing and pickles in both."""
    imported = {alias.name for node in ast.walk(TREE)
                if isinstance(node, ast.ImportFrom) and node.module == "render_plan"
                for alias in node.names}
    assert {"ActivePlan", "global_plan", "validate_plan"} <= imported


def test_the_worker_starts_holding_a_plan():
    """Not None: "no plan yet" would be a third state the frame loop has to know."""
    assert _calls_named(WORKER, "ActivePlan"), "the worker never builds an ActivePlan"
    assert _calls_named(WORKER, "global_plan"), "the startup plan is not today's behaviour"


def test_the_frame_reads_the_active_plan_exactly_once():
    """The Gate's atomicity item. Two reads in one frame is two plans in one frame."""
    assert len(_calls_named(WORKER, "begin_frame")) == 1


def test_the_drain_hands_a_submitted_plan_to_the_holder_and_nowhere_else():
    """One submit in the control drain, and every submit goes to the holder.

    The worker's other one is issue #8's hardcoded demo plan, submitted once at
    startup - which is the same door and lands at the same frame boundary.
    """
    # The innermost loop around the transition: the outer one is the frame loop.
    drain = min([node for node in ast.walk(WORKER)
                 if isinstance(node, ast.While)
                 and _calls_named(node, "_control_transition")],
                key=lambda node: node.end_lineno - node.lineno)
    assert len(_calls_named(drain, "submit")) == 1
    assert all(getattr(call.func.value, "id", None) == "active_plan"
               for call in _calls_named(WORKER, "submit"))


def test_a_rejected_plan_reaches_the_user_rather_than_a_swallowed_exception():
    """spec 8.7: the previous plan keeps rendering and the reason is surfaced."""
    statuses = _calls_named(WORKER, "_status")
    said = {ast.dump(call) for call in statuses}
    assert any("plan_error" in dump for dump in said), \
        "the worker never puts a plan rejection on the status queue"


def test_the_control_transition_is_given_the_active_plan():
    """Without it the validator cannot count versions up from the plan in force."""
    call, = _calls_named(WORKER, "_control_transition")
    assert len(call.args) == 3


def test_the_capture_thread_still_sheds_the_frames_it_cannot_keep():
    """Issue #6's third trap. Newest-frame-wins is the loop's overload behaviour."""
    capture = _function("_screen_capture_loop_dx")
    source = ast.get_source_segment(TEXT, capture)
    assert "popleft" in source or "maxlen" in source
