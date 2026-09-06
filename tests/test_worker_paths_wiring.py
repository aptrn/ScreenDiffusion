"""The worker's end of the shared cache roots (issue #9), checked structurally.

`image_generation_process` cannot be called here - it imports torch, spawns a capture
thread and owns the GPU - and the GUI hands it 32 *positional* arguments, so a
signature change that nobody re-counts fails only at runtime, in the app. These
assertions read the source instead.
"""

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parent.parent / "main_gpu_addon.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8-sig"), filename=str(SOURCE))


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"main_gpu_addon.py defines no {name}")


def _names_used_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


WORKER = _function("image_generation_process")


def test_the_worker_takes_an_engine_dir():
    params = [a.arg for a in WORKER.args.args + WORKER.args.kwonlyargs]
    assert "engine_dir" in params


def test_the_worker_resolves_both_roots_and_logs_them():
    used = _names_used_in(WORKER)
    assert {"resolve_models_dir", "resolve_engines_dir", "_cache_paths_banner"} <= used


@pytest.mark.parametrize("sink", ["print", "_status"])
def test_the_banner_reaches_the_startup_log(sink):
    """It goes to stdout *and* to the GUI status queue - the worker has no other log."""
    banner_calls = [
        call for call in ast.walk(WORKER)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == sink
        and any(isinstance(a, ast.Name) and a.id == "banner" for a in call.args)
    ]
    assert banner_calls, f"the cache-paths banner is never passed to {sink}()"


def test_the_engines_root_goes_through_the_shared_rule():
    """Not `Path(engine_dir)`: a relative value must not re-anchor to the worker's cwd."""
    call = next(n for n in ast.walk(WORKER)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "resolve_engines_dir")
    assert [a.id for a in call.args if isinstance(a, ast.Name)] == ["engine_dir"]


def test_the_wrapper_is_built_with_an_explicit_engine_dir():
    builder = next(n for n in ast.walk(WORKER)
                   if isinstance(n, ast.FunctionDef) and n.name == "_build_stream")
    construction = next(n for n in ast.walk(builder)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id == "StreamDiffusionWrapper")
    engine_kwarg = next(k for k in construction.keywords if k.arg == "engine_dir")
    assert "engines_root" in _names_used_in(engine_kwarg.value)


def _spawn_args_tuple() -> ast.Tuple:
    for call in ast.walk(TREE):
        if not isinstance(call, ast.Call):
            continue
        target = next((k for k in call.keywords if k.arg == "target"), None)
        if target and isinstance(target.value, ast.Name) and target.value.id == "image_generation_process":
            return next(k for k in call.keywords if k.arg == "args").value
    raise AssertionError("the GUI never spawns image_generation_process")


def test_the_gui_fills_every_positional_parameter():
    assert len(_spawn_args_tuple().elts) == len(WORKER.args.args)


def test_the_gui_passes_the_resolved_engines_root():
    last = _spawn_args_tuple().elts[-1]
    assert "resolve_engines_dir" in _names_used_in(last)
