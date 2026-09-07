"""The swap the bench measures is the swap the worker performs (issue #30).

`image_generation_process` owns the GPU and a capture thread and cannot be called
here, and `bench.plan_swap_runner` imports torch inside its functions - so these
assertions read both sources instead. What they hold up is the one thing the
measurement rests on: when the frame loop finds the plan it bound has changed, it
does three things, and the harness does the same three. A run that skipped one -
the detector's re-warm, say - would measure a path the app never takes, and every
figure in spec 8.9 would be about the harness rather than about the product.

The other half is what the harness must *not* do: build a second engine. Criterion
3's "no TensorRT rebuild" is checked in the record from the engine's own identity,
and a run that constructed a second stream would be reporting on one while
rendering through the other.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKER_SOURCE = ROOT / "main_gpu_addon.py"
BENCH_SOURCE = ROOT / "bench" / "plan_swap_runner.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"no top-level {name}")


def _calls_named(node: ast.AST, name: str) -> list:
    return [call for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and getattr(call.func, "attr", getattr(call.func, "id", None)) == name]


WORKER = _function(_tree(WORKER_SOURCE), "image_generation_process")
BENCH = _tree(BENCH_SOURCE)
APPLY = _function(BENCH, "apply_plan")
RUN = _function(BENCH, "run_swap")

# The three things a changed plan does in the worker's frame loop: the engine takes
# the new prompt, the detector thread takes the new vocabulary, and the denoise
# reaches the engine as a schedule value.
SWAP_CALLS = ("update_prompt", "follow", "set_t_index_list")


def test_the_worker_still_does_these_three_things_on_a_plan_change():
    """If this fails, the harness is measuring a swap the worker no longer makes -
    fix `apply_plan` to match rather than this list."""
    for name in ("_apply_prompt", "follow", "set_t_index_list"):
        assert _calls_named(WORKER, name), f"the worker never calls {name}"


def test_the_harness_applies_the_same_three_and_nothing_else():
    for name in SWAP_CALLS:
        assert _calls_named(APPLY, name), f"apply_plan never calls {name}"


def test_the_harness_applies_them_only_when_the_frame_reports_a_change():
    """`begin_frame` is the worker's single read, and the cold path hangs off its
    `changed` flag; a harness that re-applied every frame would time a re-encode
    that the app pays once."""
    assert len(_calls_named(RUN, "begin_frame")) == 1
    assert len(_calls_named(RUN, "apply_plan")) == 1


def test_the_new_instruction_goes_through_the_shipped_holder():
    """`submit` moves what the *next* frame picks up, so the swap lands at a frame
    boundary exactly as it does in the worker."""
    assert len(_calls_named(RUN, "submit")) == 1


def test_the_run_builds_exactly_one_engine():
    """Otherwise "no TensorRT rebuild" would be a claim about a stream the run had
    stopped rendering through."""
    assert len(_calls_named(RUN, "build_stream")) == 1


def test_the_detector_runs_on_its_own_thread_as_it_does_in_the_worker():
    """The re-warm a vocabulary change owes is the measurement (spec 8.1), and it
    is only off the frame path if the thread is actually started."""
    assert _calls_named(RUN, "start"), "the detector is never started"
    assert _calls_named(RUN, "offer"), "the frame loop never offers a capture"
