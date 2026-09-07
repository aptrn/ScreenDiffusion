"""The worker's end of the two §8.5 levers (issue #32), checked structurally.

`image_generation_process` owns the GPU and a capture thread, so what is held here is
the wiring the unit tests cannot see: that the plan's `seed_policy` reaches the noise
field, that the noise is written *before* the call that reads it, that the EMA's
coefficient comes off the plan rather than a constant, and that both histories end
where they have to - a frame that rendered nothing, and an engine that was rebuilt.

The Gate's last item is the first test here: `seed_policy` is read by the shipped
path, so the plan field does what its name says.
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


WORKER = _function("image_generation_process")
WORKER_TEXT = ast.get_source_segment(TEXT, WORKER)


def _calls(node: ast.AST) -> list:
    return [ast.unparse(call.func) for call in ast.walk(node)
            if isinstance(call, ast.Call)]


CALLS = _calls(WORKER)


# --- seed_policy is read by the shipped path ---------------------------------


def test_the_noise_field_is_imported_at_module_scope():
    """Stdlib-only from the GUI process's point of view, like every other module on
    the frame loop's import path."""
    imported = {alias.name for node in ast.walk(TREE)
                if isinstance(node, ast.ImportFrom) and node.module == "seeding"
                for alias in node.names}
    assert imported == {"CanvasGeometry", "NoiseField"}


def test_the_noise_field_is_built_once_and_held_across_frames():
    """One per worker, beside the scheduler's cursor and the compositor's alpha: a
    field rebuilt per frame would redraw every track's realisation every frame,
    which is the opposite of pinning it."""
    assert CALLS.count("NoiseField") == 1


def test_the_noise_field_takes_the_engine_s_own_seed():
    """So the seed box the app already has re-seeds every track together."""
    built, = [call for call in ast.walk(WORKER)
              if isinstance(call, ast.Call)
              and getattr(call.func, "id", None) == "NoiseField"]
    assert {keyword.arg: ast.unparse(keyword.value)
            for keyword in built.keywords} == {"base_seed": "seed"}


def test_a_plan_change_reaches_the_noise_field():
    """The Gate's last item: `seed_policy` is read by the shipped path."""
    assert CALLS.count("noise.follow") == 1


def test_the_noise_is_written_before_the_call_that_reads_it():
    """`init_noise` is added to the latent inside `img2img`, so a write after it is
    a write the frame renders without."""
    body = WORKER_TEXT
    assert body.index("noise.apply(") < body.index("stream.img2img(canvas, output_type")


def test_the_noise_is_applied_once_per_frame():
    assert CALLS.count("noise.apply") == 1


def test_an_engine_swap_forgets_the_noise_it_prepared():
    """The prepared field belongs to the engine that drew it; carrying it across a
    rebuild would restore `fixed` to a field the new engine never had."""
    assert CALLS.count("noise.reset") == 1


# --- the output EMA ----------------------------------------------------------


def test_the_ema_coefficient_comes_off_the_plan():
    """A plan field, not a constant - so it goes through `validate_plan`'s
    clamp-and-say-so path like every other number."""
    assert CALLS.count("compositor.set_output_ema") == 1
    assert "frame_plan.plan.settings.output_ema" in WORKER_TEXT


def test_the_ema_history_ends_where_a_frame_rendered_nothing_and_at_a_rebuild():
    """Two gaps, two resets: a passthrough frame has no render to average with, and
    a rebuilt engine's first render has nothing to do with the last one's."""
    assert CALLS.count("compositor.reset_ema") == 2


def test_the_ema_is_never_applied_to_the_composited_frame():
    """The issue's first trap. The EMA smooths the *render*, inside `blend_device`
    and before the mask; a `smooth` call out here would be one on the frame that
    already has captured pixels in it, and history would reach the background."""
    assert "compositor.smooth" not in WORKER_TEXT
