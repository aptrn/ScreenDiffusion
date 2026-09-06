"""Does the installed engine builder support dynamic-shape batch profiles? (issue #3 step 4)

Spec 7.3(c) proposes one TensorRT engine with a min/max batch *range*, so a Render
Plan with a varying number of crops needs neither padding (option a) nor several
resident engines (option b). Whether that is available is a property of the pinned
StreamDiffusion 0.1.1 and of how `wrapper.py` calls it - a code question, answerable
without building the 61 GB matrix the issue refuses to build.

The finding these tests pin down has two halves:

1. **The library supports it.** `accelerate_with_tensorrt` takes `min_batch_size` /
   `max_batch_size`, `EngineBuilder.build` takes `build_static_batch=False`, and
   `get_minmax_dims` widens the profile to `self.min_batch .. self.max_batch`
   whenever `static_batch` is false. Nothing needs patching upstream.

2. **This app collapses it.** `wrapper.py` passes the *same* value as both the min
   and the max at all three call sites, so every profile TensorRT sees is a single
   point and each batch size keys a separate engine. That is the reason 7.3(c) is
   untested here, and it is a change to `wrapper.py` rather than to the pinned
   dependency.

These read source rather than importing it: `streamdiffusion.acceleration.tensorrt`
imports torch and tensorrt, and `wrapper.py` imports torch, none of which may happen
in the GPU-free tier.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

from sourceloader import ROOT

TENSORRT_PKG = "streamdiffusion/acceleration/tensorrt"


def streamdiffusion_root() -> Path:
    """Locate the installed package without executing its `__init__`."""
    spec = importlib.util.find_spec("streamdiffusion")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("streamdiffusion is not installed in this environment")
    return Path(list(spec.submodule_search_locations)[0]).parent


def parse(path: Path) -> ast.Module:
    # utf-8-sig: wrapper.py carries a BOM, which `ast.parse` rejects as non-printable.
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function named {name}")


def defaults_by_name(func: ast.FunctionDef) -> dict:
    """`{parameter: literal default}` for the keyword parameters that have one."""
    args = func.args.posonlyargs + func.args.args
    padded = [None] * (len(args) - len(func.args.defaults)) + list(func.args.defaults)
    return {
        arg.arg: (ast.literal_eval(default) if default is not None else None)
        for arg, default in zip(args, padded)
        if default is not None
    }


def test_the_library_accepts_a_batch_range_rather_than_one_batch_size():
    """`accelerate_with_tensorrt(min_batch_size=..., max_batch_size=...)` - spec 7.3(c)."""
    tree = parse(streamdiffusion_root() / TENSORRT_PKG / "__init__.py")
    params = {arg.arg for arg in find_function(tree, "accelerate_with_tensorrt").args.args}
    assert {"min_batch_size", "max_batch_size"} <= params
    # `opt_batch_size` is the shape TensorRT tunes its tactics for inside that range.
    assert "opt_batch_size" in {
        arg.arg for arg in find_function(tree, "compile_unet").args.args
    }


def test_the_builder_defaults_to_a_non_static_batch():
    """`build_static_batch=False` by default: the range is used unless it is switched off."""
    build = find_function(parse(streamdiffusion_root() / TENSORRT_PKG / "builder.py"), "build")
    defaults = defaults_by_name(build)
    assert defaults["build_static_batch"] is False
    # Dynamic *resolution* is a separate axis, and it does default to off.
    assert defaults["build_dynamic_shape"] is False


def test_the_profile_widens_to_the_models_batch_range_when_not_static():
    """`get_minmax_dims` is where a range becomes a TensorRT optimisation profile."""
    source = (streamdiffusion_root() / TENSORRT_PKG / "models.py").read_text(encoding="utf-8")
    assert "min_batch = batch_size if static_batch else self.min_batch" in source
    assert "max_batch = batch_size if static_batch else self.max_batch" in source


def test_the_app_collapses_the_range_so_every_batch_size_keys_its_own_engine():
    """The finding: `wrapper.py` passes min == max, so 7.3(c) is available but unused.

    Asserted over the call sites rather than the engine directory names, because the
    names are a consequence - `--max_batch-1--min_batch-1--` is what this produces.
    """
    tree = parse(Path(ROOT) / "wrapper.py")
    collapsed = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        if "min_batch_size" not in keywords or "max_batch_size" not in keywords:
            continue
        collapsed.append(ast.dump(keywords["min_batch_size"])
                         == ast.dump(keywords["max_batch_size"]))

    assert collapsed, "wrapper.py no longer names a batch range at all"
    assert all(collapsed), (
        "wrapper.py now passes a real min/max batch range - spec 7.3(c) has become "
        "reachable, so update the recommendation in the spec along with this test"
    )


def test_cuda_graphs_are_off_so_a_dynamic_profile_would_not_conflict():
    """A captured CUDA graph fixes the shapes; 7.3(c) would be blocked if one were used."""
    source = (Path(ROOT) / "wrapper.py").read_text(encoding="utf-8-sig")
    assert "use_cuda_graph=True" not in source
    assert "use_cuda_graph=False" in source
