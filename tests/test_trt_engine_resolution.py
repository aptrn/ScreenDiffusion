"""Every TensorRT engine this app builds is 512x512, whatever the directory says.

Found while running issue #3 step 3. `bench img2img-tensorrt-256x256-b1` built an
engine into a directory named `--res-256x256--` and then, at inference, printed

    Static dimension mismatch while setting input shape for images.
    Set dimensions are [1,3,256,256]. Expected dimensions are [1,3,512,512].

for `images` (VAE encoder), `sample` (UNet) and `latent` (VAE decoder) alike. The
timing that came back - 53.7 ms/frame - is the *512* engine's number, so it is not a
256 measurement and was not committed.

The cause is entirely in this repo, not upstream. `EngineBuilder.build` takes
`opt_image_height` / `opt_image_width`, both defaulting to 512, and
`build_dynamic_shape=False`, which pins the profile to that one size. `wrapper.py`
calls `compile_unet` / `compile_vae_encoder` / `compile_vae_decoder` without any of
them - it passes `opt_batch_size` and nothing else - so the resolution the caller
asked for never reaches the builder. Meanwhile `create_prefix` writes
`res-{self.width}x{self.height}` into the cache directory name, so the *name* varies
by resolution while the *contents* never do.

That is the same stale-engine failure ef85e1a fixed by adding the resolution to the
cache key: the key now varies, but nothing downstream of it does. It is also why the
384x384 engine raises a CUDA illegal memory access (noted at the end of issue #2)
while the 512x512 one is fine - 512 is the only directory whose label happens to be
true.

These tests characterise the bug rather than fix it. Fixing it is a `wrapper.py`
change plus ~10 GB of engine rebuilds, which is neither of issue #3's steps 3-6, so
it wants its own issue. When that lands, these tests fail - which is the point: the
finding cannot rot into folklore, and the fix has to come past this file.

Source-level, like `test_trt_dynamic_shape.py`: `streamdiffusion.acceleration.tensorrt`
imports torch and tensorrt and `wrapper.py` imports torch, none of which may happen in
the GPU-free tier.
"""

import ast
from pathlib import Path

import pytest

from sourceloader import ROOT
from test_trt_dynamic_shape import (
    defaults_by_name,
    find_function,
    parse,
    streamdiffusion_root,
)

COMPILE_CALLS = ("compile_unet", "compile_vae_encoder", "compile_vae_decoder")
# What `EngineBuilder.build` uses when the caller says nothing, and what therefore
# ends up baked into every engine on this machine.
BUILDER_DEFAULT_RESOLUTION = 512


def builder_build() -> ast.FunctionDef:
    path = streamdiffusion_root() / "streamdiffusion/acceleration/tensorrt/builder.py"
    if not path.is_file():
        pytest.skip(f"{path} is not present in this environment")
    return find_function(parse(path), "build")


def keyword_names(call: ast.Call) -> set:
    return {keyword.arg for keyword in call.keywords}


def wrapper_calls(name: str) -> list:
    """Every call to `name` in wrapper.py, as AST nodes."""
    tree = parse(Path(ROOT) / "wrapper.py")
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == name]


def test_the_builder_bakes_in_512_unless_the_caller_says_otherwise():
    """`opt_image_height` / `opt_image_width` default to 512, and the shape is static."""
    defaults = defaults_by_name(builder_build())

    assert defaults["opt_image_height"] == BUILDER_DEFAULT_RESOLUTION
    assert defaults["opt_image_width"] == BUILDER_DEFAULT_RESOLUTION
    assert defaults["build_dynamic_shape"] is False, (
        "a dynamic-shape profile would have absorbed the mismatch instead of raising"
    )


def test_wrapper_never_tells_the_builder_which_resolution_it_asked_for():
    """The bug in one assertion: the requested size reaches no compile call site."""
    for name in COMPILE_CALLS:
        calls = wrapper_calls(name)
        assert calls, f"no call to {name} in wrapper.py"
        for call in calls:
            passed = keyword_names(call)
            assert "opt_image_height" not in passed
            assert "opt_image_width" not in passed
            assert "engine_build_options" not in passed, (
                "engine_build_options is the other door the resolution could go through"
            )
            assert "opt_batch_size" in passed, (
                "batch *is* forwarded - which is why the batch axis of the issue #3 "
                "sweep is trustworthy on TensorRT and the resolution axis is not"
            )


def test_the_cache_key_promises_a_resolution_the_engine_does_not_have():
    """`create_prefix` varies by resolution; nothing downstream of it does.

    So a directory named `--res-256x256--` holds a 512x512 engine, and the app loads
    it without complaint until the shapes collide at inference time.
    """
    prefix = find_function(parse(Path(ROOT) / "wrapper.py"), "create_prefix")
    source = ast.unparse(prefix)

    assert "res-" in source and "self.width" in source and "self.height" in source, (
        "the directory name still claims a resolution"
    )
