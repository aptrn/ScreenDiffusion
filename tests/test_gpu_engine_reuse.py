"""A cached TensorRT engine is reused when the process starts from another cwd (issue #9).

Needs the shared caches: point `SD_MODELS_DIR` / `SD_ENGINES_DIR` at a checkout that
already holds `sd-turbo-fp16` and its 512x512 batch-1 engine, or run from one. Without
them the test skips rather than spending several minutes compiling ~5.1 GB of engine.

The load itself is the assertion: it happens from a cwd with a space in it, and
afterwards neither the shared engines root nor that cwd may have gained a file.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from sourceloader import ROOT, load_symbols

pytestmark = pytest.mark.gpu

MODEL_NAME = "sd-turbo-fp16"
# The configuration the app builds by default: one step, one frame, 512x512, no LoRA.
ENGINE_CONFIG = (
    f"{MODEL_NAME}--lcm_lora-False--tiny_vae-True--max_batch-1--min_batch-1"
    "--res-512x512--lora-none--mode-img2img"
)
ENGINE_FILES = ("unet.engine", "vae_encoder.engine", "vae_decoder.engine")

_paths = load_symbols(
    "main_gpu_addon.py",
    ["SD_MODELS_DIR_ENV", "SD_ENGINES_DIR_ENV", "_resolve_cache_dir",
     "resolve_models_dir", "resolve_engines_dir"],
    extra_globals={"os": os, "Path": Path, "APP_ROOT": ROOT},
)
_resolve_engine_dir = load_symbols(
    "wrapper.py", ["_resolve_engine_dir"],
    extra_globals={"os": os, "Path": Path, "REPO_ROOT": ROOT},
)["_resolve_engine_dir"]


def _load_wrapper_module():
    """Load wrapper.py by path, the way the worker does - it is not an importable module."""
    spec = importlib.util.spec_from_file_location("wrapper", ROOT / "wrapper.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wrapper"] = mod
    spec.loader.exec_module(mod)
    return mod


def _snapshot(engines_root: Path):
    """Every file under the engines root, with size and mtime. A build would change it."""
    return {
        str(p.relative_to(engines_root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in engines_root.rglob("*") if p.is_file()
    }


@pytest.fixture
def shared_caches():
    models_root = _paths["resolve_models_dir"]()
    engines_root = _paths["resolve_engines_dir"]()
    model_dir = models_root / MODEL_NAME
    if not model_dir.is_dir():
        pytest.skip(f"no {MODEL_NAME} under {models_root} - set {_paths['SD_MODELS_DIR_ENV']}")
    cached = engines_root / ENGINE_CONFIG
    if not all((cached / name).is_file() for name in ENGINE_FILES):
        pytest.skip(f"no cached engine at {cached} - set {_paths['SD_ENGINES_DIR_ENV']}")
    return model_dir, engines_root


def test_a_cached_engine_is_reused_from_a_different_cwd(shared_caches, monkeypatch, tmp_path):
    model_dir, engines_root = shared_caches
    elsewhere = tmp_path / "a cwd with spaces"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    # Fail here, in milliseconds, rather than after a multi-minute engine build.
    assert _resolve_engine_dir(None, environ=dict(os.environ), base_dir=ROOT) == engines_root

    before = _snapshot(engines_root)
    wrapper = _load_wrapper_module()
    stream = wrapper.StreamDiffusionWrapper(
        model_id_or_path=str(model_dir), t_index_list=[35], frame_buffer_size=1,
        width=512, height=512, warmup=1, acceleration="tensorrt", mode="img2img",
        use_denoising_batch=True, cfg_type="none", use_lcm_lora=False, use_tiny_vae=True,
        engine_dir=str(engines_root),
    )
    stream.prepare(prompt="a test frame", num_inference_steps=50)

    from PIL import Image

    out = stream(image=Image.new("RGB", (512, 512), (32, 96, 160)))
    assert out is not None

    assert _snapshot(engines_root) == before, "the engine cache was rewritten, not reused"
    assert not (elsewhere / "engines").exists(), "an engines dir was created next to the cwd"
