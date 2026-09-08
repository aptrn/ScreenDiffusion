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
    ["SD_MODELS_DIR_ENV", "SD_ENGINES_DIR_ENV", "_unquoted_path",
     "_resolve_cache_dir", "resolve_models_dir", "resolve_engines_dir"],
    extra_globals={"os": os, "Path": Path, "APP_ROOT": ROOT},
)
_resolve_engine_dir = load_symbols(
    "wrapper.py", ["SD_ENGINES_DIR_ENV", "_unquoted_path", "_resolve_engine_dir"],
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


# --- a style LoRA's engine, found from the spelling the window produces -------
#
# Issue #44's Gate: the committed style engines have to still be found after the
# key was normalised, or ~5 GB apiece is orphaned. That is a question about this
# machine's caches rather than about its GPU, which is why it sits here with the
# other cache-dependent tests instead of in the merge gate's tier - a machine with
# no `engines/` cannot answer it either way.

STYLE_MODEL = "sd-v1-5-fp16"
STYLE_LORA = "loras/style-loving-vincent.safetensors"
# `bench.scenarios.ScenarioConfig.lora_scale`, which is what the committed
# `bench/results/styles/` arms were fused at - and `DEFAULT_LORA_SCALE` in the
# window, so the two ask about one engine.
STYLE_SCALE = 1.0


def test_a_style_lora_engine_is_found_from_either_spelling_of_its_path():
    """Tk's dialog returns forward slashes and `pathlib` returns backslashes. Both
    have to name the directory the harness built, or the window can never find a
    release's engines and every style costs a second ~5 GB build (issue #44)."""
    import engine_cache

    models_root = _paths["resolve_models_dir"]()
    engines_root = _paths["resolve_engines_dir"]()
    lora = models_root / STYLE_LORA
    if not lora.is_file():
        pytest.skip(f"no {STYLE_LORA} under {models_root}")

    from_pathlib = str(lora)
    from_dialog = from_pathlib.replace("\\", "/")
    names = {
        spelling: engine_cache.engine_dir_name(
            models_root / STYLE_MODEL, use_lcm_lora=True, use_tiny_vae=True,
            unet_batch=engine_cache.unet_batch_size(frame_buffer_size=1, steps=4),
            width=512, height=512, lora_dict={spelling: STYLE_SCALE})
        for spelling in (from_pathlib, from_dialog)
    }

    assert len(set(names.values())) == 1, f"two spellings, two engines: {names}"
    built = engines_root / next(iter(names.values()))
    if not built.is_dir():
        pytest.skip(f"no style engine at {built} - nothing committed to check")
    assert engine_cache.engine_is_cached(engines_root, built.name), \
        f"{built} holds no {engine_cache.UNET_ENGINE}"
