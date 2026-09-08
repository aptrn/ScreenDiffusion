"""Which TensorRT engine a configuration needs, and whether it is already built.

`wrapper.py` decides this at the moment it is about to spend the time:
`create_prefix()` names a directory from the model, the LCM-LoRA flag, the tiny-VAE
flag, the UNet batch, the resolution, a fingerprint of the fused style LoRAs and the
mode, and if `unet.engine` is not in it, it builds one. On this hardware that is
about 5 GB and several minutes, and the user finds out by watching the app go quiet.

Two other places have to answer the same question *before* that happens, and neither
can ask `wrapper.py`: it imports torch, and the GUI process must not.

- `StreamGUI`, so Start can say "this configuration has no engine yet" rather than
  going quiet (issue #38 step 6).
- `bench.cli`'s engine-build guard, so a typo does not cost ~5 GB and a coffee.

So the naming rule lives here, in stdlib, and both import it. It is a mirror of
`create_prefix` and has to stay one - `tests/test_engine_cache.py` runs the two over
the same inputs and asserts they still agree, the way `tests/test_bench_paths.py`
holds the three copies of the cache-root rule together.

The free-disk floor is here for the same reason: `bench.disk` refuses a build below
it and the GUI has to refuse the same build at the same number, and two floors is
one floor too many.
"""

from __future__ import annotations

import hashlib
import ntpath
import shutil
from pathlib import Path
from typing import Mapping, Optional, Union

# What `create_prefix` writes when nothing is fused.
NO_LORA = "none"

# The file whose absence means "not built". The VAE engines sit beside it under
# their own prefixes; the UNet is the one that costs the minutes.
UNET_ENGINE = "unet.engine"

ENGINE_DIR_TEMPLATE = ("{model}--lcm_lora-{lcm}--tiny_vae-{tiny}--max_batch-{batch}"
                       "--min_batch-{batch}--res-{width}x{height}--lora-{lora}"
                       "--mode-{mode}")

BYTES_PER_GIB = 1024 ** 3
# 20 GB: room for the engine about to be built (~5 GB), its ONNX scratch, and the
# one already there that it does not replace. Issue #38's Gate names this figure;
# issue #3 set it at 15 and issue #38's arms build several engines in a session.
MIN_FREE_BYTES_FOR_ENGINE_BUILD = 20 * BYTES_PER_GIB

# Measured on this repo's two machines: ~5.0 GB on disk, and minutes rather than
# seconds - 15-25 on an RTX 3080 laptop, ~5 on a 4090. Quoted to a user before a
# build, so it is a range rather than a figure.
ENGINE_BUILD_SIZE = "~5 GB"
ENGINE_BUILD_TIME = "5-25 minutes, depending on the GPU"

_PathLike = Union[str, Path]


def model_key(model_id_or_path: _PathLike) -> str:
    """The `base` part of the directory name, as `create_prefix` computes it.

    A local folder contributes its own name; a repo id contributes itself whole.
    """
    candidate = Path(str(model_id_or_path))
    return candidate.stem if candidate.exists() else str(model_id_or_path)


def is_turbo_model(model_id_or_path: _PathLike) -> bool:
    """Is this a turbo model? `wrapper.py`'s own rule: `"turbo" in the path`.

    It decides whether LCM-LoRA is fused at all, so the GUI needs the same answer
    to offer the right companion settings - a non-turbo model rendered at one step
    with no LCM-LoRA produces noise, and nothing in the window said so.
    """
    return "turbo" in str(model_id_or_path)


def _is_filesystem_path(name: str) -> bool:
    """A path, as opposed to a Hugging Face repo id like `org/model`.

    A drive letter or a backslash is only ever a path. Anything else has to name
    something on disk to be treated as one, because `org/model` and
    `loras/style.safetensors` are the same shape and only the disk tells them
    apart.
    """
    if not name:
        return False
    if ntpath.splitdrive(name)[0] or "\\" in name:
        return True
    return Path(name).exists()


def normalize_lora_key(name: _PathLike) -> str:
    """The spelling of one fused LoRA that keys the engine - one per *file*.

    Issue #44: the key was the raw string, and two callers spell one file two ways.
    Tk's file dialog returns forward slashes on Windows; `pathlib` returns
    backslashes. So the same LoRA at the same scale hashed to two directories, the
    window could never find an engine the harness had built, and adding one LoRA by
    two routes cost two ~5 GB builds.

    `Path.resolve()` is the normalisation: it absolutises, folds the separators,
    follows links and on Windows returns the file's own case - the file rather than
    the route taken to it. An absolute backslash spelling is already its own
    resolution, which is why the engines built before this change are still found.

    A name that is not a path is left alone: `stream.load_lora` also takes a repo
    id, and absolutising `org/model` would key the engine on whatever directory the
    app happened to be started from.
    """
    text = str(name)
    if not _is_filesystem_path(text):
        return text
    return str(Path(text).resolve())


def lora_fingerprint(lora_dict: Optional[Mapping[str, float]]) -> str:
    """The 8 hex characters `create_prefix` puts in the name for a fused LoRA set.

    This is why a style LoRA is a release-time decision on the TensorRT path: every
    distinct set of LoRAs and scales is a different engine. Both halves of an entry
    are read for what they *are* rather than for how they were written - the path
    through `normalize_lora_key`, the scale as a float, so `1` and `1.0` are one
    scale rather than two `repr`s.
    """
    if not lora_dict:
        return NO_LORA
    fused = {normalize_lora_key(name): float(scale)
             for name, scale in lora_dict.items()}
    return hashlib.sha1(
        repr(sorted(fused.items())).encode("utf-8")).hexdigest()[:8]


# The classifier-free-guidance vocabulary, spelt the way `StreamDiffusion` and
# `wrapper.py` spell it. It belongs here because it *keys an engine*: two of the
# four run extra latents through the UNet, so a cfg type is a build as much as a
# step count is (issue #45).
CFG_NONE = "none"
CFG_SELF = "self"
CFG_INITIALIZE = "initialize"
CFG_FULL = "full"
CFG_TYPES = (CFG_NONE, CFG_SELF, CFG_INITIALIZE, CFG_FULL)


def unet_batch_size(frame_buffer_size: int, steps: int,
                    use_denoising_batch: bool = True,
                    cfg_type: str = CFG_NONE) -> int:
    """The batch the UNet engine is compiled for - `stream.trt_unet_batch_size`.

    With `use_denoising_batch`, the denoising steps go through the UNet as a batch,
    so the step *count* is part of the engine key: four steps is a batch-4 engine
    and a different build from a one-step one.

    So is the **cfg type**, and that is the part issue #45 found missing. The
    pipeline's own `__init__` runs one extra unconditional latent under
    `initialize` and a second copy of every latent under `full`, so those two key
    engines the app has never built; `self` and `none` key the one it has. Getting
    this wrong is not a cosmetic error - it is the guard and the window answering
    "cached" about a directory the build never writes.
    """
    if cfg_type not in CFG_TYPES:
        raise ValueError(f"{cfg_type!r} is not a cfg type; expected one of {CFG_TYPES}")
    frames, count = int(frame_buffer_size), int(steps)
    if not use_denoising_batch:
        return frames
    if cfg_type == CFG_INITIALIZE:
        return (count + 1) * frames
    if cfg_type == CFG_FULL:
        return 2 * count * frames
    return count * frames


# The denoising step counts a user may pick at runtime, and the rungs a release
# builds engines for (issue #46, spec 8.12). Here rather than in the window because
# it is the same question this module already exists to answer once: on the batched
# route each rung is its own engine, so the list of rungs the window offers and the
# list the harness sweeps have to be one list or the window offers a quality nobody
# ever measured. 1 is what the app ships and what every committed figure belongs to.
STEP_LADDER = (1, 2, 4, 8)


def engine_dir_name(model_id_or_path: _PathLike, use_lcm_lora: bool,
                    use_tiny_vae: bool, unet_batch: int, width: int, height: int,
                    lora_dict: Optional[Mapping[str, float]] = None,
                    mode: str = "img2img") -> str:
    """The directory `wrapper.py` would look for this configuration's UNet in."""
    return ENGINE_DIR_TEMPLATE.format(
        model=model_key(model_id_or_path), lcm=bool(use_lcm_lora),
        tiny=bool(use_tiny_vae), batch=int(unet_batch), width=int(width),
        height=int(height), lora=lora_fingerprint(lora_dict), mode=mode,
    )


def engine_is_cached(engines_root: _PathLike, dir_name: str) -> bool:
    """Is there a built UNet engine under this name? The whole "will it stall" test."""
    return (Path(engines_root) / dir_name / UNET_ENGINE).is_file()


def free_bytes(path: _PathLike, usage=shutil.disk_usage) -> int:
    """Free bytes on the volume `path` sits on, or 0 when it cannot be read.

    Zero rather than an exception: a volume that cannot be read has not been shown
    to have room, which is the same answer a full one gives.
    """
    try:
        return int(usage(str(path))[2])
    except OSError:
        return 0


def enough_free_space(path: _PathLike,
                      required_bytes: int = MIN_FREE_BYTES_FOR_ENGINE_BUILD,
                      usage=shutil.disk_usage) -> bool:
    """Can this volume hold the engine that is about to be built?"""
    return free_bytes(path, usage=usage) >= int(required_bytes)
