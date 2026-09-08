"""The engine cache key, and the mirror rule that keeps it honest (issue #38).

`create_prefix` inside `wrapper.py` is the real one. It cannot be imported here -
`wrapper.py` imports torch, and this tier has no CUDA device - so `engine_cache.py`
is a stdlib copy of the same rule, and this file is what stops the two drifting:
the wrapper's own source is read and its naming expression exercised against the
copy's, over the same inputs.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from sourceloader import ROOT

from engine_cache import (
    CFG_FULL,
    CFG_INITIALIZE,
    CFG_NONE,
    CFG_SELF,
    CFG_TYPES,
    ENGINE_BUILD_SIZE,
    ENGINE_BUILD_TIME,
    MIN_FREE_BYTES_FOR_ENGINE_BUILD,
    NO_LORA,
    UNET_ENGINE,
    engine_dir_name,
    engine_is_cached,
    enough_free_space,
    free_bytes,
    is_turbo_model,
    lora_fingerprint,
    model_key,
    unet_batch_size,
)

WRAPPER = (ROOT / "wrapper.py").read_text(encoding="utf-8")


def pipeline_source() -> str:
    """StreamDiffusion's own `pipeline.py`, out of the installed package.

    Located rather than imported, and located by the *top-level* package: a
    `find_spec` for a submodule imports its parent, and `streamdiffusion/__init__`
    imports torch. This tier has no CUDA device.
    """
    import importlib.util

    spec = importlib.util.find_spec("streamdiffusion")
    assert spec is not None, "streamdiffusion is not installed"
    locations = list(spec.submodule_search_locations or [])
    assert locations, "streamdiffusion is not a package"
    return (Path(locations[0]) / "pipeline.py").read_text(encoding="utf-8")


def test_the_directory_name_matches_the_one_the_wrapper_builds():
    """The mirror rule. `create_prefix`'s f-string, read out of the wrapper's own
    source and rendered with the same values - if either side is edited alone, the
    guard starts answering about a directory the build never writes."""
    match = re.search(r'return \(\s*(f"[^)]*?)\s*\)\n', WRAPPER, re.S)
    assert match, "create_prefix's return expression moved"
    expression = match.group(1)
    lora = {"a.safetensors": 0.9}
    values = dict(base="sd-turbo-fp16", use_lcm_lora=False, use_tiny_vae=True,
                  max_batch_size=4, min_batch_size=4,
                  lora_fingerprint=lora_fingerprint(lora))

    class _Self:
        width, height, mode = 512, 512, "img2img"

    rendered = eval(  # noqa: S307 - the wrapper's own literal, not user input
        "".join(line.strip() for line in expression.splitlines()),
        {"self": _Self()}, values)
    assert rendered == engine_dir_name(
        "sd-turbo-fp16", use_lcm_lora=False, use_tiny_vae=True, unet_batch=4,
        width=512, height=512, lora_dict=lora)


def test_the_lora_fingerprint_is_the_wrapper_s_sha1_of_the_fused_set():
    lora = {"C:/loras/vincent.safetensors": 0.9}
    expected = hashlib.sha1(
        repr(sorted(lora.items())).encode("utf-8")).hexdigest()[:8]
    assert lora_fingerprint(lora) == expected
    assert lora_fingerprint(None) == NO_LORA
    assert lora_fingerprint({}) == NO_LORA


def test_two_scales_of_one_lora_are_two_engines():
    """Which is the whole reason a style is a release-time decision on this path."""
    assert lora_fingerprint({"a": 0.5}) != lora_fingerprint({"a": 0.9})


def test_the_step_count_is_part_of_the_batch_the_unet_is_built_for():
    assert unet_batch_size(frame_buffer_size=1, steps=1) == 1
    assert unet_batch_size(frame_buffer_size=1, steps=4) == 4
    assert unet_batch_size(frame_buffer_size=2, steps=4) == 8
    assert unet_batch_size(frame_buffer_size=2, steps=4,
                           use_denoising_batch=False) == 2


def test_a_local_folder_contributes_its_own_name_and_a_repo_id_itself(tmp_path):
    folder = tmp_path / "sd-v1-5-fp16"
    folder.mkdir()
    assert model_key(folder) == "sd-v1-5-fp16"
    assert model_key("stabilityai/sd-turbo") == "stabilityai/sd-turbo"


def test_turbo_is_recognised_by_the_rule_the_wrapper_uses():
    """`self.sd_turbo = "turbo" in model_id_or_path` - and it decides whether
    LCM-LoRA is fused at all, so the GUI has to answer it the same way."""
    assert 'self.sd_turbo = "turbo" in model_id_or_path' in WRAPPER
    assert is_turbo_model("C:/models/sd-turbo-fp16")
    assert not is_turbo_model("C:/models/sd-v1-5-fp16")


def test_cached_means_there_is_a_built_unet_engine(tmp_path):
    name = engine_dir_name("m", use_lcm_lora=True, use_tiny_vae=True,
                           unet_batch=4, width=512, height=512)
    assert not engine_is_cached(tmp_path, name)
    (tmp_path / name).mkdir(parents=True)
    assert not engine_is_cached(tmp_path, name)
    (tmp_path / name / UNET_ENGINE).write_bytes(b"")
    assert engine_is_cached(tmp_path, name)


def test_the_free_disk_floor_is_the_twenty_gigabytes_the_gate_names():
    assert MIN_FREE_BYTES_FOR_ENGINE_BUILD == 20 * 1024 ** 3


def test_a_volume_that_cannot_be_read_has_not_been_shown_to_have_room():
    def unreadable(_path):
        raise OSError("no such volume")

    assert free_bytes("nowhere", usage=unreadable) == 0
    assert not enough_free_space("nowhere", usage=unreadable)


def test_enough_space_is_a_comparison_against_that_floor():
    plenty = lambda _path: (0, 0, MIN_FREE_BYTES_FOR_ENGINE_BUILD)  # noqa: E731
    scarce = lambda _path: (0, 0, MIN_FREE_BYTES_FOR_ENGINE_BUILD - 1)  # noqa: E731
    assert enough_free_space("x", usage=plenty)
    assert not enough_free_space("x", usage=scarce)


def test_the_cost_quoted_to_a_user_is_a_range_over_the_two_measured_machines():
    assert "5 GB" in ENGINE_BUILD_SIZE
    assert "minutes" in ENGINE_BUILD_TIME


# --- classifier-free guidance keys its own batch (issue #45) ------------------


def test_the_cfg_type_is_part_of_the_batch_the_unet_is_built_for():
    """`StreamDiffusion.__init__` derives `trt_unet_batch_size` from `cfg_type`:
    `initialize` runs one extra unconditional latent through the UNet and `full`
    runs a second copy of every one. Both are therefore a different engine from
    the `none` the app ships, and `self` is not."""
    for cfg_type in (CFG_NONE, CFG_SELF):
        assert unet_batch_size(1, 1, cfg_type=cfg_type) == 1
        assert unet_batch_size(1, 4, cfg_type=cfg_type) == 4
    assert unet_batch_size(1, 1, cfg_type=CFG_INITIALIZE) == 2
    assert unet_batch_size(1, 4, cfg_type=CFG_INITIALIZE) == 5
    assert unet_batch_size(1, 1, cfg_type=CFG_FULL) == 2
    assert unet_batch_size(1, 4, cfg_type=CFG_FULL) == 8
    assert unet_batch_size(2, 4, cfg_type=CFG_FULL) == 16


def test_that_batch_rule_is_the_pipeline_s_own():
    """Read out of the installed StreamDiffusion rather than restated: this is a
    mirror of a third-party expression, and the mirror is the only thing that
    stops it drifting on an upgrade."""
    source = pipeline_source()
    assert "self.denoising_steps_num + 1" in source
    assert "2 * self.denoising_steps_num * self.frame_bff_size" in source


def test_without_the_denoising_batch_the_cfg_type_does_not_move_the_batch():
    """`trt_unet_batch_size = self.frame_bff_size` in that branch, whatever the
    guidance does - so a cfg arm there keys the engine the app already has."""
    for cfg_type in CFG_TYPES:
        assert unet_batch_size(2, 4, use_denoising_batch=False,
                               cfg_type=cfg_type) == 2


def test_an_unknown_cfg_type_is_refused_rather_than_silently_batched_as_none():
    """A typo that answered "cached" about an engine the build never writes is
    exactly the failure `engine_cache` exists to prevent."""
    with pytest.raises(ValueError):
        unet_batch_size(1, 1, cfg_type="selfish")
