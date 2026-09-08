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

from sourceloader import ROOT

from engine_cache import (
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
    normalize_lora_key,
    unet_batch_size,
)

WRAPPER = (ROOT / "wrapper.py").read_text(encoding="utf-8")


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
    """The hash is unchanged; what goes into it is the normalised set (issue #44)."""
    lora = {"C:/loras/vincent.safetensors": 0.9}
    normalised = {normalize_lora_key(name): scale for name, scale in lora.items()}
    expected = hashlib.sha1(
        repr(sorted(normalised.items())).encode("utf-8")).hexdigest()[:8]
    assert lora_fingerprint(lora) == expected
    assert lora_fingerprint(None) == NO_LORA
    assert lora_fingerprint({}) == NO_LORA


def test_two_scales_of_one_lora_are_two_engines():
    """Which is the whole reason a style is a release-time decision on this path."""
    assert lora_fingerprint({"a": 0.5}) != lora_fingerprint({"a": 0.9})


# --- one file is one key, however its path was spelled (issue #44) ------------


def test_the_two_windows_spellings_of_one_file_are_one_engine(tmp_path):
    """The defect. Tk's file dialog returns forward slashes on Windows and
    `pathlib` returns backslashes, so the window could never find an engine the
    harness had built - and adding one LoRA by two routes cost two ~5 GB builds."""
    lora = tmp_path / "loras" / "style-loving-vincent.safetensors"
    lora.parent.mkdir(parents=True)
    lora.write_bytes(b"")
    backslashes = str(lora)
    forward_slashes = backslashes.replace("\\", "/")

    assert backslashes != forward_slashes, "no separator to disagree about here"
    assert (lora_fingerprint({backslashes: 1.0})
            == lora_fingerprint({forward_slashes: 1.0}))


def test_the_route_taken_to_a_file_is_not_part_of_its_key(tmp_path):
    """A relative spelling, and a detour through `..`, name the same engine."""
    lora = tmp_path / "loras" / "style.safetensors"
    lora.parent.mkdir(parents=True)
    lora.write_bytes(b"")
    detour = tmp_path / "loras" / ".." / "loras" / "style.safetensors"

    assert lora_fingerprint({str(lora): 0.9}) == lora_fingerprint({str(detour): 0.9})


def test_the_scale_is_read_as_a_number_rather_than_as_whatever_was_typed():
    """`1` and `1.0` are one scale. They are two `repr`s, and the key is a hash of
    one - the same defect as the two separators, on the other half of the pair."""
    lora = r"C:\loras\a.safetensors"
    assert lora_fingerprint({lora: 1}) == lora_fingerprint({lora: 1.0})


def test_normalising_does_not_rename_an_already_normal_key(tmp_path):
    """Issue #44's second trap: the committed style engines are keyed on the
    absolute backslash spelling `bench.models.lora_path` produces, so the
    normalisation has to leave that spelling exactly where it was or orphan
    ~5 GB apiece. Checked as the hash the old rule computed, not as prose."""
    lora = tmp_path / "loras" / "style-loving-vincent.safetensors"
    lora.parent.mkdir(parents=True)
    lora.write_bytes(b"")
    fused = {str(lora): 1.0}
    old_rule = hashlib.sha1(
        repr(sorted(fused.items())).encode("utf-8")).hexdigest()[:8]

    assert lora_fingerprint(fused) == old_rule


def test_a_repo_id_is_a_name_and_is_not_absolutised(tmp_path, monkeypatch):
    """`stream.load_lora` also takes a Hugging Face id, and `org/model` is the same
    shape as `loras/style.safetensors`. Absolutising it would key the engine on the
    working directory the app happened to be started from."""
    monkeypatch.chdir(tmp_path)
    first = lora_fingerprint({"latent-consistency/lcm-lora-sdv1-5": 1.0})
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")

    assert lora_fingerprint({"latent-consistency/lcm-lora-sdv1-5": 1.0}) == first


def test_the_wrapper_asks_engine_cache_rather_than_hashing_its_own():
    """The issue's first trap. The fingerprint is shared with the harness and the
    window, so a second copy inside `wrapper.py` is exactly the two-rules bug this
    module exists to prevent - and it was the copy that keyed the engines."""
    assert "from engine_cache import" in WRAPPER
    assert "lora_fingerprint(lora_dict)" in WRAPPER
    assert "hashlib.sha1" not in WRAPPER


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
