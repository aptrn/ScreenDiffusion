"""Picking a style LoRA from a list, and being told whether it has an engine.

Issue #44's second half. The style LoRAs live in one known directory, there are
three of them, and the window offered a file browser - which is the wrong control
for three files in a known place, and is also how the wrong path spelling got into
the cache key in the first place.

The list mirrors the base-model picker (`local_model_paths` / `model_label` /
`_model_choices`, issue #38 step 6) for the same reasons and with the same escape
hatch: Browse stays, because a LoRA outside the models root is still a legal
answer.

Tk is never instantiated. The listing rules and the engine sentence are pure
top-level functions executed straight out of `main_gpu_addon.py`; where they reach
the widgets is read off the source, like every other GUI wiring test here.
"""

from __future__ import annotations

from pathlib import Path

from guinamespace import helpers
from guisource import calls_named, gui_method, mentions

import engine_cache

LORAS = ("local_lora_paths", "lora_label", "LORAS_SUBDIR", "LORA_SUFFIXES",
         "LCM_LORA_FILENAME", "NO_LOCAL_LORAS", "ADD_LORA_PROMPT",
         "DEFAULT_LORA_SCALE", "_lora_phrase")
ENGINE = ("engine_configuration", "EngineConfiguration", "_engine_state_line",
          "_engine_missing_warning", "engine_rebuild_needed", "TENSORRT",
          "model_label", "_steps_phrase", "_lora_phrase", "lora_label")


def a_loras_dir(root: Path, *names: str) -> Path:
    symbols = helpers(*LORAS)
    folder = root / symbols["LORAS_SUBDIR"]
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"")
    return folder


# --- the dropdown lists what is on disk --------------------------------------


def test_every_lora_file_under_the_models_root_is_offered(tmp_path):
    a_loras_dir(tmp_path, "style-loving-vincent.safetensors",
                "style-illusion-pattern.safetensors")

    paths = helpers(*LORAS)["local_lora_paths"](models_root=tmp_path)

    assert [Path(path).name for path in paths] == [
        "style-illusion-pattern.safetensors", "style-loving-vincent.safetensors"]


def test_the_lcm_lora_is_not_a_style_and_is_not_offered(tmp_path):
    """Issue #44's third trap. It sits in the same directory and is applied by the
    LCM-LoRA switch, so listing it here invites fusing it twice."""
    symbols = helpers(*LORAS)
    a_loras_dir(tmp_path, symbols["LCM_LORA_FILENAME"], "style-a.safetensors")

    paths = symbols["local_lora_paths"](models_root=tmp_path)

    assert [Path(path).name for path in paths] == ["style-a.safetensors"]


def test_a_file_that_is_not_lora_weights_is_not_offered(tmp_path):
    a_loras_dir(tmp_path, "style-a.safetensors", "notes.txt")
    paths = helpers(*LORAS)["local_lora_paths"](models_root=tmp_path)
    assert [Path(path).name for path in paths] == ["style-a.safetensors"]


def test_a_missing_loras_directory_is_not_an_error(tmp_path):
    assert helpers(*LORAS)["local_lora_paths"](models_root=tmp_path) == []


def test_every_offered_suffix_is_one_the_browse_dialog_would_accept():
    symbols = helpers(*LORAS)
    for suffix in symbols["LORA_SUFFIXES"]:
        assert suffix.startswith(".")


def test_a_lora_is_labelled_by_its_filename(tmp_path):
    label = helpers(*LORAS)["lora_label"]
    assert label(str(tmp_path / "style-a.safetensors")) == "style-a.safetensors"
    assert label("") == ""


# --- and says, for that LoRA at that scale, whether the engine exists ---------


def a_configuration(tmp_path, **overrides):
    model = tmp_path / "models" / "sd-v1-5-fp16"
    model.mkdir(parents=True, exist_ok=True)
    fields = dict(model_path=str(model), acceleration="tensorrt",
                  use_lcm_lora=True, steps=4, frame_buffer_size=1,
                  lora_dict=None, engines_root=tmp_path / "engines")
    fields.update(overrides)
    return helpers(*ENGINE)["engine_configuration"](**fields)


def test_the_engine_sentence_names_the_fused_lora_and_its_scale(tmp_path):
    """Issue #44's fourth trap: the scale keys the engine, so a sentence that named
    only the file would report `cached` about a different build."""
    configuration = a_configuration(
        tmp_path, lora_dict={r"C:\loras\style-loving-vincent.safetensors": 0.9})

    line = helpers(*ENGINE)["_engine_state_line"](configuration)

    assert "style-loving-vincent.safetensors" in line
    assert "0.90" in line


def test_the_sentence_about_a_bare_model_is_the_one_it_always_was(tmp_path):
    line = helpers(*ENGINE)["_engine_state_line"](a_configuration(tmp_path))
    assert "sd-v1-5-fp16" in line and "4 steps" in line
    assert "@" not in line


def test_a_cached_lora_engine_is_reported_as_cached(tmp_path):
    lora = {r"C:\loras\style-a.safetensors": 0.9}
    configuration = a_configuration(tmp_path, lora_dict=lora)
    built = Path(configuration.engines_root) / configuration.engine_dir
    built.mkdir(parents=True)
    (built / engine_cache.UNET_ENGINE).write_bytes(b"")

    line = helpers(*ENGINE)["_engine_state_line"](
        a_configuration(tmp_path, lora_dict=lora))

    assert line.startswith("Engine: cached")


def test_the_same_lora_at_another_scale_is_another_engine(tmp_path):
    lora = r"C:\loras\style-a.safetensors"
    assert a_configuration(tmp_path, lora_dict={lora: 0.9}).engine_dir != \
        a_configuration(tmp_path, lora_dict={lora: 1.0}).engine_dir


def test_two_spellings_of_one_lora_name_one_engine(tmp_path):
    """The window asks the same question the harness does, through the same rule."""
    lora = tmp_path / "loras" / "style-a.safetensors"
    lora.parent.mkdir(parents=True)
    lora.write_bytes(b"")
    assert a_configuration(tmp_path, lora_dict={str(lora): 0.9}).engine_dir == \
        a_configuration(
            tmp_path,
            lora_dict={str(lora).replace("\\", "/"): 0.9}).engine_dir


def test_the_build_warning_names_the_lora_set_too(tmp_path):
    configuration = a_configuration(tmp_path,
                                    lora_dict={r"C:\loras\style-a.safetensors": 0.9})
    message = helpers(*ENGINE)["_engine_missing_warning"](configuration)
    assert "style-a.safetensors" in message


# --- where the window puts it ------------------------------------------------


def test_the_picker_offers_what_is_on_disk():
    method = gui_method("_lora_choices")
    assert calls_named(method, "local_lora_paths"), "the picker reads nothing"
    assert calls_named(method, "lora_label")


def test_neither_resting_value_of_the_picker_is_a_filename():
    """The menu adds rather than selects, so it rests on a prompt - and both
    prompts have to fall through `_on_lora_chosen`'s search for a listed file."""
    symbols = helpers(*LORAS)
    for value in (symbols["ADD_LORA_PROMPT"], symbols["NO_LOCAL_LORAS"]):
        assert not any(value.endswith(suffix)
                       for suffix in symbols["LORA_SUFFIXES"])


def test_the_window_and_the_harness_look_in_the_same_directory():
    """`bench.models` stages a LoRA where the window offers one, or the harness
    builds engines for files the window can never list."""
    from bench import models

    symbols = helpers(*LORAS)
    assert symbols["LORAS_SUBDIR"] == models.LORAS_SUBDIR
    assert symbols["LCM_LORA_FILENAME"] == models.LCM_LORA_FILENAME


def test_a_new_lora_is_fused_at_the_scale_the_style_engines_were_built_at():
    """Issue #44's fourth trap: a default that drifted off the built value would
    report `not cached` about a build that is on disk."""
    from bench.scenarios import ScenarioConfig

    assert helpers(*LORAS)["DEFAULT_LORA_SCALE"] == ScenarioConfig.lora_scale


def test_choosing_from_the_list_and_browsing_go_through_one_place():
    """Two doors on one setting; only one of them refreshing the engine line is how
    the window starts reporting an engine for a configuration it is not in."""
    for method in ("_on_lora_chosen", "_add_lora"):
        assert calls_named(gui_method(method), "_add_lora_path"), method


def test_browse_is_kept():
    """Issue #44 step 4: a LoRA outside the models root is still a legal answer."""
    assert calls_named(gui_method("_add_lora"), "askopenfilenames")


def test_every_change_to_the_lora_set_refreshes_the_engine_line():
    """Whether this configuration has an engine has to be visible *before* Start,
    and the scale is part of the configuration."""
    for method in ("_add_lora_path", "_remove_lora", "_on_lora_scale_changed"):
        assert calls_named(gui_method(method), "_refresh_engine_state"), method


def test_the_engine_line_is_the_one_sentence_builder():
    assert calls_named(gui_method("_refresh_engine_state"), "_engine_state_line")


def test_one_file_added_twice_is_one_entry():
    """Two spellings of one path are one LoRA - `lora_dict` would otherwise fuse it
    twice at the same scale while keying a single engine."""
    method = gui_method("_add_lora_path")
    assert mentions(method, "normalize_lora_key"), \
        "duplicates are matched on the raw string again"
