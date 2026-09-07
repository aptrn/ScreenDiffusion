"""The two settings the app could not start correctly on, and what a rebuild costs.

Issue #40, steps 1 and 4. Both defects were found by driving the app rather than by
reading it: the model path started blank, so nothing could start until someone
pasted an absolute path, and acceleration started on `xformers` while every
committed benchmark in this repo is a `tensorrt` run against a cached engine.

Tk is never instantiated. The resolution rule and the warning text are pure
top-level functions executed straight out of `main_gpu_addon.py`; where they reach
the widgets is read off the source, like every other GUI wiring test here.
"""

import os
from pathlib import Path

import pytest

from guisource import assignment_to, calls_named, gui_method, mentions
from sourceloader import load_symbols

FAKE_APP_ROOT = Path(r"C:\Program Files\Screen Diffusion")

_symbols = load_symbols(
    "main_gpu_addon.py",
    [
        "ACCELERATIONS",
        "NO_ACCELERATION",
        "DEFAULT_ACCELERATION",
        "ENGINE_BUILD_SIZE",
        "ENGINE_BUILD_TIME",
        "ENGINE_KEYED_SETTINGS",
        "LOCAL_MODEL_NAMES",
        "MODEL_INDEX",
        "SD_MODELS_DIR_ENV",
        "TENSORRT",
        "XFORMERS",
        "_engine_rebuild_warning",
        "_resolve_cache_dir",
        "_unquoted_path",
        "engine_rebuild_needed",
        "is_diffusers_dir",
        "resolve_local_model_path",
        "resolve_models_dir",
    ],
    extra_globals={"os": os, "Path": Path, "APP_ROOT": FAKE_APP_ROOT},
)
ACCELERATIONS = _symbols["ACCELERATIONS"]
DEFAULT_ACCELERATION = _symbols["DEFAULT_ACCELERATION"]
ENGINE_BUILD_SIZE = _symbols["ENGINE_BUILD_SIZE"]
ENGINE_BUILD_TIME = _symbols["ENGINE_BUILD_TIME"]
ENGINE_KEYED_SETTINGS = _symbols["ENGINE_KEYED_SETTINGS"]
LOCAL_MODEL_NAMES = _symbols["LOCAL_MODEL_NAMES"]
MODEL_INDEX = _symbols["MODEL_INDEX"]
SD_MODELS_DIR_ENV = _symbols["SD_MODELS_DIR_ENV"]
TENSORRT = _symbols["TENSORRT"]
_engine_rebuild_warning = _symbols["_engine_rebuild_warning"]
engine_rebuild_needed = _symbols["engine_rebuild_needed"]
is_diffusers_dir = _symbols["is_diffusers_dir"]
resolve_local_model_path = _symbols["resolve_local_model_path"]


def _a_model_in(root: Path, name: str) -> Path:
    """A directory the worker could actually load: a diffusers folder."""
    folder = root / name
    folder.mkdir(parents=True)
    (folder / MODEL_INDEX).write_text("{}", encoding="utf-8")
    return folder


# --- the model path ----------------------------------------------------------


def test_a_local_model_fills_the_field_that_used_to_start_blank(tmp_path):
    model = _a_model_in(tmp_path, LOCAL_MODEL_NAMES[0])
    assert resolve_local_model_path(models_root=tmp_path) == str(model)


def test_nothing_local_is_still_blank(tmp_path):
    """The old behaviour, kept: Browse and Download are what a bare machine has."""
    assert resolve_local_model_path(models_root=tmp_path) == ""


def test_a_missing_models_root_is_not_an_error(tmp_path):
    assert resolve_local_model_path(models_root=tmp_path / "never downloaded") == ""


def test_a_folder_without_a_model_index_is_not_a_model(tmp_path):
    """A cancelled download leaves a directory behind; starting on it would fail."""
    (tmp_path / LOCAL_MODEL_NAMES[0]).mkdir()
    assert resolve_local_model_path(models_root=tmp_path) == ""
    assert is_diffusers_dir(tmp_path / LOCAL_MODEL_NAMES[0]) is False


def test_an_explicit_path_still_wins(tmp_path):
    """`LOCAL_MODEL_PATH` is an override, and overriding is what it is for."""
    _a_model_in(tmp_path, LOCAL_MODEL_NAMES[0])
    assert resolve_local_model_path(r"D:\elsewhere\sd-turbo", models_root=tmp_path) \
        == r"D:\elsewhere\sd-turbo"


def test_a_quoted_override_is_unquoted_like_every_other_path(tmp_path):
    assert resolve_local_model_path('"D:\\elsewhere"', models_root=tmp_path) == "D:\\elsewhere"


def test_any_diffusers_folder_under_the_root_will_do(tmp_path):
    model = _a_model_in(tmp_path, "some-other-checkpoint")
    assert resolve_local_model_path(models_root=tmp_path) == str(model)


def test_the_named_model_is_preferred_over_one_that_merely_sorts_first(tmp_path):
    _a_model_in(tmp_path, "aardvark-diffusion")
    named = _a_model_in(tmp_path, LOCAL_MODEL_NAMES[0])
    assert resolve_local_model_path(models_root=tmp_path) == str(named)


def test_the_models_root_is_the_shared_one_when_no_root_is_given(tmp_path):
    """`SD_MODELS_DIR` points every worktree at one copy - the same door as the caches."""
    model = _a_model_in(tmp_path, LOCAL_MODEL_NAMES[0])
    resolved = resolve_local_model_path(environ={SD_MODELS_DIR_ENV: str(tmp_path)})
    assert Path(resolved) == model


def test_the_gui_starts_its_model_field_from_the_resolution():
    assign = assignment_to(gui_method("__init__"), "model_var")
    assert mentions(assign, "resolve_local_model_path"), \
        "the model field is still seeded from a constant that starts blank"


# --- acceleration ------------------------------------------------------------


def test_acceleration_defaults_to_the_path_every_benchmark_measured():
    assert DEFAULT_ACCELERATION == TENSORRT


def test_the_default_is_one_of_the_offered_values():
    assert DEFAULT_ACCELERATION in ACCELERATIONS


def test_the_gui_starts_its_acceleration_field_from_that_default():
    assign = assignment_to(gui_method("__init__"), "accel_var")
    assert mentions(assign, "DEFAULT_ACCELERATION"), \
        "the acceleration field is seeded from a literal, which is how it drifted"


def test_the_combo_offers_exactly_the_known_accelerations():
    build = gui_method("_build_ui")
    combo = assignment_to(build, "_w_accel_combo")
    assert mentions(combo, "ACCELERATIONS"), \
        "the combo's values are spelt a second time and can disagree with the default"


# --- the engine rebuild warning ----------------------------------------------


@pytest.mark.parametrize("setting", ["step count", "batch size", "LoRA set"])
def test_every_engine_keyed_setting_has_a_warning(setting):
    assert setting in ENGINE_KEYED_SETTINGS


@pytest.mark.parametrize("setting", sorted(ENGINE_KEYED_SETTINGS))
def test_the_warning_names_the_setting_and_both_halves_of_the_cost(setting):
    warning = _engine_rebuild_warning(setting)
    assert setting in warning
    assert ENGINE_BUILD_TIME in warning, "the user is not told it takes minutes"
    assert ENGINE_BUILD_SIZE in warning, "the user is not told it costs gigabytes"
    assert ENGINE_KEYED_SETTINGS[setting] in warning


def test_the_cost_is_the_measured_one():
    """Measured at 512^2 on an RTX 3080 laptop - CLAUDE.md and spec 7.2."""
    assert "5 GB" in ENGINE_BUILD_SIZE
    assert ENGINE_BUILD_TIME == "15-25 minutes"


def test_only_the_path_that_compiles_an_engine_warns():
    assert engine_rebuild_needed(TENSORRT) is True
    for other in ACCELERATIONS:
        if other != TENSORRT:
            assert engine_rebuild_needed(other) is False


def test_the_confirmation_goes_through_that_rule_and_asks_the_user():
    confirm = gui_method("_confirm_engine_rebuild")
    assert mentions(confirm, "engine_rebuild_needed"), \
        "the warning fires on paths that build no engine, or not on the one that does"
    assert mentions(confirm, "_engine_rebuild_warning")
    assert mentions(confirm, "askokcancel")


@pytest.mark.parametrize("handler", ["_add_step", "_remove_step"])
def test_changing_the_step_count_asks_before_it_happens(handler):
    """The step *count* keys a distinct engine; a step's *value* is a runtime update."""
    assert mentions(gui_method(handler), "_confirm_engine_rebuild")


def test_the_loras_and_the_batch_size_ask_at_start():
    """Neither is editable while running, so start is before the build either way."""
    asked = calls_named(gui_method("_on_start"), "_confirm_engine_rebuild")
    settings = {call.args[0].value for call in asked if call.args}
    assert {"LoRA set", "batch size"} <= settings, f"only warned about {settings}"


def test_a_blank_model_path_says_where_to_point_it():
    """Step 1's other half: the failure named the empty string and nothing else."""
    verify = load_symbols(
        "main_gpu_addon.py",
        ["verify_local_model_path_dir", "_resolve_cache_dir", "_unquoted_path",
         "resolve_models_dir", "SD_MODELS_DIR_ENV"],
        extra_globals={"os": os, "Path": Path, "APP_ROOT": FAKE_APP_ROOT},
    )["verify_local_model_path_dir"]
    with pytest.raises(FileNotFoundError) as raised:
        verify("")
    assert "models" in str(raised.value), \
        "the user is told the path is missing but not where a model would be"
