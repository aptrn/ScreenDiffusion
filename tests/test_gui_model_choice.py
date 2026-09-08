"""Choosing the base model from the window, and being told what it will cost.

Issue #38 step 6. Two things the window could not do: offer a second base model
without a code edit, and say - *before* Start - that the configuration in front of
the user has no compiled engine and will spend minutes making one.

The second is not a new warning so much as a real one. `_confirm_engine_rebuild`
already existed but nothing ever looked on disk: it fired whenever the batch size
was not the default or any LoRA was listed, and never when the *model* changed,
which is the one setting that guarantees a different engine.
"""

from __future__ import annotations

from pathlib import Path

from guinamespace import helpers
from guisource import calls_named, gui_method, mentions

import engine_cache

MODELS = ("local_model_paths", "model_label", "model_companions", "ModelCompanions",
          "MODEL_STEPS_SD15", "MODEL_STEPS_TURBO", "DEFAULT_T_INDEX_LIST")
ENGINE = ("engine_configuration", "EngineConfiguration", "_engine_missing_warning",
          "_not_enough_disk_message", "engine_rebuild_needed", "TENSORRT",
          "model_label", "_steps_phrase", "_lora_phrase", "lora_label")


def a_model(root: Path, name: str) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "model_index.json").write_text("{}", encoding="utf-8")
    return folder


# --- a second model, without a code edit -------------------------------------


def test_every_local_diffusers_folder_is_offered(tmp_path):
    a_model(tmp_path, "sd-turbo-fp16")
    a_model(tmp_path, "sd-v1-5-fp16")
    (tmp_path / "half-downloaded").mkdir()

    paths = helpers(*MODELS)["local_model_paths"](models_root=tmp_path)

    assert [Path(path).name for path in paths] == ["sd-turbo-fp16", "sd-v1-5-fp16"]


def test_the_download_s_own_folder_is_offered_first(tmp_path):
    """The same preference `resolve_local_model_path` applies, so the list the user
    picks from opens on what the Download button writes."""
    a_model(tmp_path, "another-model")
    a_model(tmp_path, "sd-turbo-fp16")

    paths = helpers(*MODELS)["local_model_paths"](models_root=tmp_path)

    assert Path(paths[0]).name == "sd-turbo-fp16"


def test_a_folder_with_no_model_index_is_not_offered(tmp_path):
    """A cancelled download leaves a directory the worker cannot load, and
    starting on one fails minutes later inside another process."""
    (tmp_path / "half-downloaded").mkdir()
    assert helpers(*MODELS)["local_model_paths"](models_root=tmp_path) == []


def test_a_model_is_labelled_by_its_folder_name(tmp_path):
    label = helpers(*MODELS)["model_label"]
    assert label(str(tmp_path / "sd-v1-5-fp16")) == "sd-v1-5-fp16"
    assert label("") == ""


# --- what a model needs to render at all -------------------------------------


def test_a_turbo_model_is_one_step_with_no_lcm_lora():
    companions = helpers(*MODELS)["model_companions"]("C:/models/sd-turbo-fp16")
    assert companions.use_lcm_lora is False
    assert len(companions.t_index_list) == 1


def test_a_non_turbo_model_gets_lcm_lora_and_four_steps():
    """`wrapper.py` only fuses LCM-LoRA when the model is not a turbo one, and SD
    1.5 at one step with none is noise rather than a weaker restyle. Measured at
    four steps in issue #38's step-count sweep."""
    companions = helpers(*MODELS)["model_companions"]("C:/models/sd-v1-5-fp16")
    assert companions.use_lcm_lora is True
    assert len(companions.t_index_list) == 4


def test_the_step_ladder_is_the_shipped_one():
    """`render_plan.t_index_ladder`, so what the window builds and what a plan asks
    for are the same schedule."""
    from render_plan import t_index_ladder

    companions = helpers(*MODELS)["model_companions"]("C:/models/sd-v1-5-fp16")
    assert companions.t_index_list == t_index_ladder(companions.t_index_list[0], 4)


def test_the_companions_of_a_blank_model_are_the_shipped_defaults():
    companions = helpers(*MODELS)["model_companions"]("")
    assert companions.use_lcm_lora is False
    assert len(companions.t_index_list) == 1


# --- is there an engine for this, and room to build one? ---------------------


def a_configuration(tmp_path, **overrides):
    # A folder that exists, because `create_prefix` keys the directory on a local
    # model's *name* and on a repo id verbatim - and `engine_cache.model_key`
    # mirrors that, so a fixture pointing at nothing would exercise the other half.
    model = tmp_path / "models" / "sd-v1-5-fp16"
    model.mkdir(parents=True, exist_ok=True)
    fields = dict(model_path=str(model), acceleration="tensorrt",
                  use_lcm_lora=True, steps=4, frame_buffer_size=1,
                  lora_dict=None, engines_root=tmp_path / "engines")
    fields.update(overrides)
    return helpers(*ENGINE)["engine_configuration"](**fields)


def test_a_configuration_with_no_engine_is_reported_as_missing(tmp_path):
    configuration = a_configuration(tmp_path)
    assert configuration.builds is True
    assert configuration.cached is False
    assert "max_batch-4" in configuration.engine_dir


def test_a_configuration_whose_engine_is_on_disk_is_not_a_build(tmp_path):
    configuration = a_configuration(tmp_path)
    built = Path(configuration.engines_root) / configuration.engine_dir
    built.mkdir(parents=True)
    (built / engine_cache.UNET_ENGINE).write_bytes(b"")
    assert a_configuration(tmp_path).cached is True
    assert a_configuration(tmp_path).builds is False


def test_a_path_that_compiles_nothing_never_reports_a_build(tmp_path):
    """The warning fires where a build happens, which is the rule issue #40 set for
    `_confirm_engine_rebuild` and this keeps."""
    for acceleration in ("none", "xformers"):
        configuration = a_configuration(tmp_path, acceleration=acceleration)
        assert configuration.builds is False


def test_the_step_count_keys_the_engine_the_configuration_names(tmp_path):
    assert a_configuration(tmp_path, steps=1).engine_dir != \
        a_configuration(tmp_path, steps=4).engine_dir


def test_a_fused_lora_keys_the_engine_too(tmp_path):
    assert a_configuration(tmp_path).engine_dir != \
        a_configuration(tmp_path, lora_dict={"a.safetensors": 0.9}).engine_dir


def test_the_warning_names_the_model_the_cost_and_the_directory(tmp_path):
    configuration = a_configuration(tmp_path)
    message = helpers(*ENGINE)["_engine_missing_warning"](configuration)
    assert "sd-v1-5-fp16" in message
    assert engine_cache.ENGINE_BUILD_TIME in message
    assert "4 denoising step" in message


def test_the_disk_message_names_the_floor_and_what_is_free():
    message = helpers(*ENGINE)["_not_enough_disk_message"](
        Path("E:/engines"), free_bytes=3 * 1024 ** 3)
    assert "3.0" in message and "20.0" in message
    assert "E:" in message


# --- and Start actually asks -------------------------------------------------


def test_start_checks_the_engine_cache_before_it_spawns_the_worker():
    """The check has to be *before* the process starts, or the user finds out by
    watching the app go quiet - which is the whole complaint. Compared on line
    numbers rather than on walk order, which is breadth-first and says nothing
    about which statement runs first."""
    source = gui_method("_on_start")
    checked = _line_of(source, "_confirm_engine_available")
    spawned = _line_of(source, "Process")
    assert checked is not None, "_on_start does not check the engine cache"
    assert spawned is not None
    assert checked < spawned


def _line_of(node, name):
    import ast

    lines = [child.lineno for child in ast.walk(node)
             if getattr(child, "attr", getattr(child, "id", None)) == name]
    return min(lines) if lines else None


def test_choosing_a_model_applies_its_companion_settings():
    """Otherwise picking SD 1.5 in the window renders noise and nothing says why."""
    method = gui_method("_apply_model")
    assert calls_named(method, "model_companions"), "the companions are not read"
    assert mentions(method, "use_lcm_lora_var")
    assert mentions(method, "t_index_list")


def test_both_ways_of_setting_a_model_go_through_the_same_place():
    """Browse and the picker are two doors on one setting; only one of them
    applying the companions is how the two disagree."""
    for method in ("_on_model_chosen", "_browse_model"):
        assert calls_named(gui_method(method), "_apply_model"), method
