"""Choosing quality at runtime, and being told which rungs are already built.

Issue #46. The step-count control existed and was hidden behind
`SHOW["step_count"] = False`; what was missing was the answer to "what does
clicking this cost", which for a rung whose engine is on disk is a load and for one
without is ~5 GB and minutes. The picker's labels are that answer, read off the
real cache through the same door Start uses.

The helpers are pure functions and are executed; where they reach the widgets is
read off the source through `guisource`, because `StreamGUI` cannot be instantiated
in this tier.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from guinamespace import helpers
from guisource import calls_named, gui_method, mentions, method_text

import engine_cache

LADDER = ("STEP_LADDER", "StepChoice", "step_label", "step_choices",
          "steps_of_label", "engine_configuration", "EngineConfiguration",
          "engine_rebuild_needed", "TENSORRT", "_steps_phrase", "_lora_phrase",
          "lora_label", "model_label")

MODEL = "sd-turbo-fp16"


def built(root: Path, steps: int, batch: int = 1) -> Path:
    """A compiled UNet engine for `steps` on the shipped batched route."""
    directory = root / engine_cache.engine_dir_name(
        MODEL, use_lcm_lora=False, use_tiny_vae=True,
        unet_batch=engine_cache.unet_batch_size(batch, steps),
        width=512, height=512)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / engine_cache.UNET_ENGINE).write_bytes(b"engine")
    return directory


def choices(root: Path, acceleration: str = "tensorrt"):
    module = helpers(*LADDER)
    return module["step_choices"](
        model_path=MODEL, acceleration=acceleration, use_lcm_lora=False,
        frame_buffer_size=1, engines_root=root)


def call_line(method, name: str) -> int:
    """The line a method first calls `self.<name>()` on - two of them have to be
    in the right order, and this is how that order is read."""
    lines = [node.lineno for node in ast.walk(method)
             if isinstance(node, ast.Call)
             and getattr(node.func, "attr", None) == name]
    assert lines, f"{method.name} never calls self.{name}()"
    return min(lines)


# --- the ladder is the one the sweep measured --------------------------------


def test_the_window_offers_the_rungs_the_sweep_measured():
    """A rung the window offers and `bench.quality` never swept is a quality nobody
    priced, so both read one list."""
    from bench.quality import STEP_LADDER as SWEPT

    assert tuple(helpers(*LADDER)["STEP_LADDER"]) == tuple(SWEPT)
    assert tuple(SWEPT) == engine_cache.STEP_LADDER


def test_the_shipped_count_is_on_the_ladder():
    """Every committed figure in this repo is one step; a ladder that could not
    return to it would be a one-way trip."""
    assert 1 in helpers(*LADDER)["STEP_LADDER"]


# --- cached rungs are distinguished before the click -------------------------


def test_a_rung_whose_engine_exists_is_labelled_ready(tmp_path):
    built(tmp_path, 1)

    offered = {choice.steps: choice for choice in choices(tmp_path)}

    assert offered[1].cached is True
    assert "engine ready" in offered[1].label


def test_a_rung_with_no_engine_says_it_builds_one(tmp_path):
    built(tmp_path, 1)

    offered = {choice.steps: choice for choice in choices(tmp_path)}

    assert offered[4].cached is False
    assert "builds an engine" in offered[4].label


def test_the_lookup_is_the_one_start_makes(tmp_path):
    """Not a guess from whether a setting is at its default: the directory the
    worker will look in, named by `engine_cache`."""
    directory = built(tmp_path, 4)
    assert "max_batch-4" in directory.name

    offered = {choice.steps: choice for choice in choices(tmp_path)}

    assert offered[4].cached is True
    assert offered[1].cached is False


@pytest.mark.parametrize("acceleration", ["none", "xformers"])
def test_a_path_that_compiles_nothing_annotates_nothing(tmp_path, acceleration):
    """"Engine ready" on a path with no engines would be describing something that
    does not exist - and on those paths every rung is equally reachable."""
    for choice in choices(tmp_path, acceleration):
        assert choice.cached is False
        assert "engine" not in choice.label
        assert choice.label == f"{choice.steps} step" + ("" if choice.steps == 1
                                                         else "s")


def test_a_label_maps_back_to_the_rung_it_names(tmp_path):
    offered = choices(tmp_path)
    steps_of_label = helpers(*LADDER)["steps_of_label"]

    for choice in offered:
        assert steps_of_label(choice.label, offered) == choice.steps
    assert steps_of_label("17 steps", offered) is None


def test_the_label_pluralises_once(tmp_path):
    step_label = helpers(*LADDER)["step_label"]

    assert step_label(1, True).startswith("1 step ")
    assert step_label(4, True).startswith("4 steps ")


# --- where it is in the window, and what it asks before it changes -----------


def test_the_picker_is_beside_the_strength_sliders_not_in_advanced():
    """The trade a user asked for by name is not an engine knob to be disclosed."""
    build = gui_method("_build_ui")
    assert mentions(build, "_w_step_count")
    assert mentions(build, "_on_quality_chosen")


def test_the_picker_is_not_locked_while_a_run_is_live():
    """"Chosen at runtime" is the whole request: a control that goes grey the
    moment generation starts answers a different one."""
    build = gui_method("_build_ui")
    for call in calls_named(build, "_register_lockables"):
        assert not mentions(call, "_w_step_count")


def test_adopting_a_rung_keeps_the_strength_the_sliders_are_on():
    """The issue's fifth trap: more steps at a different opening index is two
    changes at once, so the ladder opens where the sliders already are."""
    apply_steps = gui_method("_apply_steps")
    assert mentions(apply_steps, "t_index_ladder")
    assert mentions(apply_steps, "_build_steps_ui")


def test_adopting_a_rung_tells_a_live_worker_about_it():
    """The message type is a string on the control queue, so it is read out of the
    method's text rather than off a name in it."""
    assert "set_t_index_list" in method_text("_apply_steps")


def test_a_refused_rung_puts_the_picker_back():
    """Otherwise the menu shows a quality that was never adopted."""
    chosen = gui_method("_on_quality_chosen")
    assert mentions(chosen, "_apply_steps")
    assert mentions(chosen, "_refresh_step_choices")


def test_the_ladder_is_redrawn_by_everything_that_keys_an_engine():
    """The model, the LoRA set, the batch size and the cfg type all change which
    rungs are built, and a stale "engine ready" is the one thing this must not
    say."""
    assert mentions(gui_method("_refresh_engine_state"), "_refresh_step_choices")


def test_the_resting_label_is_drawn_after_the_widget_exists():
    """`step_count_var` is constructed before the engine cache is consulted - it
    cannot be, the widget is not built yet - so the window has to redraw the ladder
    once it is, or it opens saying `builds an engine` about an engine it has."""
    init = gui_method("__init__")
    build = call_line(init, "_build_ui")
    refresh = call_line(init, "_refresh_engine_state")
    assert build < refresh, \
        "the ladder is drawn before the picker exists, so its labels never land"
