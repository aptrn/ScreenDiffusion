"""The guidance controls, reachable and coherent (issue #45).

Spec 8.11 measured that classifier-free guidance is the lever behind the weak
prompt adherence, that it works, and that it is not free - so the default does not
move and the control does ship. That leaves three ways for the window to hand a
user a setting that silently does nothing, and each of them is a test here.

- The pipeline ignores guidance entirely at or below 1.0 (`if self.guidance_scale >
  1.0`), and the window opened on 0.0. So picking a cfg type without moving the
  scale is picking nothing, with the app looking as though it obeyed.
- `delta` is read only under `self` and `initialize`. Offering it under `full` is
  offering a control the pipeline never looks at.
- `initialize` and `full` change `trt_unet_batch_size`, so each is a ~5 GB build.
  A change that keys an engine has to go through the same warning every other one
  does, and the engine-state line under the model has to name the right directory.

GPU-free: `StreamGUI`'s widget code is read out of the source (`guisource`) and the
pure helpers beside it are executed out of it (`sourceloader`), because importing
`main_gpu_addon` pulls in the Tk stack.
"""

from __future__ import annotations

import ast
from typing import NamedTuple, Tuple

from guisource import TREE, gui_method, method_text
from sourceloader import load_symbols

import engine_cache
from bench.guidance import GUIDANCE_LADDER
from engine_cache import CFG_FULL, CFG_INITIALIZE, CFG_NONE, CFG_SELF, CFG_TYPES

_symbols = load_symbols(
    "main_gpu_addon.py",
    ["ADVANCED", "CfgCompanions", "CUSTOM_COLORS", "DEFAULT_CFG_TYPE",
     "DEFAULT_DELTA", "DEFAULT_FRAME_BUFFER_SIZE", "ENGINE_KEYED_SETTINGS",
     "EngineConfiguration", "engine_configuration", "GUIDANCE_OFF", "GUIDANCE_WHEN_ON",
     "PLAN_NOTE_COLOR", "SHOW",
     "_as_scale", "cfg_companions", "cfg_keys_new_engine", "cfg_note",
     "cfg_uses_delta", "guidance_is_off",
     # What `engine_configuration` and the engine sentence close over since
     # issue #44 put the fused LoRA set in the same record.
     "_engine_state_line", "_lora_phrase", "_steps_phrase", "lora_label",
     "model_label"],
    extra_globals={
        "engine_cache": engine_cache, "Path": __import__("pathlib").Path,
        "Dict": dict, "Optional": object, "DIFFUSION_CANVAS": 512,
        "resolve_engines_dir": lambda: __import__("pathlib").Path("."),
        "engine_rebuild_needed": lambda acceleration: acceleration == "tensorrt",
        "CFG_NONE": CFG_NONE, "CFG_SELF": CFG_SELF,
        "CFG_INITIALIZE": CFG_INITIALIZE, "CFG_FULL": CFG_FULL,
        "CFG_TYPES": CFG_TYPES,
        "ENGINE_BUILD_SIZE": engine_cache.ENGINE_BUILD_SIZE,
        "ENGINE_BUILD_TIME": engine_cache.ENGINE_BUILD_TIME,
        "NamedTuple": NamedTuple, "Tuple": Tuple, "List": list,
        "LOCAL_MODEL_NAMES": ("sd-turbo-fp16", "sd-turbo"),
    },
)
ADVANCED = _symbols["ADVANCED"]
DEFAULT_CFG_TYPE = _symbols["DEFAULT_CFG_TYPE"]
ENGINE_KEYED_SETTINGS = _symbols["ENGINE_KEYED_SETTINGS"]
GUIDANCE_WHEN_ON = _symbols["GUIDANCE_WHEN_ON"]
SHOW = _symbols["SHOW"]
cfg_companions = _symbols["cfg_companions"]
cfg_note = _symbols["cfg_note"]
guidance_is_off = _symbols["guidance_is_off"]

# --- the three controls exist ------------------------------------------------


def test_the_three_guidance_controls_are_shown():
    """Hidden is how the finding becomes unusable: spec 8.11 measured a lever that
    works and is priced, and a priced lever nobody can reach is no lever."""
    for setting in ("cfg_type", "guidance_scale", "delta"):
        assert SHOW[setting] is True, setting


def test_they_live_in_advanced_beside_the_other_engine_settings():
    """They are properties of how the app runs rather than of what it makes - and
    two of the four cfg types key an engine, which is what `ADVANCED` collects."""
    for setting in ("cfg_type", "guidance_scale", "delta"):
        assert setting in ADVANCED, setting


def test_the_offered_cfg_types_are_the_vocabulary_the_pipeline_accepts():
    """Spelt from `engine_cache.CFG_TYPES` rather than typed into the combo, so a
    fifth name cannot be offered that the wrapper would refuse."""
    assert "values=list(CFG_TYPES)" in method_text("_build_ui")


# --- guidance below 1.0 is guidance off --------------------------------------


def test_a_scale_at_or_below_one_is_recognised_as_off():
    assert guidance_is_off(CFG_SELF, 1.0)
    assert guidance_is_off(CFG_SELF, 0.0)
    assert not guidance_is_off(CFG_SELF, 1.05)


def test_the_control_arm_is_off_at_any_scale():
    """`prepare` forces the scale to 1.0 under `none`, so 3.0 there is not a
    strong pull, it is no pull."""
    assert guidance_is_off(CFG_NONE, 3.0)


def test_picking_a_cfg_type_brings_a_scale_that_does_something_with_it():
    """The same shape `model_companions` has: a setting whose companion was left
    behind is a user watching nothing happen with nothing in the window saying
    why."""
    assert cfg_companions(CFG_NONE).guidance_scale <= 1.0
    for cfg_type in (CFG_SELF, CFG_INITIALIZE, CFG_FULL):
        companions = cfg_companions(cfg_type)
        assert companions.guidance_scale == GUIDANCE_WHEN_ON
        assert not guidance_is_off(cfg_type, companions.guidance_scale)


def test_the_starting_scale_is_a_rung_the_sweep_actually_measured():
    assert GUIDANCE_WHEN_ON in GUIDANCE_LADDER


def test_the_window_opens_with_guidance_off_because_the_default_did_not_move():
    """Spec 8.11's verdict. Every committed figure in this repo belongs to
    `cfg_type: none`, and the block says the lever is not worth a default."""
    assert DEFAULT_CFG_TYPE == CFG_NONE
    assert "self.cfg_type_var = ctk.StringVar(value=DEFAULT_CFG_TYPE)" in \
        method_text("__init__")


# --- what the note says ------------------------------------------------------


def test_a_cfg_type_with_the_scale_left_down_is_called_out():
    note, _ = cfg_note(CFG_SELF, 1.0, 1.0)
    assert "1.0" in note and "nothing" in note.lower()


def test_delta_is_named_as_ignored_where_the_pipeline_ignores_it():
    assert "delta" in cfg_note(CFG_FULL, 1.4, 0.5)[0].lower()
    assert "delta" not in cfg_note(CFG_SELF, 1.4, 0.5)[0].lower()


def test_a_cfg_type_that_keys_a_build_says_so():
    assert "engine" in cfg_note(CFG_FULL, 1.4, 1.0)[0].lower()
    assert "engine" not in cfg_note(CFG_SELF, 1.4, 1.0)[0].lower()


def test_guidance_off_under_none_says_nothing_at_all():
    """The shipped configuration is not a warning."""
    assert cfg_note(CFG_NONE, 1.0, 1.0)[0] == ""


# --- the engine key knows about it -------------------------------------------


def test_the_window_reads_its_own_cfg_type_when_it_asks_which_engine():
    """`initialize` and `full` change `trt_unet_batch_size`, so a lookup that
    ignored the cfg type would answer "cached" about a directory the build never
    writes."""
    assert "cfg_type=self.cfg_type_var.get()" in method_text("_engine_configuration")


def test_the_engine_configuration_passes_the_cfg_type_to_the_batch_rule():
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == "engine_configuration":
            body = ast.unparse(node)
            break
    else:  # pragma: no cover - the helper is the subject of the test
        raise AssertionError("main_gpu_addon.py defines no engine_configuration")
    assert "unet_batch_size(frame_buffer_size, steps, cfg_type=cfg_type)" in body


def test_the_engine_lines_name_the_cfg_type_only_when_it_is_not_the_shipped_one():
    """`full` at one step needs a batch-2 engine, so a line reading "no engine for
    sd-turbo-fp16 at 1 step" would be denying the existence of one that is built.
    Silent on the default, so the sentence issue #38 wrote does not churn."""
    configuration = _symbols["EngineConfiguration"](
        model_path="m", engine_dir="d", engines_root="r", cached=False,
        builds=True, steps=1, free_bytes=0, enough_disk=True, cfg_type=CFG_FULL)
    assert configuration.cfg_phrase == f", CFG {CFG_FULL}"
    assert configuration._replace(cfg_type=CFG_NONE).cfg_phrase == ""
    # `_refresh_engine_state` is one line onto `_engine_state_line` since issue
    # #44, so the sentence itself is what gets read rather than the method's text.
    line = _symbols["_engine_state_line"]
    assert f", CFG {CFG_FULL}" in line(configuration)
    assert "CFG" not in line(configuration._replace(cfg_type=CFG_NONE))
    assert "_engine_state_line" in method_text("_refresh_engine_state")
    assert "cfg_phrase" in ast.unparse(_engine_missing_warning_node())


def test_the_configuration_carries_the_cfg_type_it_was_asked_about(tmp_path):
    """It reached the *key* and not the record once, so the directory was right
    and every message about it named the shipped setting."""
    built = _symbols["engine_configuration"](
        model_path="m", acceleration="tensorrt", use_lcm_lora=False, steps=1,
        frame_buffer_size=1, engines_root=tmp_path, cfg_type=CFG_FULL)
    assert built.cfg_type == CFG_FULL
    assert "max_batch-2" in built.engine_dir


def _engine_missing_warning_node() -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_engine_missing_warning"):
            return node
    raise AssertionError("main_gpu_addon.py defines no _engine_missing_warning")


def test_changing_the_cfg_type_goes_through_the_rebuild_warning():
    """It is a setting that keys an engine and has a control here, which is
    exactly what `ENGINE_KEYED_SETTINGS` is the list of."""
    assert "CFG type" in ENGINE_KEYED_SETTINGS
    assert "_confirm_engine_rebuild" in method_text("_on_cfg_type")


def test_the_cfg_type_is_applied_through_one_place():
    """Like `_apply_model`: the type, its scale and its delta move together or a
    user gets a setting that does nothing."""
    assert isinstance(gui_method("_apply_cfg_type"), ast.FunctionDef)


def test_every_cfg_type_the_combo_offers_has_companions():
    for cfg_type in CFG_TYPES:
        assert cfg_companions(cfg_type) is not None
