"""Guidance reaches the engine, and the engine key knows about it (issue #45).

Three seams the arithmetic tests cannot see, and each one is a way the sweep could
have measured nothing at all.

- A `ScenarioConfig` carries `cfg_type` and has since the registry was written, but
  it carried no `guidance_scale` - so `build_stream` prepared every arm at the
  wrapper's own default of 1.2 and a "guidance 3.0" arm would have rendered at 1.2.
- The pipeline disables guidance entirely below 1.0 (`if self.guidance_scale >
  1.0`), so an arm that forgot to raise it is the control wearing another name.
- `initialize` and `full` change `trt_unet_batch_size`, so a scenario's own idea of
  which engine it needs has to know the cfg type or the build guard answers about a
  directory that does not exist.

Structural where it has to be: `bench.runner` imports torch inside its functions,
but `build_stream`'s call to `prepare` is read out of the source rather than run.
"""

from __future__ import annotations

import ast
from pathlib import Path

from bench.guidance import ADHERENCE_CONF, GUIDANCE_LADDER, ladder, uses_delta
from bench.scenarios import SCENARIOS, ScenarioConfig
from engine_cache import CFG_FULL, CFG_INITIALIZE, CFG_NONE, CFG_SELF

RUNNER = Path(__file__).resolve().parent.parent / "bench" / "runner.py"
RUNNER_TEXT = RUNNER.read_text(encoding="utf-8")
RUNNER_TREE = ast.parse(RUNNER_TEXT, filename=str(RUNNER))


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no {name}")


BUILD_STREAM = ast.get_source_segment(RUNNER_TEXT, _function(RUNNER_TREE,
                                                             "build_stream"))


# --- the scenario carries what the pipeline reads ----------------------------


def test_a_scenario_carries_the_guidance_scale_and_the_delta():
    scenario = ScenarioConfig(name="x", cfg_type=CFG_SELF, guidance_scale=1.4,
                              delta=0.5)
    assert scenario.guidance_scale == 1.4
    assert scenario.delta == 0.5


def test_the_registry_ships_with_guidance_off_like_the_app_does():
    """Every committed figure in this repo was measured at `cfg_type: none`, and
    a default that quietly moved would re-key every one of them."""
    for scenario in SCENARIOS.values():
        assert scenario.cfg_type == CFG_NONE
        assert scenario.guidance_scale <= 1.0


def test_build_stream_prepares_at_the_scenario_s_own_guidance_and_delta():
    """Not at the wrapper's default of 1.2: an arm that asked for 3.0 and got 1.2
    would report the wrong figure for the right shape of measurement."""
    assert "guidance_scale=scenario.guidance_scale" in BUILD_STREAM
    assert "delta=scenario.delta" in BUILD_STREAM


def test_build_stream_still_passes_the_cfg_type_to_the_constructor():
    """It is a constructor argument, not a runtime one - which is why an arm is a
    fresh stream rather than another `prepare` call."""
    assert "cfg_type=scenario.cfg_type" in BUILD_STREAM


# --- the engine key knows about the cfg type ---------------------------------


def test_a_scenario_s_unet_batch_follows_its_cfg_type():
    base = ScenarioConfig(name="x", t_index_list=[35])
    assert base.unet_batch_size == 1
    assert base.replace(cfg_type=CFG_SELF).unet_batch_size == 1
    assert base.replace(cfg_type=CFG_INITIALIZE).unet_batch_size == 2
    assert base.replace(cfg_type=CFG_FULL).unet_batch_size == 2
    four = base.replace(t_index_list=[35, 38, 41, 44])
    assert four.unet_batch_size == 4
    assert four.replace(cfg_type=CFG_INITIALIZE).unet_batch_size == 5
    assert four.replace(cfg_type=CFG_FULL).unet_batch_size == 8


def test_the_engine_directory_a_cfg_arm_needs_is_not_the_shipped_one():
    from bench.cli import engine_dir_name

    shipped = SCENARIOS["img2img-tensorrt-512x512-b1"]
    assert engine_dir_name(shipped) == engine_dir_name(
        shipped.replace(cfg_type=CFG_SELF, guidance_scale=1.4))
    assert engine_dir_name(shipped) != engine_dir_name(
        shipped.replace(cfg_type=CFG_FULL, guidance_scale=1.4))


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_starts_with_the_control_and_sweeps_every_other_cfg_type():
    specs = ladder(GUIDANCE_LADDER, (0.5, 1.0))
    assert specs[0].cfg_type == CFG_NONE
    assert {spec.cfg_type for spec in specs} == {CFG_NONE, CFG_SELF,
                                                CFG_INITIALIZE, CFG_FULL}


def test_the_ladder_never_sweeps_delta_where_the_pipeline_ignores_it():
    deltas_by_type = {}
    for spec in ladder(GUIDANCE_LADDER, (0.5, 1.0)):
        deltas_by_type.setdefault(spec.cfg_type, set()).add(spec.delta)
    for cfg_type, deltas in deltas_by_type.items():
        assert (len(deltas) > 1) == uses_delta(cfg_type), cfg_type


def test_every_guidance_rung_is_above_the_threshold_the_pipeline_ignores_below():
    """`if self.guidance_scale > 1.0` gates the whole mechanism, so a rung at or
    below 1.0 is the control arm with a different label."""
    assert all(scale > 1.0 for scale in GUIDANCE_LADDER)


def test_the_adherence_probe_uses_the_identity_probe_s_own_confidence():
    from bench.primitive_runner import IDENTITY_CONF

    assert ADHERENCE_CONF == IDENTITY_CONF
