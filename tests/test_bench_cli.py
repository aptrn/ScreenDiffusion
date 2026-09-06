"""The `python -m bench` entry point (issue #2).

The CLI parses and lists scenarios without a GPU: importing it must not drag torch
in, or the merge gate's tier stops being GPU-free. Only `run` touches the device.
"""

import subprocess
import sys

import pytest

from sourceloader import ROOT

from bench.cli import build_parser, resolve_scenario
from bench.scenarios import SCENARIOS, ScenarioConfig


def test_importing_the_cli_does_not_import_torch():
    assert "torch" not in sys.modules


def test_help_exits_zero():
    """The issue's gate: `uv run python -m bench --help` exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "bench", "--help"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "scenario" in result.stdout


def test_list_exits_zero_and_names_the_scenarios():
    result = subprocess.run(
        [sys.executable, "-m", "bench", "--list"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "img2img-none-512x512-b1" in result.stdout


def test_an_unknown_scenario_fails_rather_than_guessing():
    result = subprocess.run(
        [sys.executable, "-m", "bench", "no-such-scenario"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "no-such-scenario" in (result.stdout + result.stderr)


def test_cooldown_is_on_by_default_and_skippable_by_flag():
    parser = build_parser()
    assert parser.parse_args(["img2img-none-512x512-b1"]).cooldown is True
    assert parser.parse_args(["img2img-none-512x512-b1", "--no-cooldown"]).cooldown is False


def test_per_module_timing_is_behind_a_flag_and_off_by_default():
    parser = build_parser()
    assert parser.parse_args(["img2img-none-512x512-b1"]).per_module is False
    assert parser.parse_args(["img2img-none-512x512-b1", "--per-module"]).per_module is True


def test_the_registry_covers_the_sweep_the_spec_asks_for():
    """Spec 7.2 items 1-3: TRT vs none, batch 1/2/4/8, 256 / 384 / 512."""
    assert {s.acceleration for s in SCENARIOS.values()} == {"none", "tensorrt"}
    assert {s.batch_size for s in SCENARIOS.values()} >= {1, 2, 4, 8}
    assert {s.width for s in SCENARIOS.values()} >= {256, 384, 512}
    for name, scenario in SCENARIOS.items():
        assert scenario.name == name


def test_flags_override_the_scenario_config():
    parser = build_parser()
    args = parser.parse_args(["img2img-none-512x512-b1", "--reps", "5", "--warmup", "1"])
    scenario = resolve_scenario(args)
    assert isinstance(scenario, ScenarioConfig)
    assert scenario.reps == 5
    assert scenario.warmup_reps == 1
    assert scenario.width == 512


def test_a_scenario_config_serialises_whole():
    """Step 4: the *full* scenario config lands in the result, not a summary of it."""
    data = SCENARIOS["img2img-none-512x512-b1"].to_dict()
    assert data["name"] == "img2img-none-512x512-b1"
    for key in ("acceleration", "width", "height", "batch_size", "t_index_list",
                "model", "prompt", "seed", "reps", "warmup_reps"):
        assert key in data, key


def test_a_tensorrt_scenario_needs_an_explicit_opt_in_to_build_an_engine(tmp_path):
    """An uncached TRT engine is ~5.1 GB and several minutes. Never a surprise."""
    from bench.cli import engine_build_guard

    trt = SCENARIOS["img2img-tensorrt-512x512-b1"]
    with pytest.raises(SystemExit):
        engine_build_guard(trt, engines_root=tmp_path, allow_build=False)
    engine_build_guard(trt, engines_root=tmp_path, allow_build=True)
    engine_build_guard(SCENARIOS["img2img-none-512x512-b1"], engines_root=tmp_path,
                       allow_build=False)
