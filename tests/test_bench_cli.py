"""The `python -m bench` entry point (issue #2).

The CLI parses and lists scenarios without a GPU: importing it must not drag torch
in, or the merge gate's tier stops being GPU-free. Only `run` touches the device.
"""

import subprocess
import sys

import pytest

from sourceloader import ROOT

from bench.clocks import LOCKED, UNKNOWN, UNLOCKED
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


def test_marginal_reads_the_committed_results_and_exits_zero():
    """Issue #3: the marginal cost is recomputed from the JSON, not retyped from it."""
    result = subprocess.run(
        [sys.executable, "-m", "bench", "--marginal"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "marginal ms/item" in result.stdout
    assert "sublinear" in result.stdout
    assert "Normalised to" in result.stdout, "the normalised table names the clock it assumes"


def test_a_cached_tensorrt_run_still_records_its_free_disk(tmp_path):
    """Issue #3's gate: *each* TensorRT confirmation run carries the free-disk check.

    A cached engine compiles nothing, so there is nothing to refuse - but the gate
    asks for the check to be recorded, not merely applied, and a reader months later
    cannot tell "the volume had room" from "nobody looked" unless the reading is in
    the file. Reading is unconditional for TensorRT; refusing stays tied to a build.
    """
    from bench.cli import engine_build_guard, engine_dir_name

    trt = SCENARIOS["img2img-tensorrt-512x512-b1"]
    cached = tmp_path / engine_dir_name(trt)
    cached.mkdir()
    (cached / "unet.engine").write_bytes(b"")

    record = engine_build_guard(trt, engines_root=tmp_path, allow_build=False)
    assert record is not None, "a cached TensorRT run records its headroom too"
    assert record.free_bytes > 0
    assert "free_gib" in record.to_dict()


# --- clock regime gate (issue #13) -----------------------------------------

def test_require_locked_clocks_is_off_by_default():
    parser = build_parser()
    args = parser.parse_args(["img2img-none-512x512-b1"])
    assert args.require_locked_clocks is False
    assert parser.parse_args(
        ["img2img-none-512x512-b1", "--require-locked-clocks"]
    ).require_locked_clocks is True


def a_lock(state):
    from bench.clocks import ClockLock

    return ClockLock(state=state, applied_clock_mhz=1200.0 if state == LOCKED else None,
                     max_sm_clock_mhz=2100.0, current_sm_clock_mhz=210.0,
                     evidence=f"clocks_event_reasons.applications_clocks_setting={state}")


def test_the_gate_lets_a_locked_gpu_through():
    from bench.cli import clock_lock_guard

    lock = clock_lock_guard(True, read=lambda: a_lock(LOCKED))
    assert lock.locked is True


@pytest.mark.parametrize("state", [UNLOCKED, UNKNOWN])
def test_the_gate_stops_a_run_that_would_decide_something_on_an_unlocked_gpu(state):
    """Issue #13 step 4, and its trap: `unknown` is not permission to proceed."""
    from bench.cli import clock_lock_guard

    with pytest.raises(SystemExit) as exit_info:
        clock_lock_guard(True, read=lambda: a_lock(state))
    message = str(exit_info.value)
    assert state in message
    assert "--lock-gpu-clocks" in message, "the message has to say how to fix it"
    assert "elevated" in message, "and that it takes a shell this loop does not have"


@pytest.mark.parametrize("state", [LOCKED, UNLOCKED, UNKNOWN])
def test_without_the_flag_the_gate_only_reports(state):
    from bench.cli import clock_lock_guard

    assert clock_lock_guard(False, read=lambda: a_lock(state)).state == state


def test_an_unlocked_run_that_demands_a_lock_exits_non_zero():
    """End to end on this machine, which has no elevated shell to lock with."""
    result = subprocess.run(
        [sys.executable, "-m", "bench", "img2img-none-256x256-b1",
         "--require-locked-clocks", "--reps", "1"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "--lock-gpu-clocks" in (result.stdout + result.stderr)
