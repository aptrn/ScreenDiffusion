"""The `python -m bench` entry point (issue #2).

The CLI parses and lists scenarios without a GPU: importing it must not drag torch
in, or the merge gate's tier stops being GPU-free. Only `run` touches the device.
"""

import subprocess
import sys

import pytest

from sourceloader import ROOT

from bench.clocks import LOCKED, UNKNOWN, UNLOCKED, ClockLock
from bench.cli import build_parser, resolve_target
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
    kind, scenario = resolve_target(args)
    assert kind == "scenario"
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


def test_an_unlocked_run_that_demands_a_lock_exits_non_zero(tmp_path):
    """End to end on this machine, which has no elevated shell to lock with.

    Pointed at a throwaway results dir: the gate is expected to refuse before the
    run starts, and if a machine ever does have its clocks locked this must not
    append a real run to the tracked `bench/results/`.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bench", "img2img-none-256x256-b1",
         "--require-locked-clocks", "--reps", "1", "--results-dir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "--lock-gpu-clocks" in (result.stdout + result.stderr)


# --- detectors (issue #4) ----------------------------------------------------

def test_the_detectors_share_the_positional_slot_with_the_scenarios():
    """One name, one run. `--list` shows both registries, so a name is discoverable."""
    from bench.detectors import DETECTORS, PRIMARY_DETECTOR

    kind, target = resolve_target(build_parser().parse_args([PRIMARY_DETECTOR]))
    assert kind == "detector" and target is DETECTORS[PRIMARY_DETECTOR]

    kind, target = resolve_target(build_parser().parse_args(["img2img-none-512x512-b1"]))
    assert kind == "scenario" and target is SCENARIOS["img2img-none-512x512-b1"]


def test_an_unknown_name_names_both_registries_rather_than_one():
    with pytest.raises(SystemExit) as failure:
        resolve_target(build_parser().parse_args(["no-such-thing"]))
    assert "--list" in str(failure.value)


def test_reps_override_a_detector_too():
    _, detector = resolve_target(build_parser().parse_args(
        ["yolo-world-s-640", "--reps", "7", "--warmup", "2"]))
    assert (detector.reps, detector.warmup_reps) == (7, 2)


def test_list_names_the_detectors_as_well_as_the_scenarios():
    result = subprocess.run(
        [sys.executable, "-m", "bench", "--list"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "yolo-world-s-640" in result.stdout and "yolov8n-640" in result.stdout


def test_downloading_weights_needs_an_explicit_opt_in():
    """Hundreds of MB is not a surprise anyone should get from a typo."""
    parser = build_parser()
    assert parser.parse_args(["yolo-world-s-640"]).allow_download is False
    assert parser.parse_args(["yolo-world-s-640", "--allow-download"]).allow_download is True


def test_the_diffusion_engine_is_resident_by_default():
    """The issue's trap: a detector benchmarked alone says nothing about whether it fits."""
    from bench.detectors import DEFAULT_DIFFUSION_SCENARIO

    parser = build_parser()
    assert parser.parse_args(["yolo-world-s-640"]).with_diffusion == DEFAULT_DIFFUSION_SCENARIO
    assert parser.parse_args(["yolo-world-s-640", "--no-diffusion"]).with_diffusion is None


def test_the_detector_report_reads_the_committed_results_and_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "bench", "--detector-report"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Recommendation:" in result.stdout


def test_a_diffusion_scenario_that_does_not_exist_is_refused_by_name():
    """`--with-diffusion` takes a scenario name, so a typo has to read like one."""
    import bench.cli as cli

    with pytest.raises(SystemExit) as failure:
        cli.main(["yolo-world-s-640", "--with-diffusion", "no-such-scenario"])
    assert "no-such-scenario" in str(failure.value)


def test_a_detector_run_gates_the_engine_it_would_have_to_build(tmp_path, monkeypatch):
    """`--with-diffusion` names a TensorRT configuration, so it meets the same gate.

    Refused before `bench.detector_runner` is imported, which is what makes this
    checkable in the GPU-free tier at all.
    """
    import bench.cli as cli

    monkeypatch.setattr(cli, "resolve_engines_dir", lambda: tmp_path)
    with pytest.raises(SystemExit) as failure:
        cli.main(["yolo-world-s-640", "--with-diffusion", "img2img-tensorrt-512x512-b8"])
    assert "--allow-engine-build" in str(failure.value)


# --- rendering primitives (issue #5) -----------------------------------------

def test_the_primitive_cases_share_the_positional_slot_too():
    """Three registries, one slot. A name is a name; the CLI works out its kind."""
    from bench.primitives import CASES, RESTYLE_CASE

    kind, target = resolve_target(build_parser().parse_args([RESTYLE_CASE]))
    assert kind == "primitive" and target is CASES[RESTYLE_CASE]


def test_the_frame_count_can_be_shortened_for_a_development_run():
    from bench.primitives import RESTYLE_CASE

    _, case = resolve_target(build_parser().parse_args([RESTYLE_CASE, "--frames", "6"]))
    assert case.frames == 6


def test_list_names_the_primitive_cases_as_well():
    result = subprocess.run(
        [sys.executable, "-m", "bench", "--list"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "restyle-people" in result.stdout
    assert "priority case" in result.stdout


def test_the_primitive_report_reads_the_committed_comparisons_and_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "bench", "--primitive-report"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Decision:" in result.stdout


def test_a_primitive_run_gates_the_engine_both_primitives_share(tmp_path, monkeypatch):
    """One engine, and it meets the same gate a diffusion run does - refused before
    `bench.primitive_runner` imports torch."""
    import bench.cli as cli
    from bench.primitives import RESTYLE_CASE

    monkeypatch.setattr(cli, "resolve_engines_dir", lambda: tmp_path)
    with pytest.raises(SystemExit) as failure:
        cli.main([RESTYLE_CASE])
    assert "--allow-engine-build" in str(failure.value)


def test_writing_a_track_needs_no_engine_at_all():
    """It is a detector pass over the clip, so the engine guard must not fire first."""
    import bench.cli as cli
    from bench.primitives import CASES, RESTYLE_CASE

    parser = build_parser()
    assert parser.parse_args([RESTYLE_CASE, "--write-track"]).write_track is True
    source = ROOT / "bench" / "cli.py"
    body = source.read_text(encoding="utf-8").split("def run_primitive_target", 1)[1]
    assert body.index("args.write_track") < body.index("engine_build_guard")
    assert RESTYLE_CASE in CASES


# --- the selective render path (issue #8) ------------------------------------

def test_the_selective_case_shares_the_positional_slot_too():
    """A fourth registry, still one slot: the CLI works out the kind from the name."""
    from bench.selective import CASES, PRIORITY_CASE

    kind, target = resolve_target(build_parser().parse_args([PRIORITY_CASE]))
    assert kind == "selective" and target is CASES[PRIORITY_CASE]


def test_a_selective_run_can_be_shortened_for_a_development_run():
    from bench.selective import PRIORITY_CASE

    _, case = resolve_target(build_parser().parse_args([PRIORITY_CASE, "--frames", "6"]))
    assert case.frames == 6


def test_list_names_the_selective_case_as_well():
    result = subprocess.run(
        [sys.executable, "-m", "bench", "--list"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "selective-people" in result.stdout
    assert "the shipped path end to end" in result.stdout


def test_the_selective_report_reads_the_committed_runs_and_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "bench", "--selective-report"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "selective-people" in result.stdout


def test_a_selective_run_gates_the_engine_it_renders_through(tmp_path, monkeypatch):
    """Refused before `bench.selective_runner` imports torch, like every other run."""
    import bench.cli as cli
    from bench.selective import PRIORITY_CASE

    monkeypatch.setattr(cli, "resolve_engines_dir", lambda: tmp_path)
    with pytest.raises(SystemExit) as failure:
        cli.main([PRIORITY_CASE])
    assert "--allow-engine-build" in str(failure.value)
