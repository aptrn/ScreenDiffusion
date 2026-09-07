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


def test_the_portability_report_reads_the_same_runs_and_exits_zero():
    """Issue #24: the dev-vs-deploy block spec 7.4 carries, off the selective runs."""
    result = subprocess.run(
        [sys.executable, "-m", "bench", "--portability-report"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Acceptance criterion 2" in result.stdout


def test_a_selective_run_gates_the_engine_it_renders_through(tmp_path, monkeypatch):
    """Refused before `bench.selective_runner` imports torch, like every other run."""
    import bench.cli as cli
    from bench.selective import PRIORITY_CASE

    monkeypatch.setattr(cli, "resolve_engines_dir", lambda: tmp_path)
    with pytest.raises(SystemExit) as failure:
        cli.main([PRIORITY_CASE])
    assert "--allow-engine-build" in str(failure.value)


# --- the detect_every_n sweep (issue #23) ------------------------------------

def test_a_selective_run_can_be_asked_for_one_cadence():
    """The sweep's one moving part: `--detect-every-n` overrides the plan field
    and names the arm after it, so a run's filename says what it measured."""
    from bench.selective import PRIORITY_CASE

    _, case = resolve_target(
        build_parser().parse_args([PRIORITY_CASE, "--detect-every-n", "8"]))
    assert case.detect_every_n == 8
    assert case.name == f"{PRIORITY_CASE}-n8"


def test_a_swept_arm_is_written_beside_the_other_arms_not_beside_the_baselines():
    """The rule the whole sweep rests on: `--selective-report` and
    `--portability-report` reduce `bench/results/selective/` to the newest run per
    (case, GPU), so an arm at another cadence landing there would silently become
    the row spec 8.8 and 7.4 quote."""
    from bench.paths import CADENCE_RESULTS_SUBDIR, SELECTIVE_RESULTS_SUBDIR
    from bench.selective import CASES, PRIORITY_CASE, results_subdir

    assert results_subdir(CASES[PRIORITY_CASE]) == SELECTIVE_RESULTS_SUBDIR
    assert results_subdir(
        CASES[PRIORITY_CASE].replace(detect_every_n=5)) == CADENCE_RESULTS_SUBDIR


def test_the_cadence_report_reads_the_committed_arms_and_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "bench", "--cadence-report"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "detect_every_n" in result.stdout


# --- the plan swap (issue #30) -----------------------------------------------

def test_a_swap_case_shares_the_positional_slot_too():
    """A fifth registry, still one slot: the CLI works out the kind from the name."""
    from bench.plan_swap import CASES, TARGET_CASE

    kind, target = resolve_target(build_parser().parse_args([TARGET_CASE]))
    assert kind == "swap" and target is CASES[TARGET_CASE]


def test_a_swap_run_can_be_shortened_for_a_development_run():
    """`--frames` shortens the clip and moves the swap with it, or a short run
    would submit the new instruction after the last frame."""
    from bench.plan_swap import TARGET_CASE

    _, case = resolve_target(build_parser().parse_args([TARGET_CASE, "--frames", "12"]))
    assert case.frames == 12
    assert case.swap_frame < case.frames


def test_list_names_both_swap_cases():
    result = subprocess.run(
        [sys.executable, "-m", "bench", "--list"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "swap-target" in result.stdout and "swap-style" in result.stdout


def test_the_swap_report_reads_the_committed_runs_and_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "bench", "--swap-report"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "criterion 1" in result.stdout.lower()


def test_a_swap_run_gates_the_engine_it_renders_through(tmp_path, monkeypatch):
    """Refused before `bench.plan_swap_runner` imports torch, like every other run."""
    import bench.cli as cli
    from bench.plan_swap import TARGET_CASE

    monkeypatch.setattr(cli, "resolve_engines_dir", lambda: tmp_path)
    with pytest.raises(SystemExit) as failure:
        cli.main([TARGET_CASE])
    assert "--allow-engine-build" in str(failure.value)


def test_a_swap_lands_in_its_own_directory_rather_than_beside_the_selective_runs():
    """The same routing rule the cadence arms have: `--selective-report` and
    `--portability-report` read every JSON in `bench/results/selective/`, and a
    swap record is not a selective one."""
    from bench.paths import SELECTIVE_RESULTS_SUBDIR, SWAP_RESULTS_SUBDIR

    assert SWAP_RESULTS_SUBDIR != SELECTIVE_RESULTS_SUBDIR
    source = (ROOT / "bench" / "cli.py").read_text(encoding="utf-8")
    body = source.split("def run_swap_target", 1)[1].split("\ndef ", 1)[0]
    assert "SWAP_RESULTS_SUBDIR" in body


# --- the occupancy gate (issue #33) ------------------------------------------


def test_a_clear_card_passes_the_gate_that_demands_one():
    from bench.cli import idle_gpu_guard
    from bench.contention import CLEAR, OccupancyRecord

    record = idle_gpu_guard(
        True, measure=lambda: OccupancyRecord(outcome=CLEAR, mean_utilization_pct=1.0))
    assert record.clear is True


@pytest.mark.parametrize("outcome,utilization", [("busy", 49.0), ("unknown", None)])
def test_the_gate_stops_a_run_that_would_decide_something_on_a_shared_card(
        outcome, utilization):
    """Issue #33: the run that measured 3.2x its own committed baseline passed every
    other door. `unknown` refuses too, for the reason it refuses a clock lock."""
    from bench.cli import idle_gpu_guard
    from bench.contention import OccupancyRecord

    with pytest.raises(SystemExit) as exit_info:
        idle_gpu_guard(True, measure=lambda: OccupancyRecord(
            outcome=outcome, mean_utilization_pct=utilization))
    message = str(exit_info.value)
    assert outcome in message
    assert "--require-idle-gpu" in message, "the message has to name the flag"
    assert "Close whatever else" in message, "and say how to fix it"


def test_without_the_flag_the_gate_does_not_even_sample():
    """A gate nobody asked for must not lengthen every run by two seconds."""
    from bench.cli import idle_gpu_guard

    def measure():
        raise AssertionError("sampled without --require-idle-gpu")

    assert idle_gpu_guard(False, measure=measure) is None


# --- the step-count sweep (issue #38) ----------------------------------------

def test_a_scenario_can_be_asked_for_one_step_count():
    """`--steps` moves the count and nothing else: the opening index is the
    scenario's, and the extra steps are spent after it."""
    base = SCENARIOS["img2img-none-512x512-b1"]
    kind, scenario = resolve_target(
        build_parser().parse_args(["img2img-none-512x512-b1", "--steps", "4"]))
    assert kind == "scenario"
    assert scenario.steps == 4
    assert scenario.t_index_list[0] == base.t_index_list[0]
    assert scenario.name == "img2img-none-512x512-b1-s4"


def test_the_one_step_control_is_an_arm_too_and_lands_with_the_arms():
    """`bench --marginal` reads every JSON in `bench/results/` as a batch cell, so
    even the control has to be written under `steps/` - and it has to be measured
    on this machine, because the committed batch-1 cell is a laptop's."""
    from bench.paths import RESULTS_DIR, STEPS_RESULTS_SUBDIR
    from bench.cli import scenario_results_dir

    _, arm = resolve_target(
        build_parser().parse_args(["img2img-none-512x512-b1", "--steps", "1"]))
    plain = SCENARIOS["img2img-none-512x512-b1"]
    assert scenario_results_dir(arm, RESULTS_DIR) == RESULTS_DIR / STEPS_RESULTS_SUBDIR
    assert scenario_results_dir(plain, RESULTS_DIR) == RESULTS_DIR


def test_a_four_step_tensorrt_arm_is_gated_on_its_own_engine(tmp_path):
    """Four steps go through the UNet as a batch of four, which is a different
    engine - ~5 GB and 15-25 minutes - so it needs the same explicit opt-in."""
    from bench.cli import engine_build_guard

    _, arm = resolve_target(
        build_parser().parse_args(["img2img-tensorrt-512x512-b1", "--steps", "4"]))
    with pytest.raises(SystemExit) as excinfo:
        engine_build_guard(arm, engines_root=tmp_path, allow_build=False)
    assert "img2img-tensorrt-512x512-b1-s4" in str(excinfo.value)


def test_the_steps_report_exits_zero_even_before_anything_is_committed():
    result = subprocess.run(
        [sys.executable, "-m", "bench", "--steps-report"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "step" in result.stdout


# --- the base-model arm (issue #38) ------------------------------------------

def test_a_selective_run_can_be_asked_for_another_base_model():
    from bench.selective import PRIORITY_CASE

    _, case = resolve_target(
        build_parser().parse_args([PRIORITY_CASE, "--base-model", "sd15"]))
    assert case.base_model == "sd15"
    assert case.name == f"{PRIORITY_CASE}-sd15"


def test_a_base_model_arm_is_written_beside_the_other_arms_not_the_baselines():
    from bench.paths import MODEL_RESULTS_SUBDIR, SELECTIVE_RESULTS_SUBDIR
    from bench.selective import CASES, PRIORITY_CASE, results_subdir

    assert results_subdir(CASES[PRIORITY_CASE]) == SELECTIVE_RESULTS_SUBDIR
    assert results_subdir(
        CASES[PRIORITY_CASE].replace(base_model="sd15")) == MODEL_RESULTS_SUBDIR


def test_a_base_model_arm_renders_through_that_model_s_own_engine():
    """At the step count `BASE_MODELS` says it needs - the arm differs from the
    baseline in the model, and in what the model cannot render without."""
    from bench.selective import CASES, PRIORITY_CASE, engine_scenario_for

    scenario = engine_scenario_for(CASES[PRIORITY_CASE].replace(base_model="sd15"))
    assert scenario.model == "sd-v1-5-fp16"
    assert scenario.use_lcm_lora is True
    assert scenario.steps == 4
    assert scenario.acceleration == "tensorrt"


def test_the_shipped_case_still_renders_through_the_shipped_engine():
    """Every committed baseline was measured through it; the arm must not move it."""
    from bench.selective import CASES, ENGINE_SCENARIO, PRIORITY_CASE, \
        engine_scenario_for

    assert engine_scenario_for(CASES[PRIORITY_CASE]) is SCENARIOS[ENGINE_SCENARIO]
