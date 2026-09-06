"""One real end-to-end bench run (issue #2). GPU tier.

Short and on the `none` accelerator, so it needs the downloaded model but never
compiles an engine. It asserts what the committed result files must satisfy: every
field of steps 4 and 5 present, and a fingerprint that matches this machine.
"""

import json
import os
from pathlib import Path

import pytest

from sourceloader import ROOT, load_symbols

from bench.results import require_fingerprint
from bench.scenarios import SCENARIOS

pytestmark = pytest.mark.gpu

MODEL_NAME = "sd-turbo-fp16"

resolve_models_dir = load_symbols(
    "main_gpu_addon.py",
    ["SD_MODELS_DIR_ENV", "SD_ENGINES_DIR_ENV", "_unquoted_path", "_resolve_cache_dir",
     "resolve_models_dir", "resolve_engines_dir"],
    extra_globals={"os": os, "Path": Path, "APP_ROOT": ROOT},
)["resolve_models_dir"]


@pytest.fixture
def model_dir():
    models_root = resolve_models_dir()
    path = models_root / MODEL_NAME
    if not path.is_dir():
        pytest.skip(f"no {MODEL_NAME} under {models_root} - set SD_MODELS_DIR")
    return path


def test_a_short_run_emits_a_complete_result(model_dir, tmp_path):
    from bench.runner import run_scenario

    scenario = SCENARIOS["img2img-none-256x256-b1"].replace(warmup_reps=1, reps=3)
    result = run_scenario(scenario, cooldown=False, per_module=True, results_dir=tmp_path)

    data = result.to_dict()
    require_fingerprint(data)
    assert data["run"]["mean_ms_per_frame"] > 0
    assert data["run"]["fps"] > 0
    assert data["run"]["peak_vram_bytes"] > 0
    assert data["run"]["max_temperature_c"] is not None
    assert data["run"]["mean_sm_clock_mhz"] is not None
    assert len(data["run"]["per_rep_ms"]) == 3
    assert data["cooldown"]["outcome"] == "skipped"
    assert set(data["run"]["per_module_ms"]) == {"unet", "vae_encode", "vae_decode"}
    assert data["hardware"]["gpu_name"]

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text(encoding="utf-8")) == data
    assert (tmp_path / "README.md").exists()


def test_the_cooldown_gate_reads_a_real_temperature():
    from bench.fingerprint import read_gpu_sample

    sample = read_gpu_sample()
    assert sample.temperature_c is not None and sample.temperature_c > 0
    assert sample.sm_clock_mhz is not None and sample.sm_clock_mhz > 0


def test_the_lock_state_of_this_machine_is_readable():
    """Issue #13 step 1, against the real driver rather than a fake one."""
    from bench.clocks import REGIMES
    from bench.fingerprint import read_clock_lock

    lock = read_clock_lock()
    assert lock.state in REGIMES
    assert "applications_clocks_setting" in lock.evidence
    assert lock.max_sm_clock_mhz and lock.max_sm_clock_mhz > 0, (
        "clocks.max.sm is the normalisation basis; without it there is no estimate"
    )


def test_a_real_run_records_its_clock_regime(model_dir, tmp_path):
    """Issue #13's gate, end to end: the regime is in the result and in the table.

    Whichever regime this machine is in - it is unlocked unless someone ran
    `nvidia-smi --lock-gpu-clocks` from an elevated shell first - the result has to
    say which, and carry an estimate exactly when the raw figure needs one.
    """
    from bench.clocks import LOCKED
    from bench.results import require_clock_lock_state
    from bench.runner import run_scenario

    scenario = SCENARIOS["img2img-none-256x256-b1"].replace(warmup_reps=1, reps=3)
    data = run_scenario(scenario, cooldown=False, results_dir=tmp_path).to_dict()

    require_clock_lock_state(data)
    regime = data["hardware"]["clock_lock"]["state"]
    normalisation = data["clock_normalization"]
    assert normalisation["regime"] == regime

    if regime == LOCKED:
        assert normalisation["ms_per_frame"] is None, "a locked run needs no estimate"
    else:
        assert normalisation["ms_per_frame"] > 0
        assert normalisation["basis_mhz"] > 0
        assert normalisation["basis_source"] == "clocks.max.sm"
        assert "estimate" in normalisation["note"].lower()

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "| clock regime |" in readme and f"| {regime} |" in readme
