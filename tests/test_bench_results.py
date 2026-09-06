"""Result records (issue #2): serialisation, and the fingerprint-required rule.

The rule is the one the merge gate cannot check for itself - a JSON file with a
number in it proves nothing unless a machine is attached to it.
"""

import json

import pytest

from sourceloader import ROOT

from bench.clocks import LOCKED, UNLOCKED, ClockLock, clock_normalization
from bench.cooldown import REACHED, CooldownRecord
from bench.fingerprint import Fingerprint
from bench.results import (
    README_HEADER,
    README_SEPARATOR,
    BenchResult,
    FingerprintError,
    RunMetrics,
    append_readme_row,
    require_clock_lock_state,
    require_fingerprint,
    result_filename,
    write_result,
)
from bench.scenarios import ScenarioConfig


def a_lock(**overrides) -> ClockLock:
    fields = dict(state=UNLOCKED, applied_clock_mhz=None, max_sm_clock_mhz=2100.0,
                  current_sm_clock_mhz=210.0,
                  evidence="clocks_event_reasons.applications_clocks_setting=Not Active")
    fields.update(overrides)
    return ClockLock(**fields)


def a_fingerprint(**overrides) -> Fingerprint:
    fields = dict(
        gpu_name="NVIDIA GeForce RTX 3080 Laptop GPU",
        total_vram_mib=16384.0,
        driver_version="595.79",
        power_limit_w=None,
        enforced_power_limit_w=120.0,
        clock_lock=a_lock(),
        torch_version="2.7.0+cu128",
        cuda_version="12.8",
        hostname="dev-laptop",
        nvidia_smi_captured_utc="2026-09-06T14:00:00Z",
        nvidia_smi_raw="+---- raw nvidia-smi ----+",
    )
    fields.update(overrides)
    return Fingerprint(**fields)


def a_result(**overrides) -> BenchResult:
    fields = dict(
        scenario=ScenarioConfig(name="img2img-none-512x512-b1", acceleration="none",
                                width=512, height=512, batch_size=1, t_index_list=[35]),
        run=RunMetrics(
            started_utc="2026-09-06T14:00:01Z", finished_utc="2026-09-06T14:00:31Z",
            warmup_reps=3, reps=30, per_rep_ms=[10.0, 12.0, 14.0],
            mean_ms_per_frame=12.0, median_ms_per_frame=12.0, p95_ms_per_frame=14.0,
            min_ms_per_frame=10.0, max_ms_per_frame=14.0, stdev_ms_per_frame=2.0,
            fps=1000.0 / 12.0, mean_sm_clock_mhz=1650.0, max_temperature_c=71.0,
            peak_vram_bytes=2 * 1024 ** 3, per_module_ms=None,
        ),
        cooldown=CooldownRecord(enabled=True, outcome=REACHED, threshold_c=62.0, cap_s=120.0,
                                waited_s=18.0, final_temperature_c=60.0, samples=[[0.0, 70.0]]),
        hardware=a_fingerprint(),
        clock_normalization=clock_normalization(
            a_lock(), samples=[[0.0, 1050.0, 70.0], [0.5, 1050.0, 71.0]],
            raw_ms_per_frame=12.0,
        ),
    )
    fields.update(overrides)
    return BenchResult(**fields)


def test_a_result_serialises_to_json_with_every_field_the_issue_asks_for():
    data = a_result().to_dict()
    assert data["run"]["mean_ms_per_frame"] == 12.0
    assert data["run"]["fps"] == pytest.approx(83.333, rel=1e-3)
    assert data["run"]["mean_sm_clock_mhz"] == 1650.0
    assert data["run"]["max_temperature_c"] == 71.0
    assert data["run"]["peak_vram_bytes"] == 2 * 1024 ** 3
    assert data["cooldown"]["outcome"] == REACHED
    assert data["scenario"]["acceleration"] == "none"
    assert data["hardware"]["gpu_name"].startswith("NVIDIA")
    json.dumps(data)  # plain types only


def test_the_schema_version_is_recorded():
    assert a_result().to_dict()["schema_version"] >= 1


def test_a_result_with_a_fingerprint_passes_the_rule():
    require_fingerprint(a_result().to_dict())


@pytest.mark.parametrize("missing", ["gpu_name", "driver_version", "total_vram_mib",
                                     "nvidia_smi_raw", "nvidia_smi_captured_utc"])
def test_a_blank_fingerprint_field_is_refused(missing):
    data = a_result().to_dict()
    data["hardware"][missing] = None
    with pytest.raises(FingerprintError):
        require_fingerprint(data)


def test_one_of_the_two_power_limits_is_enough_but_neither_is_not():
    """This laptop reports `[N/A]` for `power.limit` and 120 W for the enforced one."""
    data = a_result().to_dict()
    assert data["hardware"]["power_limit_w"] is None
    require_fingerprint(data)  # the enforced limit carries it

    data["hardware"]["enforced_power_limit_w"] = None
    with pytest.raises(FingerprintError):
        require_fingerprint(data)


def test_a_missing_power_limit_key_is_refused_even_when_the_other_has_a_value():
    data = a_result().to_dict()
    del data["hardware"]["power_limit_w"]
    with pytest.raises(FingerprintError):
        require_fingerprint(data)


def test_a_result_with_no_hardware_block_at_all_is_refused():
    data = a_result().to_dict()
    del data["hardware"]
    with pytest.raises(FingerprintError):
        require_fingerprint(data)


def test_no_file_is_written_without_a_fingerprint(tmp_path):
    """The gate the issue names: a result cannot reach disk unattributed."""
    data = a_result().to_dict()
    data["hardware"]["gpu_name"] = ""
    with pytest.raises(FingerprintError):
        write_result(data, results_dir=tmp_path)
    assert list(tmp_path.glob("*.json")) == []


def test_writing_a_result_produces_the_named_json_file(tmp_path):
    path = write_result(a_result(), results_dir=tmp_path, timestamp="20260906-140031Z")
    assert path.name == "img2img-none-512x512-b1-20260906-140031Z.json"
    assert json.loads(path.read_text(encoding="utf-8"))["run"]["reps"] == 30


def test_the_filename_carries_the_scenario_and_the_timestamp():
    assert result_filename("img2img-none-512x512-b1", "20260906-140031Z") == (
        "img2img-none-512x512-b1-20260906-140031Z.json"
    )


def test_the_readme_row_names_the_gpu(tmp_path):
    readme = tmp_path / "README.md"
    append_readme_row(a_result(), readme, filename="x.json")
    text = readme.read_text(encoding="utf-8")
    assert "RTX 3080 Laptop GPU" in text
    assert "| ms/frame |" in text, "the table header is written when the file is created"
    assert text.count("RTX 3080") == 1

    append_readme_row(a_result(), readme, filename="y.json")
    text = readme.read_text(encoding="utf-8")
    assert text.count("| ms/frame |") == 1, "the header is written once, rows append under it"
    assert text.count("RTX 3080") == 2


def test_the_readme_row_cannot_be_appended_without_a_fingerprint(tmp_path):
    readme = tmp_path / "README.md"
    data = a_result().to_dict()
    data["hardware"]["driver_version"] = ""
    with pytest.raises(FingerprintError):
        append_readme_row(data, readme, filename="x.json")
    assert not readme.exists()


def test_every_committed_result_carries_a_fingerprint():
    """The rule applied to the tree, not just to the writer.

    `bench/results/` is tracked because a committed result is the deliverable. This
    is what stops one drifting out of schema - or being hand-written - unnoticed.
    """
    committed = sorted((ROOT / "bench" / "results").glob("*.json"))
    assert committed, "no benchmark result is committed yet"
    for path in committed:
        require_fingerprint(json.loads(path.read_text(encoding="utf-8")))


def test_every_committed_result_has_a_row_in_the_readme():
    results_dir = ROOT / "bench" / "results"
    readme = (results_dir / "README.md").read_text(encoding="utf-8")
    for path in sorted(results_dir.glob("*.json")):
        assert path.name in readme, f"{path.name} was written but never listed"


# --- clock regime (issue #13) ----------------------------------------------

def test_a_result_records_which_clock_regime_produced_it():
    """Issue #13's gate: every result says whether its clocks were locked."""
    data = a_result().to_dict()
    require_clock_lock_state(data)
    assert data["hardware"]["clock_lock"]["state"] == UNLOCKED
    assert data["clock_normalization"]["regime"] == UNLOCKED
    assert data["clock_normalization"]["ms_per_frame"] == pytest.approx(6.0)
    assert data["clock_normalization"]["basis_mhz"] == 2100.0


def test_a_result_with_no_lock_state_cannot_be_written(tmp_path):
    """The gate again, at the door to disk: absence is refused, not defaulted."""
    data = a_result().to_dict()
    del data["hardware"]["clock_lock"]
    with pytest.raises(FingerprintError):
        require_clock_lock_state(data)
    with pytest.raises(FingerprintError):
        write_result(data, results_dir=tmp_path)
    assert list(tmp_path.glob("*.json")) == []

    readme = tmp_path / "README.md"
    with pytest.raises(FingerprintError):
        append_readme_row(data, readme, filename="x.json")
    assert not readme.exists()


def test_a_regime_outside_the_three_is_refused():
    data = a_result().to_dict()
    data["hardware"]["clock_lock"]["state"] = "probably fine"
    with pytest.raises(FingerprintError):
        require_clock_lock_state(data)


def test_an_unknown_regime_is_writable_because_it_is_the_truth():
    """`unknown` is a state the machine can genuinely be in - a driver with no such
    field - and refusing it would throw away a real measurement. It still fails
    `--require-locked-clocks`, which is where that decision belongs."""
    data = a_result(hardware=a_fingerprint(clock_lock=a_lock(state="unknown"))).to_dict()
    require_clock_lock_state(data)


def test_the_readme_row_surfaces_the_regime_and_the_normalised_figure(tmp_path):
    """Issue #13 step 3: the regime is visible in the table, not only in the JSON."""
    readme = tmp_path / "README.md"
    append_readme_row(a_result(), readme, filename="x.json")
    text = readme.read_text(encoding="utf-8")
    assert "| clock regime |" in text
    assert f"| {UNLOCKED} |" in text
    assert "6.00 @ 2100 MHz" in text, "the normalised figure names the basis it assumes"


def test_a_locked_row_shows_the_raw_figure_rather_than_an_estimate(tmp_path):
    readme = tmp_path / "README.md"
    locked = a_lock(state=LOCKED, applied_clock_mhz=1200.0)
    append_readme_row(
        a_result(hardware=a_fingerprint(clock_lock=locked),
                 clock_normalization=clock_normalization(
                     locked, [[0.0, 1200.0, 70.0]], raw_ms_per_frame=12.0)),
        readme, filename="y.json",
    )
    text = readme.read_text(encoding="utf-8")
    assert f"| {LOCKED} |" in text
    assert "| raw (clocks locked) |" in text, (
        "a locked run has nothing to normalise, and the row must not imply it does"
    )
    assert "@ 2100 MHz" not in text


def test_every_result_written_since_issue_13_records_its_regime():
    """The tree, not just the writer - and the older results stay as they were.

    Issue #13's trap is explicit that the pre-existing results are not retro-edited:
    they were all measured unlocked on this laptop and are marked so by absence.
    Anything written since carries the field, and the schema version is what tells
    the two apart.
    """
    for path in sorted((ROOT / "bench" / "results").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("schema_version", 1) >= 2:
            require_clock_lock_state(result)


def test_the_committed_table_has_the_columns_the_writer_writes():
    """A row is appended under whatever header the file already has.

    So a new column (issue #13 added two) has to reach the committed table's header
    as well, or every future row carries cells that render into nothing. The data
    rows are never touched - the older ones simply stop before the new columns.
    """
    readme = (ROOT / "bench" / "results" / "README.md").read_text(encoding="utf-8")
    assert README_HEADER in readme
    assert README_SEPARATOR in readme
