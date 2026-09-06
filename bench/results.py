"""Result records: the shape on disk, and the one rule that guards it.

The rule is that nothing reaches `bench/results/` without a hardware fingerprint.
The merge gate cannot tell a measured number from an invented one; the fingerprint
is what makes that difference checkable, so `write_result` refuses rather than
warns, and the README row goes through the same check.

Result files are written by a run and never by hand. If a run did not happen, there
is no record.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from bench import RESULT_SCHEMA_VERSION
from bench.cooldown import CooldownRecord
from bench.disk import DiskRecord
from bench.fingerprint import Fingerprint
from bench.scenarios import ScenarioConfig

# Must be present *and* carry a value.
NON_EMPTY_FINGERPRINT_FIELDS = (
    "gpu_name", "total_vram_mib", "driver_version",
    "nvidia_smi_raw", "nvidia_smi_captured_utc",
)
# The power limit spec 7.4 asks for, under either of the two names nvidia-smi gives
# it. One of them has to have a value: this laptop reports `[N/A]` for `power.limit`
# and 120 W for `enforced.power.limit`, and a desktop reports both. Neither is a
# field that may simply be absent.
POWER_LIMIT_FIELDS = ("power_limit_w", "enforced_power_limit_w")
# Present, whether or not they carry a value.
REQUIRED_FINGERPRINT_FIELDS = NON_EMPTY_FINGERPRINT_FIELDS + POWER_LIMIT_FIELDS

BYTES_PER_MIB = 1024 * 1024


class FingerprintError(ValueError):
    """A result that does not say which machine produced it. Not writable."""


@dataclass(frozen=True)
class RunMetrics:
    """Everything measured about one run of one scenario.

    Clocks and temperature sit here next to the timings on purpose: a run that
    thermally throttled reports a real number, and the only way to see that it is
    not comparable is to read the clock beside it.
    """

    started_utc: str
    finished_utc: str
    warmup_reps: int
    reps: int
    per_rep_ms: List[float]
    mean_ms_per_frame: float
    median_ms_per_frame: float
    p95_ms_per_frame: float
    min_ms_per_frame: float
    max_ms_per_frame: float
    stdev_ms_per_frame: float
    fps: float
    mean_sm_clock_mhz: Optional[float]
    max_temperature_c: Optional[float]
    peak_vram_bytes: int
    # {"unet": ms, "vae_encode": ms, "vae_decode": ms} behind --per-module, else None.
    # The per-module timing needs a synchronise around each submodule, which perturbs
    # the total - so it is off by default and never mixed into a headline number.
    per_module_ms: Optional[Dict[str, float]] = None
    # [[elapsed_s, sm_clock_mhz, temperature_c], ...], sampled off the timed loop.
    gpu_samples: List[List[Optional[float]]] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["peak_vram_mib"] = round(self.peak_vram_bytes / BYTES_PER_MIB, 1)
        return data


@dataclass(frozen=True)
class BenchResult:
    scenario: ScenarioConfig
    run: RunMetrics
    cooldown: CooldownRecord
    hardware: Fingerprint
    # The headroom the volume had when this run was about to compile an engine, and
    # `None` when it was not about to compile one. Issue #3 asks for the free-disk
    # check to be *recorded*, not merely applied: a reviewer reading the JSON is the
    # one who has to see that the ~5.1 GB gate was cleared before the build started.
    disk: Optional[DiskRecord] = None

    def to_dict(self) -> dict:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "scenario": self.scenario.to_dict(),
            "run": self.run.to_dict(),
            "cooldown": self.cooldown.to_dict(),
            "hardware": self.hardware.to_dict(),
            "disk": None if self.disk is None else self.disk.to_dict(),
        }


ResultLike = Union[BenchResult, dict]


def _as_dict(result: ResultLike) -> dict:
    return result.to_dict() if isinstance(result, BenchResult) else result


def require_fingerprint(result: dict) -> None:
    """Raise `FingerprintError` unless the record identifies its machine.

    Called before every write and every README append, so there is no path to disk
    that skips it.
    """
    hardware = result.get("hardware")
    if not isinstance(hardware, dict):
        raise FingerprintError("result has no hardware fingerprint")
    missing = [key for key in REQUIRED_FINGERPRINT_FIELDS if key not in hardware]
    if missing:
        raise FingerprintError(f"fingerprint is missing {missing}")
    blank = [key for key in NON_EMPTY_FINGERPRINT_FIELDS
             if hardware[key] is None or str(hardware[key]).strip() == ""]
    if blank:
        raise FingerprintError(f"fingerprint fields are empty: {blank}")
    if all(hardware[key] is None for key in POWER_LIMIT_FIELDS):
        raise FingerprintError(f"fingerprint has no power limit under any of {POWER_LIMIT_FIELDS}")


def result_filename(scenario_name: str, timestamp: str) -> str:
    return f"{scenario_name}-{timestamp}.json"


def write_result(result: ResultLike, results_dir: Path, timestamp: Optional[str] = None) -> Path:
    """Write `<scenario>-<timestamp>.json` under `results_dir`; return its path."""
    data = _as_dict(result)
    require_fingerprint(data)
    if timestamp is None:
        timestamp = _timestamp_from(data)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / result_filename(data["scenario"]["name"], timestamp)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _timestamp_from(data: dict) -> str:
    """`20260906-140031Z` from the run's finish time - filename-safe, sortable."""
    finished = str(data["run"]["finished_utc"])
    return finished.replace("-", "").replace(":", "").replace("T", "-")


README_TITLE = "# Benchmark results"
README_INTRO = (
    "Written by `uv run python -m bench <scenario>`, never by hand. Each row points at\n"
    "the JSON file holding the full scenario config and hardware fingerprint.\n\n"
    "Absolute ms/frame and VRAM figures belong to the GPU in the row - spec 7.4. Compare\n"
    "rows across GPUs for curve shape and ranking only.\n"
)
README_HEADER = (
    "| finished (UTC) | scenario | GPU | accel | res | batch | steps | ms/frame | FPS |"
    " peak VRAM (MiB) | SM clock (MHz) | max temp (C) | cooldown | file |"
)
# Derived, so adding a column to the header cannot leave a separator of the wrong
# width behind - which renders the whole table as plain text.
README_SEPARATOR = "|" + "---|" * (README_HEADER.count("|") - 1)
README_NAME = "README.md"


def readme_row(result: dict, filename: str) -> str:
    scenario, run = result["scenario"], result["run"]
    cooldown, hardware = result["cooldown"], result["hardware"]

    def number(value, digits=1):
        return "-" if value is None else f"{value:.{digits}f}"

    return "| " + " | ".join([
        run["finished_utc"],
        scenario["name"],
        hardware["gpu_name"],
        scenario["acceleration"],
        f"{scenario['width']}x{scenario['height']}",
        str(scenario["batch_size"]),
        str(len(scenario["t_index_list"])),
        number(run["mean_ms_per_frame"], 2),
        number(run["fps"], 1),
        number(run["peak_vram_bytes"] / BYTES_PER_MIB, 0),
        number(run["mean_sm_clock_mhz"], 0),
        number(run["max_temperature_c"], 0),
        cooldown["outcome"],
        f"[{filename}]({filename})",
    ]) + " |"


def append_readme_row(result: ResultLike, readme_path: Path, filename: str) -> None:
    """Append one readable row, creating the table if this is the first result."""
    data = _as_dict(result)
    require_fingerprint(data)
    readme_path = Path(readme_path)
    if not readme_path.exists():
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        readme_path.write_text(
            f"{README_TITLE}\n\n{README_INTRO}\n{README_HEADER}\n{README_SEPARATOR}\n",
            encoding="utf-8",
        )
    with readme_path.open("a", encoding="utf-8") as handle:
        handle.write(readme_row(data, filename) + "\n")
