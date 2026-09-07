"""Result records: the shape on disk, and the rules that guard it.

Two rules, both enforced at the two doors to disk - `write_result` and
`append_readme_row` - and both refusing rather than warning.

The first is that nothing reaches `bench/results/` without a hardware fingerprint.
The merge gate cannot tell a measured number from an invented one; the fingerprint
is what makes that difference checkable.

The second (issue #13) is that nothing reaches it without saying which clock regime
produced it. A millisecond figure measured at an unlocked clock is not comparable
with another one at face value, and a reader months later cannot recover the regime
from the number.

Result files are written by a run and never by hand. If a run did not happen, there
is no record.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (Callable, Dict, List, Mapping, Optional, Sequence, Tuple,
                    Union)

from bench import RESULT_SCHEMA_VERSION
from bench.clocks import LOCKED, REGIMES, ClockNormalization, regime_of
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
    """A result that does not say which machine, or which clock regime, produced it.

    Not writable either way: both are claims the merge gate cannot check for itself
    after the fact.
    """


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
    # Issue #13: which clock regime produced this number, and - when the clocks were
    # not locked - what it would have been at one clock. The regime itself lives in
    # the fingerprint, because it is a fact about the machine; this is what follows
    # from it for the timing, which is why it sits beside the run instead.
    clock_normalization: Optional[ClockNormalization] = None

    def to_dict(self) -> dict:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "scenario": self.scenario.to_dict(),
            "run": self.run.to_dict(),
            "cooldown": self.cooldown.to_dict(),
            "hardware": self.hardware.to_dict(),
            "disk": None if self.disk is None else self.disk.to_dict(),
            "clock_normalization": (None if self.clock_normalization is None
                                    else self.clock_normalization.to_dict()),
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


def require_clock_lock_state(result: dict) -> None:
    """Raise `FingerprintError` unless the record says which clock regime produced it.

    Separate from `require_fingerprint` on purpose. The results committed before
    issue #13 carry no such field and are not retro-edited - they were all measured
    on this laptop with no lock in force, and `regime_of` reads their silence as
    exactly that. This rule applies at the door to disk, so everything written from
    now on answers for itself instead of relying on that reading.

    `unknown` passes: a driver that cannot answer is a real state of a real machine,
    and refusing it would discard a measurement. `--require-locked-clocks` is where
    a run that must not be unlocked says so.
    """
    hardware = result.get("hardware")
    if not isinstance(hardware, dict):
        raise FingerprintError("result has no hardware fingerprint")
    lock = hardware.get("clock_lock")
    if not isinstance(lock, dict) or "state" not in lock:
        raise FingerprintError("result does not record whether the GPU clocks were locked")
    if lock["state"] not in REGIMES:
        raise FingerprintError(f"clock lock state {lock['state']!r} is not one of {REGIMES}")


def require_recordable(result: dict) -> None:
    """Both rules, in the order a reader meets them. Every path to disk calls this.

    One function rather than two calls at each door, so a third rule cannot be added
    to one door and forgotten at the other.
    """
    require_fingerprint(result)
    require_clock_lock_state(result)


def result_filename(scenario_name: str, timestamp: str) -> str:
    return f"{scenario_name}-{timestamp}.json"


def write_record(data: dict, results_dir: Path, filename: str) -> Path:
    """One record to disk under `results_dir`, past both door rules; return its path.

    Shared with `bench.detector_results` (issue #4), which measures something else
    entirely but is subject to the same two rules. Sharing the *door* rather than
    the record shape is what keeps a second kind of result from acquiring a second,
    laxer way to reach `bench/results/`.
    """
    require_recordable(data)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_result(result: ResultLike, results_dir: Path, timestamp: Optional[str] = None) -> Path:
    """Write `<scenario>-<timestamp>.json` under `results_dir`; return its path."""
    data = _as_dict(result)
    if timestamp is None:
        timestamp = timestamp_from(data)
    return write_record(data, results_dir, result_filename(data["scenario"]["name"], timestamp))


def filename_timestamp(utc: str) -> str:
    """`20260906-140031Z` from a UTC timestamp - filename-safe, sortable.

    Public because a run that writes files *beside* its record - the primitive
    comparison writes clips (issue #5) - has to name them before the record exists,
    and a second spelling of this would let the clips and the JSON drift apart.
    """
    return utc.replace("-", "").replace(":", "").replace("T", "-")


def timestamp_from(data: dict) -> str:
    """`20260906-140031Z` from the run's finish time."""
    return filename_timestamp(str(data["run"]["finished_utc"]))


def load_records(results_dir: Path) -> Dict[str, dict]:
    """Every result JSON under `results_dir`, keyed by filename.

    One directory of records at a time: `bench/results/` holds diffusion cells and
    each other kind has its own subdirectory, so a glob here never mixes shapes.
    """
    return {path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(Path(results_dir).glob("*.json"))}


UNKNOWN_GPU = "unknown GPU"


def gpu_of(result: Mapping) -> str:
    """Which machine produced this record, or `unknown GPU` when it does not say.

    Results predating the hardware fingerprint carry no `hardware.gpu_name`. Reading
    that back as `None` would put every one of them in a single group named after
    nothing, which is the same deletion issue #25 exists to stop - so the absence
    gets a name of its own and reads as one in a table.
    """
    hardware = result.get("hardware") or {}
    return str(hardware.get("gpu_name") or "").strip() or UNKNOWN_GPU


def latest_per(results: Mapping[str, dict],
               name_of: Callable[[dict], str]) -> Dict[str, dict]:
    """One result per thing measured *per machine*: the newest run of each.

    A thing measured twice on one GPU is two honest records and both stay on disk; a
    table built from a mixture would compare a cold run against a hot one. The same
    thing measured on a *second* GPU is not a re-measurement at all - it is the
    comparison spec 7.4 is built on, and a reduction that kept one row per name
    would delete the development laptop the first time a deploy card ran (issue
    #25). So the key is the name and the GPU together.

    `name_of` says what "the same thing" means - a detector name, a case name -
    because that is the only part the three report tables disagree on.
    """
    newest: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for filename, result in results.items():
        gpu = gpu_of(result)
        # A record that does not name its machine cannot be shown to supersede
        # another one, so it stands alone under its own filename rather than
        # joining a group of anonymous results and losing to the newest of them.
        machine = gpu if gpu != UNKNOWN_GPU else filename
        finished = str(result["run"]["finished_utc"])
        key = (name_of(result), machine)
        if key not in newest or finished > newest[key][1]:
            newest[key] = (filename, finished)
    return {filename: results[filename] for filename, _ in newest.values()}


def distinct_gpus(results: Sequence[dict]) -> List[str]:
    """Every machine these records come from, sorted - the report's grouping axis."""
    return sorted({gpu_of(result) for result in results})


def measured_on(results: Sequence[dict], gpu: str) -> List[dict]:
    """The records from one machine, in the order they were given.

    A verdict - a recommendation, a decision, a Gate line - is a claim about one
    machine, so a report that spans two takes each verdict from one machine's
    records at a time (issue #25).
    """
    return [result for result in results if gpu_of(result) == gpu]


def gpu_suffix(gpu: str, gpus: Sequence[str]) -> str:
    """` (<GPU>)` to qualify a line, or nothing when the block is from one machine.

    What it avoids is two identical-looking bullets the first time a second machine
    measures the same case; with one machine the line reads as it always did.
    """
    return "" if len(gpus) == 1 else f" ({gpu})"


def table_row(cells: Sequence[str]) -> str:
    """One Markdown row from its cells."""
    return "| " + " | ".join(cells) + " |"


def table_separator(header: str) -> str:
    """The `|---|---|` under `header`, derived so it cannot be the wrong width."""
    return "|" + "---|" * (header.count("|") - 1)


def insert_cell(cells: Sequence[str], value: str, index: int) -> List[str]:
    """`cells` with `value` put at `index` - one row of an optional column.

    Shared by `GpuColumn` and `OptionalColumn` so a heading and its cells cannot
    be inserted by two rules that disagree about where the column sits.
    """
    return [*cells[:index], value, *cells[index:]]


@dataclass(frozen=True)
class OptionalColumn:
    """A column a table grows only when its rows disagree about the value.

    The rule `GpuColumn` applies to the machine, for any other dimension that is
    usually constant and occasionally is not - the base model, in issue #38's
    step-count table. Off, the table renders exactly as it did, which is what
    keeps a byte-matched spec block from churning the first time a second value
    is measured; on, no row implies a value it was not measured at.
    """

    heading: str
    value_of: Callable[[Mapping], str]
    shown: bool
    index: int = 1

    @classmethod
    def when_varied(cls, results: Sequence[dict], heading: str,
                    value_of: Callable[[Mapping], str],
                    index: int = 1) -> "OptionalColumn":
        values = {value_of(result) for result in results}
        return cls(heading=heading, value_of=value_of, shown=len(values) > 1,
                   index=index)

    def header(self, header: str) -> str:
        if not self.shown:
            return header
        return " | ".join(insert_cell(header.split(" | "), self.heading, self.index))

    def cells(self, cells: Sequence[str], result: Mapping) -> List[str]:
        if not self.shown:
            return list(cells)
        return insert_cell(cells, self.value_of(result), self.index)


GPU_COLUMN = "GPU"


@dataclass(frozen=True)
class GpuColumn:
    """The `GPU` column a report table grows when its rows span more than one machine.

    Off with one machine, and that is the point: a single-GPU repo renders exactly
    what it rendered before, so the three committed spec blocks do not churn and
    their byte-match tests keep passing while this lands (issue #25's third trap).
    On with two, so no row implies a machine it was not measured on.

    `index` is where the column sits, and it is a field rather than an argument to
    both `header` and `row` because a table whose two disagreed would put the
    heading over the wrong cells.
    """

    shown: bool
    index: int = 1

    @classmethod
    def for_gpus(cls, gpus: Sequence[str], index: int = 1) -> "GpuColumn":
        """The column a table of records from `gpus` needs: shown past one machine."""
        return cls(shown=len(gpus) > 1, index=index)

    def header(self, header: str) -> str:
        """`header` with `GPU` inserted at `self.index`, or unchanged."""
        if not self.shown:
            return header
        return " | ".join(self._inserted(header.split(" | "), GPU_COLUMN))

    def row(self, cells: Sequence[str], result: dict) -> str:
        """One row, carrying its machine when the table has that column."""
        if not self.shown:
            return table_row(cells)
        return table_row(self._inserted(cells, gpu_of(result)))

    def _inserted(self, cells: Sequence[str], value: str) -> List[str]:
        return insert_cell(cells, value, self.index)


def sentence_case(statement: str) -> str:
    """A statement built to sit mid-sentence, capitalised to open one instead.

    The `statement` field every summary dataclass carries is written lower-case so
    a report can quote it inside a sentence; a report that opens a paragraph with
    one needs the other form, and both readings come from the one stored string.
    """
    return statement[:1].upper() + statement[1:]


def format_number(value: Optional[float], digits: int = 1) -> str:
    """One table cell: a number at `digits` places, or `-` when there is none.

    A missing figure reads as `-` rather than `0.00`, which would be a measurement.
    """
    return "-" if value is None else f"{value:.{digits}f}"


README_TITLE = "# Benchmark results"
README_INTRO = (
    "Written by `uv run python -m bench <scenario>`, never by hand. Each row points at\n"
    "the JSON file holding the full scenario config and hardware fingerprint.\n\n"
    "Absolute ms/frame and VRAM figures belong to the GPU in the row - spec 7.4. Compare\n"
    "rows across GPUs for curve shape and ranking only.\n\n"
    "`clock regime` says whether the GPU clocks were locked while the row was measured.\n"
    "Unlocked, the last column carries a first-order estimate of the same work at one\n"
    "clock - an estimate, not a measurement. A row with neither cell was written before\n"
    "the field existed, and was measured unlocked (issue #13).\n"
)
# The two clock columns (issue #13) are appended *after* `file` rather than inserted
# beside the SM clock, because the rows already committed have no cells for them: a
# new column in the middle would slide every older row's values one place left and
# misreport them. Appended, an older row simply stops early - which is what "unlocked
# by absence" looks like in a table.
README_HEADER = (
    "| finished (UTC) | scenario | GPU | accel | res | batch | steps | ms/frame | FPS |"
    " peak VRAM (MiB) | SM clock (MHz) | max temp (C) | cooldown | file |"
    " clock regime | ms/frame at basis clock |"
)
# Derived, so adding a column to the header cannot leave a separator of the wrong
# width behind - which renders the whole table as plain text.
README_SEPARATOR = table_separator(README_HEADER)
README_NAME = "README.md"


def normalised_cell(result: dict) -> str:
    """The clock-normalised ms/frame, or why there is none.

    `raw` for a locked run: there is nothing to correct, and the raw column already
    holds the comparable figure. Spelt out with its basis otherwise, because a
    millisecond figure with no clock attached is what issue #13 is about.

    Public because the detector table (issue #4) carries the same column: two
    detectors measured minutes apart on an unlocked laptop are ranked by this figure,
    not by the raw one.
    """
    if regime_of(result) == LOCKED:
        return "raw (clocks locked)"
    normalisation = result.get("clock_normalization") or {}
    ms, basis = normalisation.get("ms_per_frame"), normalisation.get("basis_mhz")
    if ms is None or basis is None:
        return "-"
    return f"{ms:.2f} @ {basis:.0f} MHz"


def readme_row(result: dict, filename: str) -> str:
    scenario, run = result["scenario"], result["run"]
    cooldown, hardware = result["cooldown"], result["hardware"]
    return table_row([
        run["finished_utc"],
        scenario["name"],
        hardware["gpu_name"],
        scenario["acceleration"],
        f"{scenario['width']}x{scenario['height']}",
        str(scenario["batch_size"]),
        str(len(scenario["t_index_list"])),
        format_number(run["mean_ms_per_frame"], 2),
        format_number(run["fps"], 1),
        format_number(run["peak_vram_bytes"] / BYTES_PER_MIB, 0),
        format_number(run["mean_sm_clock_mhz"], 0),
        format_number(run["max_temperature_c"], 0),
        cooldown["outcome"],
        f"[{filename}]({filename})",
        regime_of(result),
        normalised_cell(result),
    ])


def append_row(row: str, readme_path: Path, preamble: str) -> None:
    """Append one row to a results table, writing `preamble` if the file is new.

    Shared with `bench.detector_results`, which keeps a second table of a different
    shape. A row is appended under whatever header the file already has, so the
    preamble a writer would create and the header its rows assume have to be one
    thing - which is why this takes the whole preamble rather than a path alone.
    """
    readme_path = Path(readme_path)
    if not readme_path.exists():
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        readme_path.write_text(preamble, encoding="utf-8")
    with readme_path.open("a", encoding="utf-8") as handle:
        handle.write(row + "\n")


def append_readme_row(result: ResultLike, readme_path: Path, filename: str) -> None:
    """Append one readable row, creating the table if this is the first result."""
    data = _as_dict(result)
    require_recordable(data)
    preamble = f"{README_TITLE}\n\n{README_INTRO}\n{README_HEADER}\n{README_SEPARATOR}\n"
    append_row(readme_row(data, filename), readme_path, preamble)
