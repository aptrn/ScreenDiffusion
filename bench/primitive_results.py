"""What a rendering-primitive comparison records, and the decision block spec 8.2 carries.

Issue #5. A primitive result answers a different question from a diffusion cell or a
detector run - what a primitive costs *and* what it could express - so it has its
own record and its own table, in `bench/results/primitives/`.

Its own *directory*, for the reason `bench/results/detectors/` has one: `bench
--marginal` reads every JSON beside it as a diffusion cell and would crash on this
shape rather than ignore it. What it shares is the door.
`bench.results.require_recordable` guards every write here too, so a primitive
number cannot reach disk without a machine and a clock regime attached either.

One record holds *both* primitives on one case. That is the unit the comparison is
made in: A and B are timed on the same frames, interleaved frame by frame, so
whatever the laptop's clock does under them lands on both arms. Splitting them into
two records would invite two runs minutes apart, which is exactly the confounding
issue #13 exists about.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

from bench import RESULT_SCHEMA_VERSION
from bench.clocks import LOCKED, ClockNormalization, regime_of
from bench.cooldown import CooldownRecord
from bench.detector_results import LatencySummary
from bench.fingerprint import Fingerprint
from bench.flicker import FlickerScore
from bench.primitives import (
    PRIMITIVES,
    CaseConfig,
    DenoiseRequirement,
    Decision,
    Measurement,
    SmallObjectSummary,
    decide,
)
from bench.results import (
    GpuColumn,
    append_row,
    distinct_gpus,
    format_number,
    gpu_of,
    latest_per,
    load_records,
    require_recordable,
    table_row,
    table_separator,
    timestamp_from,
    write_record,
)

RECORD_KIND = "primitive"


@dataclass(frozen=True)
class ClipRecord:
    """The committed clip a comparison ran on, identified well enough to re-run it.

    By SHA-256 as well as by name: the clip is committed, so unlike the detector
    evidence photographs the bytes are in the repo - but a record that named only
    `people.mp4` could not tell a re-encode from the original, and the flicker
    figures are sensitive to compression noise.
    """

    name: str
    sha256: str
    width: int
    height: int
    fps: float
    total_frames: int
    start_frame: int
    frames_used: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrackRecord:
    """The fixed box track the comparison rendered, and how small its regions got."""

    detector: str
    target: str
    conf: float
    region: str
    max_objects: int
    objects_per_frame: float
    regions_rendered: int
    small_objects: SmallObjectSummary

    def to_dict(self) -> dict:
        data = asdict(self)
        data["small_objects"] = self.small_objects.to_dict()
        return data


@dataclass(frozen=True)
class IdentityCheck:
    """Whether the render actually changed what the subject *is*.

    Measured with the same open-vocabulary detector the orchestrator uses (issue
    #4), asked for the old identity and the new one on the rendered frames. A change
    figure cannot answer this - a frame can move 90/255 and still be a dog - and a
    human watching the clip is the confirmation, not the metric.
    """

    detector: str
    asked_for: List[str]
    frames_probed: int
    became: int
    remained: int
    conf: float
    achieved: bool
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PrimitiveArm:
    """One primitive, on one case: what it cost, what it did, what it could express.

    `expresses` is the Gate's expressiveness half, and it is deliberately not a
    latency: it is true when the case's own criterion was met at some rung of the
    denoise ladder - a visible confined change for a restyle, the detector reading
    the new identity for an identity change.
    """

    primitive: str
    spec_option: str
    denoise: DenoiseRequirement
    latency: LatencySummary
    ms_per_frame: float
    calls_per_frame: float
    calls_total: int
    frames: int
    flicker: FlickerScore
    region_change: float
    outside_change: float
    expresses: bool
    cannot_express: str
    identity: Optional[IdentityCheck] = None
    # The rendered frames, written beside the record so a human can watch them.
    clip_file: str = ""
    # Per *arm*, not per result: the two arms are two different millisecond figures
    # measured under the same trace, and one normalisation block on the record could
    # only describe one of them.
    clock_normalization: Optional[ClockNormalization] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["denoise"] = self.denoise.to_dict()
        data["latency"] = self.latency.to_dict()
        data["flicker"] = self.flicker.to_dict()
        data["identity"] = None if self.identity is None else self.identity.to_dict()
        data["clock_normalization"] = (None if self.clock_normalization is None
                                       else self.clock_normalization.to_dict())
        return data


@dataclass(frozen=True)
class PrimitiveRunMetrics:
    """When the comparison ran and what the machine was doing underneath it."""

    started_utc: str
    finished_utc: str
    warmup_reps: int
    engine_scenario: str
    # The open-vocabulary detector is kept resident throughout, whether or not the
    # case uses it, because the orchestrator will have it resident too - and because
    # a restyle arm measured without it would not be comparable with an identity arm
    # measured with it.
    detector_resident: Optional[str]
    mean_sm_clock_mhz: Optional[float]
    max_temperature_c: Optional[float]
    peak_vram_bytes: int
    gpu_samples: List[List[Optional[float]]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PrimitiveResult:
    """One case, both primitives. The unit `bench/results/primitives/` stores."""

    case: CaseConfig
    clip: ClipRecord
    track: TrackRecord
    arms: List[PrimitiveArm]
    run: PrimitiveRunMetrics
    cooldown: CooldownRecord
    hardware: Fingerprint
    # The source-A-B triptych a human watches. The Gate's manual verification step
    # has to be pointed at a file, not at a number.
    comparison_clip: str = ""
    # A full-resolution still of the triptych. The small-crop quality floor is
    # only visible at native resolution, and the comparison clip is downscaled.
    comparison_still: str = ""
    # Set only when one-step SD-Turbo could not do the case at any strength.
    one_step_finding: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "kind": RECORD_KIND,
            "case": self.case.to_dict(),
            "clip": self.clip.to_dict(),
            "track": self.track.to_dict(),
            "arms": [arm.to_dict() for arm in self.arms],
            "run": self.run.to_dict(),
            "cooldown": self.cooldown.to_dict(),
            "hardware": self.hardware.to_dict(),
            "comparison_clip": self.comparison_clip,
            "comparison_still": self.comparison_still,
            "one_step_finding": self.one_step_finding,
        }


ResultLike = Union[PrimitiveResult, dict]


def _as_dict(result: ResultLike) -> dict:
    return result.to_dict() if isinstance(result, PrimitiveResult) else result


def primitive_result_filename(case_name: str, timestamp: str) -> str:
    return f"{case_name}-{timestamp}.json"


def write_primitive_result(result: ResultLike, results_dir: Path,
                           timestamp: Optional[str] = None) -> Path:
    """Write `<case>-<timestamp>.json` under `results_dir`; return its path."""
    data = _as_dict(result)
    if timestamp is None:
        timestamp = timestamp_from(data)
    return write_record(data, results_dir,
                        primitive_result_filename(data["case"]["name"], timestamp))


PRIMITIVE_README_TITLE = "# Rendering-primitive comparison results"
PRIMITIVE_README_INTRO = (
    "Written by `uv run python -m bench <case>`, never by hand - issue #5,\n"
    "spec 8.2. One run measures *both* primitives on one case, interleaved frame by\n"
    "frame, and writes one row per primitive plus the side-by-side clip a human\n"
    "watches.\n\n"
    "`flicker` is the mean absolute difference between consecutive outputs over the\n"
    "pixels that were static in the source *and* painted by the primitive, in 0-255\n"
    "units - lower is steadier. `t_index` is the denoise setting the case turned out\n"
    "to need; higher is *less* denoise. `expresses` is whether the case's own\n"
    "criterion was met at any rung of the ladder, and it is the half of the decision\n"
    "no millisecond figure can answer.\n\n"
    "Absolute ms/frame belongs to the GPU in the row (spec 7.4). Compare rows for the\n"
    "A-against-B ratio, not for whether 30 FPS is met.\n"
)
PRIMITIVE_README_HEADER = (
    "| finished (UTC) | case | primitive | option | GPU | clip | objects/frame |"
    " calls/frame | t_index | strength | ms/frame | flicker | expresses |"
    " cooldown | clock regime | ms/frame at basis clock | clip file | file |"
)
PRIMITIVE_README_SEPARATOR = table_separator(PRIMITIVE_README_HEADER)
PRIMITIVE_README_NAME = "README.md"

PRIMITIVE_README_PREAMBLE = (
    f"{PRIMITIVE_README_TITLE}\n\n{PRIMITIVE_README_INTRO}\n"
    f"{PRIMITIVE_README_HEADER}\n{PRIMITIVE_README_SEPARATOR}\n"
)


def arm_normalised_cell(result: dict, arm: dict) -> str:
    """The clock-normalised ms/frame for one arm, or why there is none.

    `bench.results.normalised_cell` answers the same question for a record with one
    timing in it. A comparison has two, measured under one clock trace, so the
    normalisation lives on the arm and the regime - a fact about the machine - stays
    on the record.
    """
    if regime_of(result) == LOCKED:
        return "raw (clocks locked)"
    normalisation = arm.get("clock_normalization") or {}
    ms, basis = normalisation.get("ms_per_frame"), normalisation.get("basis_mhz")
    if ms is None or basis is None:
        return "-"
    return f"{ms:.2f} @ {basis:.0f} MHz"


def primitive_readme_rows(result: dict, filename: str) -> List[str]:
    """One row per primitive: a comparison read as a table is two rows, not one."""
    case, clip, track, run = (result["case"], result["clip"], result["track"],
                              result["run"])
    rows = []
    for arm in result["arms"]:
        rows.append(table_row([
            run["finished_utc"],
            case["name"],
            arm["primitive"],
            arm["spec_option"],
            result["hardware"]["gpu_name"],
            clip["name"],
            format_number(track["objects_per_frame"], 2),
            format_number(arm["calls_per_frame"], 2),
            str(arm["denoise"]["t_index"]),
            format_number(arm["denoise"]["strength"], 2),
            format_number(arm["ms_per_frame"], 2),
            format_number(arm["flicker"]["mean_abs_diff"], 2),
            "yes" if arm["expresses"] else "no",
            result["cooldown"]["outcome"],
            regime_of(result),
            arm_normalised_cell(result, arm),
            arm["clip_file"] or "-",
            f"[{filename}]({filename})",
        ]))
    return rows


def append_primitive_readme_rows(result: ResultLike, readme_path: Path,
                                 filename: str) -> None:
    """Append this run's rows, creating the table if this is the first comparison."""
    data = _as_dict(result)
    require_recordable(data)
    for row in primitive_readme_rows(data, filename):
        append_row(row, readme_path, PRIMITIVE_README_PREAMBLE)


# --- the report spec 8.2 carries ---------------------------------------------------

def load_primitive_results(results_dir: Path) -> Dict[str, dict]:
    """Every primitive result under `results_dir`, keyed by filename."""
    return load_records(results_dir)


def latest_per_case(results: Mapping[str, dict]) -> Dict[str, dict]:
    """One result per case *per GPU*: the most recently finished run of each.

    Per GPU because a second machine's comparison does not supersede the first
    machine's - it is what makes spec 7.4's ranking claim checkable (issue #25).
    """
    return latest_per(results, lambda result: result["case"]["name"])


REPORT_HEADER = ("| case | primitive | option | denoise t_index | strength |"
                 " objects/frame | calls/frame | ms/frame |"
                 " ms/frame at basis clock | flicker (static px) | expresses |")


def _report_preamble(results: Sequence[dict], gpus: Sequence[str]) -> str:
    """One line saying which machines, which engine and which clips these are for."""
    engines = sorted({result["run"]["engine_scenario"] for result in results})
    clips = sorted({f"{result['clip']['name']} "
                    f"({result['clip']['frames_used']} consecutive frames)"
                    for result in results})
    spanning = ("" if len(gpus) == 1 else
                " Rows are one per case per GPU, and absolute figures belong to the "
                "GPU in the row (spec 7.4).")
    return (f"Measured: {', '.join(gpus)}, engine {', '.join(engines)}. "
            f"Committed clips: {', '.join(clips)}. Both primitives are timed on the "
            f"same frames, interleaved frame by frame, so the laptop's clock drift "
            f"lands on both arms.{spanning}")


def _report_rows(results: Sequence[dict], column: GpuColumn) -> List[str]:
    rows = []
    for result in results:
        for arm in result["arms"]:
            rows.append(column.row([
                result["case"]["name"],
                arm["primitive"],
                arm["spec_option"],
                str(arm["denoise"]["t_index"]),
                format_number(arm["denoise"]["strength"], 2),
                format_number(result["track"]["objects_per_frame"], 2),
                format_number(arm["calls_per_frame"], 2),
                format_number(arm["ms_per_frame"], 2),
                arm_normalised_cell(result, arm),
                f"{format_number(arm['flicker']['mean_abs_diff'], 2)} "
                f"({arm['flicker']['static_pixels']})",
                "yes" if arm["expresses"] else "no",
            ], result))
    return rows


def measurements_from(results: Sequence[dict]) -> List[Measurement]:
    """The decision's inputs, read back out of the committed records."""
    return [
        Measurement(
            case=result["case"]["name"],
            kind=result["case"]["kind"],
            priority=bool(result["case"]["priority"]),
            primitive=arm["primitive"],
            ms_per_frame=float(arm["ms_per_frame"]),
            flicker=arm["flicker"]["mean_abs_diff"],
            objects_per_frame=float(result["track"]["objects_per_frame"]),
            expresses=bool(arm["expresses"]),
            t_index=int(arm["denoise"]["t_index"]),
        )
        for result in results for arm in result["arms"]
    ]


def decision_from(results: Sequence[dict]) -> Decision:
    """The decision the committed records support, recomputed rather than quoted."""
    return decide(measurements_from(results))


def _on(result: dict, gpus: Sequence[str]) -> str:
    """` (<GPU>)`, or nothing when every line in the block is from one machine.

    A bullet that names only its case would be two identical-looking bullets once a
    second machine measured the same case (issue #25).
    """
    return "" if len(gpus) == 1 else f" ({gpu_of(result)})"


def _denoise_lines(results: Sequence[dict], gpus: Sequence[str]) -> List[str]:
    """What each case needed, per primitive. A Gate item in its own right."""
    lines = []
    for result in results:
        for arm in result["arms"]:
            lines.append(f"- **{result['case']['name']} / {arm['primitive']}"
                         f"{_on(result, gpus)}:** {arm['denoise']['statement']}")
    return lines


def _cannot_express_lines(results: Sequence[dict]) -> List[str]:
    """One bullet per primitive that was measured, in spec-option order."""
    seen = {arm["primitive"] for result in results for arm in result["arms"]}
    return [f"- **{PRIMITIVES[key].spec_option}, {PRIMITIVES[key].name} "
            f"(`{key}`) cannot express:** {PRIMITIVES[key].cannot_express}"
            for key in sorted(seen, key=lambda k: PRIMITIVES[k].spec_option)]


def _findings(results: Sequence[dict], gpus: Sequence[str]) -> List[str]:
    """Findings the Gate asks to be recorded even though they decide nothing."""
    lines = []
    for result in results:
        if result.get("one_step_finding"):
            lines.append(f"- {result['one_step_finding']}")
        able = [arm["primitive"] for arm in result["arms"] if arm["expresses"]]
        unable = [arm["primitive"] for arm in result["arms"] if not arm["expresses"]]
        if able and unable:
            lines.append(
                f"- **{result['case']['name']} separates the two primitives"
                f"{_on(result, gpus)}:** "
                f"{', '.join(unable)} did not express it at any rung of the ladder "
                f"and {', '.join(able)} did. That is the half of the comparison no "
                f"millisecond figure carries.")
        small = result["track"]["small_objects"]
        lines.append(f"- **Small objects, {result['clip']['name']}"
                     f"{_on(result, gpus)}:** {small['statement']}")
    return lines


def _decision_lines(results: Sequence[dict], gpus: Sequence[str]) -> List[str]:
    """The choice, taken once per machine.

    A cost comparison between two primitives measured on two different GPUs is not
    a comparison (spec 7.4), so each machine decides from its own arms and says
    which machine it is. With one machine the sentence is the one it always was.
    """
    lines = []
    for gpu in gpus:
        decision = decision_from([result for result in results
                                  if gpu_of(result) == gpu])
        label = "Decision" if len(gpus) == 1 else f"Decision ({gpu})"
        lines.append(
            f"**{label}: "
            f"{decision.primitive or 'none of the implemented primitives'}.** "
            f"{decision.statement}")
    return lines


def format_primitive_report(results: Mapping[str, dict]) -> str:
    """The spec 8.2 decision block: the table, what each case needed, and the choice.

    Generated from the committed JSON rather than transcribed, for the reason specs
    7.2 and 8.1 are: a table pasted into Markdown drifts the moment a case is
    re-measured, and nothing notices.

    Rows are one per (case, GPU) and a case's machines sit adjacently, so a second
    machine adds rows rather than replacing them (issue #25).
    """
    ordered = sorted(latest_per_case(results).values(),
                     key=lambda result: (not result["case"]["priority"],
                                         result["case"]["name"], gpu_of(result),
                                         result["run"]["finished_utc"]))
    if not ordered:
        return "no primitive comparison committed yet"

    gpus = distinct_gpus(ordered)
    column = GpuColumn(shown=len(gpus) > 1)
    header = column.header(REPORT_HEADER)
    clips = [f"`{result['comparison_clip']}`" for result in ordered
             if result.get("comparison_clip")]
    sections = [
        _report_preamble(ordered, gpus),
        "\n".join([header, table_separator(header)]
                  + _report_rows(ordered, column)),
        "Denoise strength each case turned out to need:",
        "\n".join(_denoise_lines(ordered, gpus)),
        "\n".join(_cannot_express_lines(ordered)),
        "Findings:",
        "\n".join(_findings(ordered, gpus)),
    ] + _decision_lines(ordered, gpus)
    if clips:
        sections.append(
            f"Side-by-side clips for human judgement (source | A | B), under "
            f"`bench/results/primitives/`: {', '.join(clips)}. **The metric ranks "
            f"cost and temporal stability, not beauty** - a human still has to watch "
            f"these and confirm the priority case is acceptable."
        )
    return "\n\n".join(sections)
