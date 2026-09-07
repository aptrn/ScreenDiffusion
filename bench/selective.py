"""What an end-to-end selective render records, and the block spec 8.8 carries.

Issue #8. The rendering-primitive comparison (issue #5) asked *which* primitive; this
asks whether the shipped path - detector, tracker, region scheduler, engine,
compositor - actually produces the priority case, and what it costs when it does.

So the case is not a configuration written out here: it is
`render_plan.priority_case_plan()`, the hardcoded plan the worker itself starts on
behind `SD_DEMO_PLAN`. The harness measures the shipped producer rather than a copy
of it, which is the only way the number means anything about the app.

Four checks make up the Gate, and each is a number with a threshold beside it rather
than a verdict on its own:

- the background is **bit-identical** to the capture, on every frame, or it is not;
- the region visibly changed, net of what the capture round trip costs;
- with more tracks than slots, every track was rendered inside `ceil(N/K)` frames;
- the loop produced a frame for every frame it took, and never waited on a detect.

Its own directory under `bench/results/selective/`, for the reason the detector and
primitive records have one - `bench --marginal` reads every JSON beside it as a
diffusion cell. What it shares is the door: `require_recordable` guards these writes
too, so a selective number cannot reach disk without a machine and a clock regime.

GPU-free: like `bench.primitives`, this module is the arithmetic and the record
shape, and `bench.selective_runner` is the half that touches a GPU.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from bench import RESULT_SCHEMA_VERSION
from bench.clocks import ClockNormalization
from bench.cooldown import CooldownRecord
from bench.detector_results import LatencySummary
from bench.fingerprint import Fingerprint
from bench.flicker import FlickerScore
from bench.primitive_results import ClipRecord
from bench.results import (
    GpuColumn,
    append_row,
    distinct_gpus,
    format_number,
    gpu_of,
    latest_per,
    load_records,
    normalised_cell,
    require_recordable,
    table_row,
    table_separator,
    timestamp_from,
    write_record,
)

RECORD_KIND = "selective"

# The one cached engine the path is rendered through - the same 512x512 batch-1
# engine issue #5 compared both primitives on, so a millisecond here is comparable
# with a millisecond there.
ENGINE_SCENARIO = "img2img-tensorrt-512x512-b1"

# Mean absolute difference inside the rendered region, in 0-255 units, below which
# the restyle is not visible. The threshold issue #5 selected its denoise by, used
# again so "visibly restyled" means the same thing in both records.
VISIBLE_CHANGE = 8.0

# What handing the newest frame to the detector may cost the frame path before it
# counts as waiting on detection. The same figure issue #7's GPU test used for the
# same call: an offer takes a lock, assigns a tuple and sets an event.
MAX_OFFER_MS = 5.0

# Slots the coverage probe forces, so the round-robin is exercised on the real
# track sequence rather than on a clip that happens to hold fewer objects than K.
COVERAGE_PROBE_SLOTS = 2

PRIORITY_CASE = "selective-people"


@dataclass(frozen=True)
class SelectiveCase:
    """One end-to-end run: a committed clip, and the shipped plan to render it under.

    `plan()` is `render_plan.priority_case_plan` - the concept, the region, the
    prompt and the denoise all come from the producer the worker uses, so this
    record cannot describe a case the app cannot be put into.
    """

    name: str
    clip: str
    note: str
    # The capture geometry the app runs at: the worker's capture thread resizes the
    # screen region to the engine's canvas *before* the frame loop sees it, so the
    # clip is resized the same way and the whole path works at one size.
    canvas: int = 512
    frames: int = 48
    start_frame: int = 0
    warmup_frames: int = 3
    # K forced down for the round-robin probe, which replays every frame's tracks
    # through the shipped scheduler; the clip holds fewer people than the plan's K.
    coverage_slots: int = COVERAGE_PROBE_SLOTS

    def plan(self):
        """The hardcoded priority-case plan, from the shipped producer."""
        from render_plan import priority_case_plan

        return priority_case_plan()

    def replace(self, **changes) -> "SelectiveCase":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)


CASES: Dict[str, SelectiveCase] = {
    PRIORITY_CASE: SelectiveCase(
        name=PRIORITY_CASE,
        clip="people.mp4",
        note="The M1 finish line: the shipped selective path end to end on the "
             "priority case - restyle the lower half of every person, gently, and "
             "leave every other pixel exactly as captured.",
    ),
}


def plan_record(plan) -> dict:
    """The plan a run rendered, as it will be read months later."""
    from render_plan import t_index_for_denoise

    target = plan.honoured_target
    return {
        "plan_version": plan.plan_version,
        "mode": plan.mode,
        "concept": None if target is None else target.concept,
        "region": None if target is None else target.region,
        "prompt": plan.effective_prompt,
        "denoise": plan.effective_denoise,
        "t_index": t_index_for_denoise(plan.effective_denoise),
        "detect_every_n": plan.settings.detect_every_n,
        "max_instances": None if target is None else target.max_instances,
    }


# --- the four checks ---------------------------------------------------------


@dataclass(frozen=True)
class BackgroundCheck:
    """The issue's sharp criterion: non-target pixels stay bit-identical.

    `worst_pixels_changed` is the count on the worst frame, not a mean: one changed
    pixel on one frame is a failure, and an average would bury it.
    """

    frames: int
    identical_frames: int
    worst_pixels_changed: int
    background_pixels: int
    passed: bool
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def background_check(pixels_changed: Sequence[int],
                     background_pixels: int) -> BackgroundCheck:
    """Read the per-frame count of background pixels that moved."""
    frames = len(pixels_changed)
    identical = sum(1 for count in pixels_changed if count == 0)
    worst = max(pixels_changed) if pixels_changed else 0
    passed = frames > 0 and identical == frames
    detail = (f" ({background_pixels} background pixels on the frame with the "
              f"most painted)" if passed else
              f"; the worst frame changed {worst} of {background_pixels} of them")
    statement = (
        f"{identical}/{frames} frames left every pixel outside the rendered regions "
        f"exactly as captured{detail}"
    )
    return BackgroundCheck(frames=frames, identical_frames=identical,
                           worst_pixels_changed=worst,
                           background_pixels=background_pixels,
                           passed=passed, statement=statement)


@dataclass(frozen=True)
class ChangeCheck:
    """Did the render actually do something where it was asked to?

    `capture_change` is the control. Issue #5 had to subtract a resize round trip
    from every change figure, because its primitives squeezed a 1280x720 frame onto
    a 512x512 canvas. The shipped path does not: the capture thread hands the frame
    loop a canvas-sized frame already, so the control here is the capture's own
    uint8 -> float -> uint8 conversion, and it is measured rather than assumed.
    """

    region_change: float
    capture_change: float
    net_change: float
    threshold: float
    passed: bool
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def change_check(region_change: float, capture_change: float,
                 threshold: float = VISIBLE_CHANGE) -> ChangeCheck:
    net = round(max(0.0, region_change - capture_change), 4)
    passed = net >= threshold
    return ChangeCheck(
        region_change=round(region_change, 4), capture_change=round(capture_change, 4),
        net_change=net, threshold=threshold, passed=passed,
        statement=(
            f"the rendered regions changed by {region_change:.1f}/255 against the "
            f"capture - {net:.1f} net of the {capture_change:.2f}/255 the capture's "
            f"own round trip costs - against a {threshold:.0f}/255 threshold"),
    )


@dataclass(frozen=True)
class CoverageCheck:
    """The round-robin bound: every track rendered inside `ceil(N/K)` frames.

    Measured by replaying the run's own per-frame track ids through a scheduler
    with `slots` forced down, because the clip holds fewer people than K and the
    bound is otherwise trivially one frame. The policy is the shipped one; only the
    slot count is the probe's.
    """

    slots: int
    max_tracks: int
    bound_frames: int
    worst_gap_frames: int
    frames: int
    passed: bool
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def with_slots(plan, slots: int):
    """`plan` with every target's instance cap forced to `slots`.

    The probe changes K and nothing else, so what it exercises is the shipped
    rotation policy rather than an imitation of it.
    """
    return dataclasses.replace(
        plan, targets=tuple(dataclasses.replace(target, max_instances=slots)
                            for target in plan.targets))


def coverage_probe(snapshots: Sequence, plan, width: int, height: int,
                   slots: int = COVERAGE_PROBE_SLOTS) -> Tuple[int, int]:
    """Worst gap, in frames, between two renders of one track, and the busiest frame.

    The run's own per-frame `Tracks` are replayed through a real `RegionScheduler`
    with K forced down - the clip holds fewer people than the plan's K, and a bound
    of one frame proves nothing about a round robin.

    A gap is counted from the frame a track first became eligible: a track that was
    not there yet was not starved. A track the rotation stops serving cannot hide at
    the end of the run either, because the gap is measured on every frame it is
    eligible for and not only when it is rendered.
    """
    from region_scheduler import RegionScheduler

    scheduler = RegionScheduler()
    probe_plan = with_slots(plan, slots)
    served: Dict[int, int] = {}
    first_seen: Dict[int, int] = {}
    worst = 0
    max_tracks = 0
    for index, tracks in enumerate(snapshots):
        selection = scheduler.select(tracks, probe_plan, width, height)
        max_tracks = max(max_tracks, selection.candidates)
        for track_id in selection.candidate_ids:
            first_seen.setdefault(track_id, index)
        for track_id in selection.track_ids:
            served[track_id] = index
        for track_id in selection.candidate_ids:
            last = served.get(track_id, first_seen[track_id] - 1)
            worst = max(worst, index - last)
    return worst, max_tracks


def coverage_check(snapshots: Sequence, plan, width: int, height: int,
                   slots: int = COVERAGE_PROBE_SLOTS) -> CoverageCheck:
    worst, max_tracks = coverage_probe(snapshots, plan, width, height, slots)
    bound = -(-max_tracks // slots) if max_tracks else 0
    passed = worst <= bound
    return CoverageCheck(
        slots=slots, max_tracks=max_tracks, bound_frames=bound,
        worst_gap_frames=worst, frames=len(snapshots), passed=passed,
        statement=(
            f"with {max_tracks} tracks over {slots} slots no track waited more than "
            f"{worst} frames to be rendered, against a ceil(N/K) bound of {bound}"),
    )


@dataclass(frozen=True)
class StallCheck:
    """The loop keeps producing frames, and never waits on the detector.

    `worst_offer_ms` is what handing the newest frame to the detector cost the
    frame path at its worst. The detector runs on its own thread in this run, as it
    does in the worker, so this is the whole of what detection costs the loop.
    """

    frames_in: int
    frames_out: int
    worst_offer_ms: float
    max_offer_ms: float
    passthrough_frames: int
    passed: bool
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def stall_check(frames_in: int, frames_out: int, worst_offer_ms: float,
                passthrough_frames: int,
                max_offer_ms: float = MAX_OFFER_MS) -> StallCheck:
    """Both halves: every frame came out, and no frame waited on a detect."""
    passed = frames_out == frames_in and worst_offer_ms <= max_offer_ms
    return StallCheck(
        frames_in=frames_in, frames_out=frames_out,
        worst_offer_ms=round(worst_offer_ms, 4), max_offer_ms=max_offer_ms,
        passthrough_frames=passthrough_frames, passed=passed,
        statement=(
            f"{frames_out}/{frames_in} frames produced an output; the worst frame "
            f"spent {worst_offer_ms:.3f} ms offering the capture to the detector "
            f"(budget {max_offer_ms:.0f} ms), and {passthrough_frames} frames had "
            f"nothing to restyle and passed the capture through"),
    )


# --- the record --------------------------------------------------------------


@dataclass(frozen=True)
class RegionSummary:
    """What the scheduler did over the run: how much was rendered, and what waited."""

    slots: int
    regions_rendered: int
    regions_per_frame: float
    tracks_per_frame: float
    deferred_total: int
    skipped_small_total: int
    min_region_px: int
    feather_px: int
    min_side_px: Optional[int]
    max_side_px: Optional[int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SelectiveRunMetrics:
    """The timings, and what produced them.

    `ms_per_frame` is the frame path alone - diffuse, composite, offer - and
    `ms_per_frame_with_detection` adds what the detector amortises to at the plan's
    cadence. Both, because the first is what the loop pays per frame and the second
    is what a frame *costs* when detection is running beside it, and quoting one as
    the other is how a budget goes missing.
    """

    started_utc: str
    finished_utc: str
    warmup_frames: int
    engine_scenario: str
    detector: Optional[str]
    frames: int
    diffusion_calls: int
    render: LatencySummary
    composite: LatencySummary
    detect: Optional[LatencySummary]
    detect_every_n: int
    detector_ticks: int
    amortised_detect_ms: float
    ms_per_frame: float
    ms_per_frame_with_detection: float
    fps: float
    mean_sm_clock_mhz: Optional[float]
    max_temperature_c: Optional[float]
    peak_vram_bytes: int
    gpu_samples: List[List[Optional[float]]] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["render"] = self.render.to_dict()
        data["composite"] = self.composite.to_dict()
        data["detect"] = None if self.detect is None else self.detect.to_dict()
        return data


@dataclass(frozen=True)
class SelectiveResult:
    """One end-to-end run of the shipped selective path."""

    case: SelectiveCase
    plan: dict
    clip: ClipRecord
    run: SelectiveRunMetrics
    regions: RegionSummary
    flicker: FlickerScore
    background: BackgroundCheck
    change: ChangeCheck
    coverage: CoverageCheck
    stall: StallCheck
    cooldown: CooldownRecord
    hardware: Fingerprint
    clock_normalization: Optional[ClockNormalization] = None
    comparison_clip: str = ""
    comparison_still: str = ""

    @property
    def gate_passed(self) -> bool:
        """Every Gate item a machine can answer. The manual one is not here."""
        return all([self.background.passed, self.change.passed,
                    self.coverage.passed, self.stall.passed])

    def to_dict(self) -> dict:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "kind": RECORD_KIND,
            "case": self.case.to_dict(),
            "plan": self.plan,
            "clip": self.clip.to_dict(),
            "run": self.run.to_dict(),
            "regions": self.regions.to_dict(),
            "flicker": self.flicker.to_dict(),
            "gate": {
                "passed": self.gate_passed,
                "background": self.background.to_dict(),
                "change": self.change.to_dict(),
                "coverage": self.coverage.to_dict(),
                "stall": self.stall.to_dict(),
            },
            "cooldown": self.cooldown.to_dict(),
            "hardware": self.hardware.to_dict(),
            "clock_normalization": (None if self.clock_normalization is None
                                    else self.clock_normalization.to_dict()),
            "comparison_clip": self.comparison_clip,
            "comparison_still": self.comparison_still,
        }


ResultLike = Union[SelectiveResult, dict]


def _as_dict(result: ResultLike) -> dict:
    return result.to_dict() if isinstance(result, SelectiveResult) else result


def selective_result_filename(case: str, timestamp: str) -> str:
    return f"{case}-{timestamp}.json"


def write_selective_result(result: ResultLike, results_dir: Path,
                           timestamp: Optional[str] = None) -> Path:
    """Write `<case>-<timestamp>.json` under `results_dir`; return its path."""
    data = _as_dict(result)
    if timestamp is None:
        timestamp = timestamp_from(data)
    return write_record(data, results_dir,
                        selective_result_filename(data["case"]["name"], timestamp))


SELECTIVE_README_TITLE = "# Selective render path results"
SELECTIVE_README_INTRO = (
    "Written by `uv run python -m bench <case>`, never by hand - issue #8, spec 8.8.\n"
    "One run drives the *shipped* path over a committed clip: the detector on its own\n"
    "thread, the tracker, the region scheduler, the 512x512 TensorRT engine and the\n"
    "compositor, under the hardcoded priority-case plan the worker itself starts on.\n\n"
    "`ms/frame` is the frame path alone; `+detect` adds what the detector amortises to\n"
    "at the plan's cadence. `background` is the sharp criterion - every pixel outside\n"
    "the rendered regions identical to the capture, on every frame. `flicker` is the\n"
    "mean absolute difference between consecutive outputs over pixels static in the\n"
    "source and painted in both, lower is steadier.\n\n"
    "Absolute figures belong to the GPU in the row (spec 7.4). 30 FPS is not a gate\n"
    "here and cannot be judged on this laptop at all.\n"
)
SELECTIVE_README_HEADER = (
    "| finished (UTC) | case | GPU | clip | plan | regions/frame | ms/frame |"
    " +detect | FPS | flicker | background | gate | cooldown | clock regime |"
    " ms/frame at basis clock | clip file | file |"
)
SELECTIVE_README_SEPARATOR = table_separator(SELECTIVE_README_HEADER)
SELECTIVE_README_NAME = "README.md"
SELECTIVE_README_PREAMBLE = (
    f"{SELECTIVE_README_TITLE}\n\n{SELECTIVE_README_INTRO}\n"
    f"{SELECTIVE_README_HEADER}\n{SELECTIVE_README_SEPARATOR}\n"
)


def selective_readme_row(result: dict, filename: str) -> str:
    run, gate = result["run"], result["gate"]
    plan, clip = result["plan"], result["clip"]
    return table_row([
        run["finished_utc"],
        result["case"]["name"],
        result["hardware"]["gpu_name"],
        f"{clip['name']} {clip['width']}x{clip['height']}",
        f"{plan['concept']} / {plan['region']} / t{plan['t_index']}",
        format_number(result["regions"]["regions_per_frame"], 2),
        format_number(run["ms_per_frame"], 2),
        format_number(run["ms_per_frame_with_detection"], 2),
        format_number(run["fps"], 1),
        format_number(result["flicker"]["mean_abs_diff"], 2),
        "identical" if gate["background"]["passed"] else "CHANGED",
        "pass" if gate["passed"] else "FAIL",
        result["cooldown"]["outcome"],
        result["hardware"]["clock_lock"]["state"],
        normalised_cell(result),
        f"[{result['comparison_clip']}]({result['comparison_clip']})"
        if result.get("comparison_clip") else "-",
        f"[{filename}]({filename})",
    ])


def append_selective_readme_row(result: ResultLike, readme_path: Path,
                                filename: str) -> None:
    """Append this run's row, creating the table if this is the first run."""
    data = _as_dict(result)
    require_recordable(data)
    append_row(selective_readme_row(data, filename), readme_path,
               SELECTIVE_README_PREAMBLE)


# --- the report the spec carries ---------------------------------------------


def load_selective_results(results_dir: Path) -> Dict[str, dict]:
    """Every selective result under `results_dir`, keyed by filename."""
    return load_records(results_dir)


def latest_per_case(results: Mapping[str, dict]) -> Dict[str, dict]:
    """One result per case *per GPU*: the most recently finished run of each.

    Per GPU because a 4090 run of `selective-people` does not supersede the 3080
    one - it is the other half of spec 7.4's portability claim (issue #25).
    """
    return latest_per(results, lambda result: result["case"]["name"])


REPORT_HEADER = ("| case | plan | regions/frame | diffusion calls/frame |"
                 " ms/frame | +detect | FPS | flicker (static px) | gate |")

GATE_ORDER = ("background", "change", "coverage", "stall")
GATE_TITLES = {
    "background": "Non-target pixels bit-identical",
    "change": "The region is visibly restyled",
    "coverage": "No track starved by the round robin",
    "stall": "The loop never stalls",
}


def _report_rows(results: Sequence[dict], column: GpuColumn) -> List[str]:
    rows = []
    for result in results:
        run, plan = result["run"], result["plan"]
        rows.append(column.row([
            result["case"]["name"],
            f"{plan['concept']} / {plan['region']} / t_index {plan['t_index']}",
            format_number(result["regions"]["regions_per_frame"], 2),
            format_number(run["diffusion_calls"] / max(1, run["frames"]), 2),
            format_number(run["ms_per_frame"], 1),
            format_number(run["ms_per_frame_with_detection"], 1),
            format_number(run["fps"], 1),
            format_number(result["flicker"]["mean_abs_diff"], 2),
            "pass" if result["gate"]["passed"] else "FAIL",
        ], result))
    return rows


def _gate_lines(result: dict) -> List[str]:
    gate = result["gate"]
    lines = []
    for name in GATE_ORDER:
        statement = gate[name]["statement"]
        lines.append(f"- **{GATE_TITLES[name]}** - "
                     f"{'yes' if gate[name]['passed'] else 'NO'}. "
                     f"{statement[:1].upper()}{statement[1:]}.")
    return lines


def _measured_phrase(result: dict) -> str:
    """The engine, the clip and the canvas one run was measured through."""
    clip, case = result["clip"], result["case"]
    return (f"{result['run']['engine_scenario']}, {clip['frames_used']} consecutive "
            f"frames of `{clip['name']}` resized to the app's "
            f"{case['canvas']}x{case['canvas']} capture canvas")


def _clock_phrase(results: Sequence[dict], gpus: Sequence[str]) -> str:
    """Which regime produced these figures, and whose figures they are.

    One machine reads as it always did. Two, and naming a single GPU would imply
    the other one's rows were measured on it (issue #25 step 3), so every machine
    is named with its own regime and the reader is pointed at the row.
    """
    def state(result: dict) -> str:
        return result["hardware"]["clock_lock"]["state"]

    if len(gpus) == 1:
        return (f"Clocks {', '.join(sorted({state(r) for r in results}))}; "
                f"absolute figures belong to this GPU (spec 7.4)")
    per_gpu = ", ".join(
        f"{gpu} {', '.join(sorted({state(r) for r in results if gpu_of(r) == gpu}))}"
        for gpu in gpus)
    return (f"Clocks: {per_gpu}; absolute figures belong to the GPU in the row "
            f"(spec 7.4)")


def _report_preamble(results: Sequence[dict], gpus: Sequence[str]) -> str:
    measured = ", ".join(sorted({_measured_phrase(result) for result in results}))
    spanning = "" if len(gpus) == 1 else " Rows are one per case per GPU."
    return (f"Measured on {', '.join(gpus)}, {measured}. "
            f"{_clock_phrase(results, gpus)}, and 30 FPS is M2's gate, "
            f"not this one's.{spanning}")


def _gate_sections(results: Sequence[dict], gpus: Sequence[str]) -> List[str]:
    """The Gate lines and the manual artefact, once per machine.

    The table lists every case; these belong to the first case *on each machine*,
    because a Gate statement is a claim about one run and the run it came from is
    the thing a reader has to be able to name.
    """
    sections = []
    for gpu in gpus:
        primary = next(result for result in results if gpu_of(result) == gpu)
        sections.append("The Gate, measured:" if len(gpus) == 1
                        else f"The Gate, measured on {gpu}:")
        sections.append("\n".join(_gate_lines(primary)))
        if primary.get("comparison_clip"):
            sections.append(
                f"Manual verification artefact: `{primary['comparison_clip']}` "
                f"(source | selective render) and `{primary['comparison_still']}`.")
    return sections


def format_selective_report(results: Mapping[str, dict]) -> str:
    """The measured block the spec carries for the selective path.

    Generated from the committed JSON rather than transcribed, for the reason specs
    7.2, 8.1 and 8.2 are: a table pasted into Markdown drifts the moment the case is
    re-measured and nothing notices.

    Rows are one per (case, GPU) and a case's machines sit adjacently, so a second
    machine adds rows rather than replacing them (issue #25).
    """
    ordered = sorted(latest_per_case(results).values(),
                     key=lambda result: (result["case"]["name"], gpu_of(result),
                                         result["run"]["finished_utc"]))
    if not ordered:
        return "no selective render run committed yet"

    gpus = distinct_gpus(ordered)
    column = GpuColumn(shown=len(gpus) > 1)
    header = column.header(REPORT_HEADER)
    sections = [
        _report_preamble(ordered, gpus),
        "\n".join([header, table_separator(header)]
                  + _report_rows(ordered, column)),
    ]
    return "\n\n".join(sections + _gate_sections(ordered, gpus))
