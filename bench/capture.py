"""What a capture bigger than the canvas costs, and what crop buys at K=1. Issue #39.

Spec 8.2 chose the masked primitive and recorded what it could not do:

> "Nor can it spend detail where it matters: the whole frame is squeezed onto one
> 512x512 canvas, so a region occupying 45 px of a 1280 px-wide frame is diffused at
> ~18 px and comes back with roughly that much detail."

Live testing hit exactly that, and the product decision that followed - *one object
at high detail is worth more than all objects at low detail* - reverses the premise
issue #5 decided under (six objects per frame) without touching the measurement it
took. This module is the arithmetic for re-taking that decision in the new regime,
and `bench.capture_runner` is the half that touches a GPU.

Five things it is built to answer, one per Gate item.

- **The bit-identity criterion does not get easier because the frame got bigger.**
  Every arm records how many non-target pixels moved, at every capture geometry,
  and an arm that broke it is disqualified rather than explained.
- **Crop against masked at K=1, at the new geometry.** Both arms render the same
  region of the same frame through the same engine at the same strength, so the
  difference between them is the primitive.
- **The cost of the larger capture, broken out per stage.** The frame path's own
  resize, the diffusion call, the composite, the device-to-host copy, and the two
  the GUI process pays - the IPC and the preview. Folded into one number the
  answer would be "it got slower", which is not a finding.
- **A 30 FPS verdict per geometry**, judged on the frame path against the same
  33.33 ms every other section here is judged against, and carrying its region
  count and its machine.
- **The object's diffusion resolution, in pixels.** The whole claim of the change,
  as a number: how many canvas pixels across the rendered region actually got. It
  is arithmetic on the boxes rather than an opinion about the clip, and it is what
  makes "high detail" checkable.

GPU-free, like every other `bench.*` results module: it reads JSON and formats it.
"""

from __future__ import annotations

import dataclasses
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from bench import RESULT_SCHEMA_VERSION
from bench.clocks import ClockNormalization
from bench.contention import OccupancyRecord
from bench.cooldown import CooldownRecord
from bench.detector_results import LatencySummary
from bench.fingerprint import Fingerprint
from bench.flicker import FlickerScore
from bench.portability import FRAME_BUDGET_MS
from bench.primitive_results import ClipRecord, IdentityCheck
from bench.primitives import (
    DENOISE_LADDER,
    IDENTITY,
    RESTYLE,
    VISIBLE_CHANGE,
    DenoisePoint,
    DenoiseRequirement,
)
from bench.results import (
    GpuColumn,
    append_row,
    distinct_gpus,
    format_number,
    gpu_of,
    latest_per,
    load_records,
    measured_on,
    require_recordable,
    sentence_case,
    table_row,
    table_separator,
    timestamp_from,
    write_record,
)
from bench.selective import BackgroundCheck
from render_plan import CROP, MASKED

RECORD_KIND = "capture-geometry"

ENGINE_SCENARIO = "img2img-tensorrt-512x512-b1"

# The engine's canvas. Fixed at 512x512 whatever a directory name claims (spec
# 7.2), which is the whole reason this comparison is about the *capture* moving.
# Spelt here rather than imported from `main_gpu_addon`, which pulls in the Tk
# stack; `tests/test_bench_capture.py` holds the two to one value, the way
# `bench.selective` pins `GLOBAL_KEY`.
CANVAS = 512

# The geometries every case is swept over. 512x512 is the control - the app as it
# shipped, and the geometry every committed measurement in this repo was taken at.
GEOMETRIES: Tuple[Tuple[int, int], ...] = ((512, 512), (1280, 720), (1920, 1080))

PRIMITIVES: Tuple[str, ...] = (MASKED, CROP)

PEOPLE_CASE = "capture-people"
DOG_CASE = "capture-dog"


# --- the cases ---------------------------------------------------------------


@dataclass(frozen=True)
class CaptureCase:
    """One clip, rendered at every capture geometry under both primitives, at K=1.

    `plan()` goes through `render_plan.validate_plan`, so a case cannot describe a
    configuration the app could not be put into - and `max_instances` is 1 because
    that is the product decision this issue is built on, not because the scheduler
    cannot do more.
    """

    name: str
    kind: str
    clip: str
    concept: str
    becomes: Optional[str]
    region: str
    prompt: str
    denoise: float
    note: str
    geometries: Tuple[Tuple[int, int], ...] = GEOMETRIES
    primitives: Tuple[str, ...] = PRIMITIVES
    frames: int = 48
    start_frame: int = 0
    warmup_frames: int = 3
    # The ladder each arm is swept over before it is timed. Per arm, because spec
    # 8.2 measured that the two primitives need different strengths on the same
    # case - crop 0.64 against masked 0.49 - and timing both at one of them would
    # be measuring the primitive that the strength was chosen for.
    denoise_ladder: Tuple[int, ...] = DENOISE_LADDER
    # Frames per swept rung. Small: the sweep selects a strength, it does not
    # measure a latency.
    sweep_frames: int = 4
    # Finished frames the three off-path stages are probed on. They are all the
    # same size, so a sample is the whole distribution - and probing every frame
    # would put a megabyte of host allocation between two timed renders.
    probe_frames: int = 8
    # K. One, which is what makes `crop` affordable: one diffusion call a frame,
    # spent on one object instead of on the whole squeezed frame.
    max_instances: int = 1

    def plan(self, primitive: str):
        """The plan one arm renders under, through the shipped producer and door."""
        from render_plan import (
            GLOBAL_KEY,
            INITIAL_PLAN_VERSION,
            plan_from_fields,
            validate_plan,
        )

        built = plan_from_fields(target=self.concept, style=self.prompt,
                                 region=self.region, denoise=self.denoise)
        if built.plan is None:  # unreachable: an open-vocabulary concept and a region
            raise AssertionError(f"the case plan did not validate: {built.reason}")
        raw = built.plan.to_dict()
        raw[GLOBAL_KEY]["primitive"] = primitive
        for target in raw["targets"]:
            target["max_instances"] = self.max_instances
        result = validate_plan(raw, previous_version=INITIAL_PLAN_VERSION)
        if result.plan is None:  # unreachable: only a validated plan is edited here
            raise AssertionError(f"the arm plan did not validate: {result.reason}")
        return result.plan

    def replace(self, **changes) -> "CaptureCase":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["geometries"] = [list(pair) for pair in self.geometries]
        data["primitives"] = list(self.primitives)
        data["denoise_ladder"] = list(self.denoise_ladder)
        return data


CASES: Dict[str, CaptureCase] = {
    PEOPLE_CASE: CaptureCase(
        name=PEOPLE_CASE,
        kind=RESTYLE,
        clip="people.mp4",
        concept="person",
        becomes=None,
        region="lower_half",
        prompt="trousers soaked through with a dark wet stain, damp fabric, "
               "wet denim, photograph",
        denoise=0.49,
        note="The priority case at K=1: one person's lower half, at three capture "
             "geometries, under both primitives. The same clip, plan and engine "
             "spec 8.2 and 8.8 were measured on.",
    ),
    DOG_CASE: CaptureCase(
        name=DOG_CASE,
        kind=IDENTITY,
        clip="dog.mp4",
        concept="dog",
        becomes="cat",
        region="full_box",
        prompt="a cat, feline face, whiskers, pointed ears, photograph",
        denoise=0.92,
        note="The identity case re-run at the new geometry (the issue's step 5). "
             "Crop lost this case at every rung of spec 8.2's ladder on 45-293 px "
             "crops out of a 1280 px frame; a 1080p capture changes that input.",
    ),
}


def arm_name(primitive: str, width: int, height: int) -> str:
    """What one arm is called on disk and in the table: `crop-1920x1080`."""
    return f"{primitive}-{width}x{height}"


# --- what one arm measured ---------------------------------------------------


@dataclass(frozen=True)
class StageCost:
    """Where one frame's milliseconds went, at one capture geometry.

    Six numbers rather than one, because the Gate asks for the cost of the larger
    capture *broken out per stage* - and because they move for different reasons:
    the resize and the copy scale with the capture's pixels, the diffusion call
    does not scale at all (the canvas is fixed), and the last two are paid in the
    other process.

    `ipc_put_ms` is what the worker's frame thread pays to hand a frame over;
    `ipc_roundtrip_ms` is what the pickle and the pipe cost end to end, most of
    which `multiprocessing.Queue` runs on its own feeder thread. Both, because the
    first is the frame budget's business and the second is the preview's latency.
    """

    resize_in_ms: float
    diffuse_ms: float
    composite_ms: float
    host_copy_ms: float
    ipc_put_ms: float
    ipc_roundtrip_ms: float
    preview_ms: float

    @property
    def frame_path_ms(self) -> float:
        """What the worker's frame loop pays: everything but the other process."""
        return round(self.resize_in_ms + self.diffuse_ms + self.composite_ms
                     + self.ipc_put_ms, 4)

    def to_dict(self) -> dict:
        return {**asdict(self), "frame_path_ms": self.frame_path_ms}


@dataclass(frozen=True)
class DetailSummary:
    """How many canvas pixels across the rendered region actually got.

    The claim the whole issue rests on, as arithmetic: under `masked` a region is
    diffused at `canvas * region / capture`, and under `crop` it is diffused at the
    canvas. `gain` is the ratio, which is what "high detail" means in numbers.
    """

    region_px: float
    canvas_px: float
    gain: float
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def detail_summary(primitive: str, region_px: float, capture_px: int,
                   canvas_px: int = CANVAS) -> DetailSummary:
    """What `primitive` spends on a region `region_px` across in a `capture_px` frame."""
    region_px = float(region_px)
    diffused = (float(canvas_px) if primitive == CROP
                else region_px * canvas_px / max(1, capture_px))
    gain = diffused / region_px if region_px else 0.0
    return DetailSummary(
        region_px=round(region_px, 2), canvas_px=round(diffused, 2),
        gain=round(gain, 3),
        statement=(
            f"a region {region_px:.0f} px across in a {capture_px} px frame is "
            f"diffused at {diffused:.0f} px under `{primitive}`"),
    )


@dataclass(frozen=True)
class CaptureArm:
    """One primitive at one capture geometry: what it cost and what it produced."""

    primitive: str
    capture_width: int
    capture_height: int
    frames: int
    diffusion_calls: int
    crop_frames: int
    regions_per_frame: float
    denoise: DenoiseRequirement
    denoise_points: List[DenoisePoint]
    stages: StageCost
    latency: LatencySummary
    ms_per_frame: float
    fps: float
    detail: DetailSummary
    region_change: float
    resample_change: float
    flicker: FlickerScore
    background: BackgroundCheck
    identity: Optional[IdentityCheck] = None
    clip_file: str = ""

    @property
    def name(self) -> str:
        return arm_name(self.primitive, self.capture_width, self.capture_height)

    @property
    def net_region_change(self) -> float:
        """Visible change net of the resize control - the issue's own criterion.

        Both primitives resize onto the canvas and back, and that round trip
        changes the pixels before anything is styled (spec 8.2's fifth gotcha), so
        the control is subtracted rather than argued about.
        """
        return round(self.region_change - self.resample_change, 4)

    @property
    def visible(self) -> bool:
        return self.net_region_change >= VISIBLE_CHANGE

    @property
    def meets_budget(self) -> bool:
        return self.ms_per_frame <= FRAME_BUDGET_MS

    def to_dict(self) -> dict:
        data = asdict(self)
        data["stages"] = self.stages.to_dict()
        data["denoise"] = self.denoise.to_dict()
        data["denoise_points"] = [point.to_dict() for point in self.denoise_points]
        data["name"] = self.name
        data["net_region_change"] = self.net_region_change
        data["visible"] = self.visible
        data["meets_budget"] = self.meets_budget
        return data


@dataclass(frozen=True)
class CaptureRunMetrics:
    started_utc: str
    finished_utc: str
    engine_scenario: str
    warmup_frames: int
    canvas: int
    mean_sm_clock_mhz: Optional[float]
    max_temperature_c: Optional[float]
    peak_vram_bytes: int
    gpu_samples: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CaptureResult:
    case: CaptureCase
    clip: ClipRecord
    arms: List[CaptureArm]
    run: CaptureRunMetrics
    cooldown: CooldownRecord
    occupancy: OccupancyRecord
    hardware: Fingerprint
    clock_normalization: Optional[ClockNormalization] = None
    comparison_clip: str = ""
    comparison_still: str = ""
    schema_version: int = RESULT_SCHEMA_VERSION
    kind: str = RECORD_KIND

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "case": self.case.to_dict(),
            "clip": self.clip.to_dict(),
            "arms": [arm.to_dict() for arm in self.arms],
            "run": self.run.to_dict(),
            "cooldown": self.cooldown.to_dict(),
            "occupancy": self.occupancy.to_dict(),
            "hardware": self.hardware.to_dict(),
            "clock_normalization": (None if self.clock_normalization is None
                                    else self.clock_normalization.to_dict()),
            "comparison_clip": self.comparison_clip,
            "comparison_still": self.comparison_still,
        }


ResultLike = Union[CaptureResult, Mapping]


def _as_dict(result: ResultLike) -> dict:
    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


# --- writing -----------------------------------------------------------------


def capture_result_filename(case_name: str, timestamp: str) -> str:
    return f"{case_name}-{timestamp}.json"


def write_capture_result(result: ResultLike, results_dir: Path,
                         timestamp: Optional[str] = None) -> Path:
    """Write `<case>-<timestamp>.json` under `results_dir`; return its path."""
    data = _as_dict(result)
    if timestamp is None:
        timestamp = timestamp_from(data)
    return write_record(data, results_dir,
                        capture_result_filename(data["case"]["name"], timestamp))


README_TITLE = "# Capture geometry results"
README_INTRO = (
    "Written by `uv run python -m bench <capture case>`, never by hand - issue\n"
    "#39, spec 8.2. One run renders the same committed clip at every capture\n"
    "geometry under both rendering primitives at K=1, through the shipped\n"
    "compositor and the one cached 512x512 engine.\n\n"
    "`frame path` is the worker's own cost - the resize onto the canvas, the\n"
    "diffusion call, the composite including its one device-to-host copy, and\n"
    "handing the frame to the GUI process. `object px` is how many canvas pixels\n"
    "across the rendered region actually got, which is what the crop primitive\n"
    "buys and the only reason to pay for a bigger capture.\n\n"
    "Absolute figures belong to the GPU in the row (spec 7.4).\n"
)
README_HEADER = (
    "| finished (UTC) | case | GPU | arm | capture | frame path (ms) | FPS |"
    " object px | net change | flicker | background | file |")
README_SEPARATOR = table_separator(README_HEADER)
README_PREAMBLE = (f"{README_TITLE}\n\n{README_INTRO}\n{README_HEADER}\n"
                   f"{README_SEPARATOR}\n")
README_NAME = "README.md"


def capture_readme_rows(result: dict, filename: str) -> List[str]:
    """One row per arm - a run measures six of them and each is a result."""
    rows = []
    for arm in result["arms"]:
        rows.append(table_row([
            result["run"]["finished_utc"],
            result["case"]["name"],
            gpu_of(result),
            arm["name"],
            f"{arm['capture_width']}x{arm['capture_height']}",
            format_number(arm["stages"]["frame_path_ms"], 2),
            format_number(arm["fps"], 1),
            format_number(arm["detail"]["canvas_px"], 0),
            format_number(arm["net_region_change"], 1),
            format_number(arm["flicker"]["mean_abs_diff"], 2),
            "identical" if arm["background"]["passed"] else "CHANGED",
            filename,
        ]))
    return rows


def append_capture_readme_rows(result: ResultLike, readme_path: Path,
                               filename: str) -> None:
    """Append this run's rows, creating the table if this is the first run."""
    data = _as_dict(result)
    require_recordable(data)
    for row in capture_readme_rows(data, filename):
        append_row(row, readme_path, README_PREAMBLE)


# --- the block spec 8.2 carries ----------------------------------------------


def load_capture_results(results_dir: Path) -> Dict[str, dict]:
    """Every capture-geometry result under `results_dir`, keyed by filename."""
    return load_records(results_dir)


def latest_per_case(results: Mapping[str, dict]) -> Dict[str, dict]:
    """One result per case *per GPU*: the most recently finished run of each."""
    return latest_per(results, lambda result: result["case"]["name"])


REPORT_HEADER = ("| case | arm | capture | denoise | object px | detail gain |"
                 " resize in | diffuse | composite | host copy | IPC put |"
                 " frame path | FPS | 30 FPS | net change | flicker | background |")


def _arm_row(result: dict, arm: dict, column: GpuColumn) -> str:
    stages = arm["stages"]
    return column.row([
        result["case"]["name"],
        arm["primitive"],
        f"{arm['capture_width']}x{arm['capture_height']}",
        format_number(arm["denoise"]["strength"], 2),
        format_number(arm["detail"]["canvas_px"], 0),
        f"{format_number(arm['detail']['gain'], 2)}x",
        format_number(stages["resize_in_ms"], 2),
        format_number(stages["diffuse_ms"], 2),
        format_number(stages["composite_ms"], 2),
        format_number(stages["host_copy_ms"], 2),
        format_number(stages["ipc_put_ms"], 2),
        format_number(stages["frame_path_ms"], 2),
        format_number(arm["fps"], 1),
        "yes" if arm["meets_budget"] else "no",
        format_number(arm["net_region_change"], 1),
        format_number(arm["flicker"]["mean_abs_diff"], 2),
        "identical" if arm["background"]["passed"] else "CHANGED",
    ], result)


def _preamble(results: Sequence[dict], gpus: Sequence[str]) -> str:
    machines = ", ".join(gpus)
    clips = ", ".join(sorted({
        f"{result['clip']['name']} ({result['clip']['frames_used']} frames)"
        for result in results}))
    return (
        f"Measured: {machines}, engine {results[0]['run']['engine_scenario']} - one "
        f"{results[0]['run']['canvas']}x{results[0]['run']['canvas']} canvas, "
        f"whatever the capture is (spec 7.2). Committed clips: {clips}, resized to "
        f"each capture geometry the way the worker's capture thread resizes a "
        f"screen region. Every arm renders the same regions of the same frames at "
        f"K=1 through the same engine at the same strength, so what differs "
        f"between two rows is the primitive and the geometry."
    )


# --- the verdicts ------------------------------------------------------------


def arms_of(result: dict) -> List[dict]:
    return list(result["arms"])


def broke_background(result: dict) -> List[dict]:
    """Arms that moved a non-target pixel. The Gate's first item, both ways."""
    return [arm for arm in arms_of(result) if not arm["background"]["passed"]]


def background_statement(results: Sequence[dict]) -> str:
    """The bit-identity criterion at every geometry, as one sentence."""
    broken = [(result["case"]["name"], arm["name"], arm["background"]["worst_pixels_changed"])
              for result in results for arm in broke_background(result)]
    geometries = sorted({f"{arm['capture_width']}x{arm['capture_height']}"
                         for result in results for arm in arms_of(result)})
    if broken:
        detail = "; ".join(f"{case} / {arm}: {pixels} pixels"
                           for case, arm, pixels in broken)
        return (f"**Non-target pixels are NOT bit-identical** at every geometry - "
                f"{detail}. The criterion does not get easier because the frame got "
                f"bigger, so those arms are disqualified.")
    frames = sum(arm["frames"] for result in results for arm in arms_of(result))
    return (f"**Non-target pixels stayed bit-identical to the capture** on all "
            f"{frames} rendered frames, at every geometry measured "
            f"({', '.join(geometries)}) and under both primitives. The criterion "
            f"does not get easier because the frame got bigger, and it did not "
            f"have to.")


def compare_primitives(result: dict, width: int, height: int) -> Optional[str]:
    """Crop against masked at one geometry: the sentence spec 8.2 quotes."""
    at = {arm["primitive"]: arm for arm in arms_of(result)
          if (arm["capture_width"], arm["capture_height"]) == (width, height)}
    if MASKED not in at or CROP not in at:
        return None
    masked, crop = at[MASKED], at[CROP]
    ratio = (crop["stages"]["frame_path_ms"] / masked["stages"]["frame_path_ms"]
             if masked["stages"]["frame_path_ms"] else 0.0)
    detail = (crop["detail"]["canvas_px"] / masked["detail"]["canvas_px"]
              if masked["detail"]["canvas_px"] else 0.0)
    return (
        f"At {width}x{height}, `crop` diffused the region at "
        f"{crop['detail']['canvas_px']:.0f} px against `masked`'s "
        f"{masked['detail']['canvas_px']:.0f} - {detail:.1f}x the detail - for "
        f"{format_number(crop['stages']['frame_path_ms'], 2)} ms against "
        f"{format_number(masked['stages']['frame_path_ms'], 2)} ms ({ratio:.2f}x). "
        f"Visible change inside the region net of the resize control: crop "
        f"{format_number(crop['net_region_change'], 1)}/255 at strength "
        f"{format_number(crop['denoise']['strength'], 2)}, masked "
        f"{format_number(masked['net_region_change'], 1)}/255 at "
        f"{format_number(masked['denoise']['strength'], 2)}, against a "
        f"{VISIBLE_CHANGE:.0f}/255 threshold. {_expression(crop, masked)}"
    )


def denoise_statement(result: dict) -> str:
    """The strength each arm turned out to need, and the rule that picked it.

    Per arm and not per case, because spec 8.2 measured that the two primitives
    need different strengths for the same job: timing both at one of them would
    measure the primitive the strength was chosen for.
    """
    lines = []
    for arm in arms_of(result):
        lines.append(f"- **{arm['name']}:** {arm['denoise']['statement']}")
    return "\n".join(lines)


def _expression(crop: dict, masked: dict) -> str:
    """Which arms expressed the case at all - the half no millisecond carries."""
    if crop.get("identity") is None:
        names = [name for name, arm in (("crop", crop), ("masked", masked))
                 if not arm["visible"]]
        if not names:
            return "Both expressed the case."
        return f"{sentence_case(' and '.join(names))} did not express the case."
    outcomes = [f"{name} {arm['identity']['became']}/"
                f"{arm['identity']['frames_probed']}"
                for name, arm in (("crop", crop), ("masked", masked))]
    return (f"Read as the new identity in: {', '.join(outcomes)} frames.")


def budget_statement(result: dict) -> str:
    """The 30 FPS verdict, per geometry, on the frame path (the Gate's fourth item)."""
    lines = []
    for arm in arms_of(result):
        verdict = "MET" if arm["meets_budget"] else "NOT MET"
        lines.append(
            f"- **{arm['name']}:** {verdict} - "
            f"{format_number(arm['stages']['frame_path_ms'], 2)} ms on the frame "
            f"path against {FRAME_BUDGET_MS:.2f} ms, "
            f"{format_number(arm['fps'], 1)} FPS at "
            f"{format_number(arm['regions_per_frame'], 2)} regions/frame on "
            f"{gpu_of(result)}.")
    lines.append(
        "\nDetection is **not** in these figures: every arm reads its boxes from "
        "the committed track, which is what makes two arms comparable (issue #5's "
        "second trap). Spec 8.8's cadence sweep measures detection at ~4.3 ms per "
        "frame amortised at `detect_every_n` 5 on this card at 512x512, so an arm "
        "with less than that in hand is not a 30 FPS claim about the shipped app.")
    return "\n".join(lines)


def scaling_statement(result: dict, primitive: str = MASKED) -> Optional[str]:
    """What the bigger capture cost the stages that are not the diffusion call."""
    arms = sorted([arm for arm in arms_of(result) if arm["primitive"] == primitive],
                  key=lambda arm: arm["capture_width"] * arm["capture_height"])
    if len(arms) < 2:
        return None
    small, large = arms[0], arms[-1]
    pixels = ((large["capture_width"] * large["capture_height"])
              / max(1, small["capture_width"] * small["capture_height"]))

    def grew(field: str) -> str:
        before = small["stages"][field]
        after = large["stages"][field]
        ratio = after / before if before else 0.0
        return (f"{field.replace('_ms', '').replace('_', ' ')} "
                f"{format_number(before, 2)} -> {format_number(after, 2)} ms "
                f"({ratio:.1f}x)")

    fields = ("resize_in_ms", "diffuse_ms", "composite_ms", "host_copy_ms",
              "ipc_put_ms", "ipc_roundtrip_ms", "preview_ms")
    return (
        f"**What {small['capture_width']}x{small['capture_height']} -> "
        f"{large['capture_width']}x{large['capture_height']} ({pixels:.1f}x the "
        f"pixels) cost, per stage, under `{primitive}`:** "
        + "; ".join(grew(field) for field in fields)
        + ". The diffusion call is the one stage that *cannot* move - the canvas "
          "is fixed at 512x512 (spec 7.2), and a bare probe on this card measures "
          "it at 15.5 / 15.3 / 16.8 ms at the three geometries - so what the table "
          "shows in that column is the allocation pressure a bigger frame puts on "
          "the same call, not a bigger call. Everything else in the list is the "
          "price of the capture, and issue #31's 9.58 ms of headroom is a 512x512 "
          "figure that does not transfer."
    )


def recommendation(result: dict) -> str:
    """Which arm this case recommends: the freshest detail that still fits.

    The same shape of rule the cadence sweep uses. An arm that broke bit-identity
    or did not express the case cannot be recommended however fast it was - the
    trap issue #5 wrote down and this issue repeats - and among the rest the pick
    is the one that gives the object the most canvas pixels inside the budget,
    because that is the whole thing being bought.
    """
    qualified = [arm for arm in arms_of(result)
                 if arm["background"]["passed"] and arm["meets_budget"]
                 and _expressed(arm)]
    if not qualified:
        return (f"**No arm of {result['case']['name']} qualifies** on "
                f"{gpu_of(result)}: every one either moved a non-target pixel, "
                f"missed the {FRAME_BUDGET_MS:.2f} ms frame budget, or did not "
                f"express the case.")
    # Canvas pixels to the nearest pixel, then the cheaper frame path. Rounded
    # because two geometries can land a hundredth of a pixel apart on the same
    # region and a recommendation decided by that is a recommendation decided by
    # rounding.
    best = max(qualified, key=lambda arm: (round(arm["detail"]["canvas_px"]),
                                           -arm["stages"]["frame_path_ms"]))
    return (
        f"**Recommended for {result['case']['name']} on {gpu_of(result)}: "
        f"`{best['name']}`** - the region diffused at "
        f"{best['detail']['canvas_px']:.0f} px "
        f"({format_number(best['detail']['gain'], 2)}x its size in the capture), "
        f"{format_number(best['stages']['frame_path_ms'], 2)} ms on the frame path "
        f"({format_number(best['fps'], 1)} FPS), background identical. It is the "
        f"most canvas an object got among the arms that fit the budget and "
        f"expressed the case." + _field_of_view(best, qualified)
    )


def _field_of_view(best: dict, qualified: Sequence[dict]) -> str:
    """What a bigger capture buys once the detail figure has stopped moving.

    Under `crop` the object gets the whole canvas whatever the capture is, so the
    rule above cannot separate 512x512 from 1920x1080 on detail and falls back to
    cost - which reads as a recommendation against the larger capture and is not
    one. What the larger capture buys is *field of view*: a screen region big
    enough to hold the object at all, which is the thing issue #39 was opened
    about and the one thing no metric here scores. Said out loud, with what it
    costs, rather than left as an inference from a table.
    """
    bigger = [arm for arm in qualified
              if arm["primitive"] == best["primitive"]
              and arm["capture_width"] * arm["capture_height"]
              > best["capture_width"] * best["capture_height"]
              and round(arm["detail"]["canvas_px"]) >= round(
                  best["detail"]["canvas_px"])]
    if not bigger:
        return ""
    largest = max(bigger, key=lambda arm: arm["capture_width"] * arm["capture_height"])
    cost = largest["stages"]["frame_path_ms"] - best["stages"]["frame_path_ms"]
    why = ("the object gets the whole canvas whatever is captured"
           if best["primitive"] == CROP else
           "the object keeps the same fraction of a frame that is squeezed onto "
           "one canvas, so it lands on the same canvas pixels either way")
    return (
        f" It is **not** a recommendation against a larger capture: under "
        f"`{best['primitive']}` {why}. Detail therefore cannot separate the "
        f"geometries and the tie falls to cost. What "
        f"{largest['capture_width']}x{largest['capture_height']} buys "
        f"instead is field of view - a screen region big enough to hold the object "
        f"at all, which is what issue #39 was opened about and the one thing no "
        f"metric here scores - and what it costs is "
        f"{format_number(cost, 2)} ms/frame "
        f"({format_number(largest['stages']['frame_path_ms'], 2)} ms, "
        f"{format_number(largest['fps'], 1)} FPS)."
    )


def _expressed(arm: dict) -> bool:
    if arm.get("identity") is not None:
        return bool(arm["identity"]["achieved"])
    return bool(arm["visible"])


def _machine_sections(results: Sequence[dict], gpus: Sequence[str]) -> List[str]:
    """The verdicts, taken once per machine - the rule every block here follows."""
    sections: List[str] = []
    for gpu in gpus:
        on_machine = measured_on(results, gpu)
        if not on_machine:
            continue
        sections.append(background_statement(on_machine))
        for result in on_machine:
            comparisons = [compare_primitives(result, width, height)
                           for width, height in _geometries(result)]
            sections.extend(line for line in comparisons if line)
            scaling = scaling_statement(result)
            if scaling:
                sections.append(scaling)
            sections.append("Denoise strength each arm turned out to need:\n"
                            + denoise_statement(result))
            sections.append("30 FPS on the frame path, per arm:\n"
                            + budget_statement(result))
            sections.append(recommendation(result))
    return sections


def _geometries(result: dict) -> List[Tuple[int, int]]:
    seen = []
    for arm in arms_of(result):
        pair = (arm["capture_width"], arm["capture_height"])
        if pair not in seen:
            seen.append(pair)
    return seen


def _artefacts(results: Sequence[dict]) -> str:
    files = [f"`{result['comparison_clip']}`" for result in results
             if result.get("comparison_clip")]
    if not files:
        return ""
    return (f"Side-by-side clips for human judgement, under "
            f"`bench/results/capture/`: {', '.join(files)}. **The metric ranks "
            f"cost, detail and steadiness, not beauty** - a human still has to "
            f"watch these and confirm the crop's upscaled invention is acceptable.")


def format_capture_report(results: Mapping[str, dict]) -> str:
    """The measured block spec 8.2 carries for the capture/primitive question."""
    ordered = sorted(latest_per_case(results).values(),
                     key=lambda result: (result["case"]["name"], gpu_of(result),
                                         result["run"]["finished_utc"]))
    if not ordered:
        return "no capture geometry measured yet (issue #39)"

    gpus = distinct_gpus(ordered)
    column = GpuColumn.for_gpus(gpus)
    header = column.header(REPORT_HEADER)
    rows = [_arm_row(result, arm, column)
            for result in ordered for arm in arms_of(result)]
    sections = [
        _preamble(ordered, gpus),
        "\n".join([header, table_separator(header)] + rows),
    ]
    sections.extend(_machine_sections(ordered, gpus))
    artefacts = _artefacts(ordered)
    if artefacts:
        sections.append(artefacts)
    return "\n\n".join(sections)


def mean_ms(samples: Sequence[float]) -> float:
    """A stage's mean, rounded the way every other figure here is rounded."""
    return round(statistics.fmean(samples), 4) if samples else 0.0
