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
from bench.flicker import FlickerScore, ResponseScore
from bench.paths import (
    CADENCE_RESULTS_SUBDIR,
    MODEL_RESULTS_SUBDIR,
    SELECTIVE_RESULTS_SUBDIR,
    STABILITY_RESULTS_SUBDIR,
)
from bench.primitive_results import ClipRecord
from bench.results import (
    GpuColumn,
    append_row,
    distinct_gpus,
    format_number,
    gpu_of,
    latest_per,
    load_records,
    measured_on,
    normalised_cell,
    require_recordable,
    sentence_case,
    table_row,
    table_separator,
    timestamp_from,
    write_record,
)

RECORD_KIND = "selective"

# The wire key the plan's `global` block travels under - `render_plan.GLOBAL_KEY`.
# Spelt here rather than imported at module scope because the shipped modules are
# imported inside functions in this file, and a test holds the two to one value.
GLOBAL_KEY = "global"

# Where a run's blend ran - `device_compositor.HOST` / `.DEVICE`, spelt here for the
# reason `GLOBAL_KEY` is, and held to those values by a test. A composite measured on
# the host and one measured on the device are two designs as much as two numbers, and
# spec 7.4 compares them across machines (issue #31).
HOST_COMPOSITE = "host"
DEVICE_COMPOSITE = "device"

# The one cached engine the path is rendered through - the same 512x512 batch-1
# engine issue #5 compared both primitives on, so a millisecond here is comparable
# with a millisecond there.
ENGINE_SCENARIO = "img2img-tensorrt-512x512-b1"


def engine_scenario_for(case: "SelectiveCase"):
    """The scenario this arm renders through, at the step count its model needs.

    The shipped one for a baseline; for a base-model arm (issue #38) the same
    accelerator and geometry with another checkpoint and `BASE_MODELS`' own step
    count, so the arm differs from the baseline in the model and nothing else a
    reader has to go looking for.
    """
    from bench.models import BASE_MODELS
    from bench.scenarios import SCENARIOS, base_model_name
    from bench.steps import step_arm

    from bench.models import DEFAULT_BASE

    if case.base_model is None:
        return SCENARIOS[ENGINE_SCENARIO]
    base = BASE_MODELS[case.base_model]
    # The shipped model is not a variant of itself: the registry's own cell is what
    # every committed baseline was measured through, and naming it any other way
    # would key an engine nothing has built. So `--base-model sd-turbo` is the
    # same-session *control* on the comparison rather than a second configuration.
    if case.base_model == DEFAULT_BASE:
        return SCENARIOS[ENGINE_SCENARIO]
    return step_arm(SCENARIOS[base_model_name("tensorrt", case.base_model)],
                    base.steps)

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
    # The one plan field the cadence sweep (issue #23) moves. `None` is the shipped
    # plan untouched, which is what the baseline runs measured.
    detect_every_n: Optional[int] = None
    # The two the temporal-stability sweep (issue #32) moves - spec 8.5's unbuilt
    # levers. `None` again means the shipped plan, so every committed baseline is
    # still a run of it.
    seed_policy: Optional[str] = None
    output_ema: Optional[float] = None
    # The base model this arm renders through (issue #38). `None` is the shipped
    # SD-Turbo engine every committed baseline was measured on; a key in
    # `bench.models.BASE_MODELS` names another, with the step count it needs.
    base_model: Optional[str] = None

    def plan(self):
        """The hardcoded priority-case plan, from the shipped producer.

        With an override, the same plan re-validated with one field changed:
        `RenderPlan.to_dict` is exactly what `validate_plan` accepts back, so a
        sweep's arms go through the same door the worker's plans do and a value
        outside its range is clamped and recorded rather than rendered. An override
        that asks for the value the plan already carries is not a change, and the
        plan is returned untouched - so an arm at the shipped setting records the
        same plan the baselines do, version and all.
        """
        from render_plan import INITIAL_PLAN_VERSION, priority_case_plan, validate_plan

        plan = priority_case_plan()
        shipped = plan.to_dict()
        raw = plan.to_dict()
        if self.detect_every_n is not None:
            raw[GLOBAL_KEY]["detect_every_n"] = self.detect_every_n
        if self.output_ema is not None:
            raw[GLOBAL_KEY]["output_ema"] = self.output_ema
        if self.seed_policy is not None:
            for target in raw["targets"]:
                target["seed_policy"] = self.seed_policy
        if raw == shipped:
            return plan
        result = validate_plan(raw, previous_version=INITIAL_PLAN_VERSION)
        if result.plan is None:  # unreachable: only a validated plan is edited here
            raise AssertionError(f"the override did not validate: {result.reason}")
        return result.plan

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


def is_cadence_arm(case: SelectiveCase) -> bool:
    """Is this a run of the cadence sweep rather than a baseline? (issue #23)

    The one predicate the two routing decisions below share, so a run cannot end
    up in the sweep's directory under the baselines' heading or the other way
    round.
    """
    return case.detect_every_n is not None


def is_model_arm(case: SelectiveCase) -> bool:
    """Is this an arm on another base model? (issue #38)

    Third predicate, third reason it exists: an arm rendered through SD 1.5 at four
    steps must not become the row spec 8.8 and 7.4 quote for the shipped path.
    """
    return case.base_model is not None


def is_stability_arm(case: SelectiveCase) -> bool:
    """Is this an arm of the temporal-stability sweep? (issue #32)

    The same predicate as `is_cadence_arm` for the other pair of swept fields, and
    it exists for the same reason: an arm rendered under a seed policy or an EMA
    the baselines were not rendered under must not become the row spec 8.8 quotes.
    """
    return case.seed_policy is not None or case.output_ema is not None


def ema_suffix(coefficient: float) -> str:
    """An output EMA as a filename-safe arm suffix: 0.5 -> `ema50`.

    Hundredths rather than the float's own text, because the coefficient lands in a
    filename and a `.` there reads as an extension to half the tools that will see
    it. The record carries the number itself; this only has to be unambiguous.
    """
    return f"ema{int(round(float(coefficient) * 100)):02d}"


def results_subdir(case: SelectiveCase) -> str:
    """Which results directory a run of `case` belongs in.

    One rule in one place: a swept arm never lands beside the baselines, because
    the selective directory is reduced to the newest run per (case, GPU) and an arm
    measured at another setting would take that row over.
    """
    if is_cadence_arm(case):
        return CADENCE_RESULTS_SUBDIR
    if is_stability_arm(case):
        return STABILITY_RESULTS_SUBDIR
    if is_model_arm(case):
        return MODEL_RESULTS_SUBDIR
    return SELECTIVE_RESULTS_SUBDIR


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
        # The two temporal-stability levers (issue #32). Read off the plan the run
        # actually rendered, like every other field here, so a clamp shows.
        "seed_policy": plan.effective_seed_policy,
        "output_ema": plan.settings.output_ema,
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


# --- how stale the boxes a frame renders are ---------------------------------


@dataclass(frozen=True)
class StalenessSummary:
    """What `detect_every_n` costs, in the only currency it is paid in.

    Issue #23 step 3. Raising the cadence is the cheap lever on the frame budget:
    it trades no image quality at all, unlike a smaller engine, and it buys back
    detector milliseconds in exact proportion. What it spends is *freshness*, and
    freshness has three readings rather than one, so all three are recorded.

    - `mean_age_frames` / `worst_age_frames` - how old the boxes a frame renders
      are, counted from the capture the detect ran on. This is the cadence plus
      however long the detect took, which is why it is measured rather than
      derived from N.
    - `mean_refresh_iou` / `mean_refresh_shift_px` - how far an object had moved by
      the time the tracker heard about it again. A box that is old and still right
      costs nothing; the shift is what a viewer sees as the mask lagging the
      subject.
    - `distinct_track_ids` against `max_concurrent_tracks` - whether identity
      survived. Spec 8.5 pins per-object seeds to track ids, so an object that
      comes back under a new id has paid the cadence in identity rather than in
      milliseconds, and no millisecond figure would show it.
    """

    detect_every_n: int
    frames: int
    ticks: int
    frames_without_tracks: int
    mean_age_frames: float
    worst_age_frames: int
    mean_age_ms: Optional[float]
    refreshes: int
    mean_refresh_iou: float
    worst_refresh_iou: float
    mean_refresh_shift_px: float
    worst_refresh_shift_px: float
    distinct_track_ids: int
    max_concurrent_tracks: int
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def _first_per_tick(snapshots: Sequence) -> List:
    """One snapshot per detector tick, in order - the refreshes the run actually had.

    The frame loop reads the same `Tracks` on every frame between two detects, so
    iterating frames would count one refresh once per frame it was read on and
    weight a slow detect as several.
    """
    seen = set()
    ordered = []
    for snapshot in snapshots:
        if snapshot.ticks and snapshot.ticks not in seen:
            seen.add(snapshot.ticks)
            ordered.append(snapshot)
    return ordered


def _centre_shift_px(before, after) -> float:
    """How far a box's centre moved, in pixels. The lag a viewer sees as the mask
    trailing the subject."""
    dx = ((after.x0 + after.x1) - (before.x0 + before.x1)) / 2.0
    dy = ((after.y0 + after.y1) - (before.y0 + before.y1)) / 2.0
    return (dx * dx + dy * dy) ** 0.5


def _refresh_movement(ticks: Sequence) -> Tuple[List[float], List[float]]:
    """Per surviving track, `(iou, centre shift)` across each pair of consecutive ticks.

    Takes the per-tick snapshots `_first_per_tick` returns, not the per-frame ones.
    """
    from detection import iou

    ious: List[float] = []
    shifts: List[float] = []
    for before, after in zip(ticks, ticks[1:]):
        later = {track.track_id: track.box for track in after.tracks}
        for track in before.tracks:
            box = later.get(track.track_id)
            if box is None:
                continue
            ious.append(iou(track.box, box))
            shifts.append(_centre_shift_px(track.box, box))
    return ious, shifts


def staleness_summary(snapshots: Sequence, detect_every_n: int,
                      ms_per_frame: Optional[float] = None) -> StalenessSummary:
    """How stale the tracks were, over one run's own per-frame snapshots.

    `snapshots[i]` is the `Tracks` frame `i` rendered under, which is what the
    frame loop actually read - so this measures the cadence as the loop experienced
    it, detect latency included, rather than the cadence as a plan field.
    """
    ages = [index - snapshot.frame_index
            for index, snapshot in enumerate(snapshots) if snapshot.ticks]
    ticks = _first_per_tick(snapshots)
    ious, shifts = _refresh_movement(ticks)
    ids = {track.track_id for snapshot in snapshots for track in snapshot.tracks}
    concurrent = max((snapshot.count for snapshot in snapshots), default=0)
    mean_age = round(statistics.fmean(ages), 4) if ages else 0.0
    summary = StalenessSummary(
        detect_every_n=detect_every_n, frames=len(snapshots), ticks=len(ticks),
        frames_without_tracks=sum(1 for snapshot in snapshots if not snapshot.ticks),
        mean_age_frames=mean_age, worst_age_frames=max(ages) if ages else 0,
        mean_age_ms=(None if ms_per_frame is None
                     else round(mean_age * ms_per_frame, 4)),
        refreshes=len(ious),
        mean_refresh_iou=round(statistics.fmean(ious), 4) if ious else 0.0,
        worst_refresh_iou=round(min(ious), 4) if ious else 0.0,
        mean_refresh_shift_px=round(statistics.fmean(shifts), 4) if shifts else 0.0,
        worst_refresh_shift_px=round(max(shifts), 4) if shifts else 0.0,
        distinct_track_ids=len(ids), max_concurrent_tracks=concurrent,
        statement="",
    )
    return dataclasses.replace(summary, statement=_staleness_statement(summary))


def _staleness_statement(summary: StalenessSummary) -> str:
    """What the numbers say, in the words the sweep table's rows are read with."""
    if not summary.ticks:
        return (f"no detect ran in {summary.frames} frames, so there is no "
                f"staleness to report at detect_every_n {summary.detect_every_n}")
    age_ms = ("" if summary.mean_age_ms is None
              else f", {summary.mean_age_ms:.0f} ms")
    if summary.refreshes:
        movement = (f"between refreshes a track's box kept "
                    f"{summary.mean_refresh_iou:.2f} IoU (worst "
                    f"{summary.worst_refresh_iou:.2f}) and its centre moved "
                    f"{summary.mean_refresh_shift_px:.1f} px (worst "
                    f"{summary.worst_refresh_shift_px:.1f})")
    else:
        movement = "no track survived a refresh to be compared"
    return (f"at detect_every_n {summary.detect_every_n} a frame rendered boxes "
            f"{summary.mean_age_frames:.1f} frames old on average (worst "
            f"{summary.worst_age_frames}{age_ms}); {movement}; "
            f"{summary.max_concurrent_tracks} concurrent objects held "
            f"{summary.distinct_track_ids} identities over {summary.ticks} detects")


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
    # Which implementation of C7 produced `composite`. Defaulted rather than
    # required: every run committed before issue #31 blended on the host, so the
    # absent field is an answer and not a gap.
    composite_path: str = HOST_COMPOSITE
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
    staleness: Optional[StalenessSummary] = None
    # Whether anything else was drawing on the card (issue #33). Optional
    # because the fourteen records committed before the gate existed cannot
    # answer it, and `None` says that rather than passing them.
    occupancy: Optional[OccupancyRecord] = None
    # Flicker's mirror: what the output did where the source *moved* (issue #32).
    # Optional because every run committed before that issue has no such figure,
    # and an absent one is "not measured" rather than "inert".
    response: Optional[ResponseScore] = None
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
            "staleness": (None if self.staleness is None
                          else self.staleness.to_dict()),
            "flicker": self.flicker.to_dict(),
            "response": None if self.response is None else self.response.to_dict(),
            "gate": {
                "passed": self.gate_passed,
                "background": self.background.to_dict(),
                "change": self.change.to_dict(),
                "coverage": self.coverage.to_dict(),
                "stall": self.stall.to_dict(),
            },
            "cooldown": self.cooldown.to_dict(),
            "occupancy": (None if self.occupancy is None
                          else self.occupancy.to_dict()),
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
    "Absolute figures belong to the GPU in the row (spec 7.4), and rows from two GPUs\n"
    "are two answers rather than one superseding the other. Whether 30 FPS is met is\n"
    "a deploy-hardware question: `python -m bench --portability-report` answers it\n"
    "from these rows, and spec 7.4 carries the answer.\n"
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


CADENCE_README_INTRO = (
    "Written by `uv run python -m bench <case> --detect-every-n N`, never by hand -\n"
    "issue #23, spec 8.8. One row is one arm of the cadence sweep: the same\n"
    "shipped path over the same clip under the same plan, with\n"
    "`global.detect_every_n` the only field that moved. So no pixel the diffusion\n"
    "produces differs between arms - what a higher cadence spends is the freshness\n"
    "of the boxes, which the record's `staleness` block measures and\n"
    "`python -m bench --cadence-report` tabulates.\n\n"
    "These rows are deliberately not in `../selective/`: that directory is reduced\n"
    "to the newest run per (case, GPU) for spec 8.8 and 7.4, and an arm at another\n"
    "cadence sitting there would quietly become the figure those sections quote.\n"
)
CADENCE_README_TITLE = "# Detector cadence sweep results"
CADENCE_README_PREAMBLE = (
    f"{CADENCE_README_TITLE}\n\n{CADENCE_README_INTRO}\n"
    f"{SELECTIVE_README_HEADER}\n{SELECTIVE_README_SEPARATOR}\n"
)


STABILITY_README_INTRO = (
    "Written by `uv run python -m bench <case> --seed-policy P --output-ema E`,\n"
    "never by hand - issue #32, spec 8.5. One row is one arm of the temporal-\n"
    "stability sweep: the same shipped path over the same clip under the same\n"
    "plan, with the seed policy and the output EMA the only fields that moved.\n\n"
    "`flicker` is what an arm was run to move, and it is only half the reading -\n"
    "`python -m bench --stability-report` puts it beside the responsiveness\n"
    "figure and the visible-change figure net of a control, because an arm that\n"
    "lowers flicker by rendering less is disqualified rather than recommended.\n\n"
    "These rows are deliberately not in `../selective/`: that directory is\n"
    "reduced to the newest run per (case, GPU) for spec 8.8 and 7.4, and an arm\n"
    "at another setting sitting there would quietly become the figure those\n"
    "sections quote.\n"
)
STABILITY_README_TITLE = "# Temporal stability sweep results"
STABILITY_README_PREAMBLE = (
    f"{STABILITY_README_TITLE}\n\n{STABILITY_README_INTRO}\n"
    f"{SELECTIVE_README_HEADER}\n{SELECTIVE_README_SEPARATOR}\n"
)


def readme_preamble(case: SelectiveCase) -> str:
    """Which table a run of `case` is appended to, heading and all.

    The other half of `results_subdir`, decided by the same predicates: the three
    directories keep the same columns, so the row builder is shared, but they exist
    for different reasons, so the paragraph above the table is not.
    """
    if is_cadence_arm(case):
        return CADENCE_README_PREAMBLE
    if is_stability_arm(case):
        return STABILITY_README_PREAMBLE
    return SELECTIVE_README_PREAMBLE


def append_selective_readme_row(result: ResultLike, readme_path: Path,
                                filename: str,
                                preamble: str = SELECTIVE_README_PREAMBLE) -> None:
    """Append this run's row, creating the table if this is the first run.

    `preamble` is `readme_preamble(case)` for a run. A parameter rather than
    something read back off the record, because a record does not know which of
    the two tables it is about to join.
    """
    data = _as_dict(result)
    require_recordable(data)
    append_row(selective_readme_row(data, filename), readme_path, preamble)


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
                     f"{sentence_case(statement)}.")
    return lines


def _measured_phrase(result: dict) -> str:
    """The engine, the clip and the canvas one run was measured through."""
    clip, case = result["clip"], result["case"]
    return (f"{result['run']['engine_scenario']}, {clip['frames_used']} consecutive "
            f"frames of `{clip['name']}` resized to the app's "
            f"{case['canvas']}x{case['canvas']} capture canvas")


def _clock_states(results: Sequence[dict]) -> str:
    """The clock regimes these records were measured under, deduplicated."""
    return ", ".join(sorted({result["hardware"]["clock_lock"]["state"]
                             for result in results}))


def _clock_phrase(results: Sequence[dict], gpus: Sequence[str]) -> str:
    """Which regime produced these figures, and whose figures they are.

    One machine reads as it always did. Two, and naming a single GPU would imply
    the other one's rows were measured on it (issue #25 step 3), so every machine
    is named with its own regime and the reader is pointed at the row.
    """
    if len(gpus) == 1:
        return (f"Clocks {_clock_states(results)}; "
                f"absolute figures belong to this GPU (spec 7.4)")
    per_gpu = ", ".join(f"{gpu} {_clock_states(measured_on(results, gpu))}"
                        for gpu in gpus)
    return (f"Clocks: {per_gpu}; absolute figures belong to the GPU in the row "
            f"(spec 7.4)")


def _report_preamble(results: Sequence[dict], gpus: Sequence[str]) -> str:
    measured = ", ".join(sorted({_measured_phrase(result) for result in results}))
    spanning = "" if len(gpus) == 1 else " Rows are one per case per GPU."
    return (f"Measured on {', '.join(gpus)}, {measured}. "
            f"{_clock_phrase(results, gpus)}, and 30 FPS is M2's gate, "
            f"not this one's.{spanning}")


def _staleness_line(result: dict) -> str:
    """What the run's cadence cost, on the machine that paid it (issue #33).

    The cadence is the one setting here whose price is not a millisecond, and the
    price is not portable: the same box age in frames is 49 ms at one card's frame
    path and 159 at another's, which is why the statement sits inside a per-machine
    section rather than under the table. A record with no block predates the
    measurement (issue #23) and says so - older than the question, not zero.
    """
    stale = result.get("staleness")
    if not stale:
        return ("What the cadence cost: **not recorded** - this run predates the "
                "staleness block (issue #23).")
    return f"What the cadence cost: {sentence_case(stale['statement'])}."


def _occupancy_line(result: dict) -> Optional[str]:
    """What the run was measured beside, but only when that is not "nothing".

    `--require-idle-gpu` is opt-in, so a contended run can still reach
    `bench/results/` - and issue #33's did, at 3.2x its own committed baseline.
    A door that fires only at run time is not a door; the block that quotes the
    number has to carry it too. `None` for a clear run and for the records written
    before the gate existed, so a caveat that would always be there is not there
    to be skipped over - the rule `GpuColumn` follows.
    """
    occupancy = result.get("occupancy")
    if not occupancy or occupancy.get("clear"):
        return None
    utilization = occupancy.get("mean_utilization_pct")
    seen = ("utilization unreadable" if utilization is None
            else f"{utilization:.0f}% of the SMs in use before the timed region")
    return (f"**This run was measured on a {occupancy['outcome']} GPU** - {seen}, "
            f"against a {occupancy['threshold_pct']:.0f}% threshold. Something else "
            f"was drawing on the card, so its milliseconds measure the neighbour as "
            f"much as the change (issue #33).")


def _gate_sections(results: Sequence[dict], gpus: Sequence[str]) -> List[str]:
    """The Gate lines, what the cadence cost, and the manual artefact, per machine.

    The table lists every case; these belong to the first case *on each machine*,
    because a Gate statement is a claim about one run and the run it came from is
    the thing a reader has to be able to name.
    """
    sections = []
    for gpu in gpus:
        primary = measured_on(results, gpu)[0]
        sections.append("The Gate, measured:" if len(gpus) == 1
                        else f"The Gate, measured on {gpu}:")
        sections.append("\n".join(_gate_lines(primary)))
        sections.append(_staleness_line(primary))
        occupancy = _occupancy_line(primary)
        if occupancy:
            sections.append(occupancy)
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
    column = GpuColumn.for_gpus(gpus)
    header = column.header(REPORT_HEADER)
    sections = [
        _report_preamble(ordered, gpus),
        "\n".join([header, table_separator(header)]
                  + _report_rows(ordered, column)),
    ]
    return "\n\n".join(sections + _gate_sections(ordered, gpus))
