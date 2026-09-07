"""What a plan swap costs: acceptance criteria 1 and 3, measured. Issue #30.

Spec 11 has five criteria. Two of them were answered on disk before this module
existed - 30 FPS (issue #24) and bit-identical non-target pixels (issue #8) - and
two had never been measured at all:

1. a typed instruction takes effect **within 3 s**, and
3. swapping it causes **no stutter** in the output stream and **no TensorRT
   rebuild**.

Everything the repo knew about them was indirect: a 400 ms debounce in the GUI, a
~108 ms `set_classes` re-warm the detector pays on its own thread, a schedule
update that is a runtime call rather than a rebuild. On paper those sum to well
under three seconds. This module is the arithmetic that answers them from a run
instead, and `bench.plan_swap_runner` is the half that touches a GPU.

Four rules, each of them one of the issue's traps made executable.

- **Criterion 1's clock starts at the keystroke.** The debounce is time the user
  waits, so it is in the figure; the worker-side half is reported beside it
  because that is the one an optimisation would move.
- **The pixels are the event, not the plan version.** A frame that bound the new
  plan but has no boxes for it yet renders the capture untouched - the *old*
  instruction is still what is on screen - so the clock stops at the first frame
  whose rendered regions came from the new plan's own tracks.
- **Criterion 3 is judged against a steady-state control from the same run.** A
  40 ms frame on a card rendering 32 ms frames has not stuttered, and 33.33 ms on
  its own cannot tell you that. The control is the worse of the two steady states
  either side of the swap, so the bar is not whichever half flatters the verdict.
- **No rebuild is a checked number.** The step *count* is what rebuilds an engine;
  a plan change moves schedule values only. Both are recorded, with the engine
  object's identity beside them, so "no rebuild" is arithmetic rather than a claim.

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
from bench.cooldown import CooldownRecord
from bench.detector_results import LatencySummary
from bench.fingerprint import Fingerprint
from bench.portability import FRAME_BUDGET_MS
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
    require_recordable,
    sentence_case,
    table_row,
    table_separator,
    timestamp_from,
    write_record,
)
from bench.selective import BackgroundCheck

RECORD_KIND = "plan-swap"

# What the GUI waits out before it builds a plan from the two fields -
# `main_gpu_addon.PLAN_DEBOUNCE_MS`. Spelt here rather than imported because
# importing `main_gpu_addon` pulls in the Tk stack and primes the DLL search path;
# a test holds the two to one value, the way `bench.selective` pins `GLOBAL_KEY`.
GUI_DEBOUNCE_MS = 400.0

# Acceptance criterion 1 (spec 11): "within 3 s", from the keystroke.
CRITERION_1_BUDGET_MS = 3000.0

# How much longer than the dearest frame in the same run's steady state a frame
# across the swap may take before the output stream has stuttered. One whole frame
# budget, and the issue's third trap is why it is a millisecond allowance rather
# than a percentage: "a swap that costs one 40 ms frame on a card rendering 32 ms
# frames has not stuttered". A stream stutters when it loses a frame, and the
# smallest unit of that is one budget's worth of extra work. The excess is recorded
# in milliseconds either way, so a stricter reading is one subtraction away.
STUTTER_ALLOWANCE_MS = FRAME_BUDGET_MS

# The two kinds of swap, which are two different paths rather than two sizes of one.
VOCABULARY_SWAP = "vocabulary"
RUNTIME_SWAP = "runtime"

TARGET_CASE = "swap-target"
STYLE_CASE = "swap-style"


# --- the cases ---------------------------------------------------------------


@dataclass(frozen=True)
class PlanFields:
    """One instruction as a user types it: the GUI's two fields, and two defaults.

    `region` and `denoise` have no widget yet (issue #22 left them to a later
    change), so they are the case's rather than the user's - but they travel the
    same producer, so a case cannot describe an instruction the GUI could not
    build.
    """

    target: str
    style: str
    region: str
    denoise: float

    def plan(self, previous_version: int = 0):
        """This instruction through `plan_from_fields`, the shipped producer."""
        from render_plan import plan_from_fields

        result = plan_from_fields(target=self.target, style=self.style,
                                  region=self.region, denoise=self.denoise,
                                  previous_version=previous_version)
        if result.plan is None:
            raise AssertionError(f"the case's plan did not validate: {result.reason}")
        return result.plan

    def to_dict(self) -> dict:
        return asdict(self)


def swap_kind(before, after) -> str:
    """Which path this swap takes, read off the two plans rather than declared.

    A changed concept re-encodes the detector's vocabulary and owes a throwaway
    detect (spec 8.1); a changed style or strength is a prompt re-encode and a
    schedule update, and the detector never hears about it. Derived, so a case
    cannot be labelled one kind and measure the other.
    """
    from detector_worker import concepts_of

    return (RUNTIME_SWAP if concepts_of(before) == concepts_of(after)
            else VOCABULARY_SWAP)


@dataclass(frozen=True)
class SwapCase:
    """One run: a committed clip, an instruction, and the one that replaces it.

    The `before` instruction is the priority case the worker itself starts on
    behind `SD_DEMO_PLAN`, so the frames before the swap are the same steady state
    issues #24 and #23 measured - and the control criterion 3 is judged against is
    a state the repo already has baselines for.
    """

    name: str
    clip: str
    note: str
    before: PlanFields
    after: PlanFields
    # The capture geometry the app runs at, as in `bench.selective`: the worker's
    # capture thread resizes to the engine's canvas before the frame loop sees it.
    canvas: int = 512
    frames: int = 96
    # The frame the new instruction is submitted on. Late enough that the steady
    # state before it is settled, early enough that the one after it is too.
    swap_frame: int = 48
    start_frame: int = 0
    warmup_frames: int = 3

    def plans(self) -> Tuple:
        """`(before, after)`, versioned as the worker would version them."""
        from render_plan import INITIAL_PLAN_VERSION

        before = self.before.plan(previous_version=INITIAL_PLAN_VERSION)
        return before, self.after.plan(previous_version=before.plan_version)

    def replace(self, **changes) -> "SwapCase":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)


# The instruction both cases start from: `render_plan.priority_case_plan()`, spelt
# as the two fields that produce it. A test holds the two to one plan.
PRIORITY_FIELDS = PlanFields(
    target="person", region="lower_half", denoise=0.49,
    style=("trousers soaked through with a dark wet stain, damp fabric, "
           "wet denim, photograph"),
)

CASES: Dict[str, SwapCase] = {
    TARGET_CASE: SwapCase(
        name=TARGET_CASE, clip="people.mp4", before=PRIORITY_FIELDS,
        # A concept the committed clip actually holds, or the clock for "when did
        # the new instruction reach the screen" would never stop. `shoes` is found
        # on every sampled frame of `people.mp4`; the detector is open-vocabulary,
        # so this is a typed word rather than a class index.
        after=PlanFields(target="shoes", region="full_box", denoise=0.62,
                         style="glossy red patent leather shoes, wet paint"),
        note="The expensive swap: a new target, so the detector's vocabulary is "
             "re-encoded and owes a throwaway detect before any box is about the "
             "new concept.",
    ),
    STYLE_CASE: SwapCase(
        name=STYLE_CASE, clip="people.mp4", before=PRIORITY_FIELDS,
        after=PlanFields(target="person", region="lower_half", denoise=0.62,
                         style="bright yellow waterproof trousers, studio light"),
        note="The cheap swap: the same target restyled, so the engine takes a new "
             "prompt embedding and a new schedule value and the detector never "
             "hears about it.",
    ),
}


# --- when the pixels arrived -------------------------------------------------


def first_pixel_frame(frames: Sequence[Mapping], plan_version: int,
                      concepts: Sequence[str]) -> Optional[int]:
    """The first frame whose pixels show the new instruction, or `None`.

    Three conditions, and all three are the same question asked of one frame: it
    bound the new plan, it actually diffused something, and the regions it
    diffused came from a `Tracks` snapshot about the new plan's concepts. The
    third is what separates the two kinds without branching on them - after a
    vocabulary change the tracks in force are empty until the re-warm and the
    detect are done, and after a style change they already serve the new plan.
    """
    wanted = tuple(concepts)
    for frame in frames:
        if frame["plan_version"] != plan_version or not frame["diffuses"]:
            continue
        if tuple(frame["tracks_concepts"]) == wanted:
            return int(frame["index"])
    return None


# --- criterion 1 -------------------------------------------------------------


@dataclass(frozen=True)
class SwapTiming:
    """Every clock the swap can be read on, and who is waiting on each.

    `accepted_to_*` starts where the worker accepts the validated plan, which is
    the figure an optimisation would move. `keystroke_to_pixel_ms` is what the
    user experiences and is the one criterion 1 is judged on.
    """

    swap_frame: int
    plan_version_before: int
    plan_version_after: int
    kind: str
    validate_ms: float
    debounce_ms: float
    accepted_to_applied_ms: Optional[float]
    accepted_to_pixel_ms: Optional[float]
    worker_ms: Optional[float]
    keystroke_to_pixel_ms: Optional[float]
    frames_to_applied: Optional[int]
    frames_to_pixel: Optional[int]
    unrestyled_frames: int
    detector_ticks_waited: int

    def to_dict(self) -> dict:
        return asdict(self)


def swap_timing(swap_frame: int, plan_version_before: int, plan_version_after: int,
                kind: str, validate_ms: float, accepted_to_applied_ms: Optional[float],
                accepted_to_pixel_ms: Optional[float], frames_to_applied: Optional[int],
                frames_to_pixel: Optional[int], unrestyled_frames: int,
                detector_ticks_waited: int,
                debounce_ms: float = GUI_DEBOUNCE_MS) -> SwapTiming:
    """The two derived clocks, computed in one place rather than at each reader."""
    worker = (None if accepted_to_pixel_ms is None
              else round(validate_ms + accepted_to_pixel_ms, 4))
    return SwapTiming(
        swap_frame=swap_frame, plan_version_before=plan_version_before,
        plan_version_after=plan_version_after, kind=kind,
        validate_ms=round(validate_ms, 4), debounce_ms=debounce_ms,
        accepted_to_applied_ms=(None if accepted_to_applied_ms is None
                                else round(accepted_to_applied_ms, 4)),
        accepted_to_pixel_ms=(None if accepted_to_pixel_ms is None
                              else round(accepted_to_pixel_ms, 4)),
        worker_ms=worker,
        keystroke_to_pixel_ms=(None if worker is None
                               else round(debounce_ms + worker, 4)),
        frames_to_applied=frames_to_applied, frames_to_pixel=frames_to_pixel,
        unrestyled_frames=unrestyled_frames,
        detector_ticks_waited=detector_ticks_waited,
    )


@dataclass(frozen=True)
class LatencyCheck:
    """Acceptance criterion 1, with the user's clock and the worker's beside it."""

    keystroke_to_pixel_ms: Optional[float]
    worker_ms: Optional[float]
    debounce_ms: float
    budget_ms: float
    passed: bool
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def latency_check(timing: SwapTiming,
                  budget_ms: float = CRITERION_1_BUDGET_MS) -> LatencyCheck:
    """Judged on the keystroke figure; the debounce is time the user waits too."""
    total = timing.keystroke_to_pixel_ms
    passed = total is not None and total <= budget_ms
    if total is None:
        statement = (f"the new instruction never reached the screen in the "
                     f"frames measured, so there is no latency to report against "
                     f"the {budget_ms / 1000.0:.2f} s the criterion allows")
    else:
        statement = (
            f"a typed instruction reached the screen {total / 1000.0:.2f} s after "
            f"the keystroke - {timing.debounce_ms:.0f} ms of GUI debounce, "
            f"{timing.validate_ms:.1f} ms to validate the plan and "
            f"{timing.accepted_to_pixel_ms:.0f} ms in the worker "
            f"({_frames(timing.frames_to_pixel)}) - against the "
            f"{budget_ms / 1000.0:.2f} s the criterion allows")
    return LatencyCheck(keystroke_to_pixel_ms=total, worker_ms=timing.worker_ms,
                        debounce_ms=timing.debounce_ms, budget_ms=budget_ms,
                        passed=passed, statement=statement)


# --- criterion 3 -------------------------------------------------------------


def intervals_of(finished_at_s: Sequence[float]) -> List[Optional[float]]:
    """The inter-frame interval series, in ms, from the frame completion times.

    `None` for the first frame, which has no interval before it. One reading of
    what an interval is, so the runner cannot grow a second.
    """
    return [None] + [round((later - earlier) * 1000.0, 4)
                     for earlier, later in zip(finished_at_s, finished_at_s[1:])]


@dataclass(frozen=True)
class Window:
    """One stretch of the interval series, summarised the same way as every other."""

    name: str
    frames: int
    mean_ms: Optional[float]
    worst_ms: Optional[float]
    over_budget: int

    def to_dict(self) -> dict:
        return asdict(self)


def _window(name: str, intervals: Sequence[Optional[float]], first: int, last: int,
            budget_ms: float) -> Window:
    """Frames `first`..`last` inclusive, skipping the frame that has no interval."""
    values = [interval for interval in intervals[max(0, first):last + 1]
              if interval is not None]
    return Window(
        name=name, frames=len(values),
        mean_ms=round(statistics.fmean(values), 4) if values else None,
        worst_ms=round(max(values), 4) if values else None,
        over_budget=sum(1 for value in values if value > budget_ms),
    )


@dataclass(frozen=True)
class StutterCheck:
    """Acceptance criterion 3's first half: did the swap cost the output stream?

    `control` is the worse of the two steady states either side of the swap, so
    the bar a stutter has to beat is the dearest normal frame in the same run and
    not whichever half flatters the verdict. `excess_ms` is what the swap actually
    cost the stream, which is the figure a stricter reading would re-judge.
    """

    swap: Window
    control: Window
    before: Window
    after: Window
    budget_ms: float
    allowance_ms: float
    excess_ms: Optional[float]
    worst_ratio: Optional[float]
    passed: bool
    statement: str

    def to_dict(self) -> dict:
        return {"swap": self.swap.to_dict(), "control": self.control.to_dict(),
                "before": self.before.to_dict(), "after": self.after.to_dict(),
                "budget_ms": self.budget_ms, "allowance_ms": self.allowance_ms,
                "excess_ms": self.excess_ms, "worst_ratio": self.worst_ratio,
                "passed": self.passed, "statement": self.statement}


def stutter_check(intervals: Sequence[Optional[float]], swap_index: int,
                  pixel_index: int, budget_ms: float = FRAME_BUDGET_MS,
                  allowance_ms: float = STUTTER_ALLOWANCE_MS,
                  warmup_frames: int = 0) -> StutterCheck:
    """The interval series across the swap, against the same run's steady state.

    The window across the swap runs from the frame the plan was submitted on to
    the frame its pixels arrived on: every frame the user is waiting through, and
    every frame the cold path could have cost the loop.
    """
    before = _window("steady state before", intervals, warmup_frames + 1,
                     swap_index - 1, budget_ms)
    across = _window("across the swap", intervals, swap_index, pixel_index, budget_ms)
    after = _window("steady state after", intervals, pixel_index + 1,
                    len(intervals) - 1, budget_ms)
    control = max([window for window in (before, after) if window.frames],
                  key=lambda window: window.worst_ms, default=None)
    if control is None or not across.frames:
        empty = Window(name="steady state", frames=0, mean_ms=None, worst_ms=None,
                       over_budget=0)
        return StutterCheck(
            swap=across, control=control or empty, before=before, after=after,
            budget_ms=budget_ms, allowance_ms=allowance_ms, excess_ms=None,
            worst_ratio=None, passed=False,
            statement="the run has no steady state either side of the swap to "
                      "judge it against, so no stutter verdict can be taken")
    ratio = round(across.worst_ms / control.worst_ms, 4) if control.worst_ms else None
    excess = round(across.worst_ms - control.worst_ms, 4)
    passed = excess <= allowance_ms
    verdict = (f"{excess:+.2f} ms, inside the {allowance_ms:.2f} ms a dropped frame "
               f"would cost" if passed else
               f"{excess:+.2f} ms, more than the {allowance_ms:.2f} ms one dropped "
               f"frame costs - the stream lost something to the swap")
    statement = (
        f"the worst inter-frame interval across the swap was {across.worst_ms:.2f} ms "
        f"over {_frames(across.frames)}, against {control.worst_ms:.2f} ms in the same "
        f"run's {control.name} ({ratio:.2f}x): {verdict}; "
        f"{across.over_budget}/{across.frames} frames across the swap were over the "
        f"{budget_ms:.2f} ms budget against {control.over_budget}/{control.frames} "
        f"in steady state")
    return StutterCheck(swap=across, control=control, before=before, after=after,
                        budget_ms=budget_ms, allowance_ms=allowance_ms,
                        excess_ms=excess, worst_ratio=ratio, passed=passed,
                        statement=statement)


# --- criterion 3's second half ----------------------------------------------


@dataclass(frozen=True)
class RebuildCheck:
    """Did the swap rebuild a TensorRT engine? A number, not an assumption.

    Two independent readings, because either alone could be fooled. The step
    *count* is what keys an engine, so a swap that moved it would have rebuilt
    whatever the objects say; and an engine object replaced under a schedule that
    happens to look the same is a rebuild the schedule cannot see.
    """

    engine_id_before: str
    engine_id_after: str
    unet_id_before: str
    unet_id_after: str
    steps_before: int
    steps_after: int
    t_index_before: List[int]
    t_index_after: List[int]
    engine_rebuilds: int
    schedule_moved: bool
    passed: bool
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def rebuild_check(engine_id_before: str, engine_id_after: str, unet_id_before: str,
                  unet_id_after: str, t_index_before: Sequence[int],
                  t_index_after: Sequence[int]) -> RebuildCheck:
    same_objects = (engine_id_before == engine_id_after
                    and unet_id_before == unet_id_after)
    steps_before, steps_after = len(t_index_before), len(t_index_after)
    rebuilt = not same_objects or steps_before != steps_after
    moved = list(t_index_before) != list(t_index_after)
    statement = (
        f"the schedule went from t_index {list(t_index_before)} to "
        f"{list(t_index_after)} on the same engine object, at {steps_before} step "
        f"either side - the step count is what keys an engine, and it did not move"
        if not rebuilt else
        f"the engine was rebuilt across the swap: {steps_before} step(s) and "
        f"{engine_id_before} became {steps_after} step(s) and {engine_id_after}")
    return RebuildCheck(
        engine_id_before=engine_id_before, engine_id_after=engine_id_after,
        unet_id_before=unet_id_before, unet_id_after=unet_id_after,
        steps_before=steps_before, steps_after=steps_after,
        t_index_before=list(t_index_before), t_index_after=list(t_index_after),
        engine_rebuilds=1 if rebuilt else 0, schedule_moved=moved,
        passed=not rebuilt, statement=statement)


# --- is the rule deciding, or is the noise? ---------------------------------


@dataclass(frozen=True)
class RepeatSpread:
    """How far apart two runs of one swap were, and whether that settles it.

    The question issue #24 asked of its 1 ms margin and issue #23 of its 3.33 ms
    one: a verdict inside the run-to-run spread is a spread, not a verdict. Both
    margins here are wide - seconds against 3 s, milliseconds against a whole
    frame budget - and `decisive` is what says so rather than a reader assuming
    it.
    """

    swaps: int
    repeats: int
    worst_latency_ms: float
    worst_interval_ms: float
    closest_latency_margin_ms: Optional[float]
    closest_stutter_margin_ms: Optional[float]
    decisive: bool
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def _spread(values: Sequence[float]) -> float:
    return max(values) - min(values) if len(values) > 1 else 0.0


def repeat_spread(results: Sequence[dict]) -> RepeatSpread:
    """Every committed run of each swap - all of them, not the newest."""
    latencies: Dict[str, List[float]] = {}
    intervals: Dict[str, List[float]] = {}
    latency_margins: List[float] = []
    stutter_margins: List[float] = []
    for result in results:
        name = result["case"]["name"]
        gate, timing = result["gate"], result["timing"]
        latency = timing["keystroke_to_pixel_ms"] or 0.0
        latencies.setdefault(name, []).append(latency)
        latency_margins.append(gate["criterion_1"]["budget_ms"] - latency)
        stutter = gate["criterion_3"]
        if stutter["excess_ms"] is not None:
            intervals.setdefault(name, []).append(stutter["swap"]["worst_ms"])
            stutter_margins.append(stutter["allowance_ms"] - stutter["excess_ms"])
    worst_latency = max((_spread(runs) for runs in latencies.values()), default=0.0)
    worst_interval = max((_spread(runs) for runs in intervals.values()), default=0.0)
    repeats = sum(len(runs) - 1 for runs in latencies.values())
    closest_latency = min(latency_margins, default=None)
    closest_stutter = min(stutter_margins, default=None)
    decisive = bool(repeats) and (
        closest_latency is not None and closest_latency > worst_latency
        and closest_stutter is not None and closest_stutter > worst_interval)
    if not repeats:
        statement = (f"each of the {len(latencies)} swaps was measured once, so "
                     f"there is no run-to-run spread to judge these verdicts "
                     f"against")
    else:
        outside = "outside" if decisive else "inside"
        statement = (
            f"each swap was measured twice and the two runs of one never differed "
            f"by more than {worst_latency:.0f} ms in keystroke-to-pixel or "
            f"{worst_interval:.2f} ms in the worst interval across the swap, "
            f"against margins of {(closest_latency or 0.0) / 1000.0:.2f} s and "
            f"{closest_stutter or 0.0:.2f} ms to the two thresholds - so both "
            f"verdicts are {outside} the run-to-run spread")
    return RepeatSpread(
        swaps=len(latencies), repeats=repeats,
        worst_latency_ms=round(worst_latency, 4),
        worst_interval_ms=round(worst_interval, 4),
        closest_latency_margin_ms=(None if closest_latency is None
                                   else round(closest_latency, 4)),
        closest_stutter_margin_ms=(None if closest_stutter is None
                                   else round(closest_stutter, 4)),
        decisive=decisive, statement=statement)


# --- the record --------------------------------------------------------------


@dataclass(frozen=True)
class SwapRunMetrics:
    """What the run cost, and what produced it.

    `regions_per_frame_before` / `_after` are here rather than in a region summary
    because they are the control on the control: two windows that rendered very
    different amounts of frame are two different steady states, and a reader
    comparing their intervals has to be able to see that.
    """

    started_utc: str
    finished_utc: str
    warmup_frames: int
    engine_scenario: str
    detector: Optional[str]
    frames: int
    diffusion_calls: int
    render: LatencySummary
    detect: Optional[LatencySummary]
    detect_every_n: int
    detector_ticks: int
    regions_per_frame_before: float
    regions_per_frame_after: float
    ms_per_frame: float
    mean_sm_clock_mhz: Optional[float]
    max_temperature_c: Optional[float]
    peak_vram_bytes: int
    gpu_samples: List[List[Optional[float]]] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["render"] = self.render.to_dict()
        data["detect"] = None if self.detect is None else self.detect.to_dict()
        return data


@dataclass(frozen=True)
class SwapResult:
    """One measured plan swap: criteria 1 and 3, and the evidence under them."""

    case: SwapCase
    plan_before: dict
    plan_after: dict
    clip: ClipRecord
    run: SwapRunMetrics
    timing: SwapTiming
    intervals_ms: List[Optional[float]]
    latency: LatencyCheck
    stutter: StutterCheck
    rebuild: RebuildCheck
    background: BackgroundCheck
    cooldown: CooldownRecord
    hardware: Fingerprint
    clock_normalization: Optional[ClockNormalization] = None
    comparison_clip: str = ""
    comparison_still: str = ""

    @property
    def gate_passed(self) -> bool:
        """Every Gate item a machine can answer. The manual one is not here."""
        return all([self.latency.passed, self.stutter.passed,
                    self.rebuild.passed, self.background.passed])

    def to_dict(self) -> dict:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "kind": RECORD_KIND,
            "case": self.case.to_dict(),
            "swap_kind": self.timing.kind,
            "plan_before": self.plan_before,
            "plan_after": self.plan_after,
            "clip": self.clip.to_dict(),
            "run": self.run.to_dict(),
            "timing": self.timing.to_dict(),
            "intervals_ms": list(self.intervals_ms),
            "gate": {
                "passed": self.gate_passed,
                "criterion_1": self.latency.to_dict(),
                "criterion_3": self.stutter.to_dict(),
                "rebuild": self.rebuild.to_dict(),
                "background": self.background.to_dict(),
            },
            "cooldown": self.cooldown.to_dict(),
            "hardware": self.hardware.to_dict(),
            "clock_normalization": (None if self.clock_normalization is None
                                    else self.clock_normalization.to_dict()),
            "comparison_clip": self.comparison_clip,
            "comparison_still": self.comparison_still,
        }


ResultLike = Union[SwapResult, dict]


def _as_dict(result: ResultLike) -> dict:
    return result.to_dict() if isinstance(result, SwapResult) else result


def swap_result_filename(case: str, timestamp: str) -> str:
    return f"{case}-{timestamp}.json"


def write_swap_result(result: ResultLike, results_dir: Path,
                      timestamp: Optional[str] = None) -> Path:
    """Write `<case>-<timestamp>.json` under `results_dir`; return its path."""
    data = _as_dict(result)
    if timestamp is None:
        timestamp = timestamp_from(data)
    return write_record(data, results_dir,
                        swap_result_filename(data["case"]["name"], timestamp))


README_TITLE = "# Plan swap results"
README_INTRO = (
    "Written by `uv run python -m bench <swap case>`, never by hand - issue #30,\n"
    "spec 8.9 and acceptance criteria 1 and 3. One run renders the committed clip\n"
    "under one instruction, submits a second instruction mid-clip through the\n"
    "shipped `ActivePlan`, and keeps rendering.\n\n"
    "`keystroke -> pixel` is criterion 1's figure and includes the GUI's 400 ms\n"
    "debounce; `worker` is the same measurement from where the worker accepts the\n"
    "plan, which is the half an optimisation would move. `worst across swap` is\n"
    "the longest inter-frame interval between the swap and the pixels, and it is\n"
    "read against `control worst` - the dearest frame in the same run's steady\n"
    "state - because 33.33 ms alone cannot say whether a swap cost anything.\n\n"
    "Absolute figures belong to the GPU in the row (spec 7.4).\n"
)
README_HEADER = (
    "| finished (UTC) | swap | GPU | kind | instruction | keystroke -> pixel (s) |"
    " worker (ms) | criterion 1 | worst across swap (ms) | control worst (ms) |"
    " over budget | criterion 3 | rebuilds | background | clip file | file |"
)
README_SEPARATOR = table_separator(README_HEADER)
README_NAME = "README.md"
README_PREAMBLE = (
    f"{README_TITLE}\n\n{README_INTRO}\n{README_HEADER}\n{README_SEPARATOR}\n"
)


def _instruction(result: Mapping) -> str:
    """What moved, in the words the case was written in."""
    before, after = result["case"]["before"], result["case"]["after"]
    if before["target"] != after["target"]:
        return f"target {before['target']} -> {after['target']}"
    return f"style, denoise {before['denoise']} -> {after['denoise']}"


def _frames(count: int) -> str:
    """`1 frame`, `11 frames` - said in three statements, spelt once."""
    return f"{count} frame{'' if count == 1 else 's'}"


def _seconds(ms: Optional[float]) -> str:
    return "-" if ms is None else f"{ms / 1000.0:.2f}"


def plan_swap_readme_row(result: dict, filename: str) -> str:
    gate, timing = result["gate"], result["timing"]
    swap, control = gate["criterion_3"]["swap"], gate["criterion_3"]["control"]
    return table_row([
        result["run"]["finished_utc"],
        result["case"]["name"],
        result["hardware"]["gpu_name"],
        result["swap_kind"],
        _instruction(result),
        _seconds(timing["keystroke_to_pixel_ms"]),
        format_number(timing["worker_ms"], 0),
        "MET" if gate["criterion_1"]["passed"] else "MISSED",
        format_number(swap["worst_ms"], 2),
        format_number(control["worst_ms"], 2),
        f"{swap['over_budget']}/{swap['frames']} vs "
        f"{control['over_budget']}/{control['frames']}",
        "MET" if gate["criterion_3"]["passed"] else "MISSED",
        str(gate["rebuild"]["engine_rebuilds"]),
        "identical" if gate["background"]["passed"] else "CHANGED",
        f"[{result['comparison_clip']}]({result['comparison_clip']})"
        if result.get("comparison_clip") else "-",
        f"[{filename}]({filename})",
    ])


def append_swap_readme_row(result: ResultLike, readme_path: Path,
                           filename: str) -> None:
    """Append this run's row, creating the table if this is the first run."""
    data = _as_dict(result)
    require_recordable(data)
    append_row(plan_swap_readme_row(data, filename), readme_path, README_PREAMBLE)


# --- the block spec 8.9 carries ----------------------------------------------


def load_swap_results(results_dir: Path) -> Dict[str, dict]:
    """Every plan-swap result under `results_dir`, keyed by filename."""
    return load_records(results_dir)


def latest_per_swap(results: Mapping[str, dict]) -> Dict[str, dict]:
    """One result per case *per GPU*: the most recently finished run of each."""
    return latest_per(results, lambda result: result["case"]["name"])


REPORT_HEADER = ("| swap | what moved | keystroke -> pixel | worker | frames to"
                 " pixel | criterion 1 | worst across swap | control worst |"
                 " over budget (swap / steady) | criterion 3 | rebuilds |"
                 " background |")


def _report_row(result: dict, column: GpuColumn) -> str:
    gate, timing = result["gate"], result["timing"]
    swap, control = gate["criterion_3"]["swap"], gate["criterion_3"]["control"]
    return column.row([
        result["case"]["name"],
        _instruction(result),
        f"{_seconds(timing['keystroke_to_pixel_ms'])} s",
        f"{format_number(timing['worker_ms'], 0)} ms",
        "-" if timing["frames_to_pixel"] is None else str(timing["frames_to_pixel"]),
        "MET" if gate["criterion_1"]["passed"] else "MISSED",
        f"{format_number(swap['worst_ms'], 2)} ms",
        f"{format_number(control['worst_ms'], 2)} ms",
        f"{swap['over_budget']}/{swap['frames']} vs "
        f"{control['over_budget']}/{control['frames']}",
        "MET" if gate["criterion_3"]["passed"] else "MISSED",
        str(gate["rebuild"]["engine_rebuilds"]),
        "identical" if gate["background"]["passed"] else "CHANGED",
    ], result)


def _measured_phrase(result: dict) -> str:
    clip, case = result["clip"], result["case"]
    return (f"{result['run']['engine_scenario']}, {clip['frames_used']} consecutive "
            f"frames of `{clip['name']}` resized to the app's "
            f"{case['canvas']}x{case['canvas']} capture canvas, the new instruction "
            f"submitted on frame {case['swap_frame']}")


def _preamble(results: Sequence[dict], gpus: Sequence[str]) -> str:
    measured = ", ".join(sorted({_measured_phrase(result) for result in results}))
    regimes = ", ".join(sorted({result["hardware"]["clock_lock"]["state"]
                                for result in results}))
    spanning = "" if len(gpus) == 1 else " Rows are one per swap per GPU."
    return (
        f"Measured on {', '.join(gpus)}, {measured}. Clocks {regimes}. The swap "
        f"travels the shipped cold path - `plan_from_fields`, `validate_plan`, "
        f"`ActivePlan.submit`, and the frame loop's own prompt re-encode, "
        f"`BackgroundDetector.follow` and `set_t_index_list` - so what is timed is "
        f"the work the worker does. It omits one hop the app has and this harness "
        f"does not: the `multiprocessing.Queue` between the GUI and the worker "
        f"process.{spanning}"
    )


def _worst_by(results: Sequence[dict], measure) -> dict:
    """The run a verdict has to survive: the slowest, dearest or least steady one."""
    return max(results, key=measure)


def _criterion_1_verdict(results: Sequence[dict], gpu: str,
                         gpus: Sequence[str]) -> str:
    met = all(result["gate"]["criterion_1"]["passed"] for result in results)
    slowest = _worst_by(results, lambda result:
                        result["timing"]["keystroke_to_pixel_ms"] or float("inf"))
    machine = f" on {gpu}" if len(gpus) > 1 else ""
    return (f"**Acceptance criterion 1 - a typed instruction takes effect within "
            f"3 s: {'MET' if met else 'MISSED'}{machine}.** The slower of the two "
            f"swaps is `{slowest['case']['name']}`, where "
            f"{slowest['gate']['criterion_1']['statement']}.")


def _criterion_3_verdict(results: Sequence[dict], gpu: str,
                         gpus: Sequence[str]) -> str:
    stutter = all(result["gate"]["criterion_3"]["passed"] for result in results)
    rebuilds = sum(result["gate"]["rebuild"]["engine_rebuilds"] for result in results)
    # By the excess, which is the quantity the verdict turns on. Ranking on the
    # ratio would put a cheap swap's 1.2x above a dear one's 1.1x of a far longer
    # frame, and name the wrong run as the one the criterion had to survive.
    worst = _worst_by(results, lambda result:
                      result["gate"]["criterion_3"]["excess_ms"] or 0.0)
    machine = f" on {gpu}" if len(gpus) > 1 else ""
    met = stutter and rebuilds == 0
    return (f"**Acceptance criterion 3 - swapping the instruction stutters "
            f"nothing and rebuilds nothing: {'MET' if met else 'MISSED'}"
            f"{machine}.** The least steady of the two swaps is "
            f"`{worst['case']['name']}`, where "
            f"{worst['gate']['criterion_3']['statement']}. Across both swaps there "
            f"were {rebuilds} TensorRT rebuilds: "
            f"{worst['gate']['rebuild']['statement']}.")


def _unrestyled_line(results: Sequence[dict]) -> Optional[str]:
    """What a swap costs when it does not cost milliseconds.

    The issue's fourth trap: a vocabulary change may cost the frame path nothing
    and cost the *output* several frames of unstyled capture, because the tracks
    are dropped the moment the plan changes and a frame with no boxes renders no
    diffusion call at all. Reporting only the millisecond half would call that
    free.
    """
    waiting = [result for result in results
               if result["timing"]["unrestyled_frames"]]
    if not waiting:
        return None
    told = ", ".join(
        f"`{result['case']['name']}` showed the capture untouched for "
        f"{result['timing']['unrestyled_frames']} of the "
        f"{result['timing']['frames_to_pixel']} frames it took to arrive "
        f"({result['timing']['detector_ticks_waited']} detects)"
        for result in waiting)
    return (f"What a swap costs when it does not cost milliseconds: {told}. A plan "
            f"change drops the tracks that were about the old concept, and a frame "
            f"with no boxes costs no diffusion call - which is why the frame path "
            f"across a vocabulary swap is *cheaper* than steady state rather than "
            f"dearer.")


def _background_line(results: Sequence[dict]) -> str:
    passed = all(result["gate"]["background"]["passed"] for result in results)
    worst = _worst_by(results, lambda result:
                      result["gate"]["background"]["worst_pixels_changed"])
    lead = ("Criterion 4 held across the swap too" if passed
            else "Criterion 4 did **not** hold across the swap")
    return f"{lead}: {worst['gate']['background']['statement']}."


def _artefact_line(results: Sequence[dict]) -> Optional[str]:
    clips = [result for result in results if result.get("comparison_clip")]
    if not clips:
        return None
    named = ", ".join(f"`{result['comparison_clip']}`" for result in clips)
    return (f"Manual verification artefacts, source | render: {named}. The still "
            f"beside each is the frame the new instruction first reached, which "
            f"is the frame there is anything to look at on.")


def _machine_sections(results: Sequence[dict], every_run: Sequence[dict],
                      gpus: Sequence[str]) -> List[str]:
    """Both verdicts, once per machine: a criterion is a claim about one card.

    `every_run` is all the committed runs rather than the newest per swap,
    because how repeatable a verdict is a fact about the repeats and not about
    the row.
    """
    sections: List[str] = []
    for gpu in gpus:
        runs = measured_on(results, gpu)
        sections.append(_criterion_1_verdict(runs, gpu, gpus))
        sections.append(_criterion_3_verdict(runs, gpu, gpus))
        unrestyled = _unrestyled_line(runs)
        if unrestyled:
            sections.append(unrestyled)
        sections.append(_background_line(runs))
        sections.append(sentence_case(
            repeat_spread(measured_on(every_run, gpu)).statement) + ".")
        artefacts = _artefact_line(runs)
        if artefacts:
            sections.append(artefacts)
    return sections


def format_swap_report(results: Mapping[str, dict]) -> str:
    """The measured block spec 8.9 carries for the plan-swap path.

    Generated from the committed JSON rather than transcribed, for the reason
    every other block in the spec is: a table pasted into Markdown drifts the
    moment a case is re-measured and nothing notices.
    """
    ordered = sorted(latest_per_swap(results).values(),
                     key=lambda result: (result["case"]["name"], gpu_of(result),
                                         result["run"]["finished_utc"]))
    if not ordered:
        return "no plan swap measured yet (issue #30)"

    gpus = distinct_gpus(ordered)
    column = GpuColumn.for_gpus(gpus)
    header = column.header(REPORT_HEADER)
    sections = [
        _preamble(ordered, gpus),
        "\n".join([header, table_separator(header)]
                  + [_report_row(result, column) for result in ordered]),
    ]
    return "\n\n".join(
        sections + _machine_sections(ordered, list(results.values()), gpus))
