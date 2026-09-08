"""What a denoising step costs, and which of the two ways to pay for it ships.

Issue #46. A user asked for the trade directly: more steps, better picture, fewer
frames per second, **chosen while the app is running**. The machinery for it exists
- a step-count change already sends `set_t_index_list`, `_control_transition` flags
`engine_swap`, and the worker tears the stream down and builds another - and the
control is simply hidden behind `SHOW["step_count"]`. What was never decided is how
the change is *paid for*, and the two ways are different products:

- **A pre-built ladder.** `use_denoising_batch` on, which is what the app ships and
  what every figure in this repo was measured under. The step count keys the engine
  (`engine_cache.unet_batch_size` is `frame_buffer_size x steps`), so each rung is
  its own ~5 GB build - but a rung that is already built is a *load*, and switching
  between two of them costs whatever that load costs. That figure is the Gate's
  second item and it is measured here rather than inferred from an `os.path.exists`.
- **`use_denoising_batch` off.** The batch is `frame_buffer_size` whatever the step
  count, so the count stops keying the engine entirely: any rung, no builds ever,
  uniformly slower.

The second route could not be measured at all until this issue: `wrapper.py` raised
`NotImplementedError("img2img mode must use denoising batch for now")`, and
upstream's own unbatched branch overwrites `init_noise` with the current frame's
latent, which is harmless for txt2img and wrong for img2img.
`wrapper.unbatched_predict_x0` is the route implemented as the batched path's
analogue; this module is the arithmetic over what it measured.

Three rules, each of them one of the issue's traps made executable.

- **The denoise is held across every arm.** More steps at the same `t_index` is not
  the same picture (the fifth trap), so every arm opens at the rung
  `t_index_for_denoise(case.denoise)` names and `render_plan.t_index_ladder` spends
  the rest after it - the same rule `bench.steps` applies for the same reason.
- **The quality figure is scored, never eyeballed.** Spec 8.2's identity probe: the
  fraction of rendered frames the open-vocabulary detector reads back as what the
  prompt asked for. Beside it sit `flicker` and `response`, because the batched
  route *pipelines* - a frame's later rungs are spent on earlier frames' latents -
  and a route that answers N-1 frames late is a quality difference no adherence
  score would show.
- **Bit-identity is a disqualifier.** The Gate's last item: whatever a step count
  does to the rendered region, a pixel nobody asked about is the captured byte.

GPU-free, like every other `bench.*` results module. `bench.quality_runner` is the
half that touches a GPU.
"""

from __future__ import annotations

import dataclasses
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from bench import RESULT_SCHEMA_VERSION
from bench.clocks import ClockNormalization
from bench.contention import OccupancyRecord
from bench.cooldown import CooldownRecord
from bench.fingerprint import Fingerprint
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
    table_row,
    table_separator,
    timestamp_from,
    write_record,
)
from bench.scenarios import SCENARIOS
from bench.selective import BackgroundCheck
from engine_cache import (
    ENGINE_BUILD_SIZE,
    ENGINE_BUILD_TIME,
    STEP_LADDER as LADDER_RUNGS,
    engine_dir_name,
    engine_is_cached,
    unet_batch_size,
)
# The app's own default strength, imported rather than restated: this sweep asks
# what more steps buy at the strength the app actually renders at, and it is the
# strength spec 8.11 swept guidance at - so the two blocks are about one baseline.
from render_plan import DEFAULT_DENOISE

RECORD_KIND = "quality"

# The two ways to pay for a runtime step count. Named, because the whole block is a
# choice between them and a reader should not have to infer which arm is which
# route from a boolean column.
ROUTE_LADDER = "pre-built ladder"
ROUTE_UNBATCHED = "unbatched"

# The rungs swept, on both routes - the window's own ladder, imported rather than
# spelt again. 1 is the shipped count and the control; 8 is past anything this app
# would ship and is in the sweep because a curve with three points cannot show
# where a route stops fitting the budget. A rung the window offers and the sweep
# never measured is a quality nobody priced.
STEP_LADDER: Tuple[int, ...] = tuple(LADDER_RUNGS)

# One frame at 30 FPS. The arms are timed with the detector out of the loop - the
# boxes come from a committed track - so an arm that lands near this line has not
# been shown to fit; spec 8.8 puts detection at ~4.3 ms/frame amortised on top.
FRAME_BUDGET_MS = 33.33
DETECTION_ALLOWANCE_MS = 4.3

# The confidence the identity probe requires, and the figure
# `bench.primitive_runner.IDENTITY_CONF` and `bench.guidance` already use, so "the
# detector saw it" means one thing in all three records.
ADHERENCE_CONF = 0.25

# How much more of the clip a deeper rung has to read back as the prompt's concept
# before more steps can be said to have bought anything. Ten points of 48 frames is
# five frames - the bar spec 8.11 set for the same kind of claim on the same probe.
MIN_ADHERENCE_GAIN = 0.10


def route_of(use_denoising_batch: bool) -> str:
    return ROUTE_LADDER if use_denoising_batch else ROUTE_UNBATCHED


@dataclass(frozen=True)
class StepSpec:
    """One point of the sweep: a step count, on one of the two routes."""

    steps: int
    use_denoising_batch: bool

    @property
    def name(self) -> str:
        return arm_name(self)

    @property
    def route(self) -> str:
        return route_of(self.use_denoising_batch)

    def to_dict(self) -> dict:
        return asdict(self)


def arm_name(spec: StepSpec) -> str:
    """`s4` on the batched route, `s4-nobatch` on the other one."""
    return f"s{int(spec.steps)}" + ("" if spec.use_denoising_batch else "-nobatch")


def ladder(steps: Sequence[int] = STEP_LADDER) -> Tuple[StepSpec, ...]:
    """Every rung on both routes, the shipped route first.

    Batched first so the record's first arm is the one every committed figure in
    this repo belongs to, and the artefact strip opens on it.
    """
    return tuple([StepSpec(count, True) for count in steps]
                 + [StepSpec(count, False) for count in steps])


@dataclass(frozen=True)
class QualityCase:
    """One clip, one prompt, one denoise, and the step arms to render it under.

    The identity case, and deliberately the one spec 8.11 swept guidance on: the
    two issues both move image quality, so sharing the clip, the prompt, the region
    and the denoise makes the second block a restatement of the first's baseline
    rather than a second baseline nobody can line up against it.
    """

    name: str
    clip: str
    # The TensorRT cell the arms render through - the shipped path, because the
    # question is which engines a release builds and that is a TensorRT question.
    base_scenario: str
    base_model: str
    concept: str
    reads_back_as: str
    region: str
    prompt: str
    denoise: float
    note: str
    arms: Tuple[StepSpec, ...] = ()
    frames: int = 48
    start_frame: int = 0
    canvas: int = 512
    max_instances: int = 1
    warmup_frames: int = 3

    def specs(self) -> Tuple[StepSpec, ...]:
        return self.arms or ladder()

    def plan(self):
        """The plan the arms render under, through the shipped producer and door.

        `bench.guidance.GuidanceCase.plan`'s shape, for its reason: a target and a
        style go in through `plan_from_fields` and what comes out is a plan the app
        itself could be put into. Nothing in it says anything about the step count -
        that is an engine setting, which is the whole point of the issue.
        """
        from render_plan import INITIAL_PLAN_VERSION, plan_from_fields, validate_plan

        built = plan_from_fields(target=self.concept, style=self.prompt,
                                 region=self.region, denoise=self.denoise)
        if built.plan is None:  # unreachable: an open-vocabulary concept and a region
            raise AssertionError(f"the case plan did not validate: {built.reason}")
        raw = built.plan.to_dict()
        for target in raw["targets"]:
            target["max_instances"] = self.max_instances
        result = validate_plan(raw, previous_version=INITIAL_PLAN_VERSION)
        if result.plan is None:  # unreachable: only a validated plan is edited here
            raise AssertionError(f"the arm plan did not validate: {result.reason}")
        return result.plan

    def replace(self, **changes) -> "QualityCase":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["arms"] = [spec.to_dict() for spec in self.specs()]
        return data


QUALITY_CASE = "steps-dog"

CASES: Dict[str, QualityCase] = {
    QUALITY_CASE: QualityCase(
        name=QUALITY_CASE,
        clip="dog.mp4",
        base_scenario="img2img-tensorrt-512x512-b1",
        base_model="sd-turbo",
        concept="dog",
        reads_back_as="cat",
        region="full_box",
        prompt="a cat, feline face, whiskers, pointed ears, photograph",
        denoise=DEFAULT_DENOISE,
        note="What does a second, fourth and eighth denoising step buy, and which "
             "of the two ways of paying for one should ship? The identity change "
             "is the sharpest quality probe this repo has - the detector reads the "
             "rendered region back and says whether it is a cat - and it is what "
             "one-step SD-Turbo is most likely to fail.",
    ),
}


@dataclass(frozen=True)
class EngineKeying:
    """Which TensorRT engine one arm needs, and whether this machine has it.

    Read from `engine_cache` - the app's own naming rule, shared with the window
    and the harness's build guard - so "cached" here is the directory the worker
    will look in a moment later rather than an approximation of it. This is also
    the whole difference between the two routes in one field: on the unbatched
    route `unet_batch` is the shipped 1 at every rung, so `keys_new_engine` is
    false all the way up the ladder.
    """

    unet_batch: int
    directory: str
    keys_new_engine: bool
    cached: Optional[bool]

    def to_dict(self) -> dict:
        return asdict(self)


def engine_keying(case: QualityCase, spec: StepSpec,
                  engines_root: Optional[Path] = None) -> EngineKeying:
    """The engine `spec` needs, against the one the shipped configuration uses."""
    scenario = SCENARIOS[case.base_scenario]
    shipped = unet_batch_size(scenario.batch_size, scenario.steps,
                              scenario.use_denoising_batch, scenario.cfg_type)
    batch = unet_batch_size(scenario.batch_size, spec.steps,
                            spec.use_denoising_batch, scenario.cfg_type)
    directory = engine_dir_name(
        scenario.model, use_lcm_lora=scenario.use_lcm_lora,
        use_tiny_vae=scenario.use_tiny_vae, unet_batch=batch,
        width=scenario.width, height=scenario.height, mode=scenario.mode)
    return EngineKeying(
        unet_batch=batch, directory=directory, keys_new_engine=batch != shipped,
        cached=(None if engines_root is None
                else engine_is_cached(engines_root, directory)))


@dataclass(frozen=True)
class QualityArm:
    """One arm: what it cost, what it bought, and how long it took to switch to.

    `swap_seconds` is the Gate's second item measured rather than inferred: the
    seconds between letting the previous arm's engine go and having this one ready
    to render, which is exactly what `image_generation_process` does when a step
    count changes. It is only a *swap* figure for an arm whose engine was already
    on disk, which is what `engine_cached` says and what `swap_summary` filters on.
    """

    steps: int
    use_denoising_batch: bool
    t_index_list: List[int]
    unet_batch: int
    engine_dir: str
    engine_cached: bool
    keys_new_engine: bool
    swap_seconds: float
    ms_per_frame: float
    adherence_hits: int
    adherence_frames: int
    adherence_conf: float
    retained_hits: int
    region_change: float
    control_change: float
    flicker: float
    response: float
    background: BackgroundCheck
    frames: int
    loaded: bool = True
    error: Optional[str] = None

    @property
    def spec(self) -> StepSpec:
        return StepSpec(self.steps, self.use_denoising_batch)

    @property
    def name(self) -> str:
        return arm_name(self.spec)

    @property
    def route(self) -> str:
        return route_of(self.use_denoising_batch)

    @property
    def measured(self) -> bool:
        """Did this arm produce numbers at all? A refusal is not a zero score."""
        return self.loaded and self.adherence_frames > 0

    @property
    def adherence(self) -> float:
        if not self.adherence_frames:
            return 0.0
        return round(self.adherence_hits / self.adherence_frames, 4)

    @property
    def net_change(self) -> float:
        """How far the region drifted from the capture, net of its round trip."""
        return round(max(0.0, self.region_change - self.control_change), 4)

    @property
    def fits_budget(self) -> bool:
        """Does this rung leave room for the detector inside one 30 FPS frame?

        The allowance is spec 8.8's own amortised figure rather than zero: these
        arms render a committed track, and an arm that fills the budget exactly has
        not been shown to fit the path the app runs.
        """
        return (self.measured
                and self.ms_per_frame + DETECTION_ALLOWANCE_MS <= FRAME_BUDGET_MS)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["name"] = self.name
        data["route"] = self.route
        data["adherence"] = self.adherence
        data["net_change"] = self.net_change
        data["fits_budget"] = self.fits_budget
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> "QualityArm":
        fields = {field.name: data[field.name] for field in dataclasses.fields(cls)}
        fields["background"] = BackgroundCheck(**dict(fields["background"]))
        return cls(**fields)


def arms_of(result: Mapping) -> List[QualityArm]:
    return [QualityArm.from_dict(arm) for arm in result["arms"]]


def on_route(arms: Sequence[QualityArm], route: str) -> List[QualityArm]:
    return sorted((arm for arm in arms if arm.route == route),
                  key=lambda arm: arm.steps)


def qualified(arms: Sequence[QualityArm]) -> List[QualityArm]:
    """The arms anything may be concluded from, and the rule in one place.

    Two disqualifiers, and each is a disqualifier rather than a caveat. An arm that
    never ran cannot be recommended. Neither can one that painted a pixel outside
    the rendered region: that is the selective path's whole promise, and an arm
    that broke it is a different renderer however good the picture inside is.
    """
    return [arm for arm in arms if arm.measured and arm.background.passed]


# --- the cached-engine swap (the Gate's second item) -------------------------


@dataclass(frozen=True)
class SwapSummary:
    """How long switching to an already-built engine actually took.

    The issue's third trap in one object: "seconds, not minutes" was an inference
    from `wrapper.py` building only `if not os.path.exists(unet_path)`, and the
    window is about to quote it. Measured over every arm whose engine was on disk
    before the run reached it - which on the unbatched route is every rung, since
    they all share the shipped engine.
    """

    arms: int
    mean_seconds: float
    worst_seconds: float
    uncached_arms: int
    statement: str

    @property
    def measured(self) -> bool:
        return self.arms > 0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["measured"] = self.measured
        return data


def swap_summary(arms: Sequence[QualityArm]) -> SwapSummary:
    """Read the swap cost off the arms whose engine already existed.

    An arm that had to compile its engine is left out rather than averaged in: it
    measures a build, which is the *other* number and one this repo already quotes.
    """
    cached = [arm for arm in arms if arm.measured and arm.engine_cached
              and arm.swap_seconds > 0.0]
    uncached = len([arm for arm in arms if arm.measured and not arm.engine_cached])
    if not cached:
        return SwapSummary(
            arms=0, mean_seconds=0.0, worst_seconds=0.0, uncached_arms=uncached,
            statement=("No arm of this run found its engine already built, so the "
                       "cost of switching between two cached step counts is not "
                       "measured here."))
    mean = statistics.fmean(arm.swap_seconds for arm in cached)
    worst = max(arm.swap_seconds for arm in cached)
    return SwapSummary(
        arms=len(cached), mean_seconds=round(mean, 3), worst_seconds=round(worst, 3),
        uncached_arms=uncached,
        statement=(
            f"Switching to a step count whose engine is already built took "
            f"**{mean:.1f} s** on average and {worst:.1f} s at worst, over "
            f"{len(cached)} arms - the whole of it: letting the previous engine go, "
            f"loading the new one and preparing it, which is exactly what the "
            f"worker does when a step count changes. That is the figure the window "
            f"may quote; the {ENGINE_BUILD_TIME} and {ENGINE_BUILD_SIZE} it warns "
            f"about belong to a count that has no engine yet."))


# --- what more steps bought --------------------------------------------------


@dataclass(frozen=True)
class StepGain:
    """Whether a deeper rung produced a better picture on one route.

    The question the whole feature rests on: a control that trades frame rate for
    nothing is worse than no control. Scored on the identity probe, against the
    same route's own one-step arm, so what is compared is the step count and not
    the route.
    """

    route: str
    control_adherence: float
    best_arm: str
    best_steps: int
    best_adherence: float
    gain: float
    worthwhile: bool
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def step_gain(arms: Sequence[QualityArm], route: str) -> Optional[StepGain]:
    """What the deepest-scoring rung on `route` bought over that route's one step."""
    on = [arm for arm in qualified(arms) if arm.route == route]
    control = next((arm for arm in on if arm.steps == 1), None)
    deeper = [arm for arm in on if arm.steps > 1]
    if control is None or not deeper:
        return None
    best = max(deeper, key=lambda arm: (arm.adherence, arm.adherence_conf))
    gain = round(best.adherence - control.adherence, 4)
    worthwhile = gain >= MIN_ADHERENCE_GAIN
    verdict = ("worth the frame rate it costs" if worthwhile else
               f"under the {MIN_ADHERENCE_GAIN:.0%} bar, so more steps have not "
               f"been shown to buy a better picture on this case")
    return StepGain(
        route=route, control_adherence=control.adherence, best_arm=best.name,
        best_steps=best.steps, best_adherence=best.adherence, gain=gain,
        worthwhile=worthwhile,
        statement=(
            f"On the {route} route the best deeper rung is `{best.name}`, which "
            f"reads back as a {best.adherence:.0%} of frames against one step's "
            f"{control.adherence:.0%} - {gain:+.0%}, {verdict}."))


# --- the recommendation (step 4) ---------------------------------------------


@dataclass(frozen=True)
class RouteRecommendation:
    """Which of the two ways to pay for a runtime step count ships, and why.

    Decided from the arms, on two numbers a reader can recompute: how deep a rung
    each route can still render inside the frame budget, and what each route costs
    to make that rung reachable - which for the ladder is one ~5 GB engine per rung
    and for the unbatched route is nothing at all.
    """

    route: str
    deepest_affordable_steps: int
    deepest_affordable_ms: float
    other_route: str
    other_deepest_steps: int
    other_deepest_ms: float
    engines_to_build: int
    rungs_shipped: List[int]
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def _deepest_affordable(arms: Sequence[QualityArm],
                        route: str) -> Optional[QualityArm]:
    """The deepest rung on `route` that still fits one 30 FPS frame."""
    affordable = [arm for arm in qualified(arms)
                  if arm.route == route and arm.fits_budget]
    return max(affordable, key=lambda arm: arm.steps) if affordable else None


def rungs_to_build(arms: Sequence[QualityArm], up_to_steps: int) -> List[int]:
    """The batched rungs a release would compile to reach `up_to_steps`.

    Step 5 of the issue, answered from the arms rather than chosen: every rung on
    the ladder route up to and including the deepest affordable one keys its own
    engine, and the one-step engine is the one the app already ships.
    """
    return sorted(arm.steps for arm in on_route(arms, ROUTE_LADDER)
                  if arm.steps <= up_to_steps)


def recommend_route(arms: Sequence[QualityArm]) -> Optional[RouteRecommendation]:
    """Which route ships. None when one of the two was not measured at all.

    The rule, in the order it decides: the route that can render the *deeper* rung
    inside the frame budget wins, because the whole feature is reaching for a
    better picture and a route that cannot get there does not offer the trade. A
    tie breaks towards the route that compiles nothing, since the ladder's rungs
    are ~5 GB and minutes apiece and a tie is not worth them.
    """
    ladder_arm = _deepest_affordable(arms, ROUTE_LADDER)
    unbatched_arm = _deepest_affordable(arms, ROUTE_UNBATCHED)
    if ladder_arm is None or unbatched_arm is None:
        return None
    # `>=`, so a tie goes to the route with no builds - that is the tie-break,
    # spelt in the comparison rather than in a branch below it.
    if unbatched_arm.steps >= ladder_arm.steps:
        winner, loser = unbatched_arm, ladder_arm
    else:
        winner, loser = ladder_arm, unbatched_arm
    if winner.route == ROUTE_UNBATCHED:
        rungs: List[int] = []
        cost = (f"and it compiles nothing at all: with `use_denoising_batch` off "
                f"the UNet batch is {winner.unet_batch} at every rung, so every "
                f"step count runs on the engine the app already ships")
    else:
        rungs = rungs_to_build(arms, winner.steps)
        keyed = [arm.steps for arm in on_route(arms, ROUTE_LADDER)
                 if arm.steps in rungs and arm.keys_new_engine]
        cost = (f"and what it costs is {len(keyed)} more engines - one per rung "
                f"past the shipped one step, {ENGINE_BUILD_SIZE} and "
                f"{ENGINE_BUILD_TIME} each, at "
                f"{', '.join(str(step) for step in rungs)} steps")
    return RouteRecommendation(
        route=winner.route, deepest_affordable_steps=winner.steps,
        deepest_affordable_ms=winner.ms_per_frame, other_route=loser.route,
        other_deepest_steps=loser.steps, other_deepest_ms=loser.ms_per_frame,
        engines_to_build=len([step for step in rungs
                              if any(arm.steps == step and arm.keys_new_engine
                                     for arm in on_route(arms, ROUTE_LADDER))]),
        rungs_shipped=rungs,
        statement=(
            f"**The {winner.route} route ships.** It renders {winner.steps} "
            f"denoising steps at {winner.ms_per_frame:.2f} ms/frame - inside a "
            f"{FRAME_BUDGET_MS:.2f} ms budget with the "
            f"{DETECTION_ALLOWANCE_MS:.1f} ms spec 8.8 amortises for detection on "
            f"top - against the {loser.route} route's {loser.steps} steps at "
            f"{loser.ms_per_frame:.2f} ms, {cost}."))


# --- the record ---------------------------------------------------------------


@dataclass(frozen=True)
class QualityResult:
    """One run: every step arm over the same frames of one clip, and the artefact."""

    case: QualityCase
    clip: ClipRecord
    arms: Tuple[QualityArm, ...]
    started_utc: str
    finished_utc: str
    cooldown: CooldownRecord
    occupancy: Optional[OccupancyRecord]
    hardware: Fingerprint
    clock_normalization: Optional[ClockNormalization]
    comparison_still: str = ""
    comparison_clip: str = ""

    def to_dict(self) -> dict:
        recommendation = recommend_route(self.arms)
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "kind": RECORD_KIND,
            "case": self.case.to_dict(),
            "clip": asdict(self.clip),
            "arms": [arm.to_dict() for arm in self.arms],
            "swap": swap_summary(self.arms).to_dict(),
            "gains": [gain.to_dict() for gain in
                      (step_gain(self.arms, route)
                       for route in (ROUTE_LADDER, ROUTE_UNBATCHED))
                      if gain is not None],
            "recommendation": (None if recommendation is None
                               else recommendation.to_dict()),
            # `run` so the shared readers - `latest_per`, `timestamp_from`,
            # `require_recordable` - find the finish time where they find it in
            # every other record shape.
            "run": {"started_utc": self.started_utc,
                    "finished_utc": self.finished_utc},
            "cooldown": self.cooldown.to_dict(),
            "occupancy": None if self.occupancy is None else self.occupancy.to_dict(),
            "hardware": self.hardware.to_dict(),
            "clock_normalization": (None if self.clock_normalization is None
                                    else self.clock_normalization.to_dict()),
            "comparison_still": self.comparison_still,
            "comparison_clip": self.comparison_clip,
        }


def write_quality_result(result: QualityResult, results_dir: Path,
                         timestamp: Optional[str] = None) -> Path:
    data = result.to_dict()
    timestamp = timestamp_from(data) if timestamp is None else timestamp
    return write_record(data, results_dir, f"{result.case.name}-{timestamp}.json")


def load_quality_results(results_dir: Path) -> Dict[str, dict]:
    return load_records(Path(results_dir))


QUALITY_README_NAME = "README.md"
QUALITY_README_TITLE = "# Denoising steps: what one buys, and how it is paid for"
QUALITY_README_PREAMBLE_TEXT = (
    "Written by `uv run python -m bench steps-dog`, never by hand. Issue #46: the\n"
    "same clip rendered at 1 / 2 / 4 / 8 denoising steps on both routes - the\n"
    "pre-built ladder (`use_denoising_batch` on, one engine per rung) and the\n"
    "unbatched one (off, one engine for every rung).\n\n"
    "`adherence` is the fraction of rendered frames the detector reads back as what\n"
    "the prompt asked for; `swap s` is how long that arm took to become the live\n"
    "engine, which is a *load* wherever `cached` says yes.\n"
)
QUALITY_README_HEADER = ("| finished (UTC) | case | GPU | arm | route | steps |"
                         " UNet batch | cached | swap s | ms/frame | adherence |"
                         " net change | flicker | response | background | file |")
QUALITY_README_SEPARATOR = table_separator(QUALITY_README_HEADER)


def append_quality_readme_rows(result: QualityResult, readme_path: Path,
                               filename: str) -> None:
    """One row per arm, the one-step control included - it is the yardstick."""
    data = result.to_dict()
    require_recordable(data)
    preamble = (f"{QUALITY_README_TITLE}\n\n{QUALITY_README_PREAMBLE_TEXT}\n"
                f"{QUALITY_README_HEADER}\n{QUALITY_README_SEPARATOR}\n")
    for arm in result.arms:
        append_row(_readme_row(result, arm, filename), readme_path, preamble)


def _readme_row(result: QualityResult, arm: QualityArm, filename: str) -> str:
    return table_row([
        result.finished_utc,
        result.case.name,
        result.hardware.gpu_name,
        arm.name,
        arm.route,
        str(arm.steps),
        str(arm.unet_batch),
        "yes" if arm.engine_cached else "no",
        format_number(arm.swap_seconds, 2),
        format_number(arm.ms_per_frame, 2),
        f"{arm.adherence:.0%}" if arm.measured else "-",
        format_number(arm.net_change, 2),
        format_number(arm.flicker, 2),
        format_number(arm.response, 2),
        f"{arm.background.identical_frames}/{arm.background.frames}",
        f"[{filename}]({filename})",
    ])


# --- the block spec 8.12 carries ---------------------------------------------


REPORT_HEADER = ("| arm | route | steps | t_index list | UNet batch | engine |"
                 " swap s | ms/frame | FPS | adherence | conf | net change |"
                 " flicker | response | background |")


def _row(arm: QualityArm, column: GpuColumn, result: dict) -> str:
    engine = "cached" if arm.engine_cached else "**built**"
    return column.row([
        f"`{arm.name}`",
        arm.route,
        str(arm.steps),
        ",".join(str(index) for index in arm.t_index_list),
        str(arm.unet_batch),
        engine,
        format_number(arm.swap_seconds, 1),
        format_number(arm.ms_per_frame, 2),
        format_number(1000.0 / arm.ms_per_frame, 1) if arm.ms_per_frame else "-",
        f"{arm.adherence:.0%}" if arm.measured else "did not run",
        format_number(arm.adherence_conf, 2),
        format_number(arm.net_change, 2),
        format_number(arm.flicker, 2),
        format_number(arm.response, 2),
        f"{arm.background.identical_frames}/{arm.background.frames}",
    ], result)


def _preamble(result: Mapping, gpus: Sequence[str]) -> str:
    case, clip = result["case"], result["clip"]
    return (
        f"{len(result['arms'])} arms over {clip['frames_used']} frames of "
        f"`{clip['name']}` at {case['canvas']}x{case['canvas']} through "
        f"`{case['base_scenario']}`, on {', '.join(gpus)}. Every arm renders the "
        f"same frames of the same committed box track at the same denoise "
        f"({case['denoise']}) under the same prompt (\"{case['prompt']}\"); only "
        f"the step count and the route move. The opening schedule index is the one "
        f"that denoise names and `render_plan.t_index_ladder` spends the extra "
        f"steps after it, so a deeper arm is not also a strength change. "
        f"`adherence` is the fraction of rendered frames the detector reads back "
        f"as `{case['reads_back_as']}`; `response` is the flicker metric over the "
        f"pixels that *moved* in the source, which is where the batched route's "
        f"pipelining shows. Detection is not in the millisecond figures - the "
        f"boxes come from the track - so a rung is judged against "
        f"{FRAME_BUDGET_MS:.2f} ms less the {DETECTION_ALLOWANCE_MS:.1f} ms spec "
        f"8.8 amortises for it."
    )


def _machine_section(result: dict, prefix: str) -> str:
    """One run's swap figure, what each route's steps bought, and the artefact."""
    arms = arms_of(result)
    lines = [f"{prefix}{swap_summary(arms).statement}"]
    for route in (ROUTE_LADDER, ROUTE_UNBATCHED):
        gain = step_gain(arms, route)
        if gain is not None:
            lines.append(gain.statement)
    broke = [arm for arm in arms if arm.measured and not arm.background.passed]
    if broke:
        lines.append(
            f"Disqualified for painting outside the rendered region: "
            f"{', '.join(f'`{arm.name}`' for arm in broke)}.")
    else:
        frames = sum(arm.background.frames for arm in arms if arm.measured)
        lines.append(f"Background bit-identity held at every arm: {frames}/{frames} "
                     f"frames left every pixel outside the rendered region exactly "
                     f"as captured.")
    artefacts = [f"`{result[key]}`" for key in ("comparison_still", "comparison_clip")
                 if result.get(key)]
    if artefacts:
        lines.append(f"The artefact a human judges this by, source | one panel per "
                     f"rung: {', '.join(artefacts)}.")
    return "\n\n".join(lines)


def format_quality_report(results: Mapping[str, dict]) -> str:
    """The measured block spec 8.12 carries, from the committed step-quality runs."""
    reduced = latest_per(results, lambda result: result["case"]["name"])
    ordered = sorted(reduced.values(),
                     key=lambda result: (result["case"]["name"], gpu_of(result)))
    if not ordered:
        return "no step-quality sweep committed yet (issue #46)"

    gpus = distinct_gpus(ordered)
    column = GpuColumn.for_gpus(gpus)
    header = column.header(REPORT_HEADER)
    sections = [_preamble(ordered[0], gpus)]
    rows = [_row(arm, column, result)
            for result in ordered for arm in arms_of(result)]
    sections.append("\n".join([header, table_separator(header)] + rows))
    for gpu in gpus:
        for result in measured_on(ordered, gpu):
            sections.append(_machine_section(
                result, "" if len(gpus) == 1 else f"{gpu}: "))
            recommendation = recommend_route(arms_of(result))
            if recommendation is not None:
                sections.append(recommendation.statement)
    return "\n\n".join(sections)
