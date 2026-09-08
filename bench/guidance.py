"""Is the weak prompt adherence the model, or a setting that has never been on?

Issue #45. The app runs with classifier-free guidance switched off - `cfg_type`
`none`, `guidance_scale` 0.0, `delta` 0.0, and all three hidden behind a `SHOW`
flag - and every benchmark this repo has ever written was measured there. CFG is
the mechanism by which a prompt pulls the output towards itself, so "followed, but
not enough" is the *expected* behaviour of that configuration rather than a
limitation of the checkpoint. This module is the arithmetic that settles which.

Four rules, each of them one of the issue's traps made executable.

- **Adherence is scored, never eyeballed.** §8.2's `identity-dog` case already
  reads identity back with the open-vocabulary detector, and this reuses that shape
  exactly: the arm's number is the fraction of rendered frames the detector reads
  back as the thing the prompt asked for, with the detector's mean confidence
  beside it because a fraction over 48 frames saturates and a confidence does not.
- **A stronger pull is not free.** The same axis that lands the prompt drives the
  output away from the captured frame - §8.2's trade, and the one the live session
  hit. Every arm carries both figures and the recommendation quotes both, because
  one of them is half an answer.
- **`delta` is meaningless for some cfg types.** `noise_pred_uncond = stock_noise *
  delta` sits under `cfg_type in ("self", "initialize")` in the pipeline and
  nowhere else, so `uses_delta` is what decides whether an arm sweeps it. Reporting
  a flat line from an argument that is never read would be reporting nothing.
- **The denoise is held fixed across the sweep.** CFG interacts with it, so an arm
  that moved both would be measuring two changes at once. The case carries one
  `denoise` and every arm renders at it.

And one thing the Gate asks for by name: **which arms key a new engine**. Two of
the four cfg types run extra latents through the UNet - `initialize` one more per
step batch, `full` a second copy of every one - so under TensorRT they are a ~5 GB
build and not a setting. `engine_cache.unet_batch_size` is that rule, shared with
the window and the harness's build guard rather than restated here.

GPU-free. `bench.guidance_runner` is the half that touches a GPU.
"""

from __future__ import annotations

import dataclasses
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
    OptionalColumn,
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
    CFG_INITIALIZE,
    CFG_NONE,
    CFG_SELF,
    CFG_TYPES,
    ENGINE_BUILD_SIZE,
    ENGINE_BUILD_TIME,
    engine_dir_name,
    engine_is_cached,
    unet_batch_size,
)
# The app's own default strength. Imported rather than restated: the sweep is
# about the setting the complaint was made at, and a copy of 0.45 here would go on
# meaning 0.45 after the app moved.
from render_plan import DEFAULT_DENOISE

RECORD_KIND = "guidance"

# The arm every other arm is measured against: the setting the app ships, which is
# guidance switched off entirely. `prepare()` forces `guidance_scale` to 1.0 under
# it, so there is exactly one `none` arm and no ladder to sweep on it.
CONTROL_ARM = CFG_NONE

# The confidence the adherence probe requires before it will say the render reads
# back as what the prompt asked for. `bench.primitive_runner.IDENTITY_CONF`'s
# figure, so "the detector saw it" means one thing in both records.
ADHERENCE_CONF = 0.25

# How much more of the clip an arm has to read back as the prompt's own concept,
# against the control, before it is worth moving a default for. Ten points of 48
# frames is five frames: below that the axis has not been shown to do the thing it
# exists to do, and the shipped setting stands.
MIN_ADHERENCE_GAIN = 0.10

# And what an arm has to gain when adopting it means compiling a new engine.
# Two and a half times the free bar, because a build is a decision of a different
# size: ~5 GB, a compile the user watches with the window gone quiet, and whatever
# the extra latents cost the frame path - which on the shipped selective budget is
# most of the headroom issue #31 bought. The repo's own instinct for a lever that
# spends the budget is that it has to fit with room to spare (issue #23's cadence
# rule); this is that instinct applied to an engine.
MIN_ADHERENCE_GAIN_FOR_BUILD = 0.25

# Arms this close in adherence are a tie, and a tie breaks towards the cheaper
# engine. Five points is two frames of 48 - inside what one more or less confident
# detect would move.
ADHERENCE_TIE = 0.05

# And how much dearer than the control an arm may be and still be a *default*.
# A ratio rather than a millisecond figure, because these arms are measured on the
# `none` accelerator and the shipped path is TensorRT - an absolute budget read off
# one would be the wrong number for the other. The figure comes from a committed
# measurement: spec 8.8 puts the shipped selective path at 23.75 ms/frame with
# detection against a 33.33 ms budget, so the frame path has about 40% of itself in
# hand before 30 FPS is missed. An arm dearer than that cannot be the default
# whatever it reads back, because the default has to run.
MAX_COST_RATIO = 1.4


def uses_delta(cfg_type: str) -> bool:
    """Does this cfg type read `delta` at all?

    The pipeline's own condition: `noise_pred_uncond = self.stock_noise *
    self.delta` runs under `self` and `initialize`. Under `full` the unconditional
    prediction comes out of the doubled batch instead and `delta` is stored and
    never read; under `none` nothing reads anything. Sweeping it where it is
    ignored and reporting the flat line is the issue's third trap.
    """
    return cfg_type in (CFG_SELF, CFG_INITIALIZE)


@dataclass(frozen=True)
class ArmSpec:
    """One point of the sweep: a cfg type, a guidance scale, and a delta."""

    cfg_type: str
    guidance_scale: float
    delta: float

    @property
    def name(self) -> str:
        return arm_name(self)

    def to_dict(self) -> dict:
        return asdict(self)


def arm_name(spec: ArmSpec) -> str:
    """An arm's name: every setting that moved it, and none that did not.

    `delta` is in the name only where the pipeline reads it, so two `full` arms at
    two deltas cannot appear as two rows measuring the same thing.
    """
    if spec.cfg_type == CFG_NONE:
        return CONTROL_ARM
    parts = [spec.cfg_type, f"g{int(round(spec.guidance_scale * 10)):02d}"]
    if uses_delta(spec.cfg_type):
        parts.append(f"d{int(round(spec.delta * 10)):02d}")
    return "-".join(parts)


def ladder(guidance_scales: Sequence[float], deltas: Sequence[float],
           cfg_types: Sequence[str] = CFG_TYPES) -> Tuple[ArmSpec, ...]:
    """The sweep: the control, then every cfg type over the guidance ladder.

    The **last** delta is the one every guidance arm carries - the pipeline's own
    default, so the guidance ladder is a ladder in guidance and nothing else - and
    the rest are swept at one middle rung, and only under a cfg type that reads
    delta at all. A full cross of the two axes would be a run nobody waits for,
    and what a delta arm answers is whether the argument moves anything, which one
    rung answers as well as six.
    """
    default_delta = deltas[-1] if deltas else 1.0
    probe = guidance_scales[len(guidance_scales) // 2] if guidance_scales else 1.0
    specs: List[ArmSpec] = [ArmSpec(CFG_NONE, 1.0, default_delta)]
    for cfg_type in cfg_types:
        if cfg_type == CFG_NONE:
            continue
        for scale in guidance_scales:
            specs.append(ArmSpec(cfg_type, scale, default_delta))
        if uses_delta(cfg_type):
            for delta in deltas[:-1]:
                specs.append(ArmSpec(cfg_type, probe, delta))
    return tuple(specs)


def showcase_specs(specs: Sequence[ArmSpec]) -> Tuple[ArmSpec, ...]:
    """The arms the comparison strip shows: what a human has to look at.

    Every arm would be a twenty-panel strip nobody can read, so the strip is one
    panel per cfg type at the ladder's lowest rung - where all four still produce
    a coherent frame and the panels differ in the cfg type and nothing else - plus
    the *top* rung of one of them. The last panel is there on purpose: the useful
    range on a short LCM schedule is narrow, and a strip taken only where guidance
    works is a picture of the good news.
    """
    chosen: Dict[str, ArmSpec] = {}
    for spec in specs:
        if spec.cfg_type in chosen:
            continue
        if spec.cfg_type == CFG_NONE or spec == _showcase_spec(spec.cfg_type):
            chosen[spec.cfg_type] = spec
    extreme = _extreme_spec(CFG_SELF)
    return tuple(chosen.values()) + ((extreme,) if extreme in specs else ())


def _showcase_spec(cfg_type: str) -> ArmSpec:
    """The rung the strip is taken at: the lowest one that turns guidance on."""
    return ArmSpec(cfg_type, GUIDANCE_LADDER[0], DELTA_LADDER[-1])


def _extreme_spec(cfg_type: str) -> ArmSpec:
    """The rung that shows what over-guiding a short schedule looks like."""
    return ArmSpec(cfg_type, GUIDANCE_LADDER[-1], DELTA_LADDER[-1])


# The ladder issue #45 sweeps. Narrow on purpose: guidance on a 1-4 step LCM
# schedule is not the same animal as on a 50-step one, and the issue's fifth trap
# asks for the range actually tried to be stated rather than implied. Below 1.0 the
# pipeline disables guidance entirely (`if self.guidance_scale > 1.0`), so 1.1 is
# the first rung that does anything at all.
GUIDANCE_LADDER: Tuple[float, ...] = (1.05, 1.1, 1.2, 1.4, 2.0, 3.0)
# 1.0 is the pipeline's own default and the rung every guidance arm carries; 0.5 is
# the probe that says whether the argument moves anything.
DELTA_LADDER: Tuple[float, ...] = (0.5, 1.0)


@dataclass(frozen=True)
class GuidanceCase:
    """One clip, one prompt, one denoise, and the cfg arms to render it under.

    `denoise` is a single number and not a ladder: CFG interacts with it, so an arm
    that moved both would be measuring two changes at once (the issue's fourth
    trap). `reads_back_as` is what the prompt asks the region to become and what
    the detector is asked for - the adherence score is that question, answered
    frame by frame.
    """

    name: str
    clip: str
    base_scenario: str
    # The TensorRT cell this case's arms *would* ship on. The arms themselves run
    # on `none` - a cfg type is a constructor argument and two of the four key a
    # ~5 GB build, so sweeping them on the accelerated path would be four builds
    # to answer a question about whether the axis does anything at all. This is
    # what `engine_keying` names, so the Gate's engine list is about the path the
    # app runs rather than the path the sweep does.
    engine_scenario: str
    base_model: str
    steps: int
    concept: str
    reads_back_as: str
    region: str
    prompt: str
    denoise: float
    note: str
    arms: Tuple[ArmSpec, ...] = ()
    frames: int = 48
    start_frame: int = 0
    canvas: int = 512
    max_instances: int = 1
    warmup_frames: int = 3

    def specs(self) -> Tuple[ArmSpec, ...]:
        return self.arms or ladder(GUIDANCE_LADDER, DELTA_LADDER)

    def plan(self):
        """The plan the arms render under, through the shipped producer and door.

        The same shape `bench.capture.CaptureCase.plan` builds: a target concept
        and a style go in through `plan_from_fields`, and what comes out is a plan
        the app itself could be put into. Nothing in the plan says anything about
        guidance - it is an engine setting, not a Render Plan field - which is
        exactly why this sweep is a bench family and not a plan override.
        """
        from render_plan import (
            INITIAL_PLAN_VERSION,
            plan_from_fields,
            validate_plan,
        )

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

    def replace(self, **changes) -> "GuidanceCase":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["arms"] = [spec.to_dict() for spec in self.specs()]
        return data


@dataclass(frozen=True)
class EngineKeying:
    """Which TensorRT engine one arm would need, and whether this machine has it.

    Computed from `engine_cache` - the app's own naming rule, shared with the
    window and the harness's build guard - rather than argued in prose. It is the
    Gate's "explicit list of which arms key a new engine", and it is the reason
    that list can be checked instead of believed.
    """

    unet_batch: int
    directory: str
    keys_new_engine: bool
    cached: Optional[bool]

    def to_dict(self) -> dict:
        return asdict(self)


def engine_keying(case: "GuidanceCase", spec: ArmSpec,
                  engines_root: Optional[Path] = None) -> EngineKeying:
    """The engine `spec` would need on the accelerated path, against the shipped one.

    `keys_new_engine` is a comparison of two batches and not a list of cfg types:
    at one step `initialize` and `full` both come out at 2 against the shipped 1,
    and at four steps they are 5 and 8 against 4. Deriving it means the answer
    stays right when the step count moves.
    """
    scenario = SCENARIOS[case.engine_scenario]
    shipped = unet_batch_size(scenario.batch_size, case.steps,
                              scenario.use_denoising_batch, CFG_NONE)
    batch = unet_batch_size(scenario.batch_size, case.steps,
                            scenario.use_denoising_batch, spec.cfg_type)
    directory = engine_dir_name(
        scenario.model, use_lcm_lora=scenario.use_lcm_lora,
        use_tiny_vae=scenario.use_tiny_vae, unet_batch=batch,
        width=scenario.width, height=scenario.height, mode=scenario.mode)
    return EngineKeying(
        unet_batch=batch, directory=directory, keys_new_engine=batch != shipped,
        cached=(None if engines_root is None
                else engine_is_cached(engines_root, directory)))


ADHERENCE_CASE = "cfg-dog"
SD15_CASE = "cfg-dog-sd15"

CASES: Dict[str, GuidanceCase] = {
    ADHERENCE_CASE: GuidanceCase(
        name=ADHERENCE_CASE,
        clip="dog.mp4",
        base_scenario="img2img-none-512x512-b1",
        engine_scenario="img2img-tensorrt-512x512-b1",
        base_model="sd-turbo",
        steps=1,
        concept="dog",
        reads_back_as="cat",
        region="full_box",
        prompt="a cat, feline face, whiskers, pointed ears, photograph",
        # `render_plan.DEFAULT_DENOISE` - the strength the app itself ships at, and
        # therefore the strength the complaint was made at. Held across every arm,
        # because CFG interacts with denoise and an arm that moved both would be
        # measuring two changes at once (the issue's fourth trap). It is also the
        # only strength at which this sweep can say anything: measured on this
        # clip, the control arm reads back as a cat on 0/12 frames at 0.32 and on
        # 11/12 at 0.72, so above about 0.56 every arm scores 100% and the axis is
        # invisible. What is being asked is exactly the product question - does
        # the prompt land at a strength that still preserves the captured frame.
        denoise=DEFAULT_DENOISE,
        note="Does turning classifier-free guidance on make the prompt land? The "
             "identity change is the sharpest adherence probe this repo has - the "
             "detector reads the rendered region back and says whether it is a cat "
             "- and it is the case one-step SD-Turbo is most likely to fail.",
    ),
    # Step 5 of the issue: does the answer differ between the two base models?
    # #43 measured them within 3% on *speed* and said out loud that a quality axis
    # does not carry, so the same sweep runs on SD 1.5 at the four steps it needs.
    SD15_CASE: GuidanceCase(
        name=SD15_CASE,
        clip="dog.mp4",
        base_scenario="img2img-none-512x512-b1-sd15",
        engine_scenario="img2img-tensorrt-512x512-b1-sd15",
        base_model="sd15",
        steps=4,
        concept="dog",
        reads_back_as="cat",
        region="full_box",
        prompt="a cat, feline face, whiskers, pointed ears, photograph",
        denoise=DEFAULT_DENOISE,
        note="The same sweep on SD 1.5 + LCM-LoRA at four steps. Guidance on a "
             "one-step schedule and on a four-step one are not the same animal, "
             "and #43's 3% speed agreement says nothing about a quality axis.",
    ),
}


@dataclass(frozen=True)
class GuidanceArm:
    """One arm: what it cost, how far the prompt landed, and how far the frame drifted.

    `loaded` and `error` carry an arm the pipeline refused - `initialize` and `full`
    reach into an unconditional embedding that only exists above guidance 1.0, and a
    configuration that raises is a result rather than a hole in the table.

    `unet_batch` / `engine_dir` / `keys_new_engine` are the Gate's explicit list,
    computed from `engine_cache` rather than argued: two of the four cfg types run
    extra latents through the UNet, so on the TensorRT path they are a build.
    """

    cfg_type: str
    guidance_scale: float
    delta: float
    delta_applies: bool
    unet_batch: int
    engine_dir: str
    keys_new_engine: bool
    engine_cached: Optional[bool]
    ms_per_frame: float
    adherence_hits: int
    adherence_frames: int
    adherence_conf: float
    retained_hits: int
    region_change: float
    control_change: float
    flicker: float
    background: BackgroundCheck
    frames: int
    loaded: bool = True
    error: Optional[str] = None

    @property
    def name(self) -> str:
        return arm_name(ArmSpec(self.cfg_type, self.guidance_scale, self.delta))

    @property
    def is_control(self) -> bool:
        return self.cfg_type == CONTROL_ARM

    @property
    def measured(self) -> bool:
        """Did this arm produce numbers at all? A refusal is not a zero score."""
        return self.loaded and self.adherence_frames > 0

    @property
    def adherence(self) -> float:
        """The fraction of rendered frames that read back as the prompt's concept."""
        if not self.adherence_frames:
            return 0.0
        return round(self.adherence_hits / self.adherence_frames, 4)

    @property
    def net_change(self) -> float:
        """How far the region drifted from the capture, net of its round trip."""
        return round(max(0.0, self.region_change - self.control_change), 4)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["name"] = self.name
        data["adherence"] = self.adherence
        data["net_change"] = self.net_change
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> "GuidanceArm":
        fields = {field.name: data[field.name] for field in dataclasses.fields(cls)}
        fields["background"] = BackgroundCheck(**dict(fields["background"]))
        return cls(**fields)


# --- the recommendation ------------------------------------------------------


@dataclass(frozen=True)
class GuidanceRecommendation:
    """Which cfg configuration should ship, and the numbers that chose it."""

    arm: str
    cfg_type: str
    guidance_scale: float
    delta: float
    moves_default: bool
    adherence: float
    control_adherence: float
    net_change: float
    control_net_change: float
    ms_per_frame: float
    control_ms_per_frame: float
    keys_new_engine: bool
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def _control_of(arms: Sequence[GuidanceArm]) -> Optional[GuidanceArm]:
    for arm in arms:
        if arm.is_control and arm.measured:
            return arm
    return None


def affordable(arm: GuidanceArm, control: GuidanceArm) -> bool:
    """Is this arm within the frame path's own headroom? See `MAX_COST_RATIO`."""
    if control.ms_per_frame <= 0.0:
        return True
    return arm.ms_per_frame <= control.ms_per_frame * MAX_COST_RATIO


def qualified(arms: Sequence[GuidanceArm],
              control: Optional[GuidanceArm] = None) -> List[GuidanceArm]:
    """The arms a recommendation may choose from, and the rule in one place.

    Three disqualifiers, and each is a disqualifier rather than a caveat. An arm
    that never ran cannot be recommended. Neither can one that painted a pixel
    outside the rendered region - that is the selective path's whole promise, and
    an arm that broke it is a different renderer however well the prompt landed.
    And neither can one that costs more of the frame path than the budget has in
    hand: the default has to run at 30 FPS, so an arm that reads back beautifully
    at half the frame rate is a setting rather than a default.
    """
    candidates = [arm for arm in arms
                  if arm.measured and not arm.is_control and arm.background.passed]
    if control is None:
        return candidates
    return [arm for arm in candidates if affordable(arm, control)]


def _no_control() -> GuidanceRecommendation:
    return GuidanceRecommendation(
        arm=CONTROL_ARM, cfg_type=CFG_NONE, guidance_scale=1.0, delta=1.0,
        moves_default=False, adherence=0.0, control_adherence=0.0,
        net_change=0.0, control_net_change=0.0, ms_per_frame=0.0,
        control_ms_per_frame=0.0, keys_new_engine=False,
        statement=("No `none` arm ran, so there is nothing to compare a guidance "
                   "arm against. The sweep's own control is what says whether "
                   "guidance did anything, and a run without it recommends "
                   "nothing."))


def _threshold_for(arm: GuidanceArm) -> float:
    """How much this arm has to gain before it is worth adopting.

    Two numbers, because two arms can buy the same adherence at wildly different
    prices. An arm on the shipped UNet batch costs nothing to adopt - a runtime
    setting and a `prepare` call - so the bar is only "the axis did something". An
    arm that keys a build spends ~5 GB, a compile the user watches, and whatever
    the extra latents cost the frame path, and it has to be plainly worth that
    rather than marginally.
    """
    return (MIN_ADHERENCE_GAIN_FOR_BUILD if arm.keys_new_engine
            else MIN_ADHERENCE_GAIN)


def _pick(arms: Sequence[GuidanceArm], control: GuidanceArm
          ) -> Optional[GuidanceArm]:
    """The best of these arms, if it cleared the bar its own cost sets.

    Arms within `ADHERENCE_TIE` of the best are a tie, and the tie breaks towards
    the smaller UNet batch and then the cheaper frame - a tie that ignored that
    would recommend an engine to buy nothing.
    """
    if not arms:
        return None
    best = max(arms, key=lambda arm: arm.adherence)
    if best.adherence - control.adherence < _threshold_for(best):
        return None
    return min([arm for arm in arms
                if arm.adherence >= best.adherence - ADHERENCE_TIE],
               key=lambda arm: (arm.unet_batch, arm.ms_per_frame))


def _stands(control: GuidanceArm, best: Optional[GuidanceArm],
            broke_identity: Sequence[GuidanceArm],
            unaffordable: Sequence[GuidanceArm] = ()) -> GuidanceRecommendation:
    """The shipped setting stands, and the sentence says which way it got here."""
    priced_out = ""
    if unaffordable:
        dearest = max(unaffordable, key=lambda arm: arm.adherence)
        priced_out = (
            f" The best-reading arm of the whole sweep was `{dearest.name}` at "
            f"{dearest.adherence:.0%}, and it is out on cost rather than on "
            f"quality: {dearest.ms_per_frame:.2f} ms/frame against the control's "
            f"{control.ms_per_frame:.2f}, "
            f"{dearest.ms_per_frame / control.ms_per_frame:.2f}x, against the "
            f"{MAX_COST_RATIO:.2f}x the frame budget has in hand.")
    if broke_identity:
        reason = (f"The best-reading arm (`{broke_identity[0].name}`, "
                  f"{broke_identity[0].adherence:.0%}) painted pixels outside the "
                  f"rendered region, so it is disqualified rather than "
                  f"recommended: the background is bit-identical to the capture "
                  f"or the selective path does not do what it says.")
    elif best is None:
        reason = "No guidance arm produced a number to compare."
    else:
        priced = (f", and it keys a **different TensorRT engine** at UNet batch "
                  f"{best.unet_batch} and costs {best.ms_per_frame:.2f} ms/frame "
                  f"against the control's {control.ms_per_frame:.2f}"
                  if best.keys_new_engine else "")
        reason = (f"The best any guidance arm read back was `{best.name}` at "
                  f"{best.adherence:.0%} against the control's "
                  f"{control.adherence:.0%}{priced} - under the "
                  f"{_threshold_for(best):.0%} gain this rule asks of an arm that "
                  f"costs what that one costs. The lever is real and it is "
                  f"priced; what it is not is a new default.")
    return GuidanceRecommendation(
        arm=CONTROL_ARM, cfg_type=CFG_NONE, guidance_scale=1.0, delta=1.0,
        moves_default=False, adherence=control.adherence,
        control_adherence=control.adherence, net_change=control.net_change,
        control_net_change=control.net_change,
        ms_per_frame=control.ms_per_frame,
        control_ms_per_frame=control.ms_per_frame, keys_new_engine=False,
        statement=(f"**Recommended: `cfg_type: none` - the shipped setting.** "
                   f"{reason}{priced_out}"))


def recommend_guidance(arms: Sequence[GuidanceArm]) -> GuidanceRecommendation:
    """The best prompt pull that is worth what it costs, or the shipped setting.

    The free arms are asked first and on their own: a cfg type that runs the
    shipped UNet batch is a runtime setting, so it only has to beat the control by
    `MIN_ADHERENCE_GAIN` to be worth adopting. Only when none of them does is the
    build asked about, and then at `MIN_ADHERENCE_GAIN_FOR_BUILD` - because
    recommending a default that makes every install compile ~5 GB is a decision of
    a different size, and a rule that scored both against one threshold would
    quietly make it on a margin.

    And the shipped setting stands unless something clears its own bar: an axis
    that does not move the number it exists to move is an axis whose default
    should not move either.
    """
    control = _control_of(arms)
    if control is None:
        return _no_control()
    broke_identity = sorted(
        [arm for arm in arms
         if arm.measured and not arm.is_control and not arm.background.passed],
        key=lambda arm: arm.adherence, reverse=True)
    candidates = qualified(arms, control)
    unaffordable = [arm for arm in qualified(arms)
                    if not affordable(arm, control)]
    if not candidates:
        return _stands(control, None, broke_identity, unaffordable)
    best = max(candidates, key=lambda arm: arm.adherence)
    if broke_identity and broke_identity[0].adherence > best.adherence:
        return _stands(control, best, broke_identity, unaffordable)
    chosen = (_pick([arm for arm in candidates if not arm.keys_new_engine], control)
              or _pick([arm for arm in candidates if arm.keys_new_engine], control))
    if chosen is None:
        return _stands(control, best, [], unaffordable)
    return GuidanceRecommendation(
        arm=chosen.name, cfg_type=chosen.cfg_type,
        guidance_scale=chosen.guidance_scale, delta=chosen.delta,
        moves_default=True, adherence=chosen.adherence,
        control_adherence=control.adherence, net_change=chosen.net_change,
        control_net_change=control.net_change,
        ms_per_frame=chosen.ms_per_frame,
        control_ms_per_frame=control.ms_per_frame,
        keys_new_engine=chosen.keys_new_engine,
        statement=_moves_statement(chosen, control))


def _moves_statement(chosen: GuidanceArm, control: GuidanceArm) -> str:
    """What the recommendation bought, what it cost, and what it costs to ship."""
    delta = (f" at delta {chosen.delta:g}" if chosen.delta_applies else "")
    engine = (f" It keys a **different TensorRT engine** - UNet batch "
              f"{chosen.unet_batch} against the control's {control.unet_batch}, "
              f"`{chosen.engine_dir}` - so adopting it on the accelerated path is "
              f"{ENGINE_BUILD_SIZE} and {ENGINE_BUILD_TIME}."
              if chosen.keys_new_engine else
              " It keys **no new engine**: this cfg type runs the same UNet batch "
              "as the shipped one, so adopting it costs no build at all.")
    cost = (f"{chosen.ms_per_frame:.2f} ms/frame against the control's "
            f"{control.ms_per_frame:.2f} - "
            f"{chosen.ms_per_frame - control.ms_per_frame:+.2f} ms the shipped "
            f"selective budget has to find")
    return (
        f"**Recommended: `cfg_type: {chosen.cfg_type}` at guidance "
        f"{chosen.guidance_scale:g}{delta}.** The rendered region reads back as "
        f"the prompt's own concept on {chosen.adherence:.0%} of frames against "
        f"{control.adherence:.0%} with guidance off, at {cost}. The price is "
        f"drift: {chosen.net_change:.1f}/255 from the capture against the "
        f"control's {control.net_change:.1f} - the same axis that lands the prompt "
        f"is the one that walks the output away from the captured frame, and both "
        f"halves are the answer.{engine}")


# --- the record --------------------------------------------------------------


@dataclass(frozen=True)
class GuidanceResult:
    """One run: every arm over the same frames of one clip, and the artefact."""

    case: GuidanceCase
    clip: ClipRecord
    arms: Tuple[GuidanceArm, ...]
    started_utc: str
    finished_utc: str
    cooldown: CooldownRecord
    occupancy: Optional[OccupancyRecord]
    hardware: Fingerprint
    clock_normalization: Optional[ClockNormalization]
    comparison_still: str = ""
    comparison_clip: str = ""

    def to_dict(self) -> dict:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "kind": RECORD_KIND,
            "case": self.case.to_dict(),
            "clip": asdict(self.clip),
            "arms": [arm.to_dict() for arm in self.arms],
            "recommendation": recommend_guidance(self.arms).to_dict(),
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


def write_guidance_result(result: GuidanceResult, results_dir: Path,
                          timestamp: Optional[str] = None) -> Path:
    data = result.to_dict()
    timestamp = timestamp_from(data) if timestamp is None else timestamp
    return write_record(data, results_dir, f"{result.case.name}-{timestamp}.json")


def load_guidance_results(results_dir: Path) -> Dict[str, dict]:
    return load_records(Path(results_dir))


GUIDANCE_README_NAME = "README.md"
GUIDANCE_README_TITLE = "# Classifier-free guidance: does the prompt land?"
GUIDANCE_README_PREAMBLE_TEXT = (
    "Written by `uv run python -m bench cfg-dog`, never by hand. Issue #45: the app\n"
    "ships with CFG switched off and has never measured anything else, so the weak\n"
    "prompt adherence has never been shown to be the checkpoint rather than a\n"
    "setting.\n\n"
    "`adherence` is the fraction of rendered frames the open-vocabulary detector\n"
    "reads back as the concept the prompt asked for - spec 8.2's identity probe -\n"
    "and `net drift` is how far the region moved from the capture, net of its own\n"
    "round trip. The two are the trade, and an arm is only worth something when\n"
    "the first one moved.\n"
)
GUIDANCE_README_HEADER = ("| finished (UTC) | case | GPU | arm | cfg | guidance |"
                          " delta | ms/frame | adherence | conf | net drift |"
                          " UNet batch | background | file |")
GUIDANCE_README_SEPARATOR = table_separator(GUIDANCE_README_HEADER)


def append_guidance_readme_rows(result: GuidanceResult, readme_path: Path,
                                filename: str) -> None:
    """One row per arm - the control too, because it is the yardstick."""
    data = result.to_dict()
    require_recordable(data)
    preamble = (f"{GUIDANCE_README_TITLE}\n\n{GUIDANCE_README_PREAMBLE_TEXT}\n"
                f"{GUIDANCE_README_HEADER}\n{GUIDANCE_README_SEPARATOR}\n")
    for arm in result.arms:
        append_row(_readme_row(result, arm, filename), readme_path, preamble)


def _readme_row(result: GuidanceResult, arm: GuidanceArm, filename: str) -> str:
    return table_row([
        result.finished_utc,
        result.case.name,
        result.hardware.gpu_name,
        arm.name,
        arm.cfg_type,
        f"{arm.guidance_scale:g}",
        f"{arm.delta:g}" if arm.delta_applies else "n/a",
        format_number(arm.ms_per_frame, 2),
        "not measured" if not arm.measured else f"{arm.adherence:.0%}",
        format_number(arm.adherence_conf, 2),
        format_number(arm.net_change, 2),
        str(arm.unet_batch),
        "identical" if arm.background.passed else "CHANGED",
        f"[{filename}]({filename})",
    ])


# --- the block spec 8.11 carries ---------------------------------------------


REPORT_HEADER = ("| arm | cfg | guidance | delta | ms/frame | adherence | conf |"
                 " retained | net drift | flicker | UNet batch | background |")


def arms_of(result: Mapping) -> List[GuidanceArm]:
    return [GuidanceArm.from_dict(arm) for arm in result["arms"]]


def base_model_column(results: Sequence[dict]) -> OptionalColumn:
    """The `base model` column the table grows when it holds more than one.

    Necessary here in a way it is not elsewhere: two cases produce arms with the
    *same names* - there is one `none` per base model and one `self-g14-d10` -
    so without it the table is two sweeps interleaved and nothing says which row
    is which. Off with one case, so a single-model block does not churn.
    """
    return OptionalColumn.when_varied(
        results, "base model",
        lambda result: f"`{result['case']['base_model']}` "
                       f"@ {result['case']['steps']} step"
                       f"{'' if result['case']['steps'] == 1 else 's'}")


def _row(result: dict, arm: GuidanceArm, column: GpuColumn,
         model: OptionalColumn) -> str:
    return column.row(model.cells([
        f"`{arm.name}`",
        arm.cfg_type,
        f"{arm.guidance_scale:g}",
        f"{arm.delta:g}" if arm.delta_applies else "n/a",
        format_number(arm.ms_per_frame, 2),
        "**did not run**" if not arm.measured else f"{arm.adherence:.0%}",
        format_number(arm.adherence_conf, 2),
        f"{arm.retained_hits}/{arm.adherence_frames}",
        format_number(arm.net_change, 2),
        format_number(arm.flicker, 2),
        str(arm.unet_batch),
        "identical" if arm.background.passed else "**CHANGED**",
    ], result), result)


def _preamble(results: Sequence[dict], gpus: Sequence[str]) -> str:
    result = results[0]
    case, clip = result["case"], result["clip"]
    models = ", ".join(f"`{one['case']['base_model']}` at "
                       f"{one['case']['steps']} step"
                       f"{'' if one['case']['steps'] == 1 else 's'}"
                       for one in results)
    return (
        f"{len(result['arms']) - 1} guidance arms and one control on each of "
        f"{models}, over {clip['frames_used']} frames of "
        f"`{clip['name']}` at {case['canvas']}x{case['canvas']}, on "
        f"{', '.join(gpus)}. Every arm renders the same frames of the same "
        f"committed box track at the same denoise ({case['denoise']}) under the "
        f"same prompt (\"{case['prompt']}\"); only `cfg_type`, `guidance_scale` and "
        f"`delta` move, because CFG interacts with denoise and an arm that moved "
        f"both would be measuring two changes at once. `adherence` is the fraction "
        f"of rendered frames the open-vocabulary detector reads back as "
        f"`{case['reads_back_as']}` at confidence {ADHERENCE_CONF}, `retained` is "
        f"how many still read as `{case['concept']}`, and `net drift` is how far "
        f"the region moved from the capture net of its own round trip. `delta` is "
        f"`n/a` where the pipeline never reads it."
    )


def engine_statement(arms: Sequence[GuidanceArm]) -> str:
    """The Gate's explicit list: which arms are a build rather than a setting."""
    keyed = [arm for arm in arms if arm.keys_new_engine]
    if not keyed:
        return ("**Which arms key a new engine: none.** Every arm above runs the "
                "same UNet batch as the shipped configuration, so on the TensorRT "
                "path each is a setting rather than a build.")
    by_batch: Dict[Tuple[int, str], List[str]] = {}
    for arm in keyed:
        by_batch.setdefault((arm.unet_batch, arm.engine_dir), []).append(arm.name)
    parts = [f"{', '.join(f'`{name}`' for name in names)} at UNet batch {batch} "
             f"(`{directory}`)" for (batch, directory), names in by_batch.items()]
    return (f"**Which arms key a new engine: {len(keyed)} of {len(arms)}.** "
            f"{'; '.join(parts)}. `initialize` runs one extra unconditional latent "
            f"through the UNet and `full` runs a second copy of every one, so on "
            f"the TensorRT path each of those batches is its own build - "
            f"{ENGINE_BUILD_SIZE} and {ENGINE_BUILD_TIME}. `self` and `none` share "
            f"the batch the app already has compiled.")


def equivalence_note(arms: Sequence[GuidanceArm]) -> Optional[str]:
    """Two cfg types that produced the *same* numbers, where they did.

    Read off the run rather than argued from the algebra: the render is
    deterministic given the clip, the plan and the seed, so two arms agreeing on
    adherence, drift and flicker to four figures agree because they computed the
    same thing. At one denoising step and one frame per call `initialize` and
    `full` do - `initialize` splices the unconditional prediction into
    `stock_noise` and multiplies it by `delta`, and at delta 1.0 that is the
    tensor `full` chunks out of its doubled batch. Worth saying, because it means
    a `full` engine buys nothing an `initialize` one does not at that shape.
    """
    def fingerprint(arm: GuidanceArm):
        return (arm.guidance_scale, arm.adherence, arm.net_change, arm.flicker)

    seen: Dict[tuple, List[GuidanceArm]] = {}
    for arm in arms:
        if arm.measured and not arm.is_control:
            seen.setdefault(fingerprint(arm), []).append(arm)
    agreed = [group for group in seen.values()
              if len({arm.cfg_type for arm in group}) > 1]
    if not agreed:
        return None
    types = sorted({arm.cfg_type for group in agreed for arm in group})
    rungs = sorted({group[0].guidance_scale for group in agreed})
    return (f"`{'` and `'.join(types)}` produced identical numbers on "
            f"{len(agreed)} of the ladder's rungs "
            f"({', '.join(f'{rung:g}' for rung in rungs)}) - adherence, drift and "
            f"flicker all to four figures. That is not a coincidence and it is not "
            f"asserted from the algebra: at one denoising step and one frame per "
            f"call, `initialize` splices the unconditional prediction into "
            f"`stock_noise` and multiplies it by `delta`, and at delta 1.0 that is "
            f"the tensor `full` chunks out of its doubled batch. So `full` buys "
            f"nothing `initialize` does not at this shape, and both cost the same "
            f"larger UNet batch.")


def background_statement(arms: Sequence[GuidanceArm]) -> str:
    """The Gate's bit-identity item, over every arm rather than over the winner."""
    measured = [arm for arm in arms if arm.measured]
    broken = [arm for arm in measured if not arm.background.passed]
    if broken:
        return (f"**{len(broken)} of {len(measured)} arms changed a pixel outside "
                f"the rendered region.** "
                + " ".join(f"`{arm.name}`: {arm.background.statement}."
                           for arm in broken))
    frames = sum(arm.background.identical_frames for arm in measured)
    total = sum(arm.background.frames for arm in measured)
    return (f"Every arm above left the background bit-identical to the capture: "
            f"{frames}/{total} frames across {len(measured)} arms, "
            f"{measured[0].background.frames}/{measured[0].background.frames} on "
            f"each.")


def _machine_section(result: dict, prefix: str) -> str:
    """One run's recommendation, its engine list, its background gate, its clip.

    Prefixed with the base model whenever there is more than one arm set in the
    block: the two read as the same paragraph with different numbers otherwise,
    and the verdict a reader quotes has to say what it is a verdict about.
    """
    arms = arms_of(result)
    lines = [f"{prefix}{recommend_guidance(arms).statement}",
             engine_statement(arms),
             background_statement(arms)]
    equivalence = equivalence_note(arms)
    if equivalence is not None:
        lines.append(equivalence)
    artefacts = [f"`{result[key]}`" for key in ("comparison_still", "comparison_clip")
                 if result.get(key)]
    if artefacts:
        lines.append(f"The artefact a human judges this by, source | control | one "
                     f"panel per arm: {', '.join(artefacts)}.")
    return "\n\n".join(lines)


def format_guidance_report(results: Mapping[str, dict]) -> str:
    """The measured block spec 8.11 carries, from the committed guidance runs."""
    reduced = latest_per(results, lambda result: result["case"]["name"])
    ordered = sorted(reduced.values(),
                     key=lambda result: (result["case"]["name"], gpu_of(result)))
    if not ordered:
        return "no guidance sweep committed yet (issue #45)"

    gpus = distinct_gpus(ordered)
    column = GpuColumn.for_gpus(gpus)
    model = base_model_column(ordered)
    header = column.header(model.header(REPORT_HEADER))
    sections = [_preamble(ordered, gpus)]
    rows = [_row(result, arm, column, model)
            for result in ordered for arm in arms_of(result)]
    sections.append("\n".join([header, table_separator(header)] + rows))
    for gpu in gpus:
        for result in measured_on(ordered, gpu):
            sections.append(_machine_section(result, _prefix(result, gpus, model)))
    return "\n\n".join(sections)


def _prefix(result: Mapping, gpus: Sequence[str], model: OptionalColumn) -> str:
    """What a verdict paragraph is a verdict *about*, when more than one is here.

    Silent when there is one machine and one base model, so a single-arm block
    reads as the one sentence it is; otherwise the machine, the model, or both.
    """
    machine = "" if len(gpus) == 1 else f"{gpu_of(result)}, "
    if not machine and not model.shown:
        return ""
    return f"{machine}`{result['case']['base_model']}`: "
