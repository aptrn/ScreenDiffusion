"""The Render Plan: the validated contract between the GUI and the worker.

Issue #6, spec section 6. A plan says *what to restyle* and *how*; the worker
renders whatever plan is active, and swaps plans between frames, never inside one.

Three things shape this module.

**No LLM.** Spec section 6 says "only C2 may write it" - C2 was the prompt
compiler, and it is cut from v1. The producer is the GUI: a **target** field whose
text goes to the open-vocabulary detector and a **style** field whose text goes to
StreamDiffusion. `plan_from_fields` is that producer; `validate_plan` is the door
every plan comes through, whichever hand wrote it.

**Stdlib only.** The plan crosses a `multiprocessing.Queue` and is validated on
both sides of it, so this module must import in the GUI process (which never
imports torch) and in the merge gate's GPU-free tier. Nothing here touches a GPU,
a filesystem or a network.

The wire form is `RenderPlan.to_dict()` - a plain dict, like every other control
message. The producer validates so it can refuse what the user typed before sending
it; the worker validates again because a plan can arrive from anywhere, and because
the version has to be counted up from the plan *the worker* is rendering.

**Rejection is a first-class outcome.** `validate_plan` returns a `PlanValidation`,
never raises at its callers, and a rejected plan carries the reason in words a user
can read - spec 8.7's failure UX: the previous plan keeps rendering, and the reason
goes on the status channel. What is clamped or dropped instead of rejected is also
said out loud, in `notes`.

The `mode`/`region` machinery is the schema the selective render path will consume
(issues #7 and #8). Until that lands the worker honours the plan's *prompt* and
renders the whole frame, which is what `mode: "global"` means and what the app does
today - see `RenderPlan.effective_prompt`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# --- vocabulary -------------------------------------------------------------

# The regions of a detected box a target can ask for. Fixed, and a value outside
# this set is rejected rather than defaulted: "upper_thrid" is a producer bug, and
# quietly restyling the whole box instead would hide it behind a plausible frame.
REGIONS: Tuple[str, ...] = (
    "full_box",
    "upper_third",
    "upper_half",
    "center",
    "lower_half",
    "lower_third",
)
DEFAULT_REGION = "full_box"

SELECTIVE = "selective"
GLOBAL = "global"
INVERSE = "inverse"
MODES: Tuple[str, ...] = (SELECTIVE, GLOBAL, INVERSE)

PER_TRACK = "per_track"
SEED_POLICIES: Tuple[str, ...] = (PER_TRACK, "fixed", "random")
DEFAULT_SEED_POLICY = PER_TRACK

PASSTHROUGH = "passthrough"
STYLIZE = "stylize"
BACKGROUND_ACTIONS: Tuple[str, ...] = (PASSTHROUGH, STYLIZE)

# `global` is a Python keyword, so `RenderPlan` cannot carry a field of that name.
# The *wire* key stays `global`, because that is what spec section 6 defines.
GLOBAL_KEY = "global"
GLOBAL_FIELD = "settings"

# --- ranges and defaults ----------------------------------------------------

# Dilation applied to a detected box before it is rendered. Below 1.0 the crop is
# smaller than the object it was found from, which is never what anyone wants; above
# 2.0 the "region" is mostly background.
BOX_SCALE_RANGE: Tuple[float, float] = (1.0, 2.0)
DEFAULT_BOX_SCALE = 1.15

DENOISE_RANGE: Tuple[float, float] = (0.0, 1.0)
DEFAULT_DENOISE = 0.45

# How many instances of one concept may be rendered per frame. The ceiling is a
# design limit, not a measured one: past it the masked region is the whole frame
# and `mode: "global"` is the cheaper way to ask for the same thing.
MAX_INSTANCES_RANGE: Tuple[int, int] = (1, 16)
DEFAULT_MAX_INSTANCES = 6

# An ordering key between targets, not a score. A plan holds a handful of targets,
# so the range only has to be wide enough to sort them.
PRIORITY_RANGE: Tuple[int, int] = (0, 99)
DEFAULT_PRIORITY = 1

FPS_TARGET_RANGE: Tuple[int, int] = (1, 240)
DEFAULT_FPS_TARGET = 30  # spec 7.1

# Detect every Nth frame. 3 is what spec 7.1 budgets and what issue #4 measured
# YOLO-World inside; past 30 the tracker is bridging a whole second on its own.
DETECT_EVERY_N_RANGE: Tuple[int, int] = (1, 30)
DEFAULT_DETECT_EVERY_N = 3

CONFIDENCE_RANGE: Tuple[float, float] = (0.0, 1.0)
# The compiler that would have reported less than full confidence is cut from v1, so
# a hand- or GUI-authored plan is certain of itself. The field survives for spec
# 8.7's banner, which is about what the *worker* could not do with a plan.
DEFAULT_CONFIDENCE = 1.0

# CLIP ViT-B/32 - what YOLO-World embeds a vocabulary with - truncates at 77 tokens.
# A concept anywhere near that is a sentence, and the detector would silently search
# for a prefix of it, so it is refused instead.
MAX_CONCEPT_CHARS = 120

# The first plan the worker holds. The first plan a producer sends is version 1.
INITIAL_PLAN_VERSION = 0


# --- denoise, and the schedule the engine indexes ---------------------------
#
# A plan says how much of the frame to replace, as a 0-1 strength. The engine takes
# a `t_index`: an index into the 50-step schedule the worker prepares, which
# *descends*, so a higher index is less denoise. These four constants are SD-Turbo's
# own scaled-linear beta schedule, and `noise_amplitude` reproduces the strengths
# issue #5's comparison read off the live scheduler - a test holds it to that
# committed measurement rather than to a curve that looks about right.

SCHEDULE_TRAIN_TIMESTEPS = 1000
SCHEDULE_STEPS = 50  # what the worker's `prepare(num_inference_steps=50)` asks for
BETA_START = 0.00085
BETA_END = 0.012
# The indices a plan may be rendered at. Index 0 is the training schedule's own
# first step, where img2img keeps nothing of the input; the worker clamps to its
# own usable range on top of this one.
SCHEDULE_T_INDEX_RANGE: Tuple[int, int] = (1, 49)


def _alphas_cumprod() -> Tuple[float, ...]:
    """The forward process's cumulative alphas, one per training timestep.

    `beta_t` is linear in *sqrt* beta - the "scaled_linear" schedule SD 1.x/2.x are
    trained with - and `alpha_bar_t` is the running product of `1 - beta`. Computed
    here rather than read off the scheduler because this module is imported by the
    GUI process, which has no torch and no engine to ask.
    """
    span = SCHEDULE_TRAIN_TIMESTEPS - 1
    alphas: List[float] = []
    running = 1.0
    for step in range(SCHEDULE_TRAIN_TIMESTEPS):
        root = BETA_START ** 0.5 + (BETA_END ** 0.5 - BETA_START ** 0.5) * step / span
        running *= 1.0 - root * root
        alphas.append(running)
    return tuple(alphas)


ALPHAS_CUMPROD: Tuple[float, ...] = _alphas_cumprod()


def timestep_of(t_index: int) -> int:
    """The training timestep a `t_index` names on the 50-step schedule.

    Index 20 is timestep 599 and index 45 is timestep 99, which is what issue #5's
    record shows and what the repo's own gotcha note says.
    """
    stride = SCHEDULE_TRAIN_TIMESTEPS // SCHEDULE_STEPS
    return SCHEDULE_TRAIN_TIMESTEPS - 1 - stride * int(t_index)


def noise_amplitude(t_index: int) -> float:
    """How much of the latent this index replaces with noise, as a 0-1 strength.

    `sqrt(1 - alpha_bar)` - the amplitude the forward process applies, which reads
    like diffusers' img2img `strength` and, unlike an index, means the same thing
    whatever the schedule.
    """
    return float((1.0 - ALPHAS_CUMPROD[timestep_of(t_index)]) ** 0.5)


def t_index_for_denoise(denoise: float) -> int:
    """The schedule index whose strength is nearest the plan's `denoise`.

    Nearest on the schedule rather than a linear map across the index range: the
    amplitude curve is far from straight, and a plan asking for 0.5 would otherwise
    get a third more denoise than it asked for. A tie goes to the gentler index -
    the plan asked for a change of that size, not for at least one.
    """
    low, high = SCHEDULE_T_INDEX_RANGE
    return min(range(high, low - 1, -1),
               key=lambda index: abs(noise_amplitude(index) - float(denoise)))


# --- what a detector can be asked for ---------------------------------------

# The 80 classes a COCO-trained detector has and cannot be asked past. Written out
# rather than read from ultralytics: that import pulls in torch, and this module is
# held to the GUI process's no-torch rule.
COCO_CLASSES: Tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)


@dataclass(frozen=True)
class DetectorVocabulary:
    """What the active detector can be asked to find, and how to say it cannot.

    The vocabulary, not the weights: this is the half of a detector the *validator*
    needs, and it has to be answerable in a process with no CUDA device. The
    measured half lives in `bench.detectors`, and a test holds the two registries to
    the same names.
    """

    name: str
    open_vocabulary: bool
    classes: Tuple[str, ...] = ()

    def refusal(self, concept: str) -> Optional[str]:
        """Why this detector cannot be asked for `concept`, or None if it can.

        `concept` is already known to be a well-formed phrase - shape is the
        validator's business, and membership is this one's.
        """
        if self.open_vocabulary:
            return None
        if concept.strip().lower() in self.classes:
            return None
        return (
            f"{self.name} has a fixed vocabulary of {len(self.classes)} classes and "
            f"none of them is '{concept}'"
        )


# The same two names `bench.detectors` measures, spelt again rather than imported:
# that module is the benchmark harness and this one is shipped code, and importing it
# from the GUI process would drag the measurement stack in behind it. A test holds
# the two registries to the same names and the same open-vocabulary flags.
PRIMARY_DETECTOR = "yolo-world-s-640"
SPEED_FLOOR_DETECTOR = "yolov8n-640"

DETECTOR_VOCABULARIES: Dict[str, DetectorVocabulary] = {
    PRIMARY_DETECTOR: DetectorVocabulary(name=PRIMARY_DETECTOR, open_vocabulary=True),
    SPEED_FLOOR_DETECTOR: DetectorVocabulary(
        name=SPEED_FLOOR_DETECTOR, open_vocabulary=False, classes=COCO_CLASSES
    ),
}

# What issue #4 measured and chose. The worker will pass the detector it actually
# loaded (issue #7); until then this is the one a plan is validated against.
ACTIVE_DETECTOR = PRIMARY_DETECTOR


# --- the plan ---------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """One concept to find and what to do with it.

    Frozen and all-primitive: a plan is shared across the frame boundary without
    defensive copies, and pickles onto a `multiprocessing.Queue` as it stands.
    """

    id: str
    concept: str
    detector_class: Optional[int] = None
    region: str = DEFAULT_REGION
    box_scale: float = DEFAULT_BOX_SCALE
    prompt: str = ""
    negative_prompt: str = ""
    denoise: float = DEFAULT_DENOISE
    seed_policy: str = DEFAULT_SEED_POLICY
    max_instances: int = DEFAULT_MAX_INSTANCES
    priority: int = DEFAULT_PRIORITY

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class Background:
    """What happens to everything that is not a target."""

    action: str = PASSTHROUGH
    prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class GlobalSettings:
    """Plan-wide knobs. Spec section 6 calls this block `global`."""

    fps_target: int = DEFAULT_FPS_TARGET
    detect_every_n: int = DEFAULT_DETECT_EVERY_N

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RenderPlan:
    """A validated plan. The only way to get one is `validate_plan`.

    `negative_prompt` is the one field spec section 6's straw-man does not have. It
    is here because the app has a negative-prompt box today and no compiler to
    distribute its text into targets; a plan that could not express it would not be
    able to reproduce the app's current behaviour, which step 5 of issue #6 requires.
    """

    plan_version: int
    source_prompt: str = ""
    negative_prompt: str = ""
    mode: str = GLOBAL
    targets: Tuple[Target, ...] = ()
    background: Background = Background()
    settings: GlobalSettings = GlobalSettings()
    confidence: float = DEFAULT_CONFIDENCE
    notes: str = ""

    @property
    def honoured_target(self) -> Optional[Target]:
        """The one target whose prompt and denoise actually reach the engine.

        Issue #5 chose primitive B: one masked full-frame diffusion per frame, so
        one prompt embedding and one denoise strength per frame. The other targets
        are kept in the schema - they are what a per-target render would need, and
        issue #6's fourth trap says to carry them rather than drop them - but they
        are not applied, and `validate_plan` says so in its notes when they differ.
        """
        return self.targets[0] if self.targets else None

    @property
    def effective_prompt(self) -> str:
        """What goes to StreamDiffusion. The honoured target's, else the plan's."""
        target = self.honoured_target
        if target is not None and target.prompt:
            return target.prompt
        return self.source_prompt

    @property
    def effective_negative_prompt(self) -> str:
        target = self.honoured_target
        if target is not None and target.negative_prompt:
            return target.negative_prompt
        return self.negative_prompt

    @property
    def effective_denoise(self) -> float:
        target = self.honoured_target
        return DEFAULT_DENOISE if target is None else target.denoise

    def to_dict(self) -> Dict[str, Any]:
        """The wire form: exactly what `validate_plan` accepts back."""
        return {
            "plan_version": self.plan_version,
            "source_prompt": self.source_prompt,
            "negative_prompt": self.negative_prompt,
            "mode": self.mode,
            "targets": [target.to_dict() for target in self.targets],
            "background": self.background.to_dict(),
            GLOBAL_KEY: self.settings.to_dict(),
            "confidence": self.confidence,
            "notes": self.notes,
        }


# What `_drop_unknown` accepts at each level, taken from the dataclasses so a
# renamed field cannot leave a stale name behind. The plan's list is the one that
# needs help: its `settings` field travels under the spec's `global` key.
PLAN_FIELDS: Tuple[str, ...] = tuple(
    GLOBAL_KEY if f.name == GLOBAL_FIELD else f.name
    for f in dataclasses.fields(RenderPlan)
)
TARGET_FIELDS: Tuple[str, ...] = tuple(f.name for f in dataclasses.fields(Target))
BACKGROUND_FIELDS: Tuple[str, ...] = tuple(f.name for f in dataclasses.fields(Background))
GLOBAL_FIELDS: Tuple[str, ...] = tuple(f.name for f in dataclasses.fields(GlobalSettings))


@dataclass(frozen=True)
class PlanValidation:
    """The outcome of validating one plan: a plan, or the reason there is not one.

    `notes` is everything the validator changed and let through - a clamp, a dropped
    field, a target the detector cannot serve. `errors` is why there is no plan at
    all. Both are for the user to read.
    """

    plan: Optional[RenderPlan]
    errors: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.plan is not None

    @property
    def reason(self) -> str:
        return "; ".join(self.errors)


class _Rejected(Exception):
    """A plan the worker must not render, carrying the reason a user is shown."""


# --- validation -------------------------------------------------------------


def _drop_unknown(raw: Mapping[str, Any], known: Sequence[str], where: str,
                  notes: List[str]) -> None:
    for key in raw:
        if key not in known:
            notes.append(f"dropped unknown {where} field '{key}'")


def _text(value: Any, field: str, default: str = "") -> str:
    """Text, or a producer bug. A list where a prompt belongs is not a prompt."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    raise _Rejected(f"`{field}` must be text, not {type(value).__name__}")


def _choice(value: Any, field: str, allowed: Sequence[str], default: Optional[str] = None) -> str:
    if value is None and default is not None:
        return default
    if isinstance(value, str) and value in allowed:
        return value
    raise _Rejected(f"`{field}` must be one of {', '.join(allowed)}, not {value!r}")


def _number(value: Any, field: str, low: float, high: float, default: float,
            notes: List[str], as_int: bool = False) -> Union[int, float]:
    """A number inside its range. Out of range is clamped and said; junk is refused.

    A numeric string is accepted, the way `set_t_index_list` already accepts one.
    Anything that is not a number at all is a producer bug, and a plan built on one
    is rejected rather than rendered at a default that looks deliberate.
    """
    if value is None:
        return int(default) if as_int else float(default)
    if isinstance(value, bool):
        raise _Rejected(f"`{field}` must be a number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise _Rejected(f"`{field}` must be a number, not {value!r}")
    if number != number:  # NaN compares false against every bound
        raise _Rejected(f"`{field}` must be a number, not {value!r}")
    clamped = max(low, min(high, number))
    if clamped != number:
        notes.append(f"clamped {field} {number:g} to {clamped:g}")
    return int(clamped) if as_int else float(clamped)


def _concept(value: Any) -> str:
    """The user's word for a target, before the detector is asked about it."""
    if not isinstance(value, str) or not value.strip():
        raise _Rejected("a target needs a `concept` - free text naming what to find")
    concept = value.strip()
    if len(concept) > MAX_CONCEPT_CHARS:
        raise _Rejected(
            f"a `concept` of {len(concept)} characters is a sentence, not a concept; "
            f"the detector's text encoder holds {MAX_CONCEPT_CHARS}"
        )
    return concept


def _detector_class(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        index = int(value)
    except (TypeError, ValueError):
        raise _Rejected(f"`detector_class` must be a class index or null, not {value!r}")
    if index < 0:
        raise _Rejected(f"`detector_class` must be a class index or null, not {index}")
    return index


def _target(raw: Any, position: int, notes: List[str]) -> Target:
    if not isinstance(raw, Mapping):
        raise _Rejected(f"target {position} must be a mapping, not {type(raw).__name__}")
    _drop_unknown(raw, TARGET_FIELDS, "target", notes)
    identifier = _text(raw.get("id"), "id").strip() or f"t{position}"
    return Target(
        id=identifier,
        concept=_concept(raw.get("concept")),
        detector_class=_detector_class(raw.get("detector_class")),
        region=_choice(raw.get("region"), "region", REGIONS, DEFAULT_REGION),
        box_scale=_number(raw.get("box_scale"), "box_scale", *BOX_SCALE_RANGE,
                          DEFAULT_BOX_SCALE, notes),
        prompt=_text(raw.get("prompt"), "prompt"),
        negative_prompt=_text(raw.get("negative_prompt"), "negative_prompt"),
        denoise=_number(raw.get("denoise"), "denoise", *DENOISE_RANGE, DEFAULT_DENOISE, notes),
        seed_policy=_choice(raw.get("seed_policy"), "seed_policy", SEED_POLICIES,
                            DEFAULT_SEED_POLICY),
        max_instances=_number(raw.get("max_instances"), "max_instances", *MAX_INSTANCES_RANGE,
                              DEFAULT_MAX_INSTANCES, notes, as_int=True),
        priority=_number(raw.get("priority"), "priority", *PRIORITY_RANGE, DEFAULT_PRIORITY,
                         notes, as_int=True),
    )


def _targets(raw: Any, detector: DetectorVocabulary,
             notes: List[str]) -> Tuple[Target, ...]:
    """Every well-formed target the detector can actually serve.

    A malformed target rejects the whole plan - the producer is broken. A target the
    detector cannot find is dropped with its reason, because that is the user asking
    for something reasonable of a detector that cannot do it, and the rest of the
    plan is still renderable.
    """
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise _Rejected(f"`targets` must be a list, not {type(raw).__name__}")

    kept: List[Target] = []
    dropped: List[str] = []
    for position, item in enumerate(raw):
        target = _target(item, position, notes)
        refusal = detector.refusal(target.concept)
        if refusal:
            dropped.append(refusal)
            notes.append(f"dropped target '{target.id}': {refusal}")
            continue
        kept.append(target)

    if dropped and not kept:
        raise _Rejected("no target the detector can find: " + "; ".join(dropped))

    identifiers = [target.id for target in kept]
    duplicates = sorted({i for i in identifiers if identifiers.count(i) > 1})
    if duplicates:
        raise _Rejected(f"target ids must be unique; {', '.join(duplicates)} is repeated")
    return tuple(kept)


def _background(raw: Any, notes: List[str]) -> Background:
    if raw is None:
        return Background()
    if not isinstance(raw, Mapping):
        raise _Rejected(f"`background` must be a mapping, not {type(raw).__name__}")
    _drop_unknown(raw, BACKGROUND_FIELDS, "background", notes)
    return Background(
        action=_choice(raw.get("action"), "background.action", BACKGROUND_ACTIONS, PASSTHROUGH),
        prompt=_text(raw.get("prompt"), "background.prompt"),
    )


def _settings(raw: Any, notes: List[str]) -> GlobalSettings:
    if raw is None:
        return GlobalSettings()
    if not isinstance(raw, Mapping):
        raise _Rejected(f"`{GLOBAL_KEY}` must be a mapping, not {type(raw).__name__}")
    _drop_unknown(raw, GLOBAL_FIELDS, GLOBAL_KEY, notes)
    return GlobalSettings(
        fps_target=_number(raw.get("fps_target"), "fps_target", *FPS_TARGET_RANGE,
                           DEFAULT_FPS_TARGET, notes, as_int=True),
        detect_every_n=_number(raw.get("detect_every_n"), "detect_every_n",
                               *DETECT_EVERY_N_RANGE, DEFAULT_DETECT_EVERY_N, notes,
                               as_int=True),
    )


def _mode(raw: Any, targets: Tuple[Target, ...]) -> str:
    """The mode the plan asked for, or the only one it could carry out.

    Defaulting reads the plan rather than picking a constant: with no target there
    is nothing selective to do, and with one there is no reason to have named it.
    """
    mode = _choice(raw, "mode", MODES, SELECTIVE if targets else GLOBAL)
    if mode in (SELECTIVE, INVERSE) and not targets:
        raise _Rejected(f"mode '{mode}' names no target to select")
    return mode


def _honoured_note(targets: Tuple[Target, ...]) -> Optional[str]:
    """Say it when a plan asks for more per-target variation than B can render."""
    if not targets:
        return None
    first = targets[0]
    others = [t for t in targets[1:]
              if t.prompt != first.prompt or t.denoise != first.denoise]
    if not others:
        return None
    return (
        f"only the first target's prompt and denoise are honoured (issue #5 chose the "
        f"full-frame masked primitive, one prompt embedding per frame); "
        f"{', '.join(t.id for t in others)} carry their own and are not applied"
    )


def validate_plan(raw: Any, previous_version: int = INITIAL_PLAN_VERSION,
                  detector: Optional[DetectorVocabulary] = None) -> PlanValidation:
    """Turn anything into a `RenderPlan`, or into the reason it is not one.

    `previous_version` is the version of the plan this one replaces; the validator
    assigns `previous_version + 1`, and any `plan_version` the producer supplied is
    ignored. The version orders plans inside the worker, so a producer that could
    choose it could send the render loop backwards.
    """
    detector = DETECTOR_VOCABULARIES[ACTIVE_DETECTOR] if detector is None else detector
    notes: List[str] = []
    try:
        if not isinstance(raw, Mapping):
            raise _Rejected(f"a render plan must be a mapping, not {type(raw).__name__}")
        _drop_unknown(raw, PLAN_FIELDS, "plan", notes)
        targets = _targets(raw.get("targets"), detector, notes)
        honoured = _honoured_note(targets)
        if honoured:
            notes.append(honoured)
        plan = RenderPlan(
            plan_version=int(previous_version) + 1,
            source_prompt=_text(raw.get("source_prompt"), "source_prompt"),
            negative_prompt=_text(raw.get("negative_prompt"), "negative_prompt"),
            mode=_mode(raw.get("mode"), targets),
            targets=targets,
            background=_background(raw.get("background"), notes),
            settings=_settings(raw.get(GLOBAL_KEY), notes),
            confidence=_number(raw.get("confidence"), "confidence", *CONFIDENCE_RANGE,
                               DEFAULT_CONFIDENCE, notes),
            notes=_text(raw.get("notes"), "notes"),
        )
    except _Rejected as rejection:
        return PlanValidation(plan=None, errors=(str(rejection),), notes=tuple(notes))
    return PlanValidation(plan=plan, notes=tuple(notes))


# --- producers --------------------------------------------------------------


def plan_from_fields(target: str, style: str, negative_prompt: str = "",
                     region: str = DEFAULT_REGION, denoise: float = DEFAULT_DENOISE,
                     previous_version: int = INITIAL_PLAN_VERSION,
                     detector: Optional[DetectorVocabulary] = None) -> PlanValidation:
    """The GUI's producer: a target field and a style field become a plan.

    That is the whole control plane in v1. `target` is free text for the
    open-vocabulary detector; blank means "everything", which is `mode: "global"` and
    is exactly what the app does today. `style` is free text for StreamDiffusion.

    Returns the validation rather than the plan: what the user typed can be refused
    (an unservable concept, an impossible region), and the GUI has to be able to say
    so instead of sending a plan the worker will silently drop.
    """
    concept = (target or "").strip()
    raw: Dict[str, Any] = {
        "source_prompt": style,
        "negative_prompt": negative_prompt,
    }
    if concept:
        raw["targets"] = [{
            "id": "t0",
            "concept": concept,
            "region": region,
            "prompt": style,
            "negative_prompt": negative_prompt,
            "denoise": denoise,
        }]
    return validate_plan(raw, previous_version=previous_version, detector=detector)


# --- the priority case, hardcoded -------------------------------------------
#
# Issue #8 step 4: the selective path has to be drivable end to end before anything
# wires the GUI up, and the plan it is driven by is the *priority* case rather than
# the spec's red-hat example - a subtle sub-region change on people, at the region
# and the strength the product actually cares about. The prompt and the denoise are
# the ones issue #5's committed comparison measured this case at, so the demo is a
# re-run of a measurement rather than a fresh guess.

PRIORITY_CONCEPT = "person"
PRIORITY_REGION = "lower_half"
PRIORITY_PROMPT = ("trousers soaked through with a dark wet stain, damp fabric, "
                   "wet denim, photograph")
# 0.49 is the strength of t_index 40, which the masked primitive needed to make a
# visible change on this case and no more than that. See `t_index_for_denoise`.
PRIORITY_DENOISE = 0.49


def priority_case_plan(previous_version: int = INITIAL_PLAN_VERSION) -> RenderPlan:
    """The hardcoded demo plan: restyle the lower half of every person, gently.

    Goes through `validate_plan` like everything else - a hardcoded plan that
    bypassed the door could carry a field the worker does not honour and nobody
    would hear about it.
    """
    result = plan_from_fields(
        target=PRIORITY_CONCEPT, style=PRIORITY_PROMPT, region=PRIORITY_REGION,
        denoise=PRIORITY_DENOISE, previous_version=previous_version,
    )
    if result.plan is None:  # unreachable: `person` is servable and the region exists
        raise AssertionError(f"the priority-case plan did not validate: {result.reason}")
    return result.plan


def global_plan(prompt: str, negative_prompt: str = "",
                previous_version: int = INITIAL_PLAN_VERSION) -> RenderPlan:
    """Today's behaviour as a plan: one prompt, the whole frame, no detector.

    Cannot fail - it names no concept and no region - so it returns the plan itself.
    It is what the worker starts holding, so that "no plan yet" is not a third state
    the frame loop has to know about.
    """
    result = validate_plan(
        {"source_prompt": prompt, "negative_prompt": negative_prompt, "mode": GLOBAL},
        previous_version=previous_version,
    )
    if result.plan is None:  # unreachable: the literal above names nothing refusable
        raise AssertionError(f"the global plan did not validate: {result.reason}")
    return result.plan


# --- holding one across the frame loop --------------------------------------


@dataclass(frozen=True)
class FramePlan:
    """The plan one frame renders, and whether this is the first frame to use it."""

    plan: RenderPlan
    changed: bool


class ActivePlan:
    """The worker's plan, swapped between frames and never inside one.

    `submit` is the cold path - the control_queue drain - and only moves what the
    *next* frame will pick up. `begin_frame` is the one read the frame loop makes,
    at the top of the frame; everything downstream of it uses the returned plan, so
    a plan arriving mid-frame cannot change the frame being rendered.
    """

    def __init__(self, plan: RenderPlan) -> None:
        self._latest = plan
        self._frame = plan
        # The startup plan is what the wrapper was prepared with, so the first frame
        # has nothing to apply and should not pay for a prompt re-encode saying so.
        self._applied_version = plan.plan_version

    @property
    def latest(self) -> RenderPlan:
        """The newest submitted plan - what the next plan counts its version from."""
        return self._latest

    @property
    def frame_plan(self) -> RenderPlan:
        """The plan the frame in flight is rendering."""
        return self._frame

    def submit(self, plan: RenderPlan) -> None:
        self._latest = plan

    def begin_frame(self) -> FramePlan:
        """Bind this frame's plan. Called once, at the top of the frame."""
        self._frame = self._latest
        changed = self._frame.plan_version != self._applied_version
        self._applied_version = self._frame.plan_version
        return FramePlan(plan=self._frame, changed=changed)
