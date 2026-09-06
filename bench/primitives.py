"""What gets compared when the question is *which rendering primitive*, and the
arithmetic that turns the measurements into a decision.

Issue #5, spec 8.2. Four options are on the table:

    A. crop -> diffuse -> composite        (implemented here as `crop`)
    B. full-frame diffuse, masked composite (implemented here as `masked`)
    C. latent-space masking                 (assessed from code, not implemented)
    D. ControlNet-conditioned               (assessed from code, not implemented)

A and B are implemented and measured because they are the two that can be built
today; C and D are assessed in the spec against these measurements. The issue's
first trap is the reason this module exists at all: **do not decide on cost alone**.
The cheapest primitive that cannot express the priority case is not the winner, so
`decide` refuses to rank a primitive that did not express it, however fast it was.

Nothing here touches a GPU, a clip or a filesystem beyond reading a committed track
JSON. The milliseconds and the flicker come from `bench.primitive_runner`; what they
*mean* is decided here, where the merge gate can check it.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from bench.paths import REPO_ROOT

# --- the region vocabulary --------------------------------------------------------
#
# Fixed by the issue's fifth trap. A region names a horizontal band of a detected
# box, so "give them a red hat" can address a head without a segmentation stage.
# `center` is the middle third, the band the two thirds leave over - the only
# reading that makes the six names one consistent set.

FULL_BOX = "full_box"
UPPER_THIRD = "upper_third"
UPPER_HALF = "upper_half"
CENTER = "center"
LOWER_HALF = "lower_half"
LOWER_THIRD = "lower_third"

REGIONS: Tuple[str, ...] = (FULL_BOX, UPPER_THIRD, UPPER_HALF, CENTER,
                            LOWER_HALF, LOWER_THIRD)

# region -> (top, bottom) as fractions of the box height.
REGION_BANDS: Dict[str, Tuple[float, float]] = {
    FULL_BOX: (0.0, 1.0),
    UPPER_THIRD: (0.0, 1.0 / 3.0),
    UPPER_HALF: (0.0, 0.5),
    CENTER: (1.0 / 3.0, 2.0 / 3.0),
    LOWER_HALF: (0.5, 1.0),
    LOWER_THIRD: (2.0 / 3.0, 1.0),
}


class Box(NamedTuple):
    """A pixel box, `x0 <= x1` and `y0 <= y1`, half-open on the far edge."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def min_side(self) -> int:
        return min(self.width, self.height)

    def to_list(self) -> List[int]:
        return [self.x0, self.y0, self.x1, self.y1]


def region_box(box: Box, region: str) -> Box:
    """The band of `box` that `region` names.

    Never empty: a one-pixel-high box still has a `lower_third`, and a region that
    rounded away to nothing would silently drop an object from the render.
    """
    if region not in REGION_BANDS:
        raise ValueError(f"{region!r} is not one of the region vocabulary {REGIONS}")
    box = Box(*box)
    top, bottom = REGION_BANDS[region]
    y0 = box.y0 + int(round(box.height * top))
    y1 = box.y0 + int(round(box.height * bottom))
    return Box(box.x0, y0, box.x1, max(y1, y0 + 1))


def clamp_box(box: Box, width: int, height: int) -> Box:
    """`box` clipped to a `width` x `height` frame, keeping at least one pixel."""
    box = Box(*box)
    x0 = min(max(0, box.x0), max(0, width - 1))
    y0 = min(max(0, box.y0), max(0, height - 1))
    return Box(x0, y0, min(max(box.x1, x0 + 1), width), min(max(box.y1, y0 + 1), height))


# The Gate asks for at least one small-object case, "source region under 96 px".
# Read as the smallest side of the region actually rendered: that is the dimension
# the crop primitive has to upscale, and upscaling is what its quality floor is
# about. Issue #17 measured the committed clips against both readings and recorded
# that the area reading is not met by either of them - see `SmallObjectSummary`.
SMALL_OBJECT_PX = 96


@dataclass(frozen=True)
class SmallObjectSummary:
    """How small the regions in a track actually got, on both readings of "small".

    Two counts rather than one because the clips satisfy the Gate on one reading and
    not the other, and issue #17 said so when it committed them. `min_side_under` is
    the reading this comparison uses; `both_sides_under` is the stricter one, and
    reporting it as zero is the honest way to carry a gap the fixtures have.
    """

    regions: int
    min_side_px: Optional[int]
    max_side_px: Optional[int]
    min_side_under: int
    both_sides_under: int
    threshold_px: int
    statement: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def small_object_summary(regions: Sequence[Box],
                         threshold_px: int = SMALL_OBJECT_PX) -> SmallObjectSummary:
    """Count the rendered regions that are small, on each reading of small."""
    boxes = [Box(*box) for box in regions]
    if not boxes:
        return SmallObjectSummary(
            regions=0, min_side_px=None, max_side_px=None, min_side_under=0,
            both_sides_under=0, threshold_px=threshold_px,
            statement="no region was rendered, so nothing was measured for size",
        )
    sides = [box.min_side for box in boxes]
    min_side_under = sum(1 for box in boxes if box.min_side < threshold_px)
    both_under = sum(1 for box in boxes
                     if box.width < threshold_px and box.height < threshold_px)
    statement = (
        f"{len(boxes)} rendered regions, smallest side {min(sides)} px and largest "
        f"smallest-side {max(sides)} px. {min_side_under} have a side under "
        f"{threshold_px} px; {both_under} are under {threshold_px} px in *both* "
        f"dimensions."
    )
    if not both_under:
        statement += (
            " The Gate's small-object case is therefore met on the smallest-side "
            "reading and not on the small-in-area one - the gap issue #17 recorded "
            "when it committed these clips, and it is still open."
        )
    return SmallObjectSummary(
        regions=len(boxes), min_side_px=min(sides), max_side_px=max(sides),
        min_side_under=min_side_under, both_sides_under=both_under,
        threshold_px=threshold_px, statement=statement,
    )


# --- the primitives ---------------------------------------------------------------

CROP = "crop"
MASKED = "masked"


@dataclass(frozen=True)
class PrimitiveConfig:
    """One rendering primitive, and - as data - what it cannot express.

    `cannot_express` is a field rather than prose in a document because the Gate
    asks for it per primitive and because it is the half of the decision that no
    benchmark produces. A record that carried only milliseconds would let the next
    reader rediscover the limitation the hard way.
    """

    key: str
    spec_option: str
    name: str
    summary: str
    # True when the primitive issues one diffusion call per object rather than one
    # per frame. This is the whole cost difference between A and B.
    per_object: bool
    cannot_express: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


PRIMITIVES: Dict[str, PrimitiveConfig] = {
    CROP: PrimitiveConfig(
        key=CROP,
        spec_option="A",
        name="crop -> diffuse -> composite",
        summary="Crop each object's region out of the frame, resize it to the "
                "engine's 512x512 canvas, diffuse it on its own, resize it back and "
                "paste it in. One diffusion call per object.",
        per_object=True,
        cannot_express=(
            "A small crop, cheaply. Every region gets the engine's full 512x512 "
            "canvas whatever its source size, so a 45 px-wide region is upscaled "
            "11x before it is diffused and the model invents the detail it finds "
            "there - the small-crop quality floor. Cost scales with the object "
            "count, so the number of objects becomes a frame-budget decision rather "
            "than a detection one; and each object is diffused with no sight of the "
            "rest of the frame, so nothing ties two objects' output together."
        ),
    ),
    MASKED: PrimitiveConfig(
        key=MASKED,
        spec_option="B",
        name="full-frame diffuse, masked composite",
        summary="Resize the whole frame to the engine's 512x512 canvas, diffuse it "
                "once, resize it back and composite only the objects' regions. One "
                "diffusion call per frame, whatever the object count.",
        per_object=False,
        cannot_express=(
            "One prompt and one denoise per frame. Every object in the frame is "
            "rendered from the same text embedding at the same strength, so "
            "\"turn the dog into a cat and the man into a statue\" is two frames' "
            "work, not one. Nor can it spend detail where it matters: the whole "
            "frame is squeezed onto one 512x512 canvas, so a region occupying 45 px "
            "of a 1280 px-wide frame is diffused at ~18 px and comes back with "
            "roughly that much detail."
        ),
    ),
}


def calls_per_frame(primitive: str, objects: int) -> int:
    """Diffusion calls one frame costs. The cost model, stated as arithmetic.

    A frame with no object costs nothing under either primitive: there is no region
    to composite into, so a full-frame pass would be work thrown away.
    """
    if primitive not in PRIMITIVES:
        raise ValueError(f"{primitive!r} is not one of {tuple(PRIMITIVES)}")
    if objects <= 0:
        return 0
    return objects if PRIMITIVES[primitive].per_object else 1


# --- the cases --------------------------------------------------------------------

CLIPS_DIR = REPO_ROOT / "bench" / "clips"
TRACK_SUFFIX = ".track.json"

RESTYLE_CASE = "restyle-people"
IDENTITY_CASE = "identity-dog"

# What the two target behaviours are called in the record, and what selects the
# denoise strength for each.
RESTYLE = "restyle"
IDENTITY = "identity"

# Mean absolute difference, in 0-255 units, inside the rendered region against the
# source. Below this a restyle is not visible and the strength did not do its job;
# it is the criterion the reported "denoise strength this case needed" is selected
# by, and it is stated rather than eyeballed so the selection can be re-checked.
VISIBLE_CHANGE = 8.0
# For the identity case a change figure is not enough - a frame can change a great
# deal and still be a dog. The criterion is the detector: at least this fraction of
# the probed frames must come back labelled as the new identity.
IDENTITY_HIT_FRACTION = 0.5

# The ladder both cases are swept over. Higher `t_index` is *less* denoise: with
# `num_inference_steps=50` on the LCM schedule, index 20 is timestep 599 (noise
# amplitude 0.92) and index 45 is timestep 99 (0.32). One shared ladder, so the two
# cases' answers are points on the same axis.
DENOISE_LADDER: Tuple[int, ...] = (20, 25, 30, 35, 40, 45)


@dataclass(frozen=True)
class CaseConfig:
    """One target behaviour, on one committed clip, with one fixed box track.

    `priority` marks the sub-region restyle the issue calls the v1 case. It is what
    `decide` ranks on: a primitive that cannot express the priority case is out
    whatever it costs elsewhere.
    """

    name: str
    kind: str
    clip: str
    # What the track was detected for, and - for an identity change - what the
    # output should come back as instead.
    target: str
    becomes: Optional[str]
    region: str
    prompt: str
    priority: bool
    note: str
    denoise_ladder: Tuple[int, ...] = DENOISE_LADDER
    # Consecutive source frames. Consecutive because a flicker metric over sampled
    # frames measures the sampling interval, not the render.
    frames: int = 48
    start_frame: int = 0
    # Frames per swept denoise point. Small: the sweep selects a strength, it does
    # not measure a latency.
    sweep_frames: int = 4
    # Objects per frame taken from the track, strongest first. A cap, because the
    # crop primitive pays one diffusion call for each of them.
    max_objects: int = 4

    def replace(self, **changes) -> "CaseConfig":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["denoise_ladder"] = list(self.denoise_ladder)
        return data


CASES: Dict[str, CaseConfig] = {
    RESTYLE_CASE: CaseConfig(
        name=RESTYLE_CASE,
        kind=RESTYLE,
        clip="people.mp4",
        target="person",
        becomes=None,
        region=LOWER_HALF,
        prompt="trousers soaked through with a dark wet stain, damp fabric, "
               "wet denim, photograph",
        priority=True,
        note="The v1 priority case: a subtle sub-region change on people, low "
             "denoise, region lower_half. This is where the decision is made.",
        # Six, not the default four. The frame holds six people and the smallest of
        # them is also the lowest-confidence, so a cap of four drops exactly the
        # small-object case the Gate asks for - a 45 px-wide region among five
        # 150-290 px ones.
        max_objects=6,
    ),
    IDENTITY_CASE: CaseConfig(
        name=IDENTITY_CASE,
        kind=IDENTITY,
        clip="dog.mp4",
        target="dog",
        becomes="cat",
        region=FULL_BOX,
        prompt="a cat, feline face, whiskers, pointed ears, photograph",
        priority=False,
        note="The eventual case: an identity change at high denoise over the whole "
             "box. Not what v1 has to do, but what the primitive has to be able to "
             "grow into - and the case one-step SD-Turbo is most likely to fail.",
        # One dog in frame, so a cap above it costs nothing and guards against a
        # spurious second detection turning into a second diffusion call.
        max_objects=2,
    ),
}


def clip_path(clip: str) -> Path:
    """The committed reference clip (issue #17)."""
    return CLIPS_DIR / clip


def track_path(clip: str) -> Path:
    """The committed fixed box track for `clip`.

    Committed, and read rather than detected at run time, because the issue's second
    trap is that a comparison whose boxes come from a live detector is not
    reproducible - two runs would render different regions and the flicker figures
    would not be comparable.
    """
    return CLIPS_DIR / (Path(clip).stem + TRACK_SUFFIX)


# --- the fixed box track ------------------------------------------------------------

@dataclass(frozen=True)
class Track:
    """Boxes per frame for one clip, as committed.

    `boxes` is indexed by frame number and each entry is strongest-detection-first,
    so `max_objects` takes a prefix rather than an arbitrary subset.
    """

    clip: str
    target: str
    detector: str
    conf: float
    width: int
    height: int
    fps: float
    frame_count: int
    generated_utc: str
    boxes: Dict[int, List[Box]]

    def at(self, index: int, limit: Optional[int] = None) -> List[Box]:
        found = self.boxes.get(index, [])
        return found if limit is None else found[:limit]

    def to_dict(self) -> dict:
        return {
            "clip": self.clip, "target": self.target, "detector": self.detector,
            "conf": self.conf, "width": self.width, "height": self.height,
            "fps": self.fps, "frame_count": self.frame_count,
            "generated_utc": self.generated_utc,
            "frames": [{"index": index, "boxes": [box.to_list() for box in boxes]}
                       for index, boxes in sorted(self.boxes.items())],
        }


def track_from_dict(data: dict) -> Track:
    return Track(
        clip=data["clip"], target=data["target"], detector=data["detector"],
        conf=float(data["conf"]), width=int(data["width"]),
        height=int(data["height"]), fps=float(data["fps"]),
        frame_count=int(data["frame_count"]),
        generated_utc=str(data["generated_utc"]),
        boxes={int(frame["index"]): [Box(*(int(v) for v in box))
                                     for box in frame["boxes"]]
               for frame in data["frames"]},
    )


def load_track(clip: str) -> Track:
    """The committed track for `clip`, or an error naming how to regenerate it."""
    path = track_path(clip)
    if not path.is_file():
        raise FileNotFoundError(
            f"no committed box track at {path}. Regenerate it with "
            f"`python -m bench <case> --write-track`, which runs the detector over "
            f"the clip once and commits the boxes."
        )
    return track_from_dict(json.loads(path.read_text(encoding="utf-8")))


# --- denoise strength ---------------------------------------------------------------

def denoise_strength(alpha_cumprod: float) -> float:
    """How much of the latent the scheduler replaces with noise at one timestep.

    `sqrt(1 - alpha_cumprod)` is the noise amplitude the forward process applies, so
    it reads as a 0-1 strength the way diffusers' img2img `strength` does - and
    unlike a `t_index`, it is comparable across schedules. Reported beside the
    `t_index` because the index is what the app's control message carries and the
    strength is what a reader can interpret.
    """
    if not 0.0 <= alpha_cumprod <= 1.0:
        raise ValueError(f"alpha_cumprod {alpha_cumprod} is outside 0..1")
    return round(float((1.0 - alpha_cumprod) ** 0.5), 6)


@dataclass(frozen=True)
class DenoisePoint:
    """One rung of the denoise ladder, measured, with its two controls.

    `outside_change` is the first: a selective primitive must leave the rest of the
    frame alone, and a point where it did not is a compositing bug rather than a
    strength that worked.

    `resample_change` is the second, and it is the one that decides things. Every
    primitive resizes the region onto the engine's 512x512 canvas and back, and that
    round trip changes the pixels on its own - for the masked primitive, which
    squeezes a 1280x720 frame onto the canvas, it changes them a lot. It is measured
    by running the identical resize path with the diffusion call taken out, and
    subtracted before the "is this visible" question is asked, so a primitive cannot
    pass the criterion on its own blur.
    """

    t_index: int
    timestep: int
    strength: float
    region_change: float
    outside_change: float
    frames: int
    resample_change: float = 0.0
    identity_hits: Optional[int] = None
    identity_frames: Optional[int] = None

    @property
    def net_region_change(self) -> float:
        """The change the diffusion is responsible for, blur control removed."""
        return round(max(0.0, self.region_change - self.resample_change), 4)

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["net_region_change"] = self.net_region_change
        return data


@dataclass(frozen=True)
class DenoiseRequirement:
    """The strength a case turned out to need, and the rule that picked it."""

    met: bool
    t_index: int
    timestep: Optional[int]
    strength: Optional[float]
    kind: str
    rule: str
    statement: str
    points: List[DenoisePoint]

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["points"] = [point.to_dict() for point in self.points]
        return data


def required_denoise(points: Sequence[DenoisePoint], kind: str,
                     visible_change: float = VISIBLE_CHANGE,
                     hit_fraction: float = IDENTITY_HIT_FRACTION) -> DenoiseRequirement:
    """The least denoise that did the case's job, by a rule stated in the record.

    Least, not most: the Gate asks what each case *needed*, and every rung of extra
    strength is structure the render throws away. Higher `t_index` is less denoise,
    so "least that worked" is the largest qualifying index.

    A restyle qualifies on the size of the change it made inside the region. An
    identity change cannot - a frame can change enormously and still be a dog - so
    it qualifies on the detector coming back with the new identity. When nothing
    qualifies the requirement is `met=False`, and the fallback `t_index` is the
    strongest denoise that was tried, so the timed comparison still has a setting
    and the record still says the case was not achieved.
    """
    if not points:
        raise ValueError("no denoise point was measured")
    ordered = sorted(points, key=lambda point: point.t_index, reverse=True)
    strongest = min(points, key=lambda point: point.t_index)

    if kind == IDENTITY:
        rule = (f"the least denoise at which the detector read at least "
                f"{hit_fraction:.0%} of the probed frames as the new identity")
        qualifying = [point for point in ordered
                      if point.identity_frames
                      and (point.identity_hits or 0) >= hit_fraction * point.identity_frames]
    else:
        rule = (f"the least denoise whose mean absolute change inside the region, "
                f"net of the resize control, reached {visible_change:.0f}/255")
        qualifying = [point for point in ordered
                      if point.net_region_change >= visible_change]

    if not qualifying:
        statement = (
            f"No rung of the ladder met the criterion ({rule}). The strongest denoise "
            f"tried was t_index {strongest.t_index} (timestep {strongest.timestep}, "
            f"strength {strongest.strength:.2f}), which changed the region by "
            f"{strongest.net_region_change:.1f}/255 net of resizing"
        )
        if kind == IDENTITY and strongest.identity_frames:
            statement += (f" and was read as the new identity in "
                          f"{strongest.identity_hits}/{strongest.identity_frames} frames")
        statement += ". The comparison was timed at that setting and the case is "
        statement += "recorded as not achieved."
        return DenoiseRequirement(
            met=False, t_index=strongest.t_index, timestep=strongest.timestep,
            strength=strongest.strength, kind=kind, rule=rule, statement=statement,
            points=list(ordered),
        )

    chosen = qualifying[0]
    statement = (
        f"t_index {chosen.t_index} - timestep {chosen.timestep}, denoise strength "
        f"{chosen.strength:.2f}. Selected as {rule}; it changed the region by "
        f"{chosen.region_change:.1f}/255 - {chosen.net_region_change:.1f} of it net "
        f"of the {chosen.resample_change:.1f}/255 the resize alone costs - against "
        f"{chosen.outside_change:.1f}/255 outside it"
    )
    if kind == IDENTITY and chosen.identity_frames:
        statement += (f", and was read as the new identity in "
                      f"{chosen.identity_hits}/{chosen.identity_frames} probed frames")
    statement += "."
    return DenoiseRequirement(
        met=True, t_index=chosen.t_index, timestep=chosen.timestep,
        strength=chosen.strength, kind=kind, rule=rule, statement=statement,
        points=list(ordered),
    )


# --- the decision --------------------------------------------------------------------

@dataclass(frozen=True)
class Measurement:
    """What the decision needs to know about one primitive on one case."""

    case: str
    kind: str
    priority: bool
    primitive: str
    ms_per_frame: float
    flicker: Optional[float]
    objects_per_frame: float
    expresses: bool
    t_index: int


@dataclass(frozen=True)
class Decision:
    """One named primitive, and the reasoning that named it.

    Rendered as prose into the spec's decision section rather than stored, which is
    why this has no `to_dict`: it is a reading of the records, not a record.
    """

    primitive: Optional[str]
    priority_case: str
    ranking: Tuple[str, ...]
    cannot_express: str
    statement: str


def decide(measurements: Sequence[Measurement]) -> Decision:
    """Name the primitive, on cost *and* expressiveness against the priority case.

    The order is the issue's first trap made executable. Expressiveness filters
    first: a primitive that did not express the priority case is not ranked at all,
    however cheap it was. Cost decides among the rest, and the statement carries the
    flicker figures and the eventual case beside it, because a primitive that wins
    v1 and cannot grow into the identity change is a decision someone has to take
    knowingly rather than discover.
    """
    if not measurements:
        raise ValueError("no measurement to decide from")
    priority = [m for m in measurements if m.priority]
    if not priority:
        raise ValueError("no measurement on the priority case, which is what decides")
    case = priority[0].case

    able = sorted([m for m in priority if m.expresses], key=lambda m: m.ms_per_frame)
    ranking = tuple(m.primitive for m in sorted(priority, key=lambda m: m.ms_per_frame))
    if not able:
        return Decision(
            primitive=None, priority_case=case, ranking=ranking, cannot_express="",
            statement=(
                f"No implemented primitive expressed the priority case ({case}), so "
                f"there is nothing to choose between on cost. Options C and D in "
                f"spec 8.2 become the next thing to build rather than a contingency."
            ),
        )

    winner = able[0]
    config = PRIMITIVES[winner.primitive]
    others = [m for m in priority if m.primitive != winner.primitive]
    statement = (
        f"**{config.spec_option}, {config.name}** (`{winner.primitive}`). On the "
        f"priority case ({case}) it costs {winner.ms_per_frame:.1f} ms/frame at "
        f"{winner.objects_per_frame:.1f} objects per frame"
    )
    if others:
        cheapest_other = min(others, key=lambda m: m.ms_per_frame)
        ratio = cheapest_other.ms_per_frame / winner.ms_per_frame if winner.ms_per_frame else 0
        verdict = "expressed it too" if cheapest_other.expresses else "did not express it"
        statement += (
            f", against {cheapest_other.ms_per_frame:.1f} ms/frame for "
            f"{cheapest_other.primitive} ({ratio:.2f}x), which {verdict}"
        )
    statement += ". "
    if winner.flicker is not None:
        flicker_parts = [f"{m.primitive} {m.flicker:.2f}" for m in priority
                         if m.flicker is not None]
        statement += (
            f"Flicker over the pixels static in the source, 0-255 units, lower "
            f"steadier: {', '.join(flicker_parts)}. "
        )
    eventual = [m for m in measurements
                if not m.priority and m.primitive == winner.primitive]
    if eventual:
        grew = all(m.expresses for m in eventual)
        statement += (
            f"On the eventual case ({eventual[0].case}) it "
            f"{'held up' if grew else 'did not hold up'} at "
            f"{eventual[0].ms_per_frame:.1f} ms/frame. "
        )
    statement += (
        f"What it cannot express: {config.cannot_express} That limitation is "
        f"accepted for v1 - one concept at a time - and is recorded here rather "
        f"than discovered later."
    )
    return Decision(primitive=winner.primitive, priority_case=case, ranking=ranking,
                    cannot_express=config.cannot_express, statement=statement)


# --- the one-step finding the Gate asks for -----------------------------------------

# What it costs to stop being a one-step pipeline, spelt out once. The Gate asks for
# the *implication* to be recorded beside the finding, because "SD-Turbo could not do
# it" reads as a model problem and it is actually a schedule, an engine and a budget
# problem.
STEP_COUNT_IMPLICATION = (
    "More steps means a longer `t_index_list`, which keys a different TensorRT "
    "engine - ~5.0 GB on disk and 15-25 minutes to build, per configuration - and "
    "multiplies the UNet cost per frame, so the 33 ms budget in spec 7.1 has to be "
    "re-derived rather than adjusted. Changing the step *count* at run time tears "
    "the wrapper down and rebuilds, so it is not a slider."
)


def one_step_finding(case: CaseConfig, requirement: DenoiseRequirement,
                     achieved: bool) -> Optional[str]:
    """The finding, when one-step SD-Turbo could not do the case at any strength.

    None when it could: a finding that fires either way says nothing.
    """
    if achieved:
        return None
    ladder = ", ".join(str(rung) for rung in case.denoise_ladder)
    return (
        f"**One-step SD-Turbo did not perform the {case.kind} change coherently at "
        f"any denoise on the ladder** (t_index {ladder}, strongest tried "
        f"{requirement.t_index}). {requirement.statement} {STEP_COUNT_IMPLICATION}"
    )


# --- turning per-frame detections into a track ---------------------------------------

# How much two boxes must overlap to be the same object one frame later.
TRACK_IOU_THRESHOLD = 0.3
# How much of a new detection is taken into the smoothed box. Spec 8.5 lists box
# smoothing in the tracker among the levers against flicker, and it is applied *here*,
# once, so both primitives render exactly the same regions. Without it the crop
# primitive would be charged for detector jitter that a real tracker would absorb -
# and the masked primitive, whose output does not depend on the box at all, would not.
TRACK_SMOOTHING = 0.4


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes. 0 when they do not overlap."""
    a, b = Box(*a), Box(*b)
    overlap_w = min(a.x1, b.x1) - max(a.x0, b.x0)
    overlap_h = min(a.y1, b.y1) - max(a.y0, b.y0)
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0
    intersection = overlap_w * overlap_h
    union = a.width * a.height + b.width * b.height - intersection
    return 0.0 if union <= 0 else intersection / union


def _blend(previous: Box, current: Box, smoothing: float) -> Box:
    return Box(*(int(round(smoothing * new + (1.0 - smoothing) * old))
                 for old, new in zip(previous, current)))


def smooth_track(per_frame: Sequence[Sequence[Box]],
                 iou_threshold: float = TRACK_IOU_THRESHOLD,
                 smoothing: float = TRACK_SMOOTHING) -> List[List[Box]]:
    """Per-frame detections, matched across frames by overlap and smoothed.

    A greedy nearest-overlap match, strongest detection first: each detection takes
    the unclaimed track it overlaps most, if that overlap clears `iou_threshold`, and
    otherwise starts one. A track nobody claimed this frame is dropped rather than
    carried - a missing detection means there is no region to render, not a region
    rendered from a stale box.

    Detection order is preserved, so `max_objects` still takes the strongest few.
    """
    tracks: Dict[int, Box] = {}
    next_id = 0
    smoothed: List[List[Box]] = []
    for boxes in per_frame:
        claimed: Dict[int, Box] = {}
        frame_boxes: List[Box] = []
        for box in boxes:
            box = Box(*box)
            candidates = [(iou(tracks[key], box), key) for key in tracks
                          if key not in claimed]
            overlap, matched = max(candidates, default=(0.0, None))
            if matched is not None and overlap >= iou_threshold:
                key = matched
                box = _blend(tracks[key], box, smoothing)
            else:
                key, next_id = next_id, next_id + 1
            claimed[key] = box
            frame_boxes.append(box)
        tracks = claimed
        smoothed.append(frame_boxes)
    return smoothed
