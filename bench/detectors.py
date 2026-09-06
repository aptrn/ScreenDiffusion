"""Which detectors get measured, and the arithmetic that turns a timing into a verdict.

Issue #4, spec 8.1. Two entries, not five. YOLO-World is the **candidate** - the
LLM prompt-compiler is cut from v1, so the detector's own text encoder is the only
path from what a user types to what gets found, and a COCO-only detector would cap
the product at 80 nouns. YOLOv8n is a **speed floor**: it exists in this registry to
say how much the open vocabulary costs, not to compete for the job. OWLv2 and
Grounding DINO are contingencies and are deliberately absent; they are worth a run
only if YOLO-World's accuracy proves inadequate.

Nothing here touches a GPU or a filesystem. The measured milliseconds come from
`bench.detector_runner`; what they *mean* - amortised over the detect cadence,
against the spec 7.1 budget - is decided here, where the merge gate can check it.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

# Spec 7.1 gives detection 4-8 ms amortised, running "every 3rd frame". Amortised
# is the operative word: one detect is paid once and spread over the frames until
# the next one, so the budget question is `ms / cadence`, never `ms`.
DETECT_BUDGET_MIN_MS = 4.0
DETECT_BUDGET_MAX_MS = 8.0
DEFAULT_CADENCE = 3

DEFAULT_INPUT_SIZE = 640

# What has to be resident while a detector is timed - the issue's fourth trap. The
# cached TensorRT configuration issue #3 already measured, so the combined-VRAM
# figure sits beside a diffusion number that exists rather than one built for it.
DEFAULT_DIFFUSION_SCENARIO = "img2img-tensorrt-512x512-b1"
# Where detector weights live under the shared models root. Hundreds of MB, and
# `models/` is gitignored - see the issue's second trap.
WEIGHTS_SUBDIR = "detectors"
WEIGHTS_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0"

# The vocabulary a run puts in front of the detector, and the one it swaps to when
# measuring what a vocabulary change costs. Two disjoint sets, so a swap cannot be
# quietly served from whatever the first encode left behind.
TARGET_VOCABULARY: Tuple[str, ...] = ("person", "red mug", "dog")
SWAP_VOCABULARY: Tuple[str, ...] = ("cat", "blue chair", "laptop")

# CLIP ViT-B/32 is what ultralytics' YOLO-World uses to embed a vocabulary. Named
# here because the cost of a vocabulary change is the cost of running *this*.
TEXT_ENCODER = "clip:ViT-B/32"


@dataclass(frozen=True)
class DetectorConfig:
    """One measurable detector, serialised whole into the result file."""

    name: str
    weights: str
    # The ultralytics entry point: a YOLO-World model has a text head and a
    # `set_classes`, a plain YOLO has 80 fixed classes and neither.
    loader: str
    open_vocabulary: bool
    # "candidate" or "speed floor" - what this row is in the table *for*.
    role: str
    note: str
    imgsz: int = DEFAULT_INPUT_SIZE
    conf: float = 0.05
    # ~4 s of detects after ~1 s of warmup. Both are longer than they look they need
    # to be: after the cooldown gate releases, this laptop's SM clock takes a second
    # or two to climb back to boost, so a short run reports the ramp rather than the
    # detector - see the note on `passes` in `frame_path_verdict`.
    reps: int = 200
    warmup_reps: int = 50
    vocabulary: Tuple[str, ...] = TARGET_VOCABULARY
    swap_vocabulary: Tuple[str, ...] = SWAP_VOCABULARY

    @property
    def weights_url(self) -> str:
        return f"{WEIGHTS_BASE_URL}/{self.weights}"

    def replace(self, **changes) -> "DetectorConfig":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["vocabulary"] = list(self.vocabulary)
        data["swap_vocabulary"] = list(self.swap_vocabulary)
        data["weights_url"] = self.weights_url
        return data


@dataclass(frozen=True)
class ConceptProbe:
    """One thing the detector is asked to find, and the picture it is asked to find it in.

    `coco_equivalent` is the nearest of the 80 COCO classes, or None when there is
    none. That is the 80-noun cap expressed as data: `person` and `dog` are classes
    a closed detector already has, and `red mug` is not - the nearest, `cup`, drops
    the attribute that made the instruction specific.
    """

    concept: str
    kind: str
    image_name: str
    image_url: str
    coco_equivalent: Optional[str]
    note: str


CONCEPT_PROBES: Tuple[ConceptProbe, ...] = (
    ConceptProbe(
        concept="person",
        kind="COCO class",
        image_name="bus.jpg",
        image_url="https://ultralytics.com/images/bus.jpg",
        coco_equivalent="person",
        note="Issue #4 step 4(a). The baseline: if this fails nothing else matters.",
    ),
    ConceptProbe(
        concept="red mug",
        kind="open vocabulary",
        image_name="red-mug.jpg",
        image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/"
                  "Red_mug%2C_red_table_-_Flickr_-_sampsyo.jpg/"
                  "1280px-Red_mug%2C_red_table_-_Flickr_-_sampsyo.jpg",
        coco_equivalent=None,
        note="Issue #4 step 4(b). A red mug on a red table, so the noun alone is not "
             "enough and the attribute has to do some work. No COCO class expresses "
             "it: the nearest, `cup`, is the whole point of the 80-noun cap.",
    ),
    ConceptProbe(
        concept="dog",
        kind="non-COCO animal",
        image_name="dog.jpg",
        image_url="https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        coco_equivalent="dog",
        note="Issue #4 step 4(c).",
    ),
)

# The raw screen, with nothing composited into it. Not a concept probe - it is the
# control: whatever it finds is what the detector says about this desktop as it
# actually was, and a run that reported only the composited frames could not tell a
# working detector from one that hallucinates.
DESKTOP_CONTROL = "desktop"

PRIMARY_DETECTOR = "yolo-world-s-640"
SPEED_FLOOR_DETECTOR = "yolov8n-640"

DETECTORS: Dict[str, DetectorConfig] = {
    PRIMARY_DETECTOR: DetectorConfig(
        name=PRIMARY_DETECTOR,
        weights="yolov8s-worldv2.pt",
        loader="YOLOWorld",
        open_vocabulary=True,
        role="candidate",
        note="The candidate. Text embeddings are precomputed for a fixed vocabulary "
             "and detection then runs at YOLOv8 speed, which is this architecture's "
             "cold-path / hot-path split exactly.",
    ),
    SPEED_FLOOR_DETECTOR: DetectorConfig(
        name=SPEED_FLOOR_DETECTOR,
        weights="yolov8n.pt",
        loader="YOLO",
        open_vocabulary=False,
        role="speed floor",
        note="For comparison only. 80 COCO classes, no text encoder - the floor the "
             "open vocabulary is paid for against.",
    ),
}


def weights_path(config: DetectorConfig, models_dir: Union[str, Path]) -> Path:
    """Where `config`'s weights live under the shared models root."""
    return Path(models_dir) / WEIGHTS_SUBDIR / config.weights


def ordinal(n: int) -> str:
    """`3` -> `3rd`. Used in the verdict, which has to name its cadence in words."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def amortised_ms(ms_per_detect: float, cadence: int) -> float:
    """What one detect costs per *frame* when it runs every `cadence` frames."""
    if cadence < 1:
        raise ValueError(f"a detect cadence of {cadence} frames is not a cadence")
    return ms_per_detect / cadence


def required_cadence(ms_per_detect: float, budget_max_ms: float = DETECT_BUDGET_MAX_MS) -> int:
    """The smallest whole-frame cadence at which `ms_per_detect` lands in budget."""
    return max(1, math.ceil(ms_per_detect / budget_max_ms))


@dataclass(frozen=True)
class BudgetVerdict:
    """Does this detector fit the spec 7.1 detection row, and at what cadence.

    The cadence is part of the verdict, never an assumption behind it: "13.5 ms
    fits" is not a claim anyone can check, and "13.5 ms is 4.5 ms/frame at one
    detect every 3rd frame, inside the 4-8 ms budget" is.
    """

    ms_per_detect: float
    cadence: int
    amortised_ms: float
    budget_min_ms: float
    budget_max_ms: float
    fits: bool
    cadence_required: int
    statement: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def budget_verdict(ms_per_detect: float, cadence: int = DEFAULT_CADENCE,
                   budget_min_ms: float = DETECT_BUDGET_MIN_MS,
                   budget_max_ms: float = DETECT_BUDGET_MAX_MS) -> BudgetVerdict:
    """`ms_per_detect` judged against the spec 7.1 budget at a named cadence."""
    amortised = amortised_ms(ms_per_detect, cadence)
    needed = required_cadence(ms_per_detect, budget_max_ms)
    fits = amortised <= budget_max_ms
    verdict = "inside" if fits else "over"
    statement = (
        f"{ms_per_detect:.2f} ms per detect at one detect every {ordinal(cadence)} "
        f"frame is {amortised:.2f} ms/frame amortised, {verdict} the "
        f"{budget_min_ms:.0f}-{budget_max_ms:.0f} ms budget"
    )
    if not fits:
        statement += f"; it would need one detect every {ordinal(needed)} frame"
    return BudgetVerdict(
        ms_per_detect=round(ms_per_detect, 4), cadence=cadence,
        amortised_ms=round(amortised, 4), budget_min_ms=budget_min_ms,
        budget_max_ms=budget_max_ms, fits=fits, cadence_required=needed,
        statement=statement,
    )


@dataclass(frozen=True)
class FramePathVerdict:
    """Whether changing the vocabulary cost anything on the frame path.

    The claim issue #4 asks to be measured rather than assumed, and measuring it
    turned up two different answers that a single number would have hidden.

    `unaffected` is about the **steady state**: does detecting against the new
    vocabulary cost more per frame than the old one. `free_on_frame_path` is the
    whole claim, and it is also about the **first detect after the change** - which
    on ultralytics is far dearer than a steady one, because `YOLOWorld.set_classes`
    drops `self.predictor` and the next `predict` rebuilds it. That cost is real,
    lands on a frame, and is avoidable: one throwaway detect on the cold path, after
    the change and before the plan goes live, pays it where nobody is watching.
    """

    before_ms: float
    after_ms: float
    delta_ms: float
    delta_fraction: float
    tolerance: float
    change_ms: float
    unaffected: bool
    free_on_frame_path: bool
    first_detect_ms: Optional[float]
    first_detect_overhead_ms: Optional[float]
    statement: str
    # The medians of the timed passes the figures came from, in the order they ran.
    # Empty when the comparison was a plain before/after.
    passes: Tuple[float, ...] = ()

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["passes"] = list(self.passes)
        return data


def frame_path_verdict(before_ms: float, after_ms: float, change_ms: float,
                       tolerance: float = 0.10,
                       passes: Sequence[float] = (),
                       first_detect_ms: Optional[float] = None) -> FramePathVerdict:
    """Compare the detect latency either side of a vocabulary change.

    `tolerance` is a fraction of the before figure rather than an absolute: a few
    tenths of a millisecond is noise on a 13 ms detect and a real regression on a
    2 ms one.

    `passes` records that the two figures were *interleaved* - alternating blocks of
    the two vocabularies, pooled per arm - rather than measured one arm after the
    other. That is not ceremony. Measured plainly on this laptop a vocabulary change
    appeared to make detection 24% *faster*, because the SM clock was still climbing
    out of the cooldown while the first arm ran. Interleaved and pooled, whatever the
    clock is doing lands on both arms.

    The values in `passes` are the median of each timed block in the order the
    blocks ran, so a reader can see how much the machine drifted underneath the
    comparison.
    """
    delta = after_ms - before_ms
    fraction = delta / before_ms if before_ms else 0.0
    unaffected = abs(fraction) <= tolerance
    overhead = None if first_detect_ms is None else first_detect_ms - before_ms
    first_detect_free = overhead is None or overhead <= tolerance * before_ms

    if unaffected:
        statement = (
            f"Steady-state detect latency is {before_ms:.2f} ms on the original "
            f"vocabulary and {after_ms:.2f} ms on the new one ({fraction:+.1%}, within "
            f"{tolerance:.0%}), so detecting against a changed vocabulary is no dearer. "
            f"The {change_ms:.1f} ms the change itself cost is paid once, on the cold path."
        )
    else:
        statement = (
            f"Steady-state detect latency moved from {before_ms:.2f} ms to "
            f"{after_ms:.2f} ms across the vocabulary change ({fraction:+.1%}, outside "
            f"{tolerance:.0%}), so detecting against the new vocabulary is not free."
        )
    if overhead is not None and not first_detect_free:
        statement += (
            f" The change is not free on the frame path either: the *first* detect "
            f"after it costs {first_detect_ms:.2f} ms, {overhead:.2f} ms more than a "
            f"steady one, because ultralytics' `set_classes` drops the predictor and "
            f"the next call rebuilds it. Re-warm the detector with one throwaway "
            f"detect on the cold path and the frame path never sees it."
        )
    elif overhead is not None:
        statement += (
            f" The first detect after the change costs {first_detect_ms:.2f} ms, "
            f"{overhead:+.2f} ms against a steady one - inside {tolerance:.0%}, so "
            f"nothing lands on the frame path."
        )
    if passes:
        blocks = ", ".join(f"{ms:.2f}" for ms in passes)
        statement += (
            f" The two vocabularies ran in alternating blocks and each arm is pooled "
            f"over all of its detects, so a drifting clock lands on both; the block "
            f"medians in order were {blocks} ms."
        )
    return FramePathVerdict(
        before_ms=round(before_ms, 4), after_ms=round(after_ms, 4),
        delta_ms=round(delta, 4), delta_fraction=round(fraction, 6),
        tolerance=tolerance, change_ms=round(change_ms, 4),
        unaffected=unaffected, free_on_frame_path=unaffected and first_detect_free,
        first_detect_ms=None if first_detect_ms is None else round(first_detect_ms, 4),
        first_detect_overhead_ms=None if overhead is None else round(overhead, 4),
        statement=statement, passes=tuple(round(ms, 4) for ms in passes),
    )


@dataclass(frozen=True)
class Candidate:
    """What the recommendation needs to know about one measured detector."""

    name: str
    ms_per_detect: float
    open_vocabulary: bool
    concepts_resolved: int
    concepts_probed: int


@dataclass(frozen=True)
class Recommendation:
    """One named detector, the ranking behind it, and why it won.

    Rendered as prose into the spec 8.1 block rather than stored, which is why this
    is the one dataclass here without a `to_dict`: it is a reading of the records,
    not a record.
    """

    name: str
    fits: bool
    cadence: int
    cadence_required: int
    ranking: Tuple[str, ...]
    reason: str


def recommend(candidates: Sequence[Candidate],
              cadence: int = DEFAULT_CADENCE,
              budget_max_ms: float = DETECT_BUDGET_MAX_MS) -> Recommendation:
    """Pick a detector by ranking, the way step 5 asks - not by absolute numbers.

    Vocabulary beats speed. With the prompt compiler cut from v1 nothing maps free
    text onto a COCO class, so a closed-vocabulary detector caps the product at 80
    nouns however fast it is; the only reason to prefer one is that no open
    -vocabulary candidate could actually find what it was asked for. A candidate
    that fits only at a slower cadence still wins - and the verdict says which
    cadence, because that is a design constraint rather than a footnote.
    """
    if not candidates:
        raise ValueError("no measured candidate to recommend")
    by_speed = sorted(candidates, key=lambda c: c.ms_per_detect)
    ranking = tuple(candidate.name for candidate in by_speed)

    # The ones that can be asked for anything *and* found what they were asked for.
    # Any of those beats every closed-vocabulary candidate; among them, speed decides.
    open_vocabulary = [candidate for candidate in by_speed
                       if candidate.open_vocabulary
                       and candidate.concepts_resolved == candidate.concepts_probed]
    winner = (open_vocabulary or by_speed)[0]
    verdict = budget_verdict(winner.ms_per_detect, cadence, budget_max_ms=budget_max_ms)

    if open_vocabulary:
        # It usually is not the fastest, and saying so is the point - but it can be,
        # and a recommendation that insisted otherwise would be reporting a fiction.
        speed_aside = (
            "It is also the fastest measured, and speed is not the criterion: "
            if winner.name == ranking[0] else
            f"It is not the fastest ({ranking[0]} is), and that is not the criterion: "
        )
        reason = (
            f"{winner.name} resolved all {winner.concepts_probed} probed concepts and "
            f"{verdict.statement}. {speed_aside}with the prompt compiler cut from v1, "
            f"a closed vocabulary caps the product at the 80 COCO nouns whatever it "
            f"costs."
        )
    else:
        unresolved = [candidate for candidate in candidates if candidate.open_vocabulary]
        missed = ", ".join(f"{c.name} did not resolve "
                           f"{c.concepts_probed - c.concepts_resolved} of "
                           f"{c.concepts_probed} concepts" for c in unresolved) or \
            "no open-vocabulary candidate was measured"
        reason = (
            f"{missed}, so the open vocabulary is not usable as measured and the "
            f"fastest candidate takes it: {winner.name}, {verdict.statement}. The 80-"
            f"noun cap comes with it, and spec 8.1's contingencies are then worth a run."
        )
    return Recommendation(
        name=winner.name, fits=verdict.fits, cadence=cadence,
        cadence_required=verdict.cadence_required, ranking=ranking, reason=reason,
    )
