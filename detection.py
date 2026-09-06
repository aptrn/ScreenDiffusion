"""Identity: what the detector found, who it is, and how often to ask again.

Issue #7, spec 5.1 (C3, C4) and 8.5. The detector says *where* things are a few
times a second; this module says *which* thing each box is, and keeps saying so on
the frames between detector ticks. Stable IDs are what later let a seed and a
prompt embedding be pinned per object, which spec 8.5 names as the main lever
against flicker - so the tracker is not a convenience, it is the thing the whole
temporal-stability plan hangs off.

Three properties are load-bearing, and each has a test.

**Stdlib only, no torch, no GPU.** Like `render_plan.py`: the GUI process imports
it to read a status payload, the worker imports it on the frame path, and the merge
gate's GPU-free tier tests all of it. The half that loads weights and runs a
forward pass is `detector_worker.py`, which is the only module here that imports
ultralytics.

**A snapshot is immutable and the frame loop reads one reference.** `Tracks` is
frozen and is published whole by the detector thread; the frame loop binds it once
and never builds anything per frame. That is step 4 of the issue - "expose the
current tracks without allocating per frame on the hot path" - and it is also what
makes the read safe without a lock: a reference swap is atomic and half a track set
never reaches a frame.

**A missing detection is not a missing object.** The capture deque sheds frames
under load and the detector runs every Nth frame at best, so `Tracker` carries a
track across `MAX_MISSES` ticks it was not seen in, with its last box rather than
an extrapolated one. Dropping on the first miss would make a track's identity a
property of the detector's luck.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

# Detect every Nth frame when nobody has said - the plan's `global.detect_every_n`
# is what actually decides, and it is taken from here when a plan does not carry
# one. Imported rather than spelt again so the fallback and the plan default
# cannot drift apart; `render_plan` is stdlib-only too, so this costs nothing.
from render_plan import DEFAULT_DETECT_EVERY_N

# --- how identity is decided ------------------------------------------------

# How much two boxes must overlap for the later one to be the earlier one's object.
# The same threshold `bench.primitives.smooth_track` uses to build the committed
# tracks in issue #5, and for the same reason: below it, two boxes a third of a
# second apart are as likely to be two objects as one.
TRACK_IOU_THRESHOLD = 0.3

# How much of a new detection is taken into a matched track's box, per tick. Spec
# 8.5 lists box smoothing in the tracker among the levers against flicker; this is
# that lever, as an exponential moving average with the new box weighted 0.4.
TRACK_SMOOTHING = 0.4

# How many detector ticks a track may go unseen before it is dropped. Two, so a
# single blink - a frame the capture thread shed, an object briefly occluded - does
# not cost an object its identity, while a departed object is gone within a few
# hundred milliseconds rather than haunting the render.
MAX_MISSES = 2


class Box(NamedTuple):
    """A pixel box in capture space, `x0 <= x1` and `y0 <= y1`.

    The same shape as `bench.primitives.Box` and deliberately not imported from it:
    that module is the benchmark harness and imports numpy, and this one is shipped
    code held to the stdlib. A test holds the two `iou` implementations to the same
    answers.
    """

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


@dataclass(frozen=True)
class Detection:
    """One box the detector returned this tick, before anyone knows who it is."""

    box: Box
    concept: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"box": self.box.to_list(), "concept": self.concept,
                "confidence": round(self.confidence, 4)}


@dataclass(frozen=True)
class Track:
    """One object, across time. `track_id` is the identity everything else pins to.

    `hits` and `misses` are how confident the tracker is that this object is still
    there: `misses` counts detector ticks since it was last seen, and a box with
    `misses > 0` is the tracker bridging a gap rather than the detector reporting.
    """

    track_id: int
    box: Box
    concept: str
    confidence: float
    hits: int = 1
    misses: int = 0
    first_frame: int = 0
    last_frame: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        data["box"] = self.box.to_list()
        return data


@dataclass(frozen=True)
class Tracks:
    """Everything the frame loop knows about objects, as of one detector tick.

    Frozen and complete: the detector thread builds a whole one and publishes it
    with a single reference assignment, so a frame either sees the previous tick or
    this one and never a mixture of the two.
    """

    tracks: Tuple[Track, ...] = ()
    # The capture frame the detection ran on - not the frame reading this snapshot,
    # which is later by however long the detect took.
    frame_index: int = -1
    plan_version: int = 0
    concepts: Tuple[str, ...] = ()
    detector_ms: float = 0.0
    # How many detector ticks have been published. The frame loop can tell a stale
    # snapshot from a fresh one without timing anything.
    ticks: int = 0

    @property
    def count(self) -> int:
        return len(self.tracks)

    @property
    def ids(self) -> Tuple[int, ...]:
        return tuple(track.track_id for track in self.tracks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracks": [track.to_dict() for track in self.tracks],
            "frame_index": self.frame_index,
            "plan_version": self.plan_version,
            "concepts": list(self.concepts),
            "detector_ms": round(self.detector_ms, 4),
            "ticks": self.ticks,
        }


# What the frame loop reads before the first detect, and after a vocabulary change
# invalidates what was there. A snapshot rather than None, so "no tracks yet" is
# never a second state the frame loop has to know about.
EMPTY_TRACKS = Tracks()


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


def blend(previous: Box, current: Box, smoothing: float) -> Box:
    """`current` eased onto `previous`. `smoothing` 1.0 is the detector unfiltered."""
    return Box(*(int(round(smoothing * new + (1.0 - smoothing) * old))
                 for old, new in zip(previous, current)))


class Tracker:
    """C4: detections in, identified tracks out, one call per detector tick.

    Greedy nearest-overlap, strongest detection first: each detection takes the
    unclaimed track of its own concept that it overlaps most, if that overlap clears
    `iou_threshold`, and otherwise starts a new track with a new id. Ids are handed
    out monotonically and never reused - a recycled id would tell the seed cache
    that a new object is an old one, which is worse than having no cache.

    Cheap on purpose. Spec 7.1 gives tracking under 1 ms/frame and it only runs on
    detector ticks, so the O(tracks x detections) match is not worth an index.
    """

    def __init__(self, iou_threshold: float = TRACK_IOU_THRESHOLD,
                 smoothing: float = TRACK_SMOOTHING,
                 max_misses: int = MAX_MISSES) -> None:
        self.iou_threshold = iou_threshold
        self.smoothing = smoothing
        self.max_misses = max_misses
        self._tracks: Dict[int, Track] = {}
        self._next_id = 0

    @property
    def next_id(self) -> int:
        """The id the next new object will get. Only ever goes up."""
        return self._next_id

    def reset(self) -> None:
        """Forget every object, keeping the id counter.

        Called when the plan's concepts change: the tracks were about the old
        vocabulary, and an id that meant "the second person" must not silently come
        to mean "the second dog".
        """
        self._tracks = {}

    def update(self, detections: Sequence[Detection], frame_index: int = 0) -> Tuple[Track, ...]:
        """One detector tick. Returns the live tracks, strongest first."""
        ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
        claimed: Dict[int, Track] = {}

        for detection in ordered:
            matched = self._best_match(detection, claimed)
            if matched is None:
                track = Track(
                    track_id=self._next_id, box=Box(*detection.box),
                    concept=detection.concept, confidence=detection.confidence,
                    hits=1, misses=0, first_frame=frame_index, last_frame=frame_index,
                )
                self._next_id += 1
            else:
                track = dataclasses.replace(
                    matched,
                    box=blend(matched.box, detection.box, self.smoothing),
                    confidence=detection.confidence,
                    hits=matched.hits + 1, misses=0, last_frame=frame_index,
                )
            claimed[track.track_id] = track

        for track_id, track in self._tracks.items():
            if track_id in claimed:
                continue
            missed = dataclasses.replace(track, misses=track.misses + 1)
            if missed.misses <= self.max_misses:
                claimed[track_id] = missed

        self._tracks = claimed
        return tuple(sorted(claimed.values(),
                            key=lambda t: (-t.confidence, t.track_id)))

    def _best_match(self, detection: Detection,
                    claimed: Dict[int, Track]) -> Optional[Track]:
        """The unclaimed track this detection is, or None if it is a new object.

        Concept first, then overlap: a dog standing where a person stood is a new
        object, and a tracker that let the box decide would hand it the person's
        seed and prompt.
        """
        # `-track_id` is the tie-break: two equal overlaps go to the older object,
        # deterministically, and `max` never has to compare two `Track`s - which are
        # frozen dataclasses and not ordered.
        candidates = [(iou(track.box, detection.box), -track_id, track)
                      for track_id, track in self._tracks.items()
                      if track_id not in claimed and track.concept == detection.concept]
        overlap, _, track = max(candidates, default=(0.0, 0, None))
        return track if track is not None and overlap >= self.iou_threshold else None


# --- the cadence ------------------------------------------------------------


def is_detect_frame(frame_index: int, detect_every_n: int = DEFAULT_DETECT_EVERY_N) -> bool:
    """Should the detector be offered this frame?

    The plan's `detect_every_n` is validated into 1..30 before it gets here, but the
    frame path is not the place to discover that something else sent a 0: a cadence
    that is not a cadence detects every frame, which is dear and correct, rather
    than raising inside the render loop.
    """
    if detect_every_n <= 0:
        return True
    return frame_index % detect_every_n == 0


def amortised_ms(detector_ms: float, detect_every_n: int) -> float:
    """What one detect costs per *frame* at this cadence - spec 7.1's detection row.

    The Gate's third item is this function being true of the running system: raising
    `detect_every_n` spreads the same detect over more frames and must measurably
    lower what detection costs a frame.
    """
    return detector_ms / detect_every_n if detect_every_n > 0 else detector_ms


def detection_status(tracks: Tracks, detect_every_n: int) -> Dict[str, Any]:
    """What the dashboard needs to know about detection, from one snapshot."""
    return {
        "detections": tracks.count,
        "detector_ms": round(tracks.detector_ms, 2),
        "detect_every_n": detect_every_n,
        "amortised_ms": round(amortised_ms(tracks.detector_ms, detect_every_n), 2),
        "concepts": tracks.concepts,
        "ticks": tracks.ticks,
    }


def fps_payload(fps: int, tracks: Tracks, detect_every_n: int) -> Dict[str, Any]:
    """The worker's per-frame message on the existing fps channel.

    Issue #7 step 5. A run with no target has no detector loaded and nothing to say
    about one, so the detection keys are absent rather than zero: "0 objects found"
    and "nobody was asked" read the same on a dashboard and are not the same thing.
    """
    payload: Dict[str, Any] = {"fps": int(fps)}
    if tracks.ticks:
        payload.update(detection_status(tracks, detect_every_n))
    return payload
