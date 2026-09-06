"""C5, the region scheduler: which regions this frame renders, and which wait.

Issue #8, spec 5.1 (C5). The detector returns however many objects are in front of
the camera; a frame has a budget. This module is the rate limiter between the two:
tracks in, at most **K** regions out, and a round-robin so that the objects which
did not fit are the ones that go first next frame.

Stdlib only, like `render_plan.py` and `detection.py` - it runs on the frame path
in the worker and is tested in the merge gate's GPU-free tier, and it allocates a
handful of tuples per frame and nothing else.

Three decisions are worth stating.

**K counts masked regions, not diffusion calls.** Issue #5 chose primitive B - one
full-frame diffusion per frame, composited through a mask - so the engine's batch
size is not what limits K, and there are no crops to snap to a tile size. That half
of the issue's step 1 does not apply to the primitive that was chosen: the single
diffusion call is already the engine's canvas. What K limits is how much of the
frame is *composited*, which is the knob that survived, and it comes from the
honoured target's `max_instances`.

**The floor is on the region, and a skip is counted.** A region a few pixels across
is composited out of a full-frame diffusion at the same few pixels, and below one
VAE cell there is no restyled content in it at all - just resampled blur. Those
regions are dropped before the slots are handed out, so a permanently tiny object
cannot starve the rotation, and the count is carried in the selection rather than
discarded (the issue's first trap).

**The rotation cursor is a track id, not a position.** Track ids are monotonic and
never reused (`detection.Tracker`), so an object appearing or leaving between two
frames moves nobody else's place in the queue. With N eligible tracks and K slots
every track is selected at least once in `ceil(N/K)` frames, which is the Gate's
second item and what `tests/test_region_scheduler.py` asserts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from detection import Box, Tracks
from render_plan import (
    DEFAULT_MAX_INSTANCES,
    GLOBAL,
    REGIONS,
    RenderPlan,
    Target,
)

# region -> (top, bottom) as fractions of the box height. `center` is the middle
# third, the band the two thirds leave over - the only reading that makes the six
# names one consistent set. The same table as `bench.primitives.REGION_BANDS`,
# which cut the regions issue #5's committed comparison rendered; spelt again
# because that module imports numpy and this one runs in the GUI process, and held
# to it by a test.
REGION_BANDS: Dict[str, Tuple[float, float]] = {
    "full_box": (0.0, 1.0),
    "upper_third": (0.0, 1.0 / 3.0),
    "upper_half": (0.0, 0.5),
    "center": (1.0 / 3.0, 2.0 / 3.0),
    "lower_half": (0.5, 1.0),
    "lower_third": (2.0 / 3.0, 1.0),
}

# The smallest side a region may have and still be rendered. The engine's VAE
# downscales by 8, so a region under two cells on the canvas carries less than one
# cell of restyled detail and composites the resampler's blur instead of a render.
# Regions below it are skipped, and how many were skipped is recorded.
MIN_REGION_PX = 16


@dataclass(frozen=True)
class Region:
    """One object's region this frame: where to composite, and who it belongs to.

    `source` is the track's own box and `box` is what is actually rendered - banded
    by the target's `region`, dilated by its `box_scale` and clipped to the frame.
    Both are kept because the second is derived from the first and a reader looking
    at a mask needs to be able to get back to the object.
    """

    track_id: int
    concept: str
    target_id: str
    box: Box
    source: Box


@dataclass(frozen=True)
class Selection:
    """What one frame renders, and what it did not get to.

    Frozen and complete: the frame loop builds one, hands it to the compositor and
    forgets it. The counts are the scheduler's own account of the frame - how many
    objects were eligible, how many are waiting for a slot, how many were too small
    to render - and they are what the bench and the status line report.
    """

    regions: Tuple[Region, ...] = ()
    mode: str = GLOBAL
    plan_version: int = 0
    slots: int = 0
    # Every track this frame *could* have rendered, selected or not. Ids rather
    # than a count, so a reader can tell which object waited - which is what the
    # round-robin bound is about, and what the bench's coverage probe replays.
    candidate_ids: Tuple[int, ...] = ()
    skipped_small: int = 0

    @property
    def count(self) -> int:
        return len(self.regions)

    @property
    def candidates(self) -> int:
        return len(self.candidate_ids)

    @property
    def deferred(self) -> int:
        """Eligible tracks this frame had no slot for. They go first next frame."""
        return self.candidates - self.count

    @property
    def boxes(self) -> Tuple[Box, ...]:
        return tuple(region.box for region in self.regions)

    @property
    def track_ids(self) -> Tuple[int, ...]:
        return tuple(region.track_id for region in self.regions)

    def to_dict(self) -> dict:
        return {
            "regions": [region.box.to_list() for region in self.regions],
            "track_ids": list(self.track_ids),
            "mode": self.mode,
            "plan_version": self.plan_version,
            "slots": self.slots,
            "candidate_ids": list(self.candidate_ids),
            "deferred": self.deferred,
            "skipped_small": self.skipped_small,
        }


# What the frame loop holds before the first selection. A selection rather than
# None, so "nothing scheduled yet" is not a second state downstream.
EMPTY_SELECTION = Selection()


# --- the geometry -----------------------------------------------------------


def region_box(box: Box, region: str) -> Box:
    """The horizontal band of `box` that `region` names.

    Never empty: a one-pixel-high box still has a `lower_third`, and a region that
    rounded away to nothing would drop an object out of the render in silence.
    """
    if region not in REGION_BANDS:
        raise ValueError(f"{region!r} is not one of the region vocabulary {REGIONS}")
    box = Box(*box)
    top, bottom = REGION_BANDS[region]
    y0 = box.y0 + int(round(box.height * top))
    y1 = box.y0 + int(round(box.height * bottom))
    return Box(box.x0, y0, box.x1, max(y1, y0 + 1))


def dilate_box(box: Box, scale: float) -> Box:
    """`box` grown by `scale` about its own centre - the plan's `box_scale`.

    Applied to the detected box *before* the band is cut, which is what the field
    means: a person's `lower_half` of a box dilated 15% is a slightly generous
    lower half, not a lower half with a margin bolted on afterwards.
    """
    box = Box(*box)
    if scale == 1.0:
        return box
    grow_x = (box.width * (scale - 1.0)) / 2.0
    grow_y = (box.height * (scale - 1.0)) / 2.0
    return Box(int(round(box.x0 - grow_x)), int(round(box.y0 - grow_y)),
               int(round(box.x1 + grow_x)), int(round(box.y1 + grow_y)))


def clamp_box(box: Box, width: int, height: int) -> Box:
    """`box` clipped to a `width` x `height` frame, keeping at least one pixel."""
    box = Box(*box)
    x0 = min(max(0, box.x0), max(0, width - 1))
    y0 = min(max(0, box.y0), max(0, height - 1))
    return Box(x0, y0, min(max(box.x1, x0 + 1), width), min(max(box.y1, y0 + 1), height))


def rendered_box(box: Box, target: Target, width: int, height: int) -> Box:
    """The region `target` asks for around `box`, inside a `width` x `height` frame."""
    return clamp_box(region_box(dilate_box(box, target.box_scale), target.region),
                     width, height)


def slots_for(plan: RenderPlan) -> int:
    """K: how many regions one frame may composite.

    The honoured target's `max_instances`, for the same reason only its prompt and
    denoise reach the engine (issue #5's primitive: one diffusion call per frame,
    so one set of per-frame settings). A plan with no target schedules nothing, and
    the default is what it would have got had it named one.
    """
    target = plan.honoured_target
    return DEFAULT_MAX_INSTANCES if target is None else target.max_instances


# --- the scheduler ----------------------------------------------------------


class RegionScheduler:
    """C5: tracks in, at most K regions out, nobody starved.

    One instance lives in the frame loop and `select` is called once per frame.
    The only state it keeps is the rotation cursor and the running skip count, so a
    plan change costs it nothing - the plan is an argument, not a setting.
    """

    def __init__(self, min_side_px: int = MIN_REGION_PX) -> None:
        self.min_side_px = min_side_px
        self._cursor = 0
        self._skipped_small_total = 0

    @property
    def skipped_small_total(self) -> int:
        """Regions dropped by the size floor since the worker started (trap 1)."""
        return self._skipped_small_total

    def select(self, tracks: Tracks, plan: RenderPlan, width: int,
               height: int) -> Selection:
        """The regions this frame renders, under `plan`, out of `tracks`."""
        if plan.mode == GLOBAL or not plan.targets:
            return Selection(mode=plan.mode, plan_version=plan.plan_version)

        # One target per concept, the first in the plan winning, so a plan that
        # names a concept twice renders it one way rather than alternating.
        targets: Dict[str, Target] = {}
        for target in plan.targets:
            targets.setdefault(target.concept, target)

        candidates: List[Region] = []
        skipped = 0
        # By id, so the rotation below has one deterministic order to walk. The
        # snapshot's own order is by confidence, which moves under the tracker.
        for track in sorted(tracks.tracks, key=lambda t: t.track_id):
            target = targets.get(track.concept)
            if target is None:
                # A snapshot outlives a plan change by a tick; those boxes are
                # about a vocabulary nobody is rendering.
                continue
            box = rendered_box(track.box, target, width, height)
            if box.min_side < self.min_side_px:
                skipped += 1
                continue
            candidates.append(Region(track_id=track.track_id, concept=track.concept,
                                     target_id=target.id, box=box, source=track.box))

        self._skipped_small_total += skipped
        slots = slots_for(plan)
        chosen = self._rotate(candidates, slots)
        return Selection(
            regions=chosen, mode=plan.mode, plan_version=plan.plan_version,
            slots=slots,
            candidate_ids=tuple(region.track_id for region in candidates),
            skipped_small=skipped,
        )

    def status(self, selection: Selection) -> dict:
        """`selection_status`, with this scheduler's running skip count in it."""
        return selection_status(selection, self._skipped_small_total)

    def _rotate(self, candidates: List[Region], slots: int) -> Tuple[Region, ...]:
        """At most `slots` of `candidates`, resuming where the last frame stopped.

        The cursor is the id of the first track *not* served last frame, so the
        laps tile the candidate list: N tracks over K slots are all served in
        `ceil(N/K)` frames, and the next lap starts where that one ended rather
        than back at the front.
        """
        if slots <= 0 or not candidates:
            return ()
        if len(candidates) <= slots:
            self._cursor = 0
            return tuple(candidates)
        ids = [region.track_id for region in candidates]
        start = next((index for index, track_id in enumerate(ids)
                      if track_id >= self._cursor), 0)
        picked = tuple(candidates[(start + offset) % len(candidates)]
                       for offset in range(slots))
        self._cursor = ids[(start + slots) % len(ids)]
        return picked


def selection_status(selection: Selection, skipped_total: int = 0) -> dict:
    """What the dashboard needs to know about one frame's selection.

    Empty for a `global` plan: no region was asked for, so "0 regions rendered" and
    "nothing was selective about this frame" would read the same on a status line
    and are not the same thing - the rule `detection.fps_payload` already follows.
    """
    if selection.mode == GLOBAL:
        return {}
    return {
        "regions": selection.count,
        "slots": selection.slots,
        "deferred": selection.deferred,
        "skipped_small": selection.skipped_small,
        "skipped_small_total": skipped_total,
    }
