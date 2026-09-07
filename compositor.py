"""C7, the compositor: the rendered regions blended back into the captured frame.

Issue #8, spec 5.1 (C7). The selective render path ends here: a diffused canvas, a
selection of regions from C5, and one rule - **everything outside the regions is the
captured pixel, unchanged**. Not "close to", unchanged: the issue's sharp criterion
is bit-identity, and `tests/test_compositor.py` asserts array equality rather than a
tolerance.

That rule is what shapes the feather. Alpha ramps up *inwards* from the region's own
edge and is exactly zero one pixel outside it, so a soft seam costs the region a few
pixels of strength and costs the background nothing. A ramp centred on the boundary
would look the same and would fail the gate (the issue's third trap).

It also holds the **output EMA** (issue #32, spec 8.5): `global.output_ema` pulls
each render back towards the one before it. That runs on the *rendered canvas*,
before the blend, and the ordering is the whole safety argument - an EMA on the
composited frame would make a background pixel a function of history, and this one
cannot, because the blend it feeds still copies the captured byte wherever alpha is
zero. What history reaches the screen is exactly what the mask lets through.

numpy, and nothing heavier. The blend is array arithmetic; it runs on the frame path
in the worker and is held whole by the merge gate's GPU-free tier. It is deliberately
*not* torch: the frame loop already has the capture on the host as an array for the
detector, and a compositor that needed a CUDA device could not be tested where the
rest of this path is tested. Moving it onto the GPU was an M2 question and issue #31
answered it in `device_compositor.py` - and the interface really was the same either
way, an alpha map and a blend, so what is here is still the reference the device path
is held to byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from detection import Box
from region_scheduler import Selection
from render_plan import CROP as PLAN_CROP
from render_plan import GLOBAL, INVERSE
from render_plan import MASKED as PLAN_MASKED

# How many pixels the alpha ramp takes to reach full strength, measured inwards
# from the region's edge. Enough to soften the rectangle the region scheduler cuts,
# small enough that a modest region still has a core at full strength - and it is
# capped per region at half the shorter side, so a small object is rendered at the
# strength that was asked for rather than at whatever the ramp had reached.
DEFAULT_FEATHER_PX = 6

# What one frame is: the capture as it stands, one full-frame diffusion, one
# full-frame diffusion composited through a mask, or one diffusion of a single
# region composited through the same mask. Actions rather than a boolean, because
# "there is nothing to restyle" and "restyle everything" are opposite answers and
# both of them are correct for some plan.
#
# The last two are the plan's `global.primitive` (issue #39, spec 8.2), so they are
# spelt where the plan spells them and `tests/test_compositor.py` holds the two
# names together. Every action costs the frame **one** diffusion call or none; what
# `crop` changes is where that call is spent, never how many there are.
PASSTHROUGH = "passthrough"
FULL_FRAME = "full_frame"
MASKED = PLAN_MASKED
CROP = PLAN_CROP


@dataclass(frozen=True)
class FrameRender:
    """What this frame needs: an action, the alpha map if it needs one, and - under
    `crop` - the box the diffusion call is spent on.

    `alpha` is None for the two actions that have no mask - the capture passed
    through, and a full-frame render composited nowhere. `crop` is None for every
    action but `crop`: it is the region the engine is handed instead of the whole
    capture, and the box the returned render has to be pasted back into.
    """

    action: str
    alpha: Optional[np.ndarray] = None
    crop: Optional[Box] = None

    @property
    def diffuses(self) -> bool:
        """Does this frame cost a diffusion call?"""
        return self.action != PASSTHROUGH

    @property
    def origin(self) -> Tuple[int, int]:
        """Where the render this frame produces sits in the captured frame.

        `(0, 0)` for a render that covers the whole capture, and the crop box's
        own corner for one that covers only it - which is exactly what `composite`
        takes, so a caller never has to branch on the action to blend.
        """
        return (0, 0) if self.crop is None else (self.crop.x0, self.crop.y0)


def _ramp(length: int, feather: int) -> np.ndarray:
    """Alpha along one axis of a region `length` pixels across.

    `feather + 1` pixels in from either edge reaches 1.0, and the outermost pixel
    *inside* the region gets `1 / (feather + 1)` - non-zero, so the region's own
    edge is still rendered, however faintly. A feather of 0 is a hard edge.
    """
    inward = np.arange(1, length + 1, dtype=np.float32)
    distance = np.minimum(inward, inward[::-1])
    return np.minimum(distance / (feather + 1.0), 1.0)


def feather_alpha(boxes: Sequence[Box], width: int, height: int,
                  feather_px: int = DEFAULT_FEATHER_PX) -> np.ndarray:
    """The alpha map for `boxes` on a `width` x `height` frame.

    Zero outside every box, ramping in from each box's edge, and the strongest of
    the overlapping values where two boxes meet - a pixel two regions both want
    rendered is rendered at the strength the more confident of them asked for,
    rather than at a seam.
    """
    alpha = np.zeros((height, width), dtype=np.float32)
    for box in boxes:
        box = Box(*box)
        if box.width <= 0 or box.height <= 0:
            continue
        # Capped per region: a ramp longer than half the shorter side would never
        # reach full strength, so the region would be rendered weaker than asked.
        feather = min(max(0, int(feather_px)), max(0, (box.min_side - 1) // 2))
        patch = np.minimum.outer(_ramp(box.height, feather), _ramp(box.width, feather))
        view = alpha[box.y0:box.y1, box.x0:box.x1]
        np.maximum(view, patch, out=view)
    return alpha


def painted_mask(alpha: np.ndarray) -> np.ndarray:
    """Where the render lands at all - what the flicker metric scores over."""
    return np.asarray(alpha) > 0.0


def alpha_bounds(alpha: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """The smallest rectangle holding every non-zero alpha, or None if there is none.

    The blend only touches this rectangle, so the rest of the frame is not merely
    computed back to the same value - it is never written at all.
    """
    rows = np.flatnonzero(alpha.any(axis=1))
    columns = np.flatnonzero(alpha.any(axis=0))
    if not rows.size or not columns.size:
        return None
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def ema_blend(previous, current, coefficient: float) -> np.ndarray:
    """`current` pulled `coefficient` of the way back towards `previous`. uint8 both.

    Written as `current + (previous - current) * coefficient` for the reason
    `composite` is written the other way up: the endpoints are exact in floating
    point, so a coefficient of 0 is the render byte for byte rather than the render
    rounded twice.
    """
    below = np.asarray(current, dtype=np.float32)
    above = np.asarray(previous, dtype=np.float32)
    return np.rint(below + (above - below) * float(coefficient)).astype(np.uint8)


def composite(source, rendered, alpha: np.ndarray,
              origin: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """`rendered` blended onto `source` through `alpha`. uint8 in, uint8 out.

    Where alpha is 0 the output is the source byte for byte, and where it is 1 it
    is the rendered byte for byte: the interpolation is written as
    `source + (rendered - source) * alpha`, whose two endpoints are exact in
    floating point, rather than as `source * (1 - alpha) + rendered * alpha`,
    whose are not.

    `origin` is where `rendered`'s own top-left corner sits in `source`. It is
    `(0, 0)` for a full-frame render and the crop box's corner for a `crop` one
    (issue #39), so the two primitives share this body rather than having one
    blend each: the alpha still decides every pixel, and a patch is only a
    smaller array to read it out of.
    """
    output = np.array(source, dtype=np.uint8, copy=True)
    bounds = alpha_bounds(alpha)
    if bounds is None:
        return output
    x0, y0, x1, y1 = bounds
    left, top = origin
    patch = np.asarray(rendered)
    if (x0 - left < 0 or y0 - top < 0
            or x1 - left > patch.shape[1] or y1 - top > patch.shape[0]):
        raise ValueError(
            f"a render of {patch.shape[1]}x{patch.shape[0]} at {origin} does not "
            f"cover the alpha's {(x0, y0, x1, y1)}")
    weights = alpha[y0:y1, x0:x1, None].astype(np.float32)
    below = output[y0:y1, x0:x1].astype(np.float32)
    above = patch[y0 - top:y1 - top, x0 - left:x1 - left].astype(np.float32)
    output[y0:y1, x0:x1] = np.rint(below + (above - below) * weights).astype(np.uint8)
    return output


class Compositor:
    """The frame loop's end of C7: a selection in, a frame's worth of work out.

    Holds the last alpha map it built and returns it again when the regions have
    not moved. Between detector ticks the boxes are identical by construction - the
    tracker publishes one snapshot and the frame loop reads it for `detect_every_n`
    frames - so without the cache two frames in three would allocate and fill a
    canvas-sized float array for a map they already had.
    """

    def __init__(self, feather_px: int = DEFAULT_FEATHER_PX,
                 output_ema: float = 0.0) -> None:
        self.feather_px = feather_px
        self.output_ema = float(output_ema)
        self._key: Optional[tuple] = None
        self._render: Optional[FrameRender] = None
        self._previous_render: Optional[np.ndarray] = None

    # --- the output EMA (issue #32, spec 8.5) -------------------------------
    #
    # `smooth` is applied to the *rendered canvas* and `blend` to what it returns,
    # never the other way round - the module docstring says why that ordering is
    # the whole safety argument.

    def set_output_ema(self, coefficient: float) -> None:
        """Take the plan's coefficient. The history is kept: a plan edit is not a
        scene change, and dropping the previous render would put a step in the
        output at the moment the user was looking at it."""
        self.output_ema = float(coefficient)

    def reset_ema(self) -> None:
        """Forget the previous render - after a frame that rendered nothing, or an
        engine swap. Averaging across such a gap blends two unrelated frames."""
        self._previous_render = None

    def smooth(self, rendered):
        """This frame's render, averaged with the ones before it. uint8 in and out.

        The state is the EMA's own output rather than the last raw render, so the
        coefficient names a time constant over the whole sequence and not a
        two-frame mean. Off, and on the first frame after a reset, the render is
        returned as it stands - starting from the capture or from grey would make
        every restyle a fade-in.
        """
        previous = self._previous_render
        if (self.output_ema <= 0.0 or previous is None
                or previous.shape != np.asarray(rendered).shape):
            self._previous_render = rendered
            return rendered
        smoothed = ema_blend(previous, rendered, self.output_ema)
        self._previous_render = smoothed
        return smoothed

    def frame(self, selection: Selection, width: int, height: int,
              primitive: str = MASKED) -> FrameRender:
        """What to do with this frame, under the plan the selection was made for.

        - `global`: diffuse the whole frame, composite nothing.
        - `selective` with regions: diffuse and composite them - the whole capture
          under `masked`, the one region under `crop`.
        - `selective` with none: pass the capture through - there is no pixel
          anyone asked to change, and a full-frame pass would be work thrown away.
        - `inverse`: the same mask, the other way up; with no region there is
          nothing to protect, so it is a whole-frame render.
        """
        key = (selection.mode, selection.boxes, width, height, self.feather_px,
               primitive)
        if key != self._key or self._render is None:
            self._key = key
            self._render = self._build(selection, width, height, primitive)
        return self._render

    def _build(self, selection: Selection, width: int, height: int,
               primitive: str) -> FrameRender:
        if selection.mode == GLOBAL:
            return FrameRender(action=FULL_FRAME)
        if not selection.regions:
            return FrameRender(
                action=FULL_FRAME if selection.mode == INVERSE else PASSTHROUGH)
        alpha = feather_alpha(selection.boxes, width, height, self.feather_px)
        if selection.mode == INVERSE:
            return FrameRender(action=MASKED,
                               alpha=(1.0 - alpha).astype(np.float32))
        # `crop` spends the frame's one call on one region, so it is only what this
        # frame is when the scheduler handed out exactly one slot's worth. Two
        # regions would be two calls - the 5.87x issue #5 measured - and that is
        # not a cost to pay by accident; `render_plan._crop_note` says so at the
        # moment a producer asks for a plan that will land here.
        if primitive == CROP and len(selection.regions) == 1:
            # The one box the alpha above was built from, so the render the engine
            # returns covers every pixel the blend will read it for.
            return FrameRender(action=CROP, alpha=alpha, crop=selection.boxes[0])
        return FrameRender(action=MASKED, alpha=alpha)

    @staticmethod
    def blend(source, rendered, alpha: np.ndarray,
              origin: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """`composite`, reachable from the object the frame loop already holds."""
        return composite(source, rendered, alpha, origin)
