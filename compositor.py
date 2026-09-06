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

numpy, and nothing heavier. The blend is array arithmetic; it runs on the frame path
in the worker and is held whole by the merge gate's GPU-free tier. It is deliberately
*not* torch: the frame loop already has the capture on the host as an array for the
detector, and a compositor that needed a CUDA device could not be tested where the
rest of this path is tested. Moving it onto the GPU is an M2 question, and the
interface here - an alpha map and a blend - is the same either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from detection import Box
from region_scheduler import Selection
from render_plan import GLOBAL, INVERSE

# How many pixels the alpha ramp takes to reach full strength, measured inwards
# from the region's edge. Enough to soften the rectangle the region scheduler cuts,
# small enough that a modest region still has a core at full strength - and it is
# capped per region at half the shorter side, so a small object is rendered at the
# strength that was asked for rather than at whatever the ramp had reached.
DEFAULT_FEATHER_PX = 6

# What one frame is: the capture as it stands, one full-frame diffusion, or one
# full-frame diffusion composited through a mask. Three actions rather than a
# boolean, because "there is nothing to restyle" and "restyle everything" are
# opposite answers and both of them are correct for some plan.
PASSTHROUGH = "passthrough"
FULL_FRAME = "full_frame"
MASKED = "masked"


@dataclass(frozen=True)
class FrameRender:
    """What this frame needs: an action, and the alpha map if it needs one.

    `alpha` is None for the two actions that have no mask - the capture passed
    through, and a full-frame render composited nowhere.
    """

    action: str
    alpha: Optional[np.ndarray] = None

    @property
    def diffuses(self) -> bool:
        """Does this frame cost a diffusion call?"""
        return self.action != PASSTHROUGH


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


def composite(source, rendered, alpha: np.ndarray) -> np.ndarray:
    """`rendered` blended onto `source` through `alpha`. uint8 in, uint8 out.

    Where alpha is 0 the output is the source byte for byte, and where it is 1 it
    is the rendered byte for byte: the interpolation is written as
    `source + (rendered - source) * alpha`, whose two endpoints are exact in
    floating point, rather than as `source * (1 - alpha) + rendered * alpha`,
    whose are not.
    """
    output = np.array(source, dtype=np.uint8, copy=True)
    bounds = alpha_bounds(alpha)
    if bounds is None:
        return output
    x0, y0, x1, y1 = bounds
    weights = alpha[y0:y1, x0:x1, None].astype(np.float32)
    below = output[y0:y1, x0:x1].astype(np.float32)
    above = np.asarray(rendered)[y0:y1, x0:x1].astype(np.float32)
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

    def __init__(self, feather_px: int = DEFAULT_FEATHER_PX) -> None:
        self.feather_px = feather_px
        self._key: Optional[tuple] = None
        self._render: Optional[FrameRender] = None

    def frame(self, selection: Selection, width: int, height: int) -> FrameRender:
        """What to do with this frame, under the plan the selection was made for.

        - `global`: diffuse the whole frame, composite nothing.
        - `selective` with regions: diffuse the whole frame and composite them.
        - `selective` with none: pass the capture through - there is no pixel
          anyone asked to change, and a full-frame pass would be work thrown away.
        - `inverse`: the same mask, the other way up; with no region there is
          nothing to protect, so it is a whole-frame render.
        """
        key = (selection.mode, selection.boxes, width, height, self.feather_px)
        if key != self._key or self._render is None:
            self._key, self._render = key, self._build(selection, width, height)
        return self._render

    def _build(self, selection: Selection, width: int, height: int) -> FrameRender:
        if selection.mode == GLOBAL:
            return FrameRender(action=FULL_FRAME)
        if not selection.regions:
            return FrameRender(
                action=FULL_FRAME if selection.mode == INVERSE else PASSTHROUGH)
        alpha = feather_alpha(selection.boxes, width, height, self.feather_px)
        if selection.mode == INVERSE:
            alpha = (1.0 - alpha).astype(np.float32)
        return FrameRender(action=MASKED, alpha=alpha)

    @staticmethod
    def blend(source, rendered, alpha: np.ndarray) -> np.ndarray:
        """`composite`, reachable from the object the frame loop already holds."""
        return composite(source, rendered, alpha)
