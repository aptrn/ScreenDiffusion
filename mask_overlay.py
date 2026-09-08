"""The mask overlay: what is being restyled, drawn over the preview.

Issue #47. The window reported detection as a **count** and nothing said *where*.
Half of the confusion that produced is correct behaviour and the docs were the only
place it was written down: under the `masked` primitive the whole frame really is
diffused, and the mask decides what is **composited back**. With `region: full_box`
and `box_scale: 1.15`, one object filling a 512x512 capture masks nearly the whole
frame - so the selective path and the global path genuinely look the same, and no
amount of staring at the preview resolves it.

Two halves, and the process split is the whole safety argument.

**The worker's half** is `restyled_boxes` / `overlay_status`: the boxes this frame
actually composited a render into, as plain integers on the *existing* fps payload.
No new channel (the issue's step 3) - a handful of 4-tuples beside a frame - and no
array is allocated to produce them, because the selection already holds them.

**The GUI's half** is `overlay_boxes` / `draw_overlay`: those capture-space boxes
transformed onto the scaled, letterboxed preview and outlined there. It runs in the
GUI process, on the preview copy, *after* the frame has left the pipeline - the
issue's first trap. An overlay drawn before the composite would break the selective
path's one promise, that a non-target pixel is the captured byte.

Stdlib, and PIL only inside the one function that draws. torch is never imported
here and neither is numpy: the transform is arithmetic on four integers, so it is
independent of the capture size (the issue's third trap - the preview already costs
9.80 ms at 1920x1080 and the overlay must not scale with it), and the whole module
is held by the merge gate's GPU-free tier.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from detection import Box
from region_scheduler import Selection
from render_plan import CROP, INVERSE, MASKED

# The one key the overlay rides under on the fps payload. Nested rather than
# flattened beside `fps` and `detections`, so a reader can tell "this frame sent no
# overlay" from "this payload predates the overlay" with one `in`.
OVERLAY_KEY = "restyled"

# Cyan, and nothing else in this window is. The overlay has to be findable against
# a restyled frame whose own colours are whatever the style asked for, and a hue no
# widget uses is the only one that cannot be mistaken for part of the render.
OVERLAY_COLOR = "#22D3EE"

# Thick enough to read at a glance on a 512 px preview, thin enough that it does
# not cover the edge of the region it is describing. Step 1: an outline, never a
# fill - the user is looking at the preview to judge the render.
OVERLAY_WIDTH_PX = 2

# What the switch beside the preview says. Off by default (step 2): the preview is
# a view of what leaves the pipeline, and a decoration drawn on it by default would
# make the one honest picture of the output into a lie the user has to remember.
OVERLAY_SWITCH_TEXT = "Show mask"


# --- the worker's half -------------------------------------------------------


def restyled_boxes(selection: Selection, render: Any) -> Tuple[Box, ...]:
    """The boxes this frame composited a render into, in capture pixels.

    Read off the *compositor's* decision rather than off the plan, which is the
    issue's fifth trap. Under `crop` the frame spends its one diffusion call on a
    single region and that region is what was rendered; a `crop` plan the
    compositor refused - two regions would be two calls - rendered the masked set
    instead, and the overlay has to say which of those actually happened.

    Empty for the two actions that outline nothing: a global frame, where a
    rectangle round the whole capture would say nothing, and a selective frame
    that found nothing, where there is no render at all. Empty for `inverse` too:
    there the boxes are what is *protected*, and outlining them as though they
    were restyled would be worse than showing nothing.
    """
    if selection.mode == INVERSE:
        return ()
    if render.action == CROP and render.crop is not None:
        return (Box(*render.crop),)
    if render.action == MASKED:
        return tuple(selection.boxes)
    return ()


def overlay_status(selection: Selection, render: Any,
                   width: int, height: int) -> Dict[str, Any]:
    """The overlay's contribution to the frame's fps payload.

    Absent rather than empty when this frame restyled nothing - the rule
    `detection.fps_payload` and `region_scheduler.selection_status` already
    follow, for the same reason: "found nothing" and "was never asked" must not
    render as the same status.

    The frame geometry travels with the boxes because the preview is scaled and
    the receiving side must not have to guess what they were measured against.
    """
    boxes = restyled_boxes(selection, render)
    if not boxes:
        return {}
    return {OVERLAY_KEY: {
        "boxes": [[int(value) for value in box] for box in boxes],
        "frame": [int(width), int(height)],
        "action": render.action,
    }}


# --- the GUI's half ----------------------------------------------------------


def _clip(value: int, low: int, high: int) -> int:
    return low if value < low else (high if value > high else value)


def _span(low: int, high: int, floor: int, ceiling: int) -> Tuple[int, int]:
    """`low`..`high` clipped into `floor`..`ceiling`, and never zero wide.

    A region that scaled below one preview pixel is still a region being restyled,
    and a rectangle of zero width would read as "this object is not in the plan" -
    so it keeps a pixel, taken from whichever side has room for one.
    """
    low = _clip(low, floor, ceiling)
    high = _clip(high, floor, ceiling)
    if high > low:
        return low, high
    return (low - 1, high) if high >= ceiling else (low, high + 1)


def overlay_boxes(payload: Any, width: int, height: int,
                  origin: Tuple[int, int] = (0, 0)
                  ) -> Tuple[Tuple[int, int, int, int], ...]:
    """The payload's boxes as rectangles on a preview `width` x `height` at `origin`.

    `origin` is where the scaled capture's own top-left corner sits on the preview
    canvas, which is not (0, 0): a non-square capture is letterboxed onto a square
    panel. Getting this wrong is the issue's fourth trap - an overlay in the wrong
    place lies about what is being restyled, which is worse than showing nothing.

    Tolerant of every payload the fps queue has ever carried: a bare number, a
    mapping with no overlay, one whose frame geometry is degenerate. All of them
    mean "nothing to draw", never an exception on the GUI's poll loop.
    """
    overlay = payload.get(OVERLAY_KEY) if isinstance(payload, dict) else None
    if not isinstance(overlay, dict):
        return ()
    frame: Sequence[Any] = overlay.get("frame") or ()
    if len(frame) != 2 or width <= 0 or height <= 0:
        return ()
    frame_width, frame_height = int(frame[0]), int(frame[1])
    if frame_width <= 0 or frame_height <= 0:
        return ()
    left, top = int(origin[0]), int(origin[1])
    scale_x = width / frame_width
    scale_y = height / frame_height
    rectangles = []
    for box in overlay.get("boxes") or ():
        if len(box) != 4:
            continue
        x0 = left + int(round(int(box[0]) * scale_x))
        y0 = top + int(round(int(box[1]) * scale_y))
        x1 = left + int(round(int(box[2]) * scale_x))
        y1 = top + int(round(int(box[3]) * scale_y))
        if x1 <= left or y1 <= top or x0 >= left + width or y0 >= top + height:
            # Wholly off the preview. Only reachable from a stale payload, since
            # the scheduler clips every region to the frame it selected on.
            continue
        x0, x1 = _span(x0, x1, left, left + width)
        y0, y1 = _span(y0, y1, top, top + height)
        rectangles.append((x0, y0, x1, y1))
    return tuple(rectangles)


def draw_overlay(image: Any, payload: Any, width: int, height: int,
                 origin: Tuple[int, int] = (0, 0)) -> int:
    """Outline this frame's restyled regions on `image`. Returns how many.

    `image` is the preview copy in the GUI process and is modified in place. It is
    already scaled down to the panel, so the cost of the overlay is a few
    rectangles on a 512 px canvas whatever the capture size - the issue's third
    trap. With nothing to draw the image is not touched at all, which is the
    Gate's third item at the level of pixels.
    """
    rectangles = overlay_boxes(payload, width, height, origin)
    if not rectangles:
        return 0
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    for x0, y0, x1, y1 in rectangles:
        # PIL's rectangle is inclusive of both corners; the boxes are half-open.
        draw.rectangle([x0, y0, x1 - 1, y1 - 1],
                       outline=OVERLAY_COLOR, width=OVERLAY_WIDTH_PX)
    return len(rectangles)
