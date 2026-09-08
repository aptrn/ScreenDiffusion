"""The mask overlay: what the frame restyled, and where it lands in the preview.

Issue #47. With a target set the window reported detection as a *count* and
nothing said **where** - and under the `masked` primitive one object filling the
capture masks nearly the whole frame, so the selective path and the global path
genuinely look the same. This module is the two halves of the answer: what the
worker puts on the existing fps payload, and the transform that turns it into
rectangles on the preview.

Both halves are pure. The worker half takes a selection and a frame render and
returns a dict; the GUI half takes that dict and a preview geometry and returns
rectangles. Neither imports torch and neither allocates an array, which is what
lets the whole overlay be held here rather than behind a GPU marker.
"""

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from compositor import CROP, FULL_FRAME, MASKED, PASSTHROUGH, Compositor
from detection import Box, Track, Tracks
from mask_overlay import (
    OVERLAY_COLOR,
    OVERLAY_KEY,
    draw_overlay,
    overlay_boxes,
    overlay_status,
    restyled_boxes,
)
from region_scheduler import RegionScheduler
from render_plan import CROP as PLAN_CROP
from render_plan import MASKED as PLAN_MASKED
from render_plan import GLOBAL, INVERSE, validate_plan

W, H = 64, 48


def plan_of(mode=None, concept="person", max_instances=6):
    raw = {"source_prompt": "wet denim",
           "targets": [{"id": "t0", "concept": concept, "region": "full_box",
                        "box_scale": 1.0, "max_instances": max_instances}]}
    if mode:
        raw["mode"] = mode
    result = validate_plan(raw)
    assert result.plan is not None, result.reason
    return result.plan


def selection_of(*boxes, mode=None, max_instances=6):
    tracks = Tracks(tracks=tuple(
        Track(track_id=index, box=Box(*box), concept="person", confidence=0.9)
        for index, box in enumerate(boxes)), ticks=1)
    return RegionScheduler().select(
        tracks, plan_of(mode, max_instances=max_instances), W, H)


def rendered(selection, primitive=PLAN_MASKED):
    return Compositor().frame(selection, W, H, primitive)


# --- the worker's half: what this frame actually restyled --------------------


def test_a_masked_frame_reports_every_region_it_composited():
    selection = selection_of((4, 4, 20, 24), (30, 10, 50, 40))
    render = rendered(selection)
    assert render.action == MASKED
    assert restyled_boxes(selection, render) == selection.boxes


def test_a_crop_frame_reports_the_one_box_the_call_was_spent_on():
    """The issue's fifth trap: under `crop` the rendered region is not the
    detected set, it is the single region the frame's one call went to."""
    selection = selection_of((4, 4, 20, 24), max_instances=1)
    render = rendered(selection, PLAN_CROP)
    assert render.action == CROP
    assert restyled_boxes(selection, render) == (render.crop,)


def test_a_crop_plan_that_fell_back_to_masked_reports_the_masked_regions():
    """Two regions are two calls, so the compositor refuses `crop`; the overlay
    has to follow the compositor rather than the plan."""
    selection = selection_of((4, 4, 20, 24), (30, 10, 50, 40))
    render = rendered(selection, PLAN_CROP)
    assert render.action == MASKED
    assert restyled_boxes(selection, render) == selection.boxes


def test_a_selective_frame_that_found_nothing_restyled_nothing():
    selection = selection_of()
    render = rendered(selection)
    assert render.action == PASSTHROUGH
    assert restyled_boxes(selection, render) == ()


def test_a_global_frame_has_no_regions_to_outline():
    """The whole frame is the render; a rectangle round it would say nothing."""
    plan = validate_plan({"source_prompt": "wet denim"}).plan
    selection = RegionScheduler().select(Tracks(), plan, W, H)
    assert selection.mode == GLOBAL
    render = rendered(selection)
    assert render.action == FULL_FRAME
    assert restyled_boxes(selection, render) == ()


def test_an_inverse_plan_outlines_nothing_rather_than_the_wrong_side():
    """Under `inverse` the boxes are what is *protected*. Outlining them as if
    they were restyled would be worse than showing nothing."""
    selection = selection_of((4, 4, 20, 24), mode=INVERSE)
    render = rendered(selection)
    assert render.action == MASKED
    assert restyled_boxes(selection, render) == ()


# --- the payload -------------------------------------------------------------


def test_the_status_rides_one_key_on_the_existing_payload():
    """Step 3: no new IPC channel. Boxes are tiny beside a frame."""
    selection = selection_of((4, 4, 20, 24))
    status = overlay_status(selection, rendered(selection), W, H)
    assert set(status) == {OVERLAY_KEY}
    assert status[OVERLAY_KEY]["boxes"] == [list(selection.boxes[0])]
    assert status[OVERLAY_KEY]["frame"] == [W, H]


def test_a_frame_that_restyled_nothing_carries_no_key_at_all():
    """Absent rather than empty, which is `fps_payload`'s own rule for the same
    reason: found nothing and was never asked must not read the same."""
    selection = selection_of()
    assert overlay_status(selection, rendered(selection), W, H) == {}


def test_the_payload_is_plain_data_a_queue_can_carry():
    selection = selection_of((4, 4, 20, 24))
    status = overlay_status(selection, rendered(selection), W, H)
    overlay = status[OVERLAY_KEY]
    assert isinstance(overlay["boxes"], list)
    assert all(isinstance(value, int) for box in overlay["boxes"] for value in box)


# --- the GUI's half: capture coordinates onto a scaled preview ---------------


def payload_of(*boxes, frame=(W, H)):
    return {"fps": 30,
            OVERLAY_KEY: {"boxes": [list(box) for box in boxes],
                          "frame": list(frame), "action": MASKED}}


def test_a_box_is_scaled_by_the_previews_own_ratio():
    """The issue's fourth trap: the boxes are capture pixels and the preview is
    scaled, so an untransformed rectangle lies about what is being restyled."""
    payload = payload_of((0, 0, 32, 24), frame=(64, 48))
    assert overlay_boxes(payload, 256, 192) == ((0, 0, 128, 96),)


def test_the_preview_offset_moves_the_rectangle_with_the_image():
    """The preview is letterboxed onto a square canvas; the image's own corner is
    where the capture's (0, 0) is."""
    payload = payload_of((0, 0, 64, 48), frame=(64, 48))
    assert overlay_boxes(payload, 256, 192, origin=(10, 40)) == ((10, 40, 266, 232),)


def test_a_box_reaching_past_the_frame_is_clipped_to_the_preview():
    """`dilate_box` grows a region before it is clipped, so a box on the edge of
    the capture can arrive wider than the frame."""
    payload = payload_of((-20, -10, 200, 100), frame=(64, 48))
    assert overlay_boxes(payload, 64, 48, origin=(5, 5)) == ((5, 5, 69, 53),)


def test_a_region_too_small_to_scale_still_gets_a_visible_rectangle():
    """A tiny object that rounded to nothing would read as not being restyled."""
    payload = payload_of((10, 10, 11, 11), frame=(1920, 1080))
    boxes = overlay_boxes(payload, 512, 288)
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert x1 > x0 and y1 > y0


def test_a_box_wholly_outside_the_preview_is_dropped():
    payload = payload_of((100, 100, 120, 120), frame=(64, 48))
    assert overlay_boxes(payload, 64, 48) == ()


def test_a_payload_with_no_overlay_draws_nothing():
    assert overlay_boxes({"fps": 30}, 256, 256) == ()


def test_the_bare_number_the_fps_queue_has_always_carried_draws_nothing():
    assert overlay_boxes(30, 256, 256) == ()
    assert overlay_boxes(None, 256, 256) == ()


def test_a_degenerate_geometry_draws_nothing_rather_than_dividing_by_zero():
    assert overlay_boxes(payload_of((0, 0, 8, 8), frame=(0, 48)), 256, 256) == ()
    assert overlay_boxes(payload_of((0, 0, 8, 8)), 0, 0) == ()


# --- the drawing -------------------------------------------------------------


def test_the_overlay_is_an_outline_and_never_a_fill():
    """Step 1: a filled overlay hides the render the user is trying to judge."""
    fills = []
    original = ImageDraw.ImageDraw.rectangle

    def record(self, xy, fill=None, outline=None, width=1):
        fills.append(fill)
        return original(self, xy, fill=fill, outline=outline, width=width)

    ImageDraw.ImageDraw.rectangle = record
    try:
        image = Image.new("RGB", (64, 48), (0, 0, 0))
        drawn = draw_overlay(image, payload_of((8, 8, 40, 40)), 64, 48)
    finally:
        ImageDraw.ImageDraw.rectangle = original
    assert drawn == 1
    assert fills == [None]


def test_the_outline_reaches_the_pixels_in_the_one_colour_the_module_names():
    image = Image.new("RGB", (64, 48), (0, 0, 0))
    assert draw_overlay(image, payload_of((8, 8, 40, 40), frame=(64, 48)), 64, 48) == 1
    pixels = np.asarray(image)
    assert pixels[8, 8].tolist() == list(ImageColor.getrgb(OVERLAY_COLOR)), \
        "no rectangle was drawn, or not in OVERLAY_COLOR"
    assert pixels[24, 24].tolist() == [0, 0, 0], "the region was filled over"


def test_drawing_nothing_leaves_the_preview_exactly_as_it_was():
    """The Gate's third item, at the level of pixels: with nothing to outline the
    preview is what it always was."""
    image = Image.new("RGB", (64, 48), (17, 23, 29))
    before = np.array(image, copy=True)
    assert draw_overlay(image, {"fps": 30}, 64, 48) == 0
    assert np.array_equal(np.asarray(image), before)


def test_every_region_gets_its_own_rectangle():
    image = Image.new("RGB", (64, 48), (0, 0, 0))
    payload = payload_of((2, 2, 20, 20), (30, 10, 60, 40))
    assert draw_overlay(image, payload, 64, 48) == 2
