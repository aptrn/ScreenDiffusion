"""C7, the compositor (issue #8): the alpha blend, and what a frame is made of.

The sharp criterion in the issue is that **non-target pixels stay bit-identical to
the capture**, so most of what is asserted here is an equality between arrays and
not a tolerance. The feather is the thing that could break it - a ramp that reached
outside the region would change a pixel nobody asked to change - so it is checked
from both sides: strictly zero everywhere outside, and strictly non-zero on the
outermost row inside.

numpy only. No torch, no GPU: the blend is array arithmetic and the merge gate can
hold all of it.
"""

import numpy as np
import pytest

from compositor import (
    CROP,
    DEFAULT_FEATHER_PX,
    FULL_FRAME,
    MASKED,
    PASSTHROUGH,
    Compositor,
    composite,
    feather_alpha,
    painted_mask,
)
from detection import Box, Track, Tracks
from region_scheduler import RegionScheduler
from render_plan import GLOBAL, INVERSE, validate_plan

W, H = 64, 48
REGION = Box(10, 8, 40, 40)


def frame(seed):
    generator = np.random.default_rng(seed)
    return generator.integers(0, 256, size=(H, W, 3), dtype=np.uint8)


def plan_of(mode=None, concept="person"):
    raw = {"source_prompt": "wet denim",
           "targets": [{"id": "t0", "concept": concept, "region": "full_box",
                        "box_scale": 1.0}]}
    if mode:
        raw["mode"] = mode
    result = validate_plan(raw)
    assert result.plan is not None, result.reason
    return result.plan


def selection_of(*boxes, mode=None):
    tracks = Tracks(tracks=tuple(
        Track(track_id=index, box=Box(*box), concept="person", confidence=0.9)
        for index, box in enumerate(boxes)), ticks=1)
    return RegionScheduler().select(tracks, plan_of(mode), W, H)


# --- the alpha map ----------------------------------------------------------


def test_alpha_is_zero_everywhere_outside_the_region():
    """The bit-identical gate, stated on the map that decides it."""
    alpha = feather_alpha([REGION], W, H)
    outside = np.ones((H, W), dtype=bool)
    outside[REGION.y0:REGION.y1, REGION.x0:REGION.x1] = False
    assert np.all(alpha[outside] == 0.0)


def test_alpha_is_one_in_the_middle_of_the_region():
    alpha = feather_alpha([REGION], W, H)
    middle_y = (REGION.y0 + REGION.y1) // 2
    middle_x = (REGION.x0 + REGION.x1) // 2
    assert alpha[middle_y, middle_x] == 1.0


def test_the_feather_ramps_inside_the_region_and_never_outside_it():
    alpha = feather_alpha([REGION], W, H, feather_px=4)
    edge = alpha[REGION.y0, REGION.x0 + 10]
    inward = alpha[REGION.y0 + 2, REGION.x0 + 10]
    assert 0.0 < edge < inward < 1.0, "the ramp does not climb inwards"
    assert alpha[REGION.y0 - 1, REGION.x0 + 10] == 0.0, "the ramp reached outside"


def test_no_feather_is_a_hard_edge():
    alpha = feather_alpha([REGION], W, H, feather_px=0)
    assert set(np.unique(alpha)) == {0.0, 1.0}


def test_a_region_smaller_than_the_feather_still_reaches_full_alpha():
    """Otherwise a small object would be rendered at a strength nobody chose."""
    small = Box(20, 20, 27, 27)
    alpha = feather_alpha([small], W, H, feather_px=DEFAULT_FEATHER_PX)
    assert alpha.max() == 1.0
    assert alpha[small.y0 - 1, small.x0 + 3] == 0.0


def test_overlapping_regions_take_the_stronger_alpha():
    alpha = feather_alpha([Box(10, 10, 30, 30), Box(28, 10, 48, 30)], W, H,
                          feather_px=4)
    seam = alpha[20, 29]
    assert seam == max(
        feather_alpha([Box(10, 10, 30, 30)], W, H, feather_px=4)[20, 29],
        feather_alpha([Box(28, 10, 48, 30)], W, H, feather_px=4)[20, 29])


def test_no_region_is_an_alpha_of_nothing():
    assert not feather_alpha([], W, H).any()


def test_painted_mask_is_where_the_render_lands():
    alpha = feather_alpha([REGION], W, H, feather_px=3)
    mask = painted_mask(alpha)
    assert mask.dtype == np.bool_
    assert mask.sum() == REGION.width * REGION.height


# --- the blend --------------------------------------------------------------


def test_outside_the_regions_the_output_is_bit_identical_to_the_capture():
    """The issue's sharp criterion, on random pixels rather than a flat colour -
    a flat frame would pass a compositor that wrote the wrong constant."""
    source, rendered = frame(1), frame(2)
    alpha = feather_alpha([REGION], W, H)
    output = composite(source, rendered, alpha)
    outside = alpha == 0.0
    assert np.array_equal(output[outside], source[outside])


def test_inside_at_full_alpha_the_output_is_the_render_exactly():
    source, rendered = frame(3), frame(4)
    alpha = feather_alpha([REGION], W, H, feather_px=0)
    output = composite(source, rendered, alpha)
    inside = alpha == 1.0
    assert np.array_equal(output[inside], rendered[inside])


def test_a_feathered_pixel_lies_between_the_two():
    source = np.zeros((H, W, 3), dtype=np.uint8)
    rendered = np.full((H, W, 3), 200, dtype=np.uint8)
    alpha = feather_alpha([REGION], W, H, feather_px=4)
    output = composite(source, rendered, alpha)
    ramp = output[REGION.y0:REGION.y0 + 5, REGION.x0 + 10, 0]
    assert 0 < ramp[0] < 200
    assert list(ramp) == sorted(ramp), f"the ramp is not monotonic: {ramp}"


def test_the_blend_returns_an_image_of_the_same_shape_and_type():
    source, rendered = frame(5), frame(6)
    output = composite(source, rendered, feather_alpha([REGION], W, H))
    assert output.shape == source.shape
    assert output.dtype == np.uint8


def test_the_capture_is_not_modified_in_place():
    source, rendered = frame(7), frame(8)
    before = source.copy()
    composite(source, rendered, feather_alpha([REGION], W, H))
    assert np.array_equal(source, before)


def test_an_empty_alpha_passes_the_capture_through_unchanged():
    source, rendered = frame(9), frame(10)
    output = composite(source, rendered, np.zeros((H, W), dtype=np.float32))
    assert np.array_equal(output, source)


# --- what a frame is made of ------------------------------------------------


def test_a_global_plan_renders_the_whole_frame():
    render = Compositor().frame(
        RegionScheduler().select(Tracks(), validate_plan({}).plan, W, H), W, H)
    assert render.action == FULL_FRAME
    assert render.alpha is None


def test_a_selective_plan_with_regions_is_a_masked_composite():
    render = Compositor().frame(selection_of(REGION), W, H)
    assert render.action == MASKED
    assert render.alpha is not None
    assert render.alpha[0, 0] == 0.0


def test_a_selective_plan_with_nothing_detected_passes_the_capture_through():
    """No region means no pixel anybody asked to change, so the frame costs no
    diffusion call at all - and the loop still produces a frame."""
    render = Compositor().frame(selection_of(), W, H)
    assert render.action == PASSTHROUGH
    assert render.alpha is None


def test_inverse_mode_protects_the_targets_and_renders_the_rest():
    render = Compositor().frame(selection_of(REGION, mode=INVERSE), W, H)
    assert render.action == MASKED
    middle_y = (REGION.y0 + REGION.y1) // 2
    middle_x = (REGION.x0 + REGION.x1) // 2
    assert render.alpha[middle_y, middle_x] == 0.0, "the target was restyled"
    assert render.alpha[0, 0] == 1.0, "the background was not"


def test_inverse_mode_with_nothing_detected_is_the_whole_frame():
    render = Compositor().frame(selection_of(mode=INVERSE), W, H)
    assert render.action == FULL_FRAME


def test_the_same_regions_twice_reuse_one_alpha_map():
    """The frame loop runs between detector ticks on identical boxes; building a
    canvas-sized float array per frame for them would be pure waste."""
    compositor = Compositor()
    first = compositor.frame(selection_of(REGION), W, H)
    second = compositor.frame(selection_of(REGION), W, H)
    assert second.alpha is first.alpha


def test_moving_a_box_rebuilds_the_alpha_map():
    compositor = Compositor()
    first = compositor.frame(selection_of(REGION), W, H)
    moved = compositor.frame(selection_of(Box(11, 8, 41, 40)), W, H)
    assert moved.alpha is not first.alpha
    assert not np.array_equal(moved.alpha, first.alpha)


@pytest.mark.parametrize("mode", [GLOBAL, INVERSE])
def test_every_mode_in_the_vocabulary_produces_a_frame(mode):
    render = Compositor().frame(selection_of(REGION, mode=mode), W, H)
    assert render.action in (FULL_FRAME, MASKED, PASSTHROUGH)


# --- the output EMA (issue #32, spec 8.5) ------------------------------------
#
# The EMA smooths the *rendered canvas*, not the composited frame, and that is the
# whole answer to the issue's first trap: the blend is unchanged, so a pixel outside
# every region is still the captured byte copied rather than a function of history.
# What is smoothed is only what the mask lets through.


def test_the_ema_is_off_by_default_and_returns_the_render_itself():
    compositor = Compositor()
    rendered = frame(3)
    assert compositor.smooth(rendered) is rendered


def test_the_first_smoothed_frame_is_the_render_itself():
    """There is no previous render to average with, and starting from the capture
    or from grey would make the first restyled frame a fade-in."""
    compositor = Compositor(output_ema=0.8)
    rendered = frame(3)
    assert np.array_equal(compositor.smooth(rendered), rendered)


def test_the_second_smoothed_frame_lies_between_the_two():
    compositor = Compositor(output_ema=0.5)
    first, second = frame(3), frame(4)
    compositor.smooth(first)
    smoothed = compositor.smooth(second)
    between = ((smoothed >= np.minimum(first, second))
               & (smoothed <= np.maximum(first, second)))
    assert between.all()


def test_the_ema_keeps_the_coefficient_s_share_of_the_previous_render():
    """`previous * ema + current * (1 - ema)`, written the way the blend is so the
    endpoints are exact: at ema 0 it is the render, at the ceiling it is nearly the
    previous one."""
    compositor = Compositor(output_ema=0.75)
    first = np.full((H, W, 3), 200, dtype=np.uint8)
    second = np.full((H, W, 3), 100, dtype=np.uint8)
    compositor.smooth(first)
    assert np.array_equal(compositor.smooth(second),
                          np.full((H, W, 3), 175, dtype=np.uint8))


def test_the_smoothed_frame_is_what_the_next_one_averages_with():
    """The state is the EMA's own output, not the last raw render - otherwise it is
    a two-frame mean however long the sequence and however high the coefficient."""
    compositor = Compositor(output_ema=0.5)
    compositor.smooth(np.full((H, W, 3), 0, dtype=np.uint8))
    compositor.smooth(np.full((H, W, 3), 100, dtype=np.uint8))
    assert np.array_equal(compositor.smooth(np.full((H, W, 3), 100, dtype=np.uint8)),
                          np.full((H, W, 3), 75, dtype=np.uint8))


def test_a_frame_that_rendered_nothing_resets_the_ema():
    """A passthrough frame has no render, and averaging across the gap would blend a
    frame with one from before an object left the screen."""
    compositor = Compositor(output_ema=0.5)
    compositor.smooth(np.full((H, W, 3), 0, dtype=np.uint8))
    compositor.reset_ema()
    fresh = np.full((H, W, 3), 100, dtype=np.uint8)
    assert np.array_equal(compositor.smooth(fresh), fresh)


def test_changing_the_coefficient_does_not_throw_the_history_away():
    """A plan edit is not a scene change: the previous render is still the previous
    render, and dropping it would put a visible step in the output."""
    compositor = Compositor(output_ema=0.5)
    compositor.smooth(np.full((H, W, 3), 200, dtype=np.uint8))
    compositor.set_output_ema(0.0)
    assert compositor._previous_render is not None


def test_a_render_of_a_different_shape_starts_the_ema_again():
    """The canvas can only change with an engine rebuild, and averaging across one
    is an exception rather than a resize."""
    compositor = Compositor(output_ema=0.5)
    compositor.smooth(frame(3))
    other = np.zeros((H * 2, W, 3), dtype=np.uint8)
    assert np.array_equal(compositor.smooth(other), other)


def test_the_ema_never_touches_the_composite_outside_the_regions():
    """The Gate's third item, end to end: two smoothed renders composited through a
    mask still leave every pixel outside it exactly as captured."""
    compositor = Compositor(output_ema=0.8)
    render = compositor.frame(selection_of(REGION), W, H)
    source, first, second = frame(1), frame(2), frame(3)
    compositor.smooth(first)
    output = compositor.blend(source, compositor.smooth(second), render.alpha)
    outside = ~painted_mask(render.alpha)
    assert np.array_equal(output[outside], source[outside])


# --- the crop action (issue #39, spec 8.2) ----------------------------------
#
# `crop` spends the frame's one diffusion call on one region instead of on the
# whole capture, so the object gets the engine's whole 512x512 canvas. The blend
# is the same blend - the same feathered alpha, the same "outside is the captured
# byte" rule - reading a render that covers only the crop box. `origin` is what
# says where that patch sits.


def crop_selection(*boxes, max_instances=1):
    raw = {"source_prompt": "wet denim",
           "targets": [{"id": "t0", "concept": "person", "region": "full_box",
                        "box_scale": 1.0, "max_instances": max_instances}],
           "global": {"primitive": "crop"}}
    result = validate_plan(raw)
    assert result.plan is not None, result.reason
    tracks = Tracks(tracks=tuple(
        Track(track_id=index, box=Box(*box), concept="person", confidence=0.9)
        for index, box in enumerate(boxes)), ticks=1)
    return RegionScheduler().select(tracks, result.plan, W, H), result.plan


def test_the_compositor_vocabulary_is_the_plan_s():
    """One spelling for the primitive the plan names and the action it selects."""
    import render_plan

    assert (CROP, MASKED) == (render_plan.CROP, render_plan.MASKED)


def test_one_region_under_crop_is_a_crop_render():
    selection, plan = crop_selection(REGION)
    render = Compositor().frame(selection, W, H, plan.settings.primitive)
    assert render.action == CROP
    assert render.crop == REGION
    assert render.diffuses


def test_two_regions_under_crop_fall_back_to_masked():
    """Crop is one call per region, so a frame that selected two would cost two.
    The frame renders rather than refusing, and the plan's note said this would
    happen."""
    selection, plan = crop_selection(REGION, Box(44, 5, 62, 30), max_instances=2)
    assert selection.count == 2
    render = Compositor().frame(selection, W, H, plan.settings.primitive)
    assert render.action == MASKED
    assert render.crop is None


def test_no_region_under_crop_still_passes_the_capture_through():
    selection, plan = crop_selection()
    render = Compositor().frame(selection, W, H, plan.settings.primitive)
    assert render.action == PASSTHROUGH


def test_a_global_plan_ignores_the_crop_primitive():
    """`crop` names one region to spend the canvas on; `global` names none."""
    selection = selection_of(REGION, mode="global")
    render = Compositor().frame(selection, W, H, CROP)
    assert render.action == FULL_FRAME


def test_the_crop_alpha_is_the_masked_alpha():
    """The same feather, so the bit-identity rule is the same rule."""
    selection, plan = crop_selection(REGION)
    render = Compositor().frame(selection, W, H, plan.settings.primitive)
    assert np.array_equal(render.alpha, feather_alpha([REGION], W, H))


def test_changing_the_primitive_rebuilds_the_render():
    compositor = Compositor()
    selection, _ = crop_selection(REGION)
    assert compositor.frame(selection, W, H, MASKED).action == MASKED
    assert compositor.frame(selection, W, H, CROP).action == CROP


# --- the blend, with the render covering only the crop box ------------------


def crop_patch(seed, box):
    generator = np.random.default_rng(seed)
    return generator.integers(0, 256, size=(box.height, box.width, 3), dtype=np.uint8)


def test_a_patch_at_its_origin_blends_exactly_as_a_full_frame_render_would():
    source = frame(1)
    full = frame(2)
    alpha = feather_alpha([REGION], W, H)
    patch = full[REGION.y0:REGION.y1, REGION.x0:REGION.x1]
    assert np.array_equal(
        composite(source, patch, alpha, origin=(REGION.x0, REGION.y0)),
        composite(source, full, alpha))


def test_outside_the_crop_the_output_is_bit_identical_to_the_capture():
    source = frame(3)
    alpha = feather_alpha([REGION], W, H)
    output = composite(source, crop_patch(4, REGION), alpha,
                       origin=(REGION.x0, REGION.y0))
    mask = painted_mask(alpha)
    assert np.array_equal(output[~mask], source[~mask])


def test_a_patch_smaller_than_the_alpha_is_a_producer_bug():
    """The patch has to cover every pixel the alpha paints; a short one would
    silently blend the wrong pixels rather than raising."""
    source = frame(5)
    alpha = feather_alpha([REGION], W, H)
    with pytest.raises(ValueError):
        composite(source, crop_patch(6, Box(0, 0, 4, 4)), alpha,
                  origin=(REGION.x0, REGION.y0))
