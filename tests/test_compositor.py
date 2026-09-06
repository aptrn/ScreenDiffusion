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
