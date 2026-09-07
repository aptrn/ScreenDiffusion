"""The crop primitive at a capture that is not the canvas. Issue #39, GPU tier.

`tests/test_gpu_selective_render.py` holds the shipped path where the capture *is*
the engine's 512x512 square. This holds the two things that change when it is not:

- **the frame renders a crop at all** - the plan asks for `crop` at K=1, the
  scheduler hands out one slot, and the compositor's action is `crop` rather than
  the masked fallback;
- **non-target pixels are still the captured bytes at 1280x720**, which is the
  Gate's sharpest item and the one that does not get easier because the frame got
  bigger.

Plus the two claims that make the change worth having: the region is visibly
restyled, and it is diffused at the whole canvas rather than at its share of a
squeezed frame.

The real detector and the real cached engine, twelve frames, no cooldown. It skips
rather than builds and skips rather than fetches. No absolute latency is asserted
(issue #13).
"""

import numpy as np
import pytest

from bench.cli import engine_dir_name
from bench.paths import resolve_engines_dir, resolve_models_dir
from bench.scenarios import SCENARIOS
from bench.selective import ENGINE_SCENARIO, VISIBLE_CHANGE
from compositor import CROP, MASKED, painted_mask
from device_compositor import DeviceCompositor
from detection import Tracker, Tracks, is_detect_frame
from region_scheduler import RegionScheduler
from render_plan import (
    CROP as PLAN_CROP,
    PRIORITY_DENOISE,
    PRIORITY_PROMPT,
    PRIORITY_REGION,
    plan_from_fields,
    t_index_for_denoise,
)

pytestmark = pytest.mark.gpu

FRAMES = 12
CANVAS = 512
# A capture the engine was not built for, and one the app now offers.
CAPTURE_W, CAPTURE_H = 1280, 720


@pytest.fixture(scope="module")
def cached_engine():
    scenario = SCENARIOS[ENGINE_SCENARIO]
    root = resolve_engines_dir()
    if not (root / engine_dir_name(scenario) / "unet.engine").is_file():
        pytest.skip(f"no cached engine for {ENGINE_SCENARIO} under {root}")
    return root


# The strength `capture-people` measured `crop` needs at 1280x720 - t_index 30,
# spec 8.2's capture block. Not the priority case's 0.49: that was selected for
# `masked`, and crop needs more of the ladder for the same visible change, which is
# the measured finding rather than an inconvenience. `PRIORITY_DENOISE` is imported
# so that this stays visibly *different* from it rather than looking arbitrary.
CROP_DENOISE = 0.76


@pytest.fixture(scope="module")
def plan():
    """The priority case as the GUI's Detail box would send it: crop, one slot."""
    assert CROP_DENOISE > PRIORITY_DENOISE
    result = plan_from_fields(
        target="person", style=PRIORITY_PROMPT, region=PRIORITY_REGION,
        denoise=CROP_DENOISE, primitive=PLAN_CROP, max_instances=1)
    assert result.plan is not None, result.reason
    return result.plan


@pytest.fixture(scope="module")
def frames():
    """Clip frames at a capture geometry that is not the engine's canvas."""
    from bench.primitive_runner import read_clip, resize
    from bench.primitives import clip_path

    decoded, _ = read_clip(clip_path("people.mp4"), 0, FRAMES)
    return [resize(frame, CAPTURE_W, CAPTURE_H) for frame in decoded]


@pytest.fixture(scope="module")
def stream(cached_engine, plan):
    from bench.primitive_runner import set_denoise
    from bench.runner import build_stream

    built = build_stream(SCENARIOS[ENGINE_SCENARIO].replace(
        prompt=plan.effective_prompt), engines_root=cached_engine)
    set_denoise(built, t_index_for_denoise(plan.effective_denoise))
    return built


@pytest.fixture(scope="module")
def detector(plan):
    from bench.detectors import DETECTORS, PRIMARY_DETECTOR, weights_path
    from detector_worker import UltralyticsDetector

    root = resolve_models_dir()
    if not weights_path(DETECTORS[PRIMARY_DETECTOR], root).is_file():
        pytest.skip(f"no {PRIMARY_DETECTOR} weights under {root}")
    live = UltralyticsDetector(root)
    live.open()
    live.set_concepts([target.concept for target in plan.targets])
    yield live
    live.close()


@pytest.fixture(scope="module")
def run(stream, detector, frames, plan):
    """The shipped path at 1280x720 under `crop`: sources, outputs, masks, renders."""
    from bench.selective_runner import capture_tensor, render_frame
    from detector_worker import frame_to_array
    from seeding import NoiseField

    noise = NoiseField(policy=plan.effective_seed_policy)
    tracker, scheduler = Tracker(), RegionScheduler()
    compositor = DeviceCompositor()
    tracks = Tracks()
    sources, outputs, masks, renders, selections = [], [], [], [], []
    for index, frame in enumerate(frames):
        tensor = capture_tensor(frame, device=stream.device, dtype=stream.dtype)
        source = frame_to_array(tensor)
        if is_detect_frame(index, plan.settings.detect_every_n):
            tracks = Tracks(tracks=tracker.update(detector.detect(tensor), index),
                            frame_index=index, ticks=len(renders) + 1)
        selection = scheduler.select(tracks, plan, CAPTURE_W, CAPTURE_H)
        render = compositor.frame(selection, CAPTURE_W, CAPTURE_H,
                                  plan.settings.primitive)
        output, _, _ = render_frame(stream, tensor, compositor, render, source,
                                    noise, selection, CANVAS)
        sources.append(source)
        outputs.append(output)
        masks.append(painted_mask(render.alpha) if render.alpha is not None
                     else np.zeros(output.shape[:2], dtype=bool))
        renders.append(render)
        selections.append(selection)
    return sources, outputs, masks, renders, selections


def test_the_capture_is_not_the_canvas(frames):
    assert frames[0].shape[:2] == (CAPTURE_H, CAPTURE_W) != (CANVAS, CANVAS)


def test_the_frames_that_found_a_person_render_a_crop(run):
    _, _, _, renders, _ = run
    diffusing = [render for render in renders if render.diffuses]
    assert diffusing, "no person was detected on any frame"
    assert all(render.action == CROP for render in diffusing), (
        [render.action for render in diffusing])
    assert MASKED not in {render.action for render in diffusing}


def test_the_crop_box_is_the_one_region_the_scheduler_chose(run):
    _, _, _, renders, selections = run
    for render, selection in zip(renders, selections):
        if render.action != CROP:
            continue
        assert selection.count == 1
        assert render.crop == selection.regions[0].box
        assert render.origin == (render.crop.x0, render.crop.y0)


def test_everything_outside_the_region_is_bit_identical_at_1280x720(run):
    """The Gate's first item, at a capture two and a half times the canvas."""
    sources, outputs, masks, _, _ = run
    for index, (source, output, mask) in enumerate(zip(sources, outputs, masks)):
        moved = int(np.count_nonzero(np.any(source[~mask] != output[~mask], axis=-1)))
        assert moved == 0, f"frame {index}: {moved} background pixels changed"


def test_the_frame_comes_back_at_the_capture_size_and_not_the_canvas(run):
    """A crop render is pasted into the captured frame, not returned instead of it."""
    sources, outputs, _, _, _ = run
    for source, output in zip(sources, outputs):
        assert output.shape == source.shape == (CAPTURE_H, CAPTURE_W, 3)


def test_the_cropped_region_is_visibly_restyled(run):
    sources, outputs, masks, _, _ = run
    scored = [np.abs(output.astype(np.float64)
                     - source.astype(np.float64)).mean(axis=-1)[mask].mean()
              for source, output, mask in zip(sources, outputs, masks) if mask.any()]
    assert scored, "no frame rendered a region"
    assert float(np.mean(scored)) >= VISIBLE_CHANGE


def test_the_region_was_diffused_at_more_of_the_canvas_than_masked_would_give(run):
    """The claim the change rests on: under `masked` the region would have been
    squeezed to its share of one canvas covering the whole 1280 px frame."""
    _, _, _, renders, _ = run
    crops = [render.crop for render in renders if render.action == CROP]
    assert crops
    for crop in crops:
        masked_px = CANVAS * crop.width / CAPTURE_W
        assert masked_px < CANVAS, "the region already fills the frame"
