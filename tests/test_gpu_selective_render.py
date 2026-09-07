"""The selective render path against the real engine and the real detector. GPU tier.

Issue #8's Verification, the two items only a GPU can answer:

- the `lower_half` region of detected people is visibly restyled and **everything
  else is bit-identical to the source frame**, asserted on the committed clip and
  not judged by eye;
- with more tracks than slots, every track is rendered inside `ceil(N/K)` frames.

What runs is the shipped path - `UltralyticsDetector`, `Tracker`,
`RegionScheduler`, the cached 512x512 TensorRT engine, `DeviceCompositor` - under
`priority_case_plan()`, the plan the worker itself starts on. Detection is driven
synchronously here so the frames are the same on every run; the *threaded* detector
is what `bench selective-people` measures and what issue #7's GPU tier covers.

Short: twelve frames, no cooldown, no clips. It skips rather than builds - an engine
is ~5.0 GB and 15-25 minutes - and skips rather than fetches the detector weights.
No absolute latency is asserted anywhere (issue #13).
"""

import math

import numpy as np
import pytest

from bench.cli import engine_dir_name
from bench.paths import resolve_engines_dir, resolve_models_dir
from bench.scenarios import SCENARIOS
from bench.selective import ENGINE_SCENARIO, VISIBLE_CHANGE, coverage_check, with_slots
from compositor import composite, painted_mask
from device_compositor import DeviceCompositor
from detection import Box, Track, Tracker, Tracks, is_detect_frame
from region_scheduler import RegionScheduler
from render_plan import plan_from_fields, priority_case_plan, t_index_for_denoise

pytestmark = pytest.mark.gpu

FRAMES = 12
CANVAS = 512
PROBE_SLOTS = 2


@pytest.fixture(scope="module")
def cached_engine():
    scenario = SCENARIOS[ENGINE_SCENARIO]
    root = resolve_engines_dir()
    engine = root / engine_dir_name(scenario) / "unet.engine"
    if not engine.is_file():
        pytest.skip(f"no cached engine for {ENGINE_SCENARIO} under {root}. Build one "
                    f"with `python -m bench {ENGINE_SCENARIO} --allow-engine-build`.")
    return root


@pytest.fixture(scope="module")
def plan():
    return priority_case_plan()


@pytest.fixture(scope="module")
def frames():
    """Clip frames at the app's capture geometry: the worker's capture thread
    resizes before the frame loop ever sees one."""
    from bench.primitive_runner import read_clip, resize
    from bench.primitives import clip_path

    decoded, _ = read_clip(clip_path("people.mp4"), 0, FRAMES)
    return [resize(frame, CANVAS, CANVAS) for frame in decoded]


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
    weights = weights_path(DETECTORS[PRIMARY_DETECTOR], root)
    if not weights.is_file():
        pytest.skip(f"no detector weights at {weights}. Run "
                    f"`python -m bench {PRIMARY_DETECTOR} --allow-download` once.")
    live = UltralyticsDetector(root)
    live.open()
    live.set_concepts([target.concept for target in plan.targets])
    yield live
    live.close()


@pytest.fixture(scope="module")
def run(stream, detector, frames, plan):
    """The shipped path over the clip: sources, outputs, masks and per-frame tracks."""
    from bench.selective_runner import capture_tensor, render_frame
    from detector_worker import frame_to_array
    from seeding import NoiseField

    noise = NoiseField(policy=plan.effective_seed_policy)
    tracker = Tracker()
    scheduler = RegionScheduler()
    compositor = DeviceCompositor()
    tracks = Tracks()
    sources, outputs, masks, snapshots, selections = [], [], [], [], []
    for index, frame in enumerate(frames):
        tensor = capture_tensor(frame, device=stream.device, dtype=stream.dtype)
        source = frame_to_array(tensor)
        if is_detect_frame(index, plan.settings.detect_every_n):
            detections = detector.detect(tensor)
            tracks = Tracks(tracks=tracker.update(detections, index),
                            frame_index=index, ticks=len(snapshots) + 1)
        selection = scheduler.select(tracks, plan, CANVAS, CANVAS)
        render = compositor.frame(selection, CANVAS, CANVAS)
        output, _, _ = render_frame(stream, tensor, compositor, render,
                                    source, noise, selection)
        sources.append(source)
        outputs.append(output)
        masks.append(painted_mask(render.alpha) if render.alpha is not None
                     else np.zeros(output.shape[:2], dtype=bool))
        snapshots.append(tracks)
        selections.append(selection)
    return sources, outputs, masks, snapshots, selections


# --- the Gate ---------------------------------------------------------------


def test_people_are_found_and_their_lower_halves_are_what_gets_rendered(run, plan):
    _, _, masks, _, selections = run
    assert all(selection.count for selection in selections), "some frame had no region"
    assert max(selection.count for selection in selections) >= 2, (
        "people.mp4 has five subjects; this scheduled under two")
    # A lower half is the bottom of the box, so every region starts below its own
    # source box's midpoint.
    for selection in selections:
        for region in selection.regions:
            middle = region.source.y0 + region.source.height // 2
            assert region.box.y0 >= middle - 1, (region.source, region.box)
    assert all(mask.any() for mask in masks)


def test_everything_outside_the_regions_is_bit_identical_to_the_capture(run):
    """The issue's sharp criterion. Array equality, not a tolerance."""
    sources, outputs, masks, _, _ = run
    for index, (source, output, mask) in enumerate(zip(sources, outputs, masks)):
        outside = ~mask
        assert outside.any(), f"frame {index} painted the whole canvas"
        assert np.array_equal(output[outside], source[outside]), (
            f"frame {index} changed "
            f"{int(np.count_nonzero(np.any(output[outside] != source[outside], axis=-1)))}"
            f" background pixels")


def test_the_rendered_regions_are_visibly_restyled(run):
    """And measurably: the mean absolute change inside the mask, against the
    threshold issue #5 selected this case's denoise by."""
    sources, outputs, masks, _, _ = run
    changes = []
    for source, output, mask in zip(sources, outputs, masks):
        difference = np.abs(output.astype(np.float64)
                            - source.astype(np.float64)).mean(axis=-1)
        changes.append(float(difference[mask].mean()))
    mean_change = sum(changes) / len(changes)
    assert mean_change >= VISIBLE_CHANGE, (
        f"the regions changed by {mean_change:.2f}/255, under the "
        f"{VISIBLE_CHANGE}/255 the case needs to be visible")


def test_no_track_is_starved_when_there_are_more_tracks_than_slots(run, plan):
    """The Gate's second item, on the real track sequence with K forced to two."""
    _, _, _, snapshots, _ = run
    check = coverage_check(snapshots, plan, CANVAS, CANVAS, slots=PROBE_SLOTS)
    assert check.max_tracks > PROBE_SLOTS, (
        "the clip did not hold more tracks than slots, so nothing was proved")
    assert check.worst_gap_frames <= math.ceil(check.max_tracks / PROBE_SLOTS)
    assert check.passed, check.statement
    print(f"\n{check.statement}")


def test_a_frame_with_nothing_detected_comes_out_as_the_capture(stream, frames):
    """A selective plan whose concept is not in the clip costs no diffusion call
    and still produces a frame - byte for byte the capture."""
    from bench.selective_runner import capture_tensor, render_frame
    from detector_worker import frame_to_array
    from seeding import NoiseField

    absent = plan_from_fields("giraffe", "a charcoal drawing").plan
    selection = RegionScheduler().select(Tracks(), absent, CANVAS, CANVAS)
    compositor = DeviceCompositor()
    render = compositor.frame(selection, CANVAS, CANVAS)
    assert render.diffuses is False

    tensor = capture_tensor(frames[0], device=stream.device, dtype=stream.dtype)
    source = frame_to_array(tensor)
    output, _, _ = render_frame(stream, tensor, compositor, render, source,
                                NoiseField(policy=absent.effective_seed_policy),
                                selection)
    assert np.array_equal(output, source)


def test_the_device_blend_and_the_host_blend_render_the_same_frame(stream, frames,
                                                                  plan):
    """Issue #31's step 3, end to end rather than on synthetic arrays: the same
    engine output, blended both ways, byte for byte.

    `tests/test_gpu_device_compositor.py` holds the two implementations to each
    other on inputs a test made up. This one holds them to each other on the two
    things the frame loop actually hands over - a real capture tensor and a real
    diffusion - because that is where the conversions the equality rests on live.
    """
    import torch
    from bench.selective_runner import capture_tensor
    from detector_worker import frame_to_array

    scheduler = RegionScheduler()
    compositor = DeviceCompositor()
    tracks = Tracks(tracks=(Track(track_id=0, box=Box(120, 180, 300, 460),
                                  concept="person", confidence=0.9),), ticks=1)
    render = compositor.frame(scheduler.select(tracks, plan, CANVAS, CANVAS),
                              CANVAS, CANVAS)
    assert render.alpha is not None

    tensor = capture_tensor(frames[0], device=stream.device, dtype=stream.dtype)
    source = frame_to_array(tensor)
    with torch.no_grad():
        on_device = compositor.blend_device(
            tensor, stream.img2img(tensor, output_type="pt"), render.alpha)[-1]
        on_host = composite(source, np.asarray(stream.img2img(tensor)),
                            render.alpha)
    assert np.array_equal(on_device, on_host), (
        f"{int(np.count_nonzero(np.any(on_device != on_host, axis=-1)))} pixels "
        f"differ between the device blend and the numpy one")


def test_the_plan_that_drives_this_is_the_one_the_worker_starts_on(plan):
    assert plan.targets[0].concept == "person"
    assert plan.targets[0].region == "lower_half"
    assert t_index_for_denoise(plan.effective_denoise) == 40


def test_forcing_the_slot_count_leaves_the_rest_of_the_plan_alone(plan):
    probed = with_slots(plan, PROBE_SLOTS)
    assert probed.effective_prompt == plan.effective_prompt
    assert probed.targets[0].region == plan.targets[0].region
