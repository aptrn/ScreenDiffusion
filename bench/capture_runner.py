"""The measuring half of the capture-geometry comparison. Touches the GPU. Issue #39.

What runs here is the **shipped** path, not a copy of it: `region_scheduler`'s own
scheduler at K=1, `device_compositor.DeviceCompositor`, and the two geometry helpers
the worker's frame loop calls - `to_canvas` and `crop_to_canvas`. The only things
this module supplies are the frame source (a committed clip instead of a screen),
the clock, and the two stages that happen in the *other* process, which it measures
against a real `multiprocessing.Queue` and the GUI's own preview arithmetic.

Four things about the numbers.

- **The boxes come from a committed track, not a live detector** - issue #5's second
  trap. Every arm therefore renders exactly the same region of exactly the same
  frame, and the difference between two rows is the primitive and the geometry.
- **The track is scaled, never regenerated.** The committed tracks are 1280x720 and
  960x540 and regenerating one invalidates every committed comparison (the issue's
  sixth trap), so the boxes are scaled onto the capture geometry the same way the
  frames are. The clip's pixels at 1920x1080 are an upscale of its own 720p, which
  the record says out loud: it makes the *cost* questions exact and leaves the
  detail question measuring what each primitive does with the same source.
- **The two primitives are interleaved frame by frame** inside one geometry, so a
  clock that drifts drifts across both arms. The geometries are not interleaved
  with each other - three sets of resident capture tensors is a VRAM decision -
  and each geometry's pair is therefore internally comparable and the pairs are
  comparable to each other only as far as the clock trace says.
- **Every stage is bracketed by a synchronise.** Without them the clock measures
  queue submission and every figure is fiction.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from bench.capture import (
    CANVAS,
    ENGINE_SCENARIO,
    README_NAME,
    CaptureArm,
    CaptureCase,
    CaptureResult,
    CaptureRunMetrics,
    StageCost,
    append_capture_readme_rows,
    arm_name,
    detail_summary,
    mean_ms,
    write_capture_result,
)
from bench.clocks import clock_normalization, regime_summary
from bench.cooldown import DEFAULT_CAP_S, DEFAULT_POLL_INTERVAL_S, DEFAULT_THRESHOLD_C
from bench.detector_results import LatencySummary
from bench.fingerprint import capture_fingerprint, utc_now
from bench.flicker import flicker_score
from bench.paths import CAPTURE_RESULTS_DIR, resolve_models_dir
from bench.primitive_results import ClipRecord
from bench.primitives import (
    Box,
    DenoisePoint,
    clamp_box,
    clip_path,
    load_track,
    region_box,
    required_denoise,
)
from bench.primitive_runner import (
    COMPARISON_PANEL_WIDTH,
    identity_check,
    load_identity_detector,
    mean_abs_diff,
    read_clip,
    resize,
    resized_panel,
    set_denoise,
    set_detector_vocabulary,
    sha256_of,
    triptych,
    write_clip,
    write_still,
)
from bench.results import filename_timestamp
from bench.runner import (
    DEFAULT_SAMPLE_INTERVAL_S,
    GpuSampler,
    build_stream,
    cooldown_gate,
    occupancy_gate,
)
from bench.scenarios import SCENARIOS
from bench.selective import background_check
from bench.selective_runner import background_pixels_changed, capture_tensor

# The preview panel the GUI letterboxes each frame into - `StreamGUI.preview_dim`.
# Spelt here for the reason `CANVAS` is: importing `main_gpu_addon` pulls in the Tk
# stack, and a test holds the two to one value.
PREVIEW_DIM = 512


# --- the geometry -------------------------------------------------------------


def scaled_box(box: Box, scale_x: float, scale_y: float) -> Box:
    """`box` in a frame scaled by `scale_x` / `scale_y`. Rounded, never emptied."""
    box = Box(*box)
    x0, y0 = int(round(box.x0 * scale_x)), int(round(box.y0 * scale_y))
    return Box(x0, y0, max(int(round(box.x1 * scale_x)), x0 + 1),
               max(int(round(box.y1 * scale_y)), y0 + 1))


def frame_boxes(track, index: int, case: CaptureCase, width: int, height: int,
                source_width: int, source_height: int) -> List[Box]:
    """The regions one frame renders, banded and scaled onto the capture geometry.

    At most K of them, K being the case's `max_instances` - the committed track is
    strongest-detection-first, so a cap of 1 is the strongest object rather than an
    arbitrary one.
    """
    scale_x, scale_y = width / source_width, height / source_height
    return [clamp_box(region_box(scaled_box(box, scale_x, scale_y), case.region),
                      width, height)
            for box in track.at(index, limit=case.max_instances)]


def tracks_for(boxes: Sequence[Box], concept: str):
    """One detector tick's worth of tracks, as the frame loop would read them."""
    from detection import Box as TrackBox, Track, Tracks

    return Tracks(tracks=tuple(
        Track(track_id=index, box=TrackBox(*box), concept=concept, confidence=0.9)
        for index, box in enumerate(boxes)), ticks=1)


# --- the stages ---------------------------------------------------------------


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def canvas_for(tensor, render, canvas: int):
    """The frame the engine is handed: the whole capture, or one region of it."""
    from device_compositor import crop_to_canvas, to_canvas

    if render.crop is None:
        return to_canvas(tensor, canvas, canvas)
    return crop_to_canvas(tensor, render.crop, canvas, canvas)


def ipc_cost(queue, image) -> Tuple[float, float]:
    """What handing one frame to the GUI process costs: the put, and the round trip.

    A real `multiprocessing.Queue`, so what is measured is the pickle and the pipe
    rather than an estimate of them. The two figures are different questions: `put`
    is what the worker's frame thread pays (the queue's feeder thread does the
    pickling), and the round trip is how long the frame takes to become available
    to the reader. Both grow with the capture, which is the trap this answers.

    The frame that comes back is dropped rather than returned - reading it is what
    closes the round trip, and it is the same frame that went in.
    """
    started = time.perf_counter()
    queue.put(image)
    put_ms = _elapsed_ms(started)
    queue.get()
    return put_ms, _elapsed_ms(started)


def preview_cost(image) -> float:
    """`StreamGUI._update_preview`'s own arithmetic, timed on this frame's size."""
    from PIL import Image

    started = time.perf_counter()
    panel = Image.new("RGB", (PREVIEW_DIM, PREVIEW_DIM), (30, 30, 30))
    thumbnail = image.copy()
    thumbnail.thumbnail((PREVIEW_DIM, PREVIEW_DIM), Image.BICUBIC)
    panel.paste(thumbnail, ((PREVIEW_DIM - thumbnail.width) // 2,
                            (PREVIEW_DIM - thumbnail.height) // 2))
    return _elapsed_ms(started)


def render_frame(stream, tensor, compositor, render, canvas: int):
    """One frame of the shipped path, with each stage timed separately.

    Returns the finished frame and the three milliseconds the *worker's* frame path
    spends on the device: the resize onto the canvas, the diffusion call, and the
    composite - which since issue #31 carries the frame's one device-to-host copy.
    """
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    frame_canvas = canvas_for(tensor, render, canvas)
    torch.cuda.synchronize()
    resize_ms = _elapsed_ms(started)

    started = time.perf_counter()
    rendered = stream.img2img(frame_canvas, output_type="pt")
    torch.cuda.synchronize()
    diffuse_ms = _elapsed_ms(started)

    started = time.perf_counter()
    output = compositor.blend_device(tensor, rendered, render.alpha, render.crop)[-1]
    composite_ms = _elapsed_ms(started)
    return output, resize_ms, diffuse_ms, composite_ms


def control_frame(tensor, compositor, render, canvas: int):
    """The same path with the diffusion call taken out - the resize control.

    Both primitives resize onto the canvas and back, and that round trip changes
    the region before anything is styled (spec 8.2). A "did the render do anything"
    criterion that did not subtract this would pass a strength that does nothing.
    """
    frame_canvas = canvas_for(tensor, render, canvas)
    return compositor.blend_device(tensor, frame_canvas, render.alpha, render.crop)[-1]


def host_copy_cost(frames) -> float:
    """What moving one finished frame off the device costs at this geometry.

    Measured on its own rather than inside the blend: the composite figure beside
    it already carries this copy, and the trap the issue names is specifically that
    at 1080p it carries ~8x the bytes it did at 512x512. Both numbers, so a reader
    can see the copy's share of the composite rather than take it on trust.
    """
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    frames.cpu().numpy()
    return _elapsed_ms(started)


# --- the run ------------------------------------------------------------------


def sweep_denoise(stream, case: CaptureCase, primitive: str, tensors: Sequence,
                  sources: Sequence, renders: Sequence, compositor, canvas: int,
                  identity_probe, log: Callable[[str], None]) -> List[DenoisePoint]:
    """Walk the ladder on a few frames and record what each rung did.

    Cheap on purpose - it selects a strength, it does not measure a latency - and
    it runs on the *same* path the timed pass runs on, so the control it subtracts
    is the resize this arm actually pays rather than a copy of it.
    """
    from compositor import painted_mask

    frames = [index for index, render in enumerate(renders) if render.diffuses]
    frames = frames[:case.sweep_frames]
    if not frames:
        raise SystemExit(f"bench: {primitive} rendered no frame to sweep")
    masks = [painted_mask(renders[index].alpha) for index in frames]
    controls = [control_frame(tensors[index], compositor, renders[index], canvas)
                for index in frames]
    control = statistics.fmean(
        [mean_abs_diff(sources[index], output, mask)
         for index, output, mask in zip(frames, controls, masks)])

    points: List[DenoisePoint] = []
    for rung in case.denoise_ladder:
        timestep, strength = set_denoise(stream, rung)
        inside, outside, rendered = [], [], []
        for position, index in enumerate(frames):
            output, _, _, _ = render_frame(stream, tensors[index], compositor,
                                           renders[index], canvas)
            inside.append(mean_abs_diff(sources[index], output, masks[position]))
            outside.append(mean_abs_diff(sources[index], output, ~masks[position]))
            rendered.append(output)
        hits, probed = identity_probe(rendered)
        points.append(DenoisePoint(
            t_index=rung, timestep=timestep, strength=strength,
            region_change=round(statistics.fmean(inside), 4),
            outside_change=round(statistics.fmean(outside), 4),
            frames=len(frames), resample_change=round(control, 4),
            identity_hits=hits, identity_frames=probed))
    log(f"denoise {primitive}: " + ", ".join(
        f"t{point.t_index} net {point.net_region_change:.1f}" for point in points))
    return points


def _arm(primitive: str, width: int, height: int,
         denoise, points: Sequence[DenoisePoint],
         stages: StageCost, per_frame_ms: Sequence[float], sources: Sequence,
         outputs: Sequence, control_change: Sequence[float], masks: Sequence,
         selections: Sequence, background_changed: Sequence[int],
         diffusion_calls: int, crop_frames: int, identity) -> CaptureArm:
    """One arm's half of the record, assembled from what the pass produced."""
    import numpy as np

    latency = LatencySummary.from_samples(list(per_frame_ms))
    inside = statistics.fmean(
        [mean_abs_diff(source, output, mask)
         for source, output, mask in zip(sources, outputs, masks) if mask.any()]
        or [0.0])
    control = statistics.fmean(list(control_change) or [0.0])
    sides = [box.min_side for selection in selections for box in selection.boxes]
    regions = sum(selection.count for selection in selections)
    frames = max(1, len(per_frame_ms))
    return CaptureArm(
        primitive=primitive, capture_width=width, capture_height=height,
        denoise=denoise, denoise_points=list(points),
        frames=len(per_frame_ms), diffusion_calls=diffusion_calls,
        crop_frames=crop_frames, regions_per_frame=round(regions / frames, 4),
        stages=stages, latency=latency,
        ms_per_frame=round(stages.frame_path_ms, 4),
        fps=round(1000.0 / stages.frame_path_ms, 4) if stages.frame_path_ms else 0.0,
        detail=detail_summary(primitive,
                              statistics.fmean(sides) if sides else 0.0, width),
        region_change=round(inside, 4), resample_change=round(control, 4),
        flicker=flicker_score(sources, outputs, masks),
        background=background_check(
            list(background_changed),
            min(int(np.count_nonzero(~mask)) for mask in masks) if masks else 0),
        identity=identity,
    )


def renders_for(case: CaptureCase, plan, regions: Sequence[Sequence[Box]],
                width: int, height: int, indices: Sequence[int]):
    """One pass's renders, and the compositor that built them.

    A fresh scheduler and compositor, walked in order, so a pass that only needs
    the renders does not advance the rotation cursor the timed pass reads - and so
    every pass over the same frames produces the same regions. The compositor comes
    back beside them because its alpha cache and its EMA history belong to this
    pass, and a caller that renders through these has to render through it.
    """
    from device_compositor import DeviceCompositor
    from region_scheduler import RegionScheduler

    scheduler, compositor = RegionScheduler(), DeviceCompositor()
    renders = []
    for index in indices:
        selection = scheduler.select(tracks_for(regions[index], case.concept), plan,
                                     width, height)
        renders.append(compositor.frame(selection, width, height,
                                        plan.settings.primitive))
    return compositor, renders


def run_capture(
    case: CaptureCase,
    cooldown: bool = True,
    results_dir: Path = CAPTURE_RESULTS_DIR,
    threshold_c: float = DEFAULT_THRESHOLD_C,
    cap_s: float = DEFAULT_CAP_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    engines_root: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    write_clips: bool = True,
    log: Callable[[str], None] = print,
) -> CaptureResult:
    """Render one case at every capture geometry under both primitives; write it.

    Four passes per geometry, and the separation is the measurement:

    1. **the ladder**, on a handful of frames, per arm;
    2. **the timed pass**, which does nothing but render - no metric, no host
       copy, no queue. A first attempt folded all of those into one loop and
       charged their allocator churn to the diffusion call, which then appeared
       to grow from 15.8 ms to 43 ms as the capture grew. It does not: the canvas
       is fixed, and a bare probe measures 15.5 / 15.3 / 16.8 ms at the three
       geometries. What is timed here has to be only what the worker's frame loop
       does;
    3. **the metric pass**, which rebuilds the same renders off a fresh scheduler
       and asks the questions - the mask, the background, the resize control;
    4. **the probes**, on a sample of the finished frames: the device-to-host
       copy, the IPC and the preview, each measured on its own.
    """
    import multiprocessing as mp

    import numpy as np
    import torch
    from PIL import Image

    from compositor import CROP as CROP_ACTION
    from compositor import painted_mask
    from device_compositor import DeviceCompositor
    from detector_worker import frame_to_array
    from region_scheduler import RegionScheduler

    # First, so a machine that cannot be fingerprinted fails before it spends
    # minutes loading an engine for a result that could never be written.
    fingerprint = capture_fingerprint()
    log(f"gpu: {fingerprint.gpu_name} | driver {fingerprint.driver_version} | "
        f"{regime_summary(fingerprint.clock_lock)}")

    path = clip_path(case.clip)
    track = load_track(case.clip)
    raw_frames, meta = read_clip(path, case.start_frame, case.frames)
    log(f"clip: {case.clip} {meta['width']}x{meta['height']} @ {meta['fps']:.1f} fps, "
        f"{len(raw_frames)} frames from {case.start_frame}")

    scenario = SCENARIOS[ENGINE_SCENARIO]
    log(f"building {scenario.name} - one {CANVAS}x{CANVAS} engine for every arm")
    stream = build_stream(scenario.replace(prompt=case.prompt),
                          engines_root=engines_root)

    def identity_probe(rendered):
        """What the detector reads the rendered subject as, or no answer at all."""
        if detector is None or not case.becomes:
            return None, None
        check = identity_check(detector, probe_case, rendered)
        return check.became, check.frames_probed

    probe_case = SimpleNamespace(target=case.concept, becomes=case.becomes)
    detector = None
    if case.becomes:
        models_root = resolve_models_dir() if models_dir is None else Path(models_dir)
        detector = load_identity_detector(models_root, log)
        if detector is not None:
            set_detector_vocabulary(detector, [case.concept, case.becomes],
                                    Image.fromarray(raw_frames[0]))

    cooldown_record = cooldown_gate(cooldown, threshold_c, cap_s, poll_interval_s, log)
    occupancy_record = occupancy_gate(log)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started_utc = utc_now()
    queue = mp.get_context("spawn").Queue()
    arms: List[CaptureArm] = []
    artefacts: Dict[str, List] = {}
    artefact_sources: List = []
    with GpuSampler(sample_interval_s) as sampler:
        for width, height in case.geometries:
            frames = [resize(frame, width, height) for frame in raw_frames]
            tensors = [capture_tensor(frame, device=stream.device, dtype=stream.dtype)
                       for frame in frames]
            # What the frame loop composites onto: the capture as the detector's
            # own conversion reads it back, which is what bit-identity is against.
            sources = [frame_to_array(tensor) for tensor in tensors]
            regions = [frame_boxes(track, case.start_frame + index, case, width,
                                   height, meta["width"], meta["height"])
                       for index in range(len(frames))]
            plans = {primitive: case.plan(primitive) for primitive in case.primitives}
            indices = list(range(len(frames)))
            outputs: Dict[str, List] = {}

            for primitive in case.primitives:
                plan = plans[primitive]

                # 1. the ladder, on its own scheduler and compositor.
                sweep_compositor, swept = renders_for(
                    case, plan, regions, width, height,
                    indices[:case.sweep_frames])
                points = sweep_denoise(
                    stream, case, primitive, tensors, sources, swept,
                    sweep_compositor, CANVAS, identity_probe, log)
                chosen = required_denoise(points, case.kind)
                set_denoise(stream, chosen.t_index)
                log(f"denoise chosen for {arm_name(primitive, width, height)}: "
                    f"t_index {chosen.t_index}")

                # 2. the timed pass. Nothing in this loop but the render.
                scheduler, compositor = RegionScheduler(), DeviceCompositor()
                rendered: List = []
                stage_ms: Dict[str, List[float]] = {
                    name: [] for name in ("resize_in", "diffuse", "composite",
                                          "frame")}
                calls = crop_frames = 0
                selections = []
                for index, tensor in enumerate(tensors):
                    selection = scheduler.select(
                        tracks_for(regions[index], case.concept), plan, width, height)
                    render = compositor.frame(selection, width, height,
                                              plan.settings.primitive)
                    selections.append(selection)
                    if not render.diffuses:
                        rendered.append(sources[index])
                        for name in stage_ms:
                            stage_ms[name].append(0.0)
                        continue
                    if index < case.warmup_frames:
                        render_frame(stream, tensor, compositor, render, CANVAS)
                    calls += 1
                    crop_frames += int(render.action == CROP_ACTION)
                    started = time.perf_counter()
                    output, resize_in, diffuse, composite = render_frame(
                        stream, tensor, compositor, render, CANVAS)
                    frame_ms = _elapsed_ms(started)
                    rendered.append(output)
                    for name, value in (("resize_in", resize_in),
                                        ("diffuse", diffuse),
                                        ("composite", composite),
                                        ("frame", frame_ms)):
                        stage_ms[name].append(value)
                outputs[primitive] = rendered

                # 3. the metric pass. Untimed, and off a fresh scheduler, so the
                # same regions are asked about that were rendered.
                masks, changed, control_change = [], [], []
                metric_compositor, metric_renders = renders_for(
                    case, plan, regions, width, height, indices)
                for index, render in enumerate(metric_renders):
                    if not render.diffuses:
                        masks.append(np.zeros(sources[index].shape[:2], dtype=bool))
                        changed.append(0)
                        continue
                    painted = painted_mask(render.alpha)
                    masks.append(painted)
                    changed.append(background_pixels_changed(
                        sources[index], rendered[index], painted))
                    control_change.append(mean_abs_diff(
                        sources[index],
                        control_frame(tensors[index], metric_compositor, render,
                                      CANVAS),
                        painted))

                # 4. the probes: the three stages that are not the render.
                stages = StageCost(
                    resize_in_ms=mean_ms(stage_ms["resize_in"]),
                    diffuse_ms=mean_ms(stage_ms["diffuse"]),
                    composite_ms=mean_ms(stage_ms["composite"]),
                    **probe_stages(queue, rendered[:case.probe_frames],
                                   stream.device))

                identity = None
                if detector is not None and case.becomes:
                    identity = identity_check(detector, probe_case, rendered)
                    log(f"identity {arm_name(primitive, width, height)}: "
                        f"{identity.statement}")
                arm = _arm(primitive, width, height, chosen, points, stages,
                           stage_ms["frame"], sources, rendered, control_change,
                           masks, selections, changed, calls, crop_frames, identity)
                arms.append(arm)
                log(f"{arm.name}: {arm.ms_per_frame:.2f} ms/frame "
                    f"({arm.fps:.1f} FPS), denoise "
                    f"{format(arm.denoise.strength or 0.0, '.2f')}, object at "
                    f"{arm.detail.canvas_px:.0f} px, net change "
                    f"{arm.net_region_change:.1f}, background "
                    f"{'identical' if arm.background.passed else 'CHANGED'}")

            # The artefact is cut from the last geometry measured - the one the
            # change exists for - and only that one's frames are kept.
            artefacts = outputs
            artefact_sources = sources
            del tensors
            torch.cuda.empty_cache()
    finished_utc = utc_now()
    peak_vram_bytes = int(torch.cuda.max_memory_allocated())
    queue.close()

    run = CaptureRunMetrics(
        started_utc=started_utc, finished_utc=finished_utc,
        engine_scenario=ENGINE_SCENARIO, warmup_frames=case.warmup_frames,
        canvas=CANVAS, mean_sm_clock_mhz=sampler.mean_sm_clock_mhz,
        max_temperature_c=sampler.max_temperature_c,
        peak_vram_bytes=peak_vram_bytes, gpu_samples=sampler.samples)

    results_dir = Path(results_dir)
    timestamp = filename_timestamp(finished_utc)
    stem = f"{case.name}-{timestamp}"
    comparison, still = "", ""
    if write_clips:
        comparison, still = write_capture_artefacts(
            artefact_sources, artefacts, case.primitives, results_dir, stem,
            meta["fps"], log)

    result = CaptureResult(
        case=case,
        clip=ClipRecord(name=case.clip, sha256=sha256_of(path), width=meta["width"],
                        height=meta["height"], fps=meta["fps"],
                        total_frames=meta["total_frames"],
                        start_frame=case.start_frame, frames_used=len(raw_frames)),
        arms=arms, run=run, cooldown=cooldown_record, occupancy=occupancy_record,
        hardware=fingerprint,
        clock_normalization=clock_normalization(
            fingerprint.clock_lock, sampler.samples,
            raw_ms_per_frame=arms[0].ms_per_frame if arms else 0.0),
        comparison_clip=comparison, comparison_still=still)

    written = write_capture_result(result, results_dir=results_dir,
                                   timestamp=timestamp)
    append_capture_readme_rows(result, results_dir / README_NAME,
                               filename=written.name)
    log(f"{case.name} -> {written.name}")
    return result


def probe_stages(queue, frames: Sequence, device) -> Dict[str, float]:
    """The three stages that are not the render, each measured on its own.

    Off the timed pass on purpose. Two of them happen in the *other* process and
    the third is a probe rather than a stage the blend exposes, so measuring them
    inside the render loop would charge their host allocations to the diffusion
    call - which is exactly the artefact the first version of this runner
    produced. Timed on a sample of the finished frames, which are all the same
    size, so a sample is the whole distribution.
    """
    import torch
    from PIL import Image

    if not frames:
        return {"host_copy_ms": 0.0, "ipc_put_ms": 0.0, "ipc_roundtrip_ms": 0.0,
                "preview_ms": 0.0}
    copies, puts, roundtrips, previews = [], [], [], []
    for frame in frames:
        image = Image.fromarray(frame)
        put_ms, roundtrip_ms = ipc_cost(queue, image)
        puts.append(put_ms)
        roundtrips.append(roundtrip_ms)
        previews.append(preview_cost(image))
        copies.append(host_copy_cost(torch.from_numpy(frame).to(device=device)))
    return {"host_copy_ms": mean_ms(copies), "ipc_put_ms": mean_ms(puts),
            "ipc_roundtrip_ms": mean_ms(roundtrips),
            "preview_ms": mean_ms(previews)}


def write_capture_artefacts(sources: Sequence, outputs: Dict[str, List],
                            primitives: Sequence[str], results_dir: Path,
                            stem: str, fps: float,
                            log: Callable[[str], None]) -> Tuple[str, str]:
    """source | masked | crop at the largest geometry, and one full-resolution still.

    The Gate's manual-verification step has to be pointed at a file: the crop arm's
    upscaled invention is exactly the thing a number cannot rule on.
    """
    arms = [outputs[primitive] for primitive in primitives]
    comparison = write_clip(triptych(sources, arms),
                            results_dir / f"{stem}-triptych.mp4", fps).name
    middle = len(sources) // 2
    still = write_still(
        triptych(sources[middle:middle + 1],
                 [arm[middle:middle + 1] for arm in arms], panel_width=None)[0],
        results_dir / f"{stem}-triptych.jpg").name
    for primitive in primitives:
        write_clip([resized_panel(frame, COMPARISON_PANEL_WIDTH)
                    for frame in outputs[primitive]],
                   results_dir / f"{stem}-{primitive}.mp4", fps)
    log(f"clips: {comparison}, {still}")
    return comparison, still
