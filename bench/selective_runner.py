"""The measuring half of the selective render path. Touches the GPU; imports late.

Issue #8, step 5. What runs here is the **shipped** path, not a copy of it:
`detector_worker.BackgroundDetector` on its own thread, `detection.Tracker`,
`region_scheduler.RegionScheduler`, `device_compositor.DeviceCompositor`, and the plan
`render_plan.priority_case_plan()` the worker itself starts on behind
`SD_DEMO_PLAN`. The only thing this module supplies is the frame source - a
committed clip instead of a screen - and the clock.

Three things about the numbers.

- **The clip is resized to the capture canvas first.** The worker's capture thread
  hands the frame loop a frame already at the engine's size, so that is where the
  measurement starts. It is also what makes the bit-identity criterion sharp: there
  is no resize inside the path to blur the background, and the control that proves
  it - the capture's own uint8 -> float -> uint8 round trip - is measured rather
  than assumed.
- **Detection runs on its thread, as it does in the worker.** So the per-frame
  figure is what the frame path actually pays, the detector's cost is reported
  separately and amortised, and the worst `offer` is recorded - which is the Gate's
  "never stalls" item measured rather than promised.
- **No absolute-latency assertion.** Every millisecond printed carries the SM clock
  it was measured at (issue #13), and the FPS figure is this laptop's, not the
  deploy hardware's (spec 7.4).
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from bench.clocks import clock_normalization, regime_summary
from bench.cooldown import DEFAULT_CAP_S, DEFAULT_POLL_INTERVAL_S, DEFAULT_THRESHOLD_C
from bench.detector_results import LatencySummary
from bench.detectors import DETECTORS, PRIMARY_DETECTOR, weights_path
from bench.fingerprint import capture_fingerprint, utc_now
from bench.flicker import flicker_score, response_score
from bench.paths import SELECTIVE_RESULTS_DIR, resolve_models_dir
from bench.primitive_results import ClipRecord
from bench.primitives import clip_path
from bench.primitive_runner import (
    COMPARISON_PANEL_WIDTH,
    mean_abs_diff,
    read_clip,
    resize,
    resized_panel,
    set_denoise,
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
from bench.selective import (
    ENGINE_SCENARIO,
    SELECTIVE_README_NAME,
    SELECTIVE_README_PREAMBLE,
    RegionSummary,
    SelectiveCase,
    SelectiveResult,
    SelectiveRunMetrics,
    append_selective_readme_row,
    background_check,
    change_check,
    coverage_check,
    plan_record,
    stall_check,
    staleness_summary,
    write_selective_result,
)


def capture_tensor(frame, device: str = "cuda", dtype=None):
    """One clip frame as the capture thread would hand it to the frame loop.

    HWC uint8 RGB -> (1, 3, H, W) float in 0..1 on the GPU. The same shape and the
    same range `_screen_capture_loop_dx` produces, so what is timed below is the
    call the worker makes and not a differently-shaped one.
    """
    import torch

    tensor = torch.from_numpy(frame).to(device=device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(
        dtype=torch.float32 if dtype is None else dtype)
    return tensor / 255.0


def open_detector(models_root: Path, concepts: Sequence[str],
                  log: Callable[[str], None] = print):
    """The shipped detector, loaded and warmed, or None with the reason.

    Never fetches: the worker is offline and so is this. A missing checkpoint is a
    refusal naming the command that would cache it, exactly as issue #4's run does.
    """
    from detector_worker import UltralyticsDetector

    weights = weights_path(DETECTORS[PRIMARY_DETECTOR], models_root)
    if not weights.is_file():
        log(f"no detector weights at {weights}; run "
            f"`python -m bench {PRIMARY_DETECTOR} --allow-download` once")
        return None
    detector = UltralyticsDetector(models_root)
    detector.open()
    detector.set_concepts(list(concepts))
    return detector


def render_frame(stream, tensor, compositor, render, source, noise, selection):
    """One frame of the shipped path: seed, diffuse, then composite through the mask.

    Returns the output frame and the milliseconds the frame path spent, split into
    the whole call and the composite alone - the composite is the part issue #8
    added to the loop, and it has to be readable next to the diffusion it rides on.

    `noise` is the frame's `seeding.NoiseField` and `selection` is what it pins a
    field to; under the default `fixed` policy it writes nothing at all.

    Since issue #31 the masked frame is blended **on the device**: the engine is
    asked for `output_type="pt"`, so the render never comes home, and the frame's
    one device-to-host copy happens inside the composite instead of inside the
    engine call. The composite figure therefore now carries that copy, which the
    host figure it is compared against did not - the comparison is conservative in
    the direction that matters.
    """
    import numpy as np
    import torch
    from compositor import MASKED

    torch.cuda.synchronize()
    started = time.perf_counter()
    composite_ms = 0.0
    if not render.diffuses:
        # Nothing to restyle: the capture itself, which is what the worker emits -
        # and the end of the EMA's history, for the reason the worker ends it.
        compositor.reset_ema()
        output = source
    elif render.action == MASKED:
        # The plan's seed policy, written into the engine's noise before the call
        # that reads it (issue #32). Inside the timed region because it is inside
        # the worker's frame loop: whatever it costs, the frame pays it.
        noise.apply(stream, selection)
        rendered = stream.img2img(tensor, output_type="pt")
        torch.cuda.synchronize()
        composite_started = time.perf_counter()
        output = compositor.blend_device(tensor, rendered, render.alpha)[-1]
        composite_ms = (time.perf_counter() - composite_started) * 1000.0
    else:
        output = np.asarray(stream.img2img(tensor))
    torch.cuda.synchronize()
    return output, (time.perf_counter() - started) * 1000.0, composite_ms


def _region_summary(selections: Sequence, plan, feather_px: int) -> RegionSummary:
    """What the scheduler did over the run, out of the selections it produced."""
    from region_scheduler import MIN_REGION_PX, slots_for

    boxes = [box for selection in selections for box in selection.boxes]
    sides = [box.min_side for box in boxes]
    frames = max(1, len(selections))
    return RegionSummary(
        slots=slots_for(plan),
        regions_rendered=len(boxes),
        regions_per_frame=round(len(boxes) / frames, 4),
        tracks_per_frame=round(
            sum(selection.candidates for selection in selections) / frames, 4),
        deferred_total=sum(selection.deferred for selection in selections),
        skipped_small_total=sum(selection.skipped_small for selection in selections),
        min_region_px=MIN_REGION_PX, feather_px=feather_px,
        min_side_px=min(sides) if sides else None,
        max_side_px=max(sides) if sides else None,
    )


def background_pixels_changed(source, output, painted) -> int:
    """How many background pixels the render moved. Zero is the whole point.

    Public because the plan-swap run (issue #30) asks the same question of its own
    frames: bit-identity has to hold *across* a swap too, and a second spelling of
    this would be a second criterion wearing the same name.
    """
    import numpy as np

    outside = ~painted
    if not outside.any():
        return 0
    return int(np.count_nonzero(np.any(source[outside] != output[outside], axis=-1)))


def run_selective(
    case: SelectiveCase,
    cooldown: bool = True,
    results_dir: Path = SELECTIVE_RESULTS_DIR,
    threshold_c: float = DEFAULT_THRESHOLD_C,
    cap_s: float = DEFAULT_CAP_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    engines_root: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    write_clips: bool = True,
    readme_preamble: str = SELECTIVE_README_PREAMBLE,
    log: Callable[[str], None] = print,
) -> SelectiveResult:
    """Drive the shipped selective path over one committed clip; write the result."""
    import numpy as np
    import torch

    from compositor import painted_mask
    from detection import EMPTY_TRACKS, is_detect_frame
    from device_compositor import DEVICE, DeviceCompositor
    from detector_worker import BackgroundDetector, frame_to_array
    from region_scheduler import RegionScheduler
    from render_plan import t_index_for_denoise
    from seeding import NoiseField

    # First, so a machine that cannot be fingerprinted fails before it spends
    # minutes loading an engine for a result that could never be written.
    fingerprint = capture_fingerprint()
    log(f"gpu: {fingerprint.gpu_name} | driver {fingerprint.driver_version} | "
        f"{regime_summary(fingerprint.clock_lock)}")

    plan = case.plan()
    concepts = tuple(target.concept for target in plan.targets)
    detect_every_n = plan.settings.detect_every_n
    canvas = case.canvas

    path = clip_path(case.clip)
    raw_frames, meta = read_clip(path, case.start_frame, case.frames)
    # The capture thread's own resize: the frame loop never sees a frame that is
    # not the engine's canvas size.
    frames = [resize(frame, canvas, canvas) for frame in raw_frames]
    log(f"clip: {case.clip} {meta['width']}x{meta['height']} -> {canvas}x{canvas}, "
        f"{len(frames)} frames from {case.start_frame}")

    scenario = SCENARIOS[ENGINE_SCENARIO]
    log(f"building {scenario.name}")
    stream = build_stream(scenario.replace(prompt=plan.effective_prompt),
                          engines_root=engines_root)
    t_index = t_index_for_denoise(plan.effective_denoise)
    timestep, strength = set_denoise(stream, t_index)
    log(f"plan: {concepts} / {plan.honoured_target.region} / denoise "
        f"{plan.effective_denoise} -> t_index {t_index} (timestep {timestep}, "
        f"strength {strength:.3f}), seed {plan.effective_seed_policy}, "
        f"output_ema {plan.settings.output_ema}")

    models_root = resolve_models_dir() if models_dir is None else Path(models_dir)
    live = open_detector(models_root, concepts, log)
    detection = None if live is None else BackgroundDetector(live, log=log)
    if detection is not None:
        detection.follow(plan)
        detection.start()

    scheduler = RegionScheduler()
    # The shipped compositor, blend and all: since issue #31 the frame loop's C7 is
    # the device one, so measuring the host one here would measure a path the app
    # no longer takes. Its output EMA is the plan's, like every other setting here
    # (issue #32) - an arm of the stability sweep is a plan, not a harness flag.
    compositor = DeviceCompositor(output_ema=plan.settings.output_ema)
    noise = NoiseField(policy=plan.effective_seed_policy)
    tensors = [capture_tensor(frame, device=stream.device, dtype=stream.dtype)
               for frame in frames]
    # What the frame loop composites onto: the capture as the detector's own
    # conversion reads it back. Its distance from the decoded frame is the control
    # the change criterion subtracts.
    sources = [frame_to_array(tensor) for tensor in tensors]
    capture_change = statistics.fmean(
        [mean_abs_diff(frame, source, np.ones(frame.shape[:2], dtype=bool))
         for frame, source in zip(frames, sources)])

    # Warm the engine and the allocator on the path's own call. The frames the
    # scheduler would have selected are not known yet - no detect has run - so the
    # warmup is a plain full-frame render, which is the dearest thing a frame does.
    for index in range(min(case.warmup_frames, len(frames))):
        stream.img2img(tensors[index])
    if detection is not None:
        # The weights load and the vocabulary re-warm happen on the first step;
        # paying them inside the timed loop would measure issue #4's cold path.
        detection.offer(tensors[0], 0)
        detection.wait_for_tick(1, timeout=120.0)

    cooldown_record = cooldown_gate(cooldown, threshold_c, cap_s, poll_interval_s, log)
    # After the cooldown and before the clock starts: the process is idle here,
    # so the utilization this reads is the card's other tenants (issue #33).
    occupancy_record = occupancy_gate(log)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started_utc = utc_now()
    outputs: List = []
    masks: List = []
    selections: List = []
    snapshots: List = []
    per_frame_ms: List[float] = []
    composite_ms: List[float] = []
    detector_ms: List[float] = []
    background_changed: List[int] = []
    worst_offer_ms = 0.0
    diffusion_calls = 0
    passthrough_frames = 0
    seen_ticks = 0
    with GpuSampler(sample_interval_s) as sampler:
        for index, tensor in enumerate(tensors):
            if detection is not None and is_detect_frame(index, detect_every_n):
                offered = time.perf_counter()
                detection.offer(tensor, index)
                worst_offer_ms = max(worst_offer_ms,
                                     (time.perf_counter() - offered) * 1000.0)
            tracks = detection.tracks if detection is not None else EMPTY_TRACKS
            if tracks.ticks > seen_ticks:
                seen_ticks = tracks.ticks
                detector_ms.append(tracks.detector_ms)
            selection = scheduler.select(tracks, plan, canvas, canvas)
            render = compositor.frame(selection, canvas, canvas)
            output, frame_ms, blend_ms = render_frame(
                stream, tensor, compositor, render, sources[index], noise, selection)
            outputs.append(output)
            per_frame_ms.append(frame_ms)
            if render.diffuses:
                diffusion_calls += 1
            else:
                passthrough_frames += 1
            if blend_ms:
                composite_ms.append(blend_ms)
            painted = (painted_mask(render.alpha) if render.alpha is not None
                       else np.zeros(output.shape[:2], dtype=bool))
            masks.append(painted)
            selections.append(selection)
            snapshots.append(tracks)
            background_changed.append(
                background_pixels_changed(sources[index], output, painted))
    finished_utc = utc_now()
    peak_vram_bytes = int(torch.cuda.max_memory_allocated())
    if detection is not None:
        detection.stop()

    region_change = statistics.fmean(
        [mean_abs_diff(source, output, mask)
         for source, output, mask in zip(sources, outputs, masks) if mask.any()]
        or [0.0])
    # The *smallest* background any frame had: the frame with the most painted
    # pixels is the one the identity claim is thinnest on.
    background = background_check(
        background_changed,
        min(int(np.count_nonzero(~mask)) for mask in masks) if masks else 0)
    change = change_check(region_change, capture_change)
    coverage = coverage_check(snapshots, plan, canvas, canvas, case.coverage_slots)
    stall = stall_check(len(frames), len(outputs), worst_offer_ms, passthrough_frames)

    render_latency = LatencySummary.from_samples(per_frame_ms)
    blend_latency = LatencySummary.from_samples(composite_ms or [0.0])
    detect_latency = (LatencySummary.from_samples(detector_ms)
                      if detector_ms else None)
    amortised = round((detect_latency.mean_ms / detect_every_n)
                      if detect_latency else 0.0, 4)
    with_detection = round(render_latency.mean_ms + amortised, 4)
    run = SelectiveRunMetrics(
        started_utc=started_utc, finished_utc=finished_utc,
        warmup_frames=case.warmup_frames, engine_scenario=ENGINE_SCENARIO,
        detector=None if detection is None else PRIMARY_DETECTOR,
        frames=len(frames), diffusion_calls=diffusion_calls,
        render=render_latency, composite=blend_latency, detect=detect_latency,
        detect_every_n=detect_every_n, detector_ticks=seen_ticks,
        amortised_detect_ms=amortised, ms_per_frame=render_latency.mean_ms,
        ms_per_frame_with_detection=with_detection,
        fps=round(1000.0 / with_detection, 4) if with_detection else 0.0,
        mean_sm_clock_mhz=sampler.mean_sm_clock_mhz,
        max_temperature_c=sampler.max_temperature_c,
        composite_path=DEVICE,
        peak_vram_bytes=peak_vram_bytes, gpu_samples=sampler.samples,
    )

    results_dir = Path(results_dir)
    timestamp = filename_timestamp(finished_utc)
    stem = f"{case.name}-{timestamp}"
    comparison, still = "", ""
    if write_clips:
        comparison, still = write_comparison_artefacts(sources, outputs, results_dir,
                                                       stem, meta["fps"], log)

    result = SelectiveResult(
        case=case, plan=plan_record(plan),
        clip=ClipRecord(name=case.clip, sha256=sha256_of(path), width=meta["width"],
                        height=meta["height"], fps=meta["fps"],
                        total_frames=meta["total_frames"],
                        start_frame=case.start_frame, frames_used=len(frames)),
        run=run, regions=_region_summary(selections, plan, compositor.feather_px),
        staleness=staleness_summary(snapshots, detect_every_n,
                                    ms_per_frame=run.ms_per_frame),
        flicker=flicker_score(sources, outputs, masks),
        response=response_score(sources, outputs, masks),
        background=background, change=change, coverage=coverage, stall=stall,
        cooldown=cooldown_record, occupancy=occupancy_record,
        hardware=fingerprint,
        clock_normalization=clock_normalization(
            fingerprint.clock_lock, sampler.samples,
            raw_ms_per_frame=run.ms_per_frame),
        comparison_clip=comparison, comparison_still=still,
    )
    log(f"{run.ms_per_frame:.2f} ms/frame ({with_detection:.2f} with detection "
        f"amortised, {run.fps:.1f} FPS), flicker {result.flicker.mean_abs_diff}, "
        f"response {result.response.mean_abs_diff}")
    for check in (background, change, coverage, stall):
        log(f"gate: {'pass' if check.passed else 'FAIL'} - {check.statement}")
    log(f"staleness: {result.staleness.statement}")

    written = write_selective_result(result, results_dir=results_dir,
                                     timestamp=timestamp)
    append_selective_readme_row(result, results_dir / SELECTIVE_README_NAME,
                                filename=written.name, preamble=readme_preamble)
    log(f"{case.name} -> {written.name}")
    return result


def write_comparison_artefacts(sources: Sequence, outputs: Sequence,
                               results_dir: Path, stem: str, fps: float,
                               log: Callable[[str], None],
                               still_index: Optional[int] = None,
                               render_clip: bool = True) -> Tuple[str, str]:
    """The files a human judges the run by: capture | render.

    The Gate's manual-verification step has to be pointed at a file, and the still
    is full resolution because a subtle low-denoise change on a person's lower half
    is not visible in a downscaled clip.

    `still_index` is the frame the still is cut from - the middle of the clip by
    default, and for a plan swap (issue #30) the frame the new instruction first
    reached, which is the one there is anything to look at on. `render_clip` is
    the render-only arm, which the swap run does not need beside the comparison.
    """
    comparison = write_clip(triptych(sources, [outputs]),
                            results_dir / f"{stem}-comparison.mp4", fps).name
    written = [comparison]
    if render_clip:
        panels = [resized_panel(frame, COMPARISON_PANEL_WIDTH) for frame in outputs]
        written.append(write_clip(panels, results_dir / f"{stem}-render.mp4", fps).name)
    chosen = len(sources) // 2 if still_index is None else still_index
    still = write_still(
        triptych(sources[chosen:chosen + 1], [outputs[chosen:chosen + 1]],
                 panel_width=None)[0],
        results_dir / f"{stem}-comparison.jpg").name
    log(f"clips: {', '.join(written)}, {still}")
    return comparison, still
