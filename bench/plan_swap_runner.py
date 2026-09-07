"""The measuring half of the plan swap. Touches the GPU; imports late.

Issue #30, steps 1-3. What runs here is the **shipped** cold path, not a
description of it: `render_plan.plan_from_fields` builds the new instruction from
the two strings a user would type, `validate_plan` accepts or refuses it,
`ActivePlan.submit` puts it where the *next* frame will pick it up, and
`apply_plan` does exactly the three things the worker's frame loop does when
`begin_frame` reports a change - re-encode the prompt on the live engine, hand the
plan to the detector thread, and move the schedule value.

Three things about what is timed.

- **The clock is the frame's completion time.** `intervals_of` turns those into
  the inter-frame interval series, which is the only quantity criterion 3 is about
  - and the swap's cold-path work lands inside the interval of the frame that
  applied it, which is where a stutter would show.
- **The detector's re-warm is inside the measurement, not skipped.** A vocabulary
  change owes one throwaway detect (spec 8.1), it happens on the detector's own
  thread, and a swap measured without it measures a path the app never takes.
- **One hop is missing and is named rather than absorbed.** The app puts
  `set_plan` on a `multiprocessing.Queue` between two processes; this harness is
  one process, so what it measures is everything either side of that queue and
  not the queue. The report says so.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from bench.clocks import clock_normalization, regime_summary
from bench.cooldown import DEFAULT_CAP_S, DEFAULT_POLL_INTERVAL_S, DEFAULT_THRESHOLD_C
from bench.detector_results import LatencySummary
from bench.detectors import PRIMARY_DETECTOR
from bench.fingerprint import capture_fingerprint, utc_now
from bench.paths import SWAP_RESULTS_DIR, resolve_models_dir
from bench.plan_swap import (
    README_NAME,
    SwapCase,
    SwapResult,
    SwapRunMetrics,
    append_swap_readme_row,
    first_pixel_frame,
    intervals_of,
    latency_check,
    rebuild_check,
    stutter_check,
    swap_kind,
    swap_timing,
    write_swap_result,
)
from bench.primitive_results import ClipRecord
from bench.primitives import clip_path
from bench.primitive_runner import read_clip, resize, set_denoise, sha256_of
from bench.results import filename_timestamp
from bench.runner import (
    DEFAULT_SAMPLE_INTERVAL_S,
    GpuSampler,
    build_stream,
    cooldown_gate,
)
from bench.scenarios import SCENARIOS
from bench.selective import ENGINE_SCENARIO, background_check, plan_record
from bench.selective_runner import (
    capture_tensor,
    open_detector,
    render_frame,
    write_comparison_artefacts,
    background_pixels_changed,
)


def engine_identity(stream) -> Tuple[str, str]:
    """`(wrapper, unet)` as identities that survive into the record.

    Two objects rather than one because either alone could be fooled: a wrapper
    rebuilt around the same engine, or the same wrapper handed a new one. The
    identity is the object's, which is exactly the question - a TensorRT rebuild
    tears the wrapper down and constructs another (`main_gpu_addon.py`'s
    `engine_swap` branch), so an unchanged pair is evidence no rebuild happened.
    """
    inner = stream.stream
    unet = getattr(inner, "unet", inner)
    return (f"{type(stream).__name__}@{id(stream):x}",
            f"{type(unet).__name__}@{id(unet):x}")


def apply_plan(stream, detection, plan, t_index_list: Sequence[int],
               log: Callable[[str], None] = print) -> List[int]:
    """What the worker's frame loop does when the plan it bound changed.

    The same three calls in the same order: the engine takes the new prompt, the
    detector thread takes the new vocabulary (and pays the re-warm off the frame
    path), and the plan's denoise reaches the engine as a *schedule value*. Only
    the values move, never the step count, which is why this is a runtime update
    and not an engine rebuild - the claim criterion 3's second half checks.
    """
    from render_plan import t_index_for_denoise

    try:
        stream.stream.update_prompt(plan.effective_prompt)
    except Exception as error:  # as in the worker: the loop keeps rendering
        log(f"prompt not applied: {error}")
    detection.follow(plan)
    honoured = plan.honoured_target
    current = list(t_index_list)
    if honoured is not None and len(current) == 1:
        wanted = t_index_for_denoise(plan.effective_denoise)
        if wanted != current[0]:
            current = [wanted]
            stream.set_t_index_list(current)
    return current


def _frame_record(index: int, plan_version: int, tracks, selection,
                  diffuses: bool, detector_ticks: int) -> dict:
    """One frame, as the two "when did it arrive" questions need to read it.

    `detector_ticks` is the detector's own running count rather than the
    snapshot's: `follow` publishes `EMPTY_TRACKS` on a vocabulary change, whose
    count is zero, so a difference taken across the swap on the snapshot would
    report every detect the run ever did.
    """
    return {"index": index, "plan_version": plan_version,
            "tracks_concepts": list(tracks.concepts),
            "detector_ticks": detector_ticks,
            "diffuses": bool(diffuses), "regions": len(selection.boxes)}


def _mean_regions(frames: Sequence[dict], first: int, last: int) -> float:
    """Regions per frame over a window - the control on the interval control."""
    window = [frame["regions"] for frame in frames[max(0, first):last + 1]]
    return round(statistics.fmean(window), 4) if window else 0.0


def run_swap(
    case: SwapCase,
    cooldown: bool = True,
    results_dir: Path = SWAP_RESULTS_DIR,
    threshold_c: float = DEFAULT_THRESHOLD_C,
    cap_s: float = DEFAULT_CAP_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    engines_root: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    write_clips: bool = True,
    log: Callable[[str], None] = print,
) -> SwapResult:
    """Render a clip under one instruction, swap it mid-clip, write the result."""
    import numpy as np
    import torch

    from compositor import Compositor, painted_mask
    from detection import EMPTY_TRACKS, is_detect_frame
    from region_scheduler import RegionScheduler
    from render_plan import ActivePlan, t_index_for_denoise

    # First, so a machine that cannot be fingerprinted fails before it spends
    # minutes loading an engine for a result that could never be written.
    fingerprint = capture_fingerprint()
    log(f"gpu: {fingerprint.gpu_name} | driver {fingerprint.driver_version} | "
        f"{regime_summary(fingerprint.clock_lock)}")

    before_plan, after_plan = case.plans()
    kind = swap_kind(before_plan, after_plan)
    canvas = case.canvas
    detect_every_n = before_plan.settings.detect_every_n

    path = clip_path(case.clip)
    raw_frames, meta = read_clip(path, case.start_frame, case.frames)
    frames = [resize(frame, canvas, canvas) for frame in raw_frames]
    log(f"clip: {case.clip} {meta['width']}x{meta['height']} -> {canvas}x{canvas}, "
        f"{len(frames)} frames from {case.start_frame}")

    scenario = SCENARIOS[ENGINE_SCENARIO]
    log(f"building {scenario.name}")
    stream = build_stream(scenario.replace(prompt=before_plan.effective_prompt),
                          engines_root=engines_root)
    t_index_list = [t_index_for_denoise(before_plan.effective_denoise)]
    set_denoise(stream, t_index_list[0])
    engine_before, unet_before = engine_identity(stream)
    log(f"{kind} swap: {before_plan.honoured_target.concept} -> "
        f"{after_plan.honoured_target.concept}, t_index {t_index_list[0]} -> "
        f"{t_index_for_denoise(after_plan.effective_denoise)}")

    models_root = resolve_models_dir() if models_dir is None else Path(models_dir)
    live = open_detector(models_root,
                         [target.concept for target in before_plan.targets], log)
    if live is None:
        raise SystemExit("bench: the plan swap needs the detector; see the message above")
    from detector_worker import BackgroundDetector

    detection = BackgroundDetector(live, log=log)
    detection.follow(before_plan)
    detection.start()

    scheduler = RegionScheduler()
    compositor = Compositor()
    tensors = [capture_tensor(frame, device=stream.device, dtype=stream.dtype)
               for frame in frames]
    from detector_worker import frame_to_array

    sources = [frame_to_array(tensor) for tensor in tensors]

    # Warm the engine, the allocator and the detector's first vocabulary, so the
    # run measures a swap rather than a startup.
    for index in range(min(case.warmup_frames, len(frames))):
        stream.img2img(tensors[index])
    detection.offer(tensors[0], 0)
    detection.wait_for_tick(1, timeout=120.0)

    cooldown_record = cooldown_gate(cooldown, threshold_c, cap_s, poll_interval_s, log)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started_utc = utc_now()
    active_plan = ActivePlan(before_plan)
    outputs: List = []
    frame_records: List[dict] = []
    finished_at: List[float] = []
    per_frame_ms: List[float] = []
    detector_ms: List[float] = []
    background_changed: List[int] = []
    masks: List = []
    accepted_at: Optional[float] = None
    validate_ms = 0.0
    diffusion_calls = 0
    seen_ticks = 0
    with GpuSampler(sample_interval_s) as sampler:
        for index, tensor in enumerate(tensors):
            if index == case.swap_frame:
                # The GUI's producer, timed: two typed strings through
                # `plan_from_fields` and the validator. The debounce before it is
                # a constant the record carries rather than a wait to sit out.
                typed = time.perf_counter()
                new_plan = case.after.plan(previous_version=active_plan.latest.plan_version)
                validate_ms = (time.perf_counter() - typed) * 1000.0
                accepted_at = time.perf_counter()
                active_plan.submit(new_plan)
            frame_plan = active_plan.begin_frame()
            if frame_plan.changed:
                t_index_list = apply_plan(stream, detection, frame_plan.plan,
                                          t_index_list, log)
            if is_detect_frame(index, detect_every_n):
                detection.offer(tensor, index)
            tracks = detection.tracks if detection is not None else EMPTY_TRACKS
            if tracks.ticks > seen_ticks:
                seen_ticks = tracks.ticks
                detector_ms.append(tracks.detector_ms)
            selection = scheduler.select(tracks, frame_plan.plan, canvas, canvas)
            render = compositor.frame(selection, canvas, canvas)
            output, frame_ms, _ = render_frame(stream, tensor, compositor, render,
                                               sources[index])
            finished_at.append(time.perf_counter())
            outputs.append(output)
            per_frame_ms.append(frame_ms)
            diffusion_calls += 1 if render.diffuses else 0
            painted = (painted_mask(render.alpha) if render.alpha is not None
                       else np.zeros(output.shape[:2], dtype=bool))
            masks.append(painted)
            background_changed.append(
                background_pixels_changed(sources[index], output, painted))
            frame_records.append(
                _frame_record(index, frame_plan.plan.plan_version, tracks, selection,
                              render.diffuses, detection.ticks))
    finished_utc = utc_now()
    peak_vram_bytes = int(torch.cuda.max_memory_allocated())
    detection.stop()
    engine_after, unet_after = engine_identity(stream)

    intervals = intervals_of(finished_at)
    applied_index = next((frame["index"] for frame in frame_records
                          if frame["plan_version"] == after_plan.plan_version), None)
    pixel_index = first_pixel_frame(
        frame_records, after_plan.plan_version,
        [target.concept for target in after_plan.targets])
    timing = swap_timing(
        swap_frame=case.swap_frame,
        plan_version_before=before_plan.plan_version,
        plan_version_after=after_plan.plan_version, kind=kind,
        validate_ms=validate_ms,
        accepted_to_applied_ms=(None if applied_index is None else
                                (finished_at[applied_index] - accepted_at) * 1000.0),
        accepted_to_pixel_ms=(None if pixel_index is None else
                              (finished_at[pixel_index] - accepted_at) * 1000.0),
        frames_to_applied=(None if applied_index is None
                           else applied_index - case.swap_frame + 1),
        frames_to_pixel=(None if pixel_index is None
                         else pixel_index - case.swap_frame + 1),
        unrestyled_frames=sum(1 for frame in frame_records[case.swap_frame:
                                                           pixel_index or len(frames)]
                              if not frame["diffuses"]),
        detector_ticks_waited=(0 if pixel_index is None else
                               frame_records[pixel_index]["detector_ticks"]
                               - frame_records[case.swap_frame]["detector_ticks"]),
    )
    # A swap whose pixels never arrived has no window to judge steadiness over
    # either, so the last frame stands in and the latency check fails the Gate.
    stutter = stutter_check(intervals, case.swap_frame,
                            len(frames) - 1 if pixel_index is None else pixel_index,
                            warmup_frames=case.warmup_frames)
    background = background_check(
        background_changed,
        min(int(np.count_nonzero(~mask)) for mask in masks) if masks else 0)
    rebuild = rebuild_check(
        engine_id_before=engine_before, engine_id_after=engine_after,
        unet_id_before=unet_before, unet_id_after=unet_after,
        t_index_before=[t_index_for_denoise(before_plan.effective_denoise)],
        t_index_after=t_index_list)

    detect_latency = (LatencySummary.from_samples(detector_ms) if detector_ms
                      else None)
    render_latency = LatencySummary.from_samples(per_frame_ms)
    run = SwapRunMetrics(
        started_utc=started_utc, finished_utc=finished_utc,
        warmup_frames=case.warmup_frames, engine_scenario=ENGINE_SCENARIO,
        detector=PRIMARY_DETECTOR, frames=len(frames),
        diffusion_calls=diffusion_calls, render=render_latency,
        detect=detect_latency, detect_every_n=detect_every_n,
        detector_ticks=seen_ticks,
        regions_per_frame_before=_mean_regions(frame_records, case.warmup_frames,
                                               case.swap_frame - 1),
        regions_per_frame_after=_mean_regions(
            frame_records, (pixel_index or case.swap_frame) + 1, len(frames) - 1),
        ms_per_frame=render_latency.mean_ms,
        mean_sm_clock_mhz=sampler.mean_sm_clock_mhz,
        max_temperature_c=sampler.max_temperature_c,
        peak_vram_bytes=peak_vram_bytes, gpu_samples=sampler.samples,
    )

    results_dir = Path(results_dir)
    timestamp = filename_timestamp(finished_utc)
    stem = f"{case.name}-{timestamp}"
    comparison, still = "", ""
    if write_clips:
        comparison, still = write_comparison_artefacts(
            sources, outputs, results_dir, stem, meta["fps"], log,
            # The still is the frame the new instruction first reached, which is
            # the one a human has to look at to confirm it is the right edit.
            still_index=pixel_index, render_clip=False)

    result = SwapResult(
        case=case, plan_before=plan_record(before_plan),
        plan_after=plan_record(after_plan),
        clip=ClipRecord(name=case.clip, sha256=sha256_of(path), width=meta["width"],
                        height=meta["height"], fps=meta["fps"],
                        total_frames=meta["total_frames"],
                        start_frame=case.start_frame, frames_used=len(frames)),
        run=run, timing=timing, intervals_ms=intervals,
        latency=latency_check(timing), stutter=stutter, rebuild=rebuild,
        background=background, cooldown=cooldown_record, hardware=fingerprint,
        clock_normalization=clock_normalization(
            fingerprint.clock_lock, sampler.samples,
            raw_ms_per_frame=run.ms_per_frame),
        comparison_clip=comparison, comparison_still=still,
    )
    for check in (result.latency, result.stutter, result.rebuild, result.background):
        log(f"gate: {'pass' if check.passed else 'FAIL'} - {check.statement}")

    written = write_swap_result(result, results_dir=results_dir, timestamp=timestamp)
    append_swap_readme_row(result, results_dir / README_NAME, filename=written.name)
    log(f"{case.name} -> {written.name}")
    return result
