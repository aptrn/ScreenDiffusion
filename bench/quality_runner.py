"""The measuring half of the step-count sweep. Touches the GPU. Issue #46.

What runs here is the shipped selective path with a committed box track in place of
the live detector - `region_scheduler`'s own scheduler at K=1,
`device_compositor.DeviceCompositor`, and the plan `render_plan.plan_from_fields`
produces from the case's two fields. Only the step count and the batching route
move between arms.

Four things about the numbers.

- **The swap is timed, not inferred.** Every arm is preceded by letting the previous
  engine go and building this one, which is exactly what
  `image_generation_process` does when a step count changes. One priming build and
  teardown happens *before* the first arm, so no arm's figure carries this
  process's CUDA start-up: every `swap_seconds` in the record is a swap.
- **`engine_cached` is read before the build**, so an arm that had to compile says
  so and is left out of the swap figure - it measured a build, which is the other
  number.
- **The denoise is one rung for the whole sweep**, opened at the index the case's
  denoise names, with `render_plan.t_index_ladder` spending the rest after it. More
  steps at the same opening index is the picture the issue's fifth trap asks for;
  more steps at a *different* one would be two changes at once.
- **The boxes come from a committed track**, so every arm renders exactly the same
  region of exactly the same frame and the difference between two rows is the arm.

Every arm is a fresh `StreamDiffusionWrapper`: the step count and the batching flag
are both constructor arguments, and the UNet batch derived from them is fixed at
construction.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from bench.capture_runner import frame_boxes, renders_for
from bench.clocks import clock_normalization, regime_summary
from bench.cooldown import DEFAULT_CAP_S, DEFAULT_POLL_INTERVAL_S, DEFAULT_THRESHOLD_C
from bench.fingerprint import capture_fingerprint, utc_now
from bench.flicker import flicker_score, response_score
# One frame of the shipped path, and letting an arm's model go before the next is
# built. Both are exactly what issue #45's sweep needs and neither is about
# guidance, so they are borrowed rather than copied - a second spelling of "render
# one frame through the compositor" is how two sweeps start measuring two things.
from bench.guidance_runner import free, render_frame
from bench.paths import QUALITY_RESULTS_DIR, resolve_engines_dir, resolve_models_dir
from bench.primitive_results import ClipRecord
from bench.primitives import clip_path, load_track
from bench.primitive_runner import (
    COMPARISON_PANEL_WIDTH,
    labels_in,
    load_identity_detector,
    mean_abs_diff,
    read_clip,
    resize,
    set_detector_vocabulary,
    sha256_of,
    triptych,
    write_clip,
    write_still,
)
from bench.quality import (
    ADHERENCE_CONF,
    QUALITY_README_NAME,
    QualityArm,
    QualityCase,
    QualityResult,
    StepSpec,
    append_quality_readme_rows,
    arm_name,
    engine_keying,
    recommend_route,
    swap_summary,
    write_quality_result,
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


def arm_scenario(case: QualityCase, spec: StepSpec, ladder: Sequence[int]):
    """The case's cell at one arm's step count and route.

    The schedule goes in through the *constructor* rather than through
    `set_t_index_list` afterwards: the count is what the pipeline derives its batch
    and its `x_t_latent_buffer` from, and a count changed after construction is the
    thing `wrapper.set_t_index_list` cannot do (which is why the worker rebuilds).
    """
    scenario = SCENARIOS[case.base_scenario]
    return scenario.replace(name=f"{scenario.name}-{arm_name(spec)}",
                            prompt=case.prompt, t_index_list=list(ladder),
                            use_denoising_batch=spec.use_denoising_batch)


def adherence_probe(detector, case: QualityCase, frames: Sequence
                    ) -> Tuple[int, int, float]:
    """How many rendered frames read back as the prompt's concept, and how strongly.

    `bench.guidance_runner`'s probe, on the same clip at the same confidence: two
    numbers rather than one, because a fraction over 48 frames saturates and the
    detector's own confidence separates two arms that both land it every time.
    """
    from PIL import Image

    hits, retained, confidences = 0, 0, []
    for frame in frames:
        labels = labels_in(detector, Image.fromarray(frame), conf=ADHERENCE_CONF)
        hits += int(case.reads_back_as in labels)
        retained += int(case.concept in labels)
        confidences.append(labels.get(case.reads_back_as, 0.0))
    return hits, retained, round(statistics.fmean(confidences or [0.0]), 4)


def build_arm(scenario, engines_root: Path) -> Tuple[object, float]:
    """Build one arm's engine and say how long it took, with the previous one gone.

    The caller frees the outgoing stream first, so what this clock covers is the
    load and the prepare - the second half of what the worker's `engine_swap`
    branch does, and the only half that takes any time.
    """
    started = time.perf_counter()
    stream = build_stream(scenario, engines_root=engines_root)
    return stream, round(time.perf_counter() - started, 3)


def run_quality(
    case: QualityCase,
    cooldown: bool = True,
    results_dir: Path = QUALITY_RESULTS_DIR,
    threshold_c: float = DEFAULT_THRESHOLD_C,
    cap_s: float = DEFAULT_CAP_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    engines_root: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    write_clips: bool = True,
    log: Callable[[str], None] = print,
) -> QualityResult:
    """Render one clip at every step count on both routes; write the result."""
    import numpy as np
    import torch
    from PIL import Image

    from compositor import painted_mask
    from detector_worker import frame_to_array
    from render_plan import t_index_for_denoise, t_index_ladder

    from bench.capture_runner import control_frame

    # First, so a machine that cannot be fingerprinted fails before it spends
    # minutes loading engines for a result that could never be written.
    fingerprint = capture_fingerprint()
    log(f"gpu: {fingerprint.gpu_name} | driver {fingerprint.driver_version} | "
        f"{regime_summary(fingerprint.clock_lock)}")

    path = clip_path(case.clip)
    track = load_track(case.clip)
    raw_frames, meta = read_clip(path, case.start_frame, case.frames)
    canvas = case.canvas
    frames = [resize(frame, canvas, canvas) for frame in raw_frames]
    log(f"clip: {case.clip} {meta['width']}x{meta['height']} -> {canvas}x{canvas}, "
        f"{len(frames)} frames from {case.start_frame}")

    plan = case.plan()
    regions = [frame_boxes(track, case.start_frame + index, case, canvas, canvas,
                           meta["width"], meta["height"])
               for index in range(len(frames))]
    indices = list(range(len(frames)))

    models_root = resolve_models_dir() if models_dir is None else Path(models_dir)
    detector = load_identity_detector(models_root, log)
    if detector is None:
        raise SystemExit(
            f"bench: {case.name} scores what a step buys with the open-vocabulary "
            f"detector, and its weights are not cached. There is no eyeball "
            f"fallback - the Gate asks for a quality figure per arm.")
    set_detector_vocabulary(detector, [case.concept, case.reads_back_as],
                            Image.fromarray(frames[0]))

    engines = resolve_engines_dir() if engines_root is None else Path(engines_root)
    specs = case.specs()
    opening = t_index_for_denoise(case.denoise)
    log(f"denoise {case.denoise} -> opening t_index {opening}, held across "
        f"{len(specs)} arms")

    cooldown_record = cooldown_gate(cooldown, threshold_c, cap_s, poll_interval_s, log)
    occupancy_record = occupancy_gate(log)

    # The priming build, thrown away: it pays this process's CUDA start-up and the
    # first read of the model off disk, so every arm's `swap_seconds` below is a
    # swap between two live engines rather than a cold start wearing that name.
    primer, prime_seconds = build_arm(
        arm_scenario(case, specs[0], t_index_ladder(opening, specs[0].steps)), engines)
    free(primer)
    log(f"primed in {prime_seconds:.1f} s (thrown away, so no arm carries it)")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started_utc = utc_now()
    arms: List[QualityArm] = []
    panels: List[List] = []
    panel_labels: List[str] = []
    sources: List = []
    masks: List = []
    control_change = 0.0
    with GpuSampler(sample_interval_s) as sampler:
        for spec in specs:
            label = arm_name(spec)
            ladder = t_index_ladder(opening, spec.steps)
            keying = engine_keying(case, spec, engines)
            scenario = arm_scenario(case, spec, ladder)
            log(f"arm {label}: {spec.route}, UNet batch {keying.unet_batch}, "
                f"engine {'cached' if keying.cached else 'NOT cached - a build'}")
            try:
                stream, swap_seconds = build_arm(scenario, engines)
            except Exception as error:  # a configuration the pipeline refuses
                log(f"arm {label}: did not run - {error}")
                arms.append(_unrun_arm(spec, ladder, keying, str(error)))
                continue
            log(f"arm {label}: engine ready in {swap_seconds:.1f} s")

            compositor, renders = renders_for(case, plan, regions, canvas, canvas,
                                              indices)
            tensors = [capture_tensor(frame, device=stream.device,
                                      dtype=stream.dtype)
                       for frame in frames]
            if not sources:
                # The capture as the frame loop reads it back, the masks the
                # renders paint through, and the drift the round trip costs before
                # anything is styled. All three are the same for every arm - the
                # regions come from a committed track and the plan does not move -
                # so they are measured once and every arm's figures are net of them.
                sources = [frame_to_array(tensor) for tensor in tensors]
                masks = [painted_mask(render.alpha) if render.alpha is not None
                         else np.zeros((canvas, canvas), dtype=bool)
                         for render in renders]
                controls = [control_frame(tensors[index], compositor,
                                          renders[index], canvas)
                            for index in indices]
                control_change = round(statistics.fmean(
                    [mean_abs_diff(sources[index], output, masks[index])
                     for index, output in enumerate(controls)]), 4)
                log(f"control: the capture round trip costs {control_change:.2f}/255")
                # A fresh scheduler and compositor for the timed pass, so this arm
                # walks the same rotation from the same cursor every other one does
                # rather than one the control pass had already advanced.
                compositor, renders = renders_for(case, plan, regions, canvas,
                                                  canvas, indices)

            for index in range(min(case.warmup_frames, len(tensors))):
                render_frame(stream, tensors[index], compositor, renders[index],
                             canvas)
            outputs, per_frame_ms = [], []
            for index in indices:
                output, frame_ms = render_frame(stream, tensors[index], compositor,
                                                renders[index], canvas)
                outputs.append(output)
                per_frame_ms.append(frame_ms)

            hits, retained, confidence = adherence_probe(detector, case, outputs)
            arm = _measured_arm(spec, ladder, keying, swap_seconds, sources, outputs,
                                masks, per_frame_ms, control_change, hits, retained,
                                confidence)
            arms.append(arm)
            log(f"arm {label}: {arm.ms_per_frame:.2f} ms/frame, adherence "
                f"{arm.adherence:.0%} at conf {confidence:.2f}, net change "
                f"{arm.net_change:.1f}/255, flicker {arm.flicker:.2f}, response "
                f"{arm.response:.2f}, background "
                f"{arm.background.identical_frames}/{arm.background.frames}")
            panels.append(outputs)
            panel_labels.append(label)
            free(stream)
    finished_utc = utc_now()

    results_dir = Path(results_dir)
    timestamp = filename_timestamp(finished_utc)
    stem = f"{case.name}-{timestamp}"
    still, clip = "", ""
    if write_clips and panels:
        still, clip = _write_artefacts(sources, panels, results_dir, stem,
                                       meta["fps"], log)

    result = QualityResult(
        case=case,
        clip=ClipRecord(name=case.clip, sha256=sha256_of(path),
                        width=meta["width"], height=meta["height"],
                        fps=meta["fps"], total_frames=meta["total_frames"],
                        start_frame=case.start_frame, frames_used=len(frames)),
        arms=tuple(arms), started_utc=started_utc, finished_utc=finished_utc,
        cooldown=cooldown_record, occupancy=occupancy_record,
        hardware=fingerprint,
        clock_normalization=clock_normalization(
            fingerprint.clock_lock, sampler.samples,
            raw_ms_per_frame=arms[0].ms_per_frame if arms else 0.0),
        comparison_still=still, comparison_clip=clip,
    )
    log(f"panels: source | {' | '.join(panel_labels)}")
    log(f"swap: {swap_summary(arms).statement}")
    recommendation = recommend_route(arms)
    if recommendation is not None:
        log(f"gate: {recommendation.statement}")
    written = write_quality_result(result, results_dir=results_dir,
                                   timestamp=timestamp)
    append_quality_readme_rows(result, results_dir / QUALITY_README_NAME,
                               filename=written.name)
    log(f"{case.name} -> {written.name} ({len(arms)} arms)")
    return result


def _unrun_arm(spec: StepSpec, ladder: Sequence[int], keying,
               error: str) -> QualityArm:
    """An arm the pipeline refused, with the reason it gave and no numbers."""
    return QualityArm(
        steps=spec.steps, use_denoising_batch=spec.use_denoising_batch,
        t_index_list=list(ladder), unet_batch=keying.unet_batch,
        engine_dir=keying.directory, engine_cached=bool(keying.cached),
        keys_new_engine=keying.keys_new_engine, swap_seconds=0.0,
        ms_per_frame=0.0, adherence_hits=0, adherence_frames=0, adherence_conf=0.0,
        retained_hits=0, region_change=0.0, control_change=0.0, flicker=0.0,
        response=0.0, background=background_check([], 0), frames=0, loaded=False,
        error=error)


def _measured_arm(spec: StepSpec, ladder: Sequence[int], keying,
                  swap_seconds: float, sources: Sequence, outputs: Sequence,
                  masks: Sequence, per_frame_ms: Sequence[float],
                  control_change: float, hits: int, retained: int,
                  confidence: float) -> QualityArm:
    import numpy as np

    region_change = statistics.fmean(
        [mean_abs_diff(source, output, mask)
         for source, output, mask in zip(sources, outputs, masks) if mask.any()]
        or [0.0])
    changed = [background_pixels_changed(source, output, mask)
               for source, output, mask in zip(sources, outputs, masks)]
    return QualityArm(
        steps=spec.steps, use_denoising_batch=spec.use_denoising_batch,
        t_index_list=list(ladder), unet_batch=keying.unet_batch,
        engine_dir=keying.directory, engine_cached=bool(keying.cached),
        keys_new_engine=keying.keys_new_engine, swap_seconds=swap_seconds,
        ms_per_frame=round(statistics.fmean(per_frame_ms), 4),
        adherence_hits=hits, adherence_frames=len(outputs),
        adherence_conf=confidence, retained_hits=retained,
        region_change=round(region_change, 4), control_change=control_change,
        flicker=flicker_score(sources, outputs, masks).mean_abs_diff,
        response=response_score(sources, outputs, masks).mean_abs_diff,
        background=background_check(
            changed,
            min(int(np.count_nonzero(~mask)) for mask in masks) if masks else 0),
        frames=len(outputs))


def _write_artefacts(sources: Sequence, panels: Sequence[Sequence],
                     results_dir: Path, stem: str, fps: float,
                     log: Callable[[str], None]) -> Tuple[str, str]:
    """source | one panel per arm, as a still and as a clip."""
    strip = triptych(sources, list(panels), panel_width=COMPARISON_PANEL_WIDTH)
    clip = write_clip(strip, results_dir / f"{stem}-arms.mp4", fps).name
    chosen = len(sources) // 2
    still = write_still(
        triptych(sources[chosen:chosen + 1],
                 [panel[chosen:chosen + 1] for panel in panels],
                 panel_width=None)[0],
        results_dir / f"{stem}-arms.jpg").name
    log(f"artefacts: {still}, {clip}")
    return still, clip
