"""The measuring half of the classifier-free-guidance sweep. Touches the GPU. Issue #45.

What runs here is the shipped selective path with a committed box track in place of
the live detector: `region_scheduler`'s own scheduler at K=1,
`device_compositor.DeviceCompositor`, and the plan `render_plan.plan_from_fields`
produces from the case's two fields. Only the engine's guidance settings move
between arms.

Five things about the numbers.

- **The boxes come from a committed track**, so every arm renders exactly the same
  region of exactly the same frame and the difference between two rows is the
  guidance. Issue #5's second trap, applied a sixth time.
- **The denoise is one number for the whole sweep.** CFG interacts with it, so an
  arm that moved both would be measuring two changes at once - the issue's fourth
  trap - and `set_denoise_ladder` puts the same rung in front of every arm.
- **The adherence probe is §8.2's identity probe**, asked of the *composited*
  frame: the detector is given the whole output and asked whether the thing inside
  the region now reads back as what the prompt asked for. Composited rather than
  the bare canvas, because that is what a viewer sees and what the bit-identity
  criterion is about.
- **The drift control is the capture's own round trip**, measured by running the
  identical path with the diffusion call taken out. The canvas is the capture here
  so there is no resize to subtract, but the uint8 -> float -> uint8 conversion is
  real and it is measured rather than assumed.
- **An arm that the pipeline refuses is a result.** `initialize` and `full` build a
  prompt embedding that only exists above guidance 1.0; a configuration that raises
  is recorded with its reason and the sweep carries on, exactly as issue #38's LoCon
  arm is.

Every arm is a fresh `StreamDiffusionWrapper`: `cfg_type` is a constructor argument
and the batch it derives from it is fixed at construction, so the arms cannot share
one. The guidance *scale* and `delta` are `prepare()` arguments, which is why they
are cheap to sweep and the cfg type is not.
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
from bench.flicker import flicker_score
from bench.guidance import (
    ADHERENCE_CONF,
    GUIDANCE_README_NAME,
    ArmSpec,
    GuidanceArm,
    GuidanceCase,
    GuidanceResult,
    append_guidance_readme_rows,
    arm_name,
    engine_keying,
    recommend_guidance,
    showcase_specs,
    uses_delta,
    write_guidance_result,
)
from bench.paths import GUIDANCE_RESULTS_DIR, resolve_engines_dir, resolve_models_dir
from bench.primitive_results import ClipRecord
from bench.primitives import clip_path, load_track
from bench.primitive_runner import (
    COMPARISON_PANEL_WIDTH,
    labels_in,
    load_identity_detector,
    mean_abs_diff,
    read_clip,
    resize,
    set_denoise_ladder,
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
from bench.steps import step_arm


def arm_scenario(case: GuidanceCase, spec: ArmSpec):
    """The case's base cell at its step count, with one arm's guidance settings.

    Renamed after the arm, because a scenario is serialised whole into a record and
    two arms that differ only in a field nobody reads back would be two rows saying
    the same thing.
    """
    scenario = step_arm(SCENARIOS[case.base_scenario], case.steps)
    return scenario.replace(name=f"{scenario.name}-{arm_name(spec)}",
                            prompt=case.prompt, cfg_type=spec.cfg_type,
                            guidance_scale=spec.guidance_scale, delta=spec.delta)


def adherence_probe(detector, case, frames: Sequence) -> Tuple[int, int, float]:
    """How many rendered frames read back as the prompt's concept, and how strongly.

    Two numbers rather than one because a fraction over 48 frames saturates: an arm
    that lands the prompt on every frame and one that barely lands it on every frame
    both score 100%, and the detector's own confidence separates them.

    `case` is any case carrying `concept` and `reads_back_as` - `GuidanceCase` here
    and `QualityCase` in `bench.quality_runner`, which borrows this rather than
    spelling it a second time, because spec 8.11 and 8.12 share one baseline and a
    probe measured two ways is two baselines.
    """
    from PIL import Image

    hits, retained, confidences = 0, 0, []
    for frame in frames:
        labels = labels_in(detector, Image.fromarray(frame), conf=ADHERENCE_CONF)
        confidence = labels.get(case.reads_back_as, 0.0)
        hits += int(case.reads_back_as in labels)
        retained += int(case.concept in labels)
        confidences.append(confidence)
    return hits, retained, round(statistics.fmean(confidences or [0.0]), 4)


def render_frame(stream, tensor, compositor, render, canvas: int):
    """One frame of the shipped path, and the milliseconds it cost.

    The same three stages `bench.capture_runner` times, summed: what this sweep
    asks of the clock is only whether one arm costs more than another, and the
    per-stage split is that module's question rather than this one's.
    """
    import torch

    from bench.capture_runner import canvas_for

    torch.cuda.synchronize()
    started = time.perf_counter()
    frame_canvas = canvas_for(tensor, render, canvas)
    rendered = stream.img2img(frame_canvas, output_type="pt")
    output = compositor.blend_device(tensor, rendered, render.alpha, render.crop)[-1]
    torch.cuda.synchronize()
    return output, (time.perf_counter() - started) * 1000.0


def free(stream) -> None:
    """Let go of one arm before the next one is built.

    `cfg_type` is fixed at construction and so is the UNet batch derived from it,
    so every arm is its own model; several of them resident at once is a needless
    several gigabytes.
    """
    import gc

    import torch

    del stream
    gc.collect()
    torch.cuda.empty_cache()


def run_guidance(
    case: GuidanceCase,
    cooldown: bool = True,
    results_dir: Path = GUIDANCE_RESULTS_DIR,
    threshold_c: float = DEFAULT_THRESHOLD_C,
    cap_s: float = DEFAULT_CAP_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    engines_root: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    write_clips: bool = True,
    log: Callable[[str], None] = print,
) -> GuidanceResult:
    """Render one clip under every cfg arm, score the adherence, write the result."""
    import numpy as np
    import torch
    from PIL import Image

    from compositor import painted_mask
    from detector_worker import frame_to_array
    from render_plan import t_index_for_denoise

    from bench.capture_runner import control_frame

    # First, so a machine that cannot be fingerprinted fails before it spends
    # minutes loading models for a result that could never be written.
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
            f"bench: {case.name} scores adherence with the open-vocabulary "
            f"detector, and its weights are not cached. There is no eyeball "
            f"fallback - the Gate asks for a number per arm.")
    set_detector_vocabulary(detector, [case.concept, case.reads_back_as],
                            Image.fromarray(frames[0]))

    engines = resolve_engines_dir() if engines_root is None else Path(engines_root)
    specs = case.specs()
    showcase = showcase_specs(specs)
    ladder_rung = t_index_for_denoise(case.denoise)
    log(f"denoise {case.denoise} -> t_index {ladder_rung} over {case.steps} step(s), "
        f"held fixed across {len(specs)} arms")

    cooldown_record = cooldown_gate(cooldown, threshold_c, cap_s, poll_interval_s, log)
    occupancy_record = occupancy_gate(log)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started_utc = utc_now()
    arms: List[GuidanceArm] = []
    panels: List[List] = []
    panel_labels: List[str] = []
    sources: List = []
    masks: List = []
    control_change = 0.0
    with GpuSampler(sample_interval_s) as sampler:
        for spec in specs:
            label = arm_name(spec)
            keying = engine_keying(case, spec, engines)
            scenario = arm_scenario(case, spec)
            log(f"arm {label}: {scenario.name} | UNet batch {keying.unet_batch}"
                f"{' (a build)' if keying.keys_new_engine else ''}")
            try:
                stream = build_stream(scenario, engines_root=engines)
                set_denoise_ladder(stream, ladder_rung, case.steps)
            except Exception as error:  # a configuration the pipeline refuses
                log(f"arm {label}: did not run - {error}")
                arms.append(_unrun_arm(spec, keying, str(error)))
                continue

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
                # so they are measured once, on the first arm that got as far as a
                # stream, and every arm's figures are net of them.
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
                # walks the same rotation from the same cursor every other one
                # does rather than one the control pass had already advanced.
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
            arm = _measured_arm(spec, keying, sources, outputs, masks, per_frame_ms,
                                control_change, hits, retained, confidence)
            arms.append(arm)
            log(f"arm {label}: {arm.ms_per_frame:.2f} ms/frame, adherence "
                f"{arm.adherence:.0%} at conf {confidence:.2f}, drift "
                f"{arm.net_change:.1f}/255, background "
                f"{arm.background.identical_frames}/{arm.background.frames}")
            if spec in showcase:
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

    result = GuidanceResult(
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
    log(f"gate: {recommend_guidance(arms).statement}")
    written = write_guidance_result(result, results_dir=results_dir,
                                    timestamp=timestamp)
    append_guidance_readme_rows(result, results_dir / GUIDANCE_README_NAME,
                                filename=written.name)
    log(f"{case.name} -> {written.name} ({len(arms)} arms)")
    return result


def _unrun_arm(spec: ArmSpec, keying, error: str) -> GuidanceArm:
    """An arm the pipeline refused, with the reason it gave and no numbers."""
    return GuidanceArm(
        cfg_type=spec.cfg_type, guidance_scale=spec.guidance_scale,
        delta=spec.delta, delta_applies=uses_delta(spec.cfg_type),
        unet_batch=keying.unet_batch, engine_dir=keying.directory,
        keys_new_engine=keying.keys_new_engine, engine_cached=keying.cached,
        ms_per_frame=0.0, adherence_hits=0, adherence_frames=0,
        adherence_conf=0.0, retained_hits=0, region_change=0.0,
        control_change=0.0, flicker=0.0,
        background=background_check([], 0), frames=0, loaded=False, error=error)


def _measured_arm(spec: ArmSpec, keying, sources: Sequence, outputs: Sequence,
                  masks: Sequence, per_frame_ms: Sequence[float],
                  control_change: float, hits: int, retained: int,
                  confidence: float) -> GuidanceArm:
    import numpy as np

    region_change = statistics.fmean(
        [mean_abs_diff(source, output, mask)
         for source, output, mask in zip(sources, outputs, masks) if mask.any()]
        or [0.0])
    changed = [background_pixels_changed(source, output, mask)
               for source, output, mask in zip(sources, outputs, masks)]
    return GuidanceArm(
        cfg_type=spec.cfg_type, guidance_scale=spec.guidance_scale,
        delta=spec.delta, delta_applies=uses_delta(spec.cfg_type),
        unet_batch=keying.unet_batch, engine_dir=keying.directory,
        keys_new_engine=keying.keys_new_engine, engine_cached=keying.cached,
        ms_per_frame=round(statistics.fmean(per_frame_ms), 4),
        adherence_hits=hits, adherence_frames=len(outputs),
        adherence_conf=confidence, retained_hits=retained,
        region_change=round(region_change, 4), control_change=control_change,
        flicker=flicker_score(sources, outputs, masks).mean_abs_diff,
        background=background_check(
            changed,
            min(int(np.count_nonzero(~mask)) for mask in masks) if masks else 0),
        frames=len(outputs))


def _write_artefacts(sources: Sequence, panels: Sequence[Sequence],
                     results_dir: Path, stem: str, fps: float,
                     log: Callable[[str], None]) -> Tuple[str, str]:
    """source | control | one panel per cfg type, as a still and as a clip."""
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
