"""The measuring half of the style-LoRA comparison. Touches the GPU; imports late.

Issue #38 steps 3-5. One run renders the same frames of one committed clip through
the same base model at the same denoise, once with no LoRA fused and once per style
LoRA, and records for each what it cost, what it changed, and - when it refused to
load at all - why.

Three things about the arithmetic.

- **The control is the round trip, not zero.** The frames are resized to the canvas
  once, before anything is timed, and the source every arm is compared against is
  what the capture's own uint8 -> float -> uint8 conversion reads back. That
  difference is measured, not assumed, and subtracted from every arm's change - the
  same rule spec 8.2 applies for the same reason.
- **The LoRA's own effect is a second measurement.** A plain SD 1.5 render already
  clears the visible-change threshold, so "did the LoRA do anything" has to be asked
  against the base *render* rather than against the source.
- **A LoRA that will not load is a result.** The wrapper raises with the reason;
  the arm records it and carries on to the next one, because "which formats fail"
  is exactly what issue #38's second trap asks to be recorded.

Every arm is a fresh `StreamDiffusionWrapper`: a fused LoRA cannot be unfused, so
the arms cannot share one.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from bench.clocks import clock_normalization, regime_summary
from bench.cooldown import DEFAULT_CAP_S, DEFAULT_POLL_INTERVAL_S, DEFAULT_THRESHOLD_C
from bench.fingerprint import capture_fingerprint, utc_now
from bench.flicker import flicker_score
from bench.models import STYLE_LORAS, lora_path, with_style
from bench.paths import STYLE_RESULTS_DIR, resolve_models_dir
from bench.primitive_results import ClipRecord
from bench.primitives import clip_path
from bench.primitive_runner import (
    mean_abs_diff,
    read_clip,
    resize,
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
from bench.selective_runner import capture_tensor
from bench.steps import step_arm
from bench.styles import (
    BASE_ARM,
    STYLE_README_NAME,
    StyleArm,
    StyleCase,
    StyleResult,
    append_style_readme_rows,
    style_verdict,
    write_style_result,
)


def arm_scenario(case: StyleCase, style: Optional[str]):
    """The base arm, or the base arm with one style LoRA fused into it."""
    scenario = SCENARIOS[case.base_scenario]
    if style is not None:
        scenario = with_style(scenario, style)
    return step_arm(scenario, case.steps)


def render_arm(stream, tensors: Sequence, ladder: Sequence[int]) -> Tuple[List, float]:
    """Every frame through one arm, full frame, and the mean ms it cost.

    Full frame rather than through the selective path on purpose: what is being
    judged is whether the *style* reached the pixels, and a masked lower half of a
    person is a small and awkward window on that question. The selective path's own
    cost is measured elsewhere and does not change with the LoRA.

    The whole ladder goes in front of the engine, not `primitive_runner.set_denoise`'s
    single rung: that helper is right everywhere else in this harness because
    everywhere else is SD-Turbo at one step, and writing it here would quietly render
    a four-step arm at one - which for SD 1.5 is not a weaker restyle, it is noise.
    """
    import numpy as np
    import torch

    from detector_worker import frame_to_array

    stream.set_t_index_list(list(ladder))
    outputs: List = []
    per_frame_ms: List[float] = []
    for tensor in tensors:
        torch.cuda.synchronize()
        started = time.perf_counter()
        rendered = stream.img2img(tensor, output_type="pt")
        torch.cuda.synchronize()
        per_frame_ms.append((time.perf_counter() - started) * 1000.0)
        outputs.append(np.asarray(frame_to_array(rendered)))
    return outputs, statistics.fmean(per_frame_ms)


def free(stream) -> None:
    """Let go of one arm before the next one is built.

    A fused LoRA cannot be unfused, so every arm is its own model; three of them
    resident at once is a needless several gigabytes.
    """
    import gc

    import torch

    del stream
    gc.collect()
    torch.cuda.empty_cache()


def run_style(
    case: StyleCase,
    cooldown: bool = True,
    results_dir: Path = STYLE_RESULTS_DIR,
    threshold_c: float = DEFAULT_THRESHOLD_C,
    cap_s: float = DEFAULT_CAP_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    engines_root: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    write_clips: bool = True,
    log: Callable[[str], None] = print,
) -> StyleResult:
    """Render one clip through the base model and each style LoRA; write the result."""
    from detector_worker import frame_to_array
    from render_plan import t_index_for_denoise, t_index_ladder

    fingerprint = capture_fingerprint()
    log(f"gpu: {fingerprint.gpu_name} | driver {fingerprint.driver_version} | "
        f"{regime_summary(fingerprint.clock_lock)}")

    path = clip_path(case.clip)
    raw_frames, meta = read_clip(path, case.start_frame, case.frames)
    frames = [resize(frame, case.canvas, case.canvas) for frame in raw_frames]
    ladder = t_index_ladder(t_index_for_denoise(case.denoise), case.steps)
    log(f"clip: {case.clip} -> {case.canvas}x{case.canvas}, {len(frames)} frames; "
        f"denoise {case.denoise} -> t_index list {ladder}")

    models_root = resolve_models_dir() if models_dir is None else Path(models_dir)
    cooldown_record = cooldown_gate(cooldown, threshold_c, cap_s, poll_interval_s, log)
    occupancy_record = occupancy_gate(log)

    arms: List[StyleArm] = []
    renders: List[List] = []
    labels: List[str] = []
    sources: List = []
    control_change = 0.0
    started_utc = utc_now()
    with GpuSampler(sample_interval_s) as sampler:
        for style in (None, *case.styles):
            label = BASE_ARM if style is None else style
            scenario = arm_scenario(case, style)
            log(f"arm {label}: {scenario.name}")
            try:
                stream = build_stream(scenario.replace(prompt=case.prompt),
                                      engines_root=engines_root)
            except Exception as error:  # a LoRA this diffusers version refuses
                log(f"arm {label}: did not load - {error}")
                arms.append(_unloaded_arm(label, models_root, str(error),
                                          len(frames)))
                continue
            tensors = [capture_tensor(frame, device=stream.device,
                                      dtype=stream.dtype)
                       for frame in frames]
            if not sources:
                # The capture as the frame loop reads it back, and how far that
                # already is from the decoded frame: the control every arm's
                # change figure is net of.
                sources = [frame_to_array(tensor) for tensor in tensors]
                control_change = statistics.fmean(
                    [mean_abs_diff(frame, source, _whole(frame))
                     for frame, source in zip(frames, sources)])
                log(f"control: the capture round trip costs "
                    f"{control_change:.2f}/255")
            for index in range(min(case.warmup_frames, len(tensors))):
                stream.img2img(tensors[index])
            outputs, ms = render_arm(stream, tensors, ladder)
            renders.append(outputs)
            labels.append(label)
            arms.append(_measured_arm(label, models_root, sources, outputs, ms,
                                      control_change, renders[0]))
            log(f"arm {label}: {ms:.2f} ms/frame, "
                f"{arms[-1].net_change:.2f}/255 net of control, "
                f"{arms[-1].change_vs_base:.2f}/255 against the base arm")
            free(stream)
    finished_utc = utc_now()

    results_dir = Path(results_dir)
    timestamp = filename_timestamp(finished_utc)
    stem = f"{case.name}-{timestamp}"
    still, clip = "", ""
    if write_clips and renders:
        still, clip = _write_artefacts(sources, renders, results_dir, stem,
                                       meta["fps"], log)

    result = StyleResult(
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
    for arm in arms:
        verdict = style_verdict(arm)
        if verdict is not None:
            log(f"gate: {'pass' if verdict.passed else 'FAIL'} - "
                f"{verdict.statement}")
    written = write_style_result(result, results_dir=results_dir,
                                 timestamp=timestamp)
    append_style_readme_rows(result, results_dir / STYLE_README_NAME,
                             filename=written.name)
    log(f"{case.name} -> {written.name} (arms: {', '.join(labels)})")
    return result


def _whole(frame):
    """Every pixel. A style LoRA is judged over the frame, not over a region."""
    import numpy as np

    return np.ones(frame.shape[:2], dtype=bool)


def _style_fields(style: str, models_root: Path) -> Tuple[str, float, str, str]:
    """Filename, scale, source and format for one arm, or the base arm's blanks."""
    if style == BASE_ARM:
        return "", 0.0, "", "no LoRA"
    lora = STYLE_LORAS[style]
    return (str(lora_path(lora.filename, models_root)), lora.scale, lora.source,
            lora.format)


def _unloaded_arm(style: str, models_root: Path, error: str,
                  frames: int) -> StyleArm:
    filename, scale, source, fmt = _style_fields(style, models_root)
    return StyleArm(style=style, filename=filename, scale=scale, source=source,
                    format=fmt, loaded=False, error=error, ms_per_frame=0.0,
                    change_vs_source=0.0, control_change=0.0, change_vs_base=0.0,
                    flicker=0.0, frames=frames)


def _measured_arm(style: str, models_root: Path, sources: Sequence,
                  outputs: Sequence, ms: float, control_change: float,
                  base_outputs: Sequence) -> StyleArm:
    filename, scale, source, fmt = _style_fields(style, models_root)
    masks = [_whole(frame) for frame in sources]
    change_vs_source = statistics.fmean(
        [mean_abs_diff(one, other, mask)
         for one, other, mask in zip(sources, outputs, masks)])
    change_vs_base = statistics.fmean(
        [mean_abs_diff(one, other, mask)
         for one, other, mask in zip(base_outputs, outputs, masks)])
    return StyleArm(
        style=style, filename=filename, scale=scale, source=source, format=fmt,
        loaded=True, error=None, ms_per_frame=round(ms, 4),
        change_vs_source=round(change_vs_source, 4),
        control_change=round(control_change, 4),
        change_vs_base=round(change_vs_base, 4),
        flicker=flicker_score(sources, outputs, masks).mean_abs_diff,
        frames=len(outputs),
    )


def _write_artefacts(sources: Sequence, renders: Sequence[Sequence],
                     results_dir: Path, stem: str, fps: float,
                     log: Callable[[str], None]) -> Tuple[str, str]:
    """source | base | one panel per style, as a still and as a clip."""
    panels = triptych(sources, list(renders))
    clip = write_clip(panels, results_dir / f"{stem}-styles.mp4", fps).name
    chosen = len(sources) // 2
    still = write_still(
        triptych(sources[chosen:chosen + 1],
                 [render[chosen:chosen + 1] for render in renders],
                 panel_width=None)[0],
        results_dir / f"{stem}-styles.jpg").name
    log(f"artefacts: {still}, {clip}")
    return still, clip
