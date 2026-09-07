"""The measuring half for rendering primitives. The only primitive module that
touches the GPU.

Issue #5, spec 8.2. torch, cv2 and ultralytics are imported inside functions, never
at module scope, so `bench.primitives` and `bench.primitive_results` stay importable
- and testable - in the merge gate's GPU-free tier, exactly as `bench.runner` does
for the diffusion half.

Four things about the numbers this produces:

- **Both primitives run on the same frames, interleaved frame by frame.** Under the
  120 W limit this laptop's SM clock moves further inside a run than the difference
  being looked for (issue #13), so one arm after the other would compare a boost
  clock against a throttled one. Frame `n` is rendered by A and then by B, and each
  arm keeps its own output sequence in order, so the flicker metric still sees
  consecutive frames.
- **The boxes come from a committed track, not from a live detector.** The issue's
  second trap. The track is generated once by `--write-track` and read back after
  that, so two runs render exactly the same regions.
- **What is timed is the whole primitive**, including the resizes and the composite,
  because the frame loop pays those too. It is not the UNet alone.
- **The denoise strength is measured, not assumed.** Each case is swept over the
  ladder before the timed pass and the strength it needed is selected by a rule the
  record carries. The identity case is judged by the detector rather than by how
  much the frame moved, because a frame can move a great deal and still be a dog.

Nothing here downloads anything. The clips are committed; the detector weights come
from the shared models root and are the ones issue #4 already fetched.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from bench.clocks import clock_normalization, regime_summary
from bench.cooldown import DEFAULT_CAP_S, DEFAULT_POLL_INTERVAL_S, DEFAULT_THRESHOLD_C
from bench.detector_results import LatencySummary
from bench.detectors import DETECTORS, PRIMARY_DETECTOR, weights_path
from bench.fingerprint import capture_fingerprint, utc_now
from bench.flicker import flicker_score
from bench.paths import PRIMITIVE_RESULTS_DIR, resolve_models_dir
from bench.primitive_results import (
    PRIMITIVE_README_NAME,
    ClipRecord,
    IdentityCheck,
    PrimitiveArm,
    PrimitiveResult,
    PrimitiveRunMetrics,
    TrackRecord,
    append_primitive_readme_rows,
    write_primitive_result,
)
from bench.primitives import (
    IDENTITY_HIT_FRACTION,
    MASKED,
    PRIMITIVES,
    TRACK_IOU_THRESHOLD,
    TRACK_SMOOTHING,
    Box,
    CaseConfig,
    DenoisePoint,
    Track,
    clamp_box,
    clip_path,
    denoise_strength,
    load_track,
    one_step_finding,
    region_box,
    required_denoise,
    small_object_summary,
    smooth_track,
    track_path,
)
from bench.results import filename_timestamp
from bench.runner import (
    DEFAULT_SAMPLE_INTERVAL_S,
    GpuSampler,
    build_stream,
    cooldown_gate,
)
from bench.scenarios import SCENARIOS

# The one cached engine every primitive is rendered through. 512x512 batch 1: both
# primitives diffuse a single 512x512 canvas per call, and the only difference
# between them is how many calls a frame costs. Using one engine for both is what
# makes the ms/frame ratio a property of the primitives rather than of two engines.
ENGINE_SCENARIO = "img2img-tensorrt-512x512-b1"

# Panel width of one column of the source | A | B triptych. Downscaled: the clip is
# for judging temporal behaviour, and the full-resolution still beside it is for
# judging detail.
COMPARISON_PANEL_WIDTH = 480
STILL_JPEG_QUALITY = 92

# Confidence the identity probe requires before it will say the subject changed.
# Higher than the detector benchmark's 0.05: this is a verdict, not a recall sweep.
IDENTITY_CONF = 0.25
# And what the track generator requires of a box it is going to render for 48 frames.
TRACK_CONF = 0.25

# Renders before the timed pass, per arm. The engine is already warmed by
# `build_stream`; these warm the resize/composite path and the allocator.
WARMUP_FRAMES = 3

# What the sweep asks about a rendered sequence: how many frames read as the new
# identity, and how many were probed. `(None, None)` for a case that asks no
# identity question, or when the detector weights are not cached.
IdentityProbe = Callable[[Sequence], Tuple[Optional[int], Optional[int]]]


def sha256_of(path: Path) -> str:
    """The clip's identity. Committed bytes, but a re-encode is a different clip."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- the clip ---------------------------------------------------------------------

def read_clip(path: Path, start: int = 0,
              count: Optional[int] = None) -> Tuple[List, dict]:
    """`count` consecutive RGB frames from `path`, starting at `start`.

    Consecutive, because a flicker metric over sampled frames measures the sampling
    interval rather than the render. Returns numpy uint8 HxWx3 arrays and the clip's
    own metadata, which the record carries so a reader knows what was decoded.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"bench: could not open the clip at {path}")
    meta = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "total_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    frames = []
    try:
        for index in range(start + (count if count is not None else meta["total_frames"])):
            ok, frame = capture.read()
            if not ok:
                break
            if index >= start:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise SystemExit(f"bench: {path.name} yielded no frame at offset {start}")
    return frames, meta


def write_clip(frames: Sequence, path: Path, fps: float) -> Path:
    """`frames` as an mp4 beside the record. RGB in, BGR to the encoder."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps or 25.0, (width, height))
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path


def resized_panel(frame, width: int):
    """`frame` scaled to `width`, with both sides even - an odd size breaks mp4v."""
    height = max(2, int(round(frame.shape[0] * width / frame.shape[1])))
    return resize(frame, width - width % 2, height - height % 2)


def triptych(source: Sequence, arms: Sequence[Sequence],
             panel_width: Optional[int] = COMPARISON_PANEL_WIDTH) -> List:
    """source | arm | arm, frame by frame. `panel_width` None keeps full resolution."""
    import numpy as np

    columns = [source, *arms]
    return [np.hstack([column[index] if panel_width is None
                       else resized_panel(column[index], panel_width)
                       for column in columns])
            for index in range(len(source))]


def write_still(frame, path: Path) -> Path:
    """One full-resolution triptych frame. The small-crop quality floor is only
    visible at native resolution, which the downscaled clip throws away."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, STILL_JPEG_QUALITY])
    return path


# --- the primitives ----------------------------------------------------------------

def regions_for(track: Track, index: int, case: CaseConfig,
                width: int, height: int) -> List[Box]:
    """The rendered regions for one frame: the track's boxes, banded and clamped."""
    return [clamp_box(region_box(box, case.region), width, height)
            for box in track.at(index, limit=case.max_objects)]


def painted_mask(regions: Sequence[Box], width: int, height: int):
    """Where a primitive writes. Identical for both: they differ in how the pixels
    are produced, not in where they land."""
    import numpy as np

    mask = np.zeros((height, width), dtype=bool)
    for region in regions:
        mask[region.y0:region.y1, region.x0:region.x1] = True
    return mask


def resize(frame, width: int, height: int):
    """`frame` at a new size, with the interpolation that suits the direction.

    cv2 rather than PIL, and not because either is more correct: what is timed here
    is the whole primitive, resizes included, and a Python-level PIL resample of a
    1280x720 frame costs more than the diffusion call it wraps - which would have
    made this a benchmark of the resampler. `INTER_AREA` shrinking and `INTER_LINEAR`
    growing is the ordinary pairing; the crop primitive is always growing, which is
    its whole problem.
    """
    import cv2

    shrinking = width * height < frame.shape[1] * frame.shape[0]
    return cv2.resize(frame, (width, height),
                      interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)


def _diffuse(stream, frame):
    """One diffusion call on a `stream.width` x `stream.height` RGB array.

    The array is already the engine's canvas size, so `preprocess_image`'s own resize
    is a no-op and the only resampling in the pipeline is the explicit one above.
    """
    import numpy as np
    from PIL import Image

    return np.asarray(stream(image=stream.preprocess_image(Image.fromarray(frame))))


def _composite(stream, frame, regions: Sequence[Box], primitive: str,
               on_canvas: Callable):
    """Both primitives' shared body: resize onto the canvas, `on_canvas`, paste back.

    `on_canvas` is what happens to a frame once it is the engine's canvas size - the
    diffusion call for a real render, the identity for the resize control. One body
    rather than two, so the control provably runs the same resize path instead of a
    copy of it that can drift - which is what makes the blur it measures subtractable.

    Returns the frame and the number of times `on_canvas` was called, which is the
    whole cost difference between the two primitives.
    """
    height, width = frame.shape[:2]
    canvas = (stream.width, stream.height)
    output = frame.copy()
    if not regions:
        return output, 0
    if primitive == MASKED:
        rendered = resize(on_canvas(resize(frame, *canvas)), width, height)
        for region in regions:
            output[region.y0:region.y1, region.x0:region.x1] = (
                rendered[region.y0:region.y1, region.x0:region.x1])
        return output, 1
    for region in regions:
        patch = resize(frame[region.y0:region.y1, region.x0:region.x1], *canvas)
        output[region.y0:region.y1, region.x0:region.x1] = resize(
            on_canvas(patch), region.width, region.height)
    return output, len(regions)


def render(stream, frame, regions: Sequence[Box], primitive: str):
    """Render one frame with one primitive. Returns the frame and the calls it cost.

    Both primitives composite into exactly the same regions. `crop` diffuses each
    region on its own 512x512 canvas - which is what upscales a 45 px region 11x -
    and `masked` diffuses the whole frame once and takes the regions out of it.
    """
    return _composite(stream, frame, regions, primitive,
                      lambda canvas_frame: _diffuse(stream, canvas_frame))


def render_resample_only(stream, frame, regions: Sequence[Box], primitive: str):
    """`render`'s resize path with the diffusion call taken out.

    The control for "did the render change anything". A primitive that squeezes a
    1280x720 frame onto a 512x512 canvas and stretches it back has changed the
    region before it has styled anything, and a visibility criterion that counted
    that blur would pass a strength that does nothing.
    """
    output, _ = _composite(stream, frame, regions, primitive,
                           lambda canvas_frame: canvas_frame)
    return output


def timed_render(stream, frame, regions: Sequence[Box], primitive: str):
    """`render`, bracketed by synchronises. Without them the clock measures queue
    submission and every figure is fiction."""
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    output, calls = render(stream, frame, regions, primitive)
    torch.cuda.synchronize()
    return output, calls, (time.perf_counter() - started) * 1000.0


def mean_abs_diff(source, output, mask) -> float:
    """Mean absolute difference between two frames over `mask`, in 0-255 units."""
    import numpy as np

    if not mask.any():
        return 0.0
    difference = np.abs(output.astype(np.float64)
                        - source.astype(np.float64)).mean(axis=-1)
    return round(float(difference[mask].mean()), 4)


# --- the denoise ladder ---------------------------------------------------------------

def set_denoise(stream, t_index: int) -> Tuple[int, float]:
    """Put one denoise setting in front of the engine; return its timestep and strength.

    A one-step schedule, so `t_index_list` is one rung. Changing the *values* is a
    runtime update - only changing the step *count* rebuilds the engine, which is
    why a ladder of strengths is affordable and a ladder of step counts is not.
    """
    stream.set_t_index_list([t_index])
    inner = stream.stream
    timestep = int(inner.sub_timesteps[0]) if getattr(inner, "sub_timesteps", None) \
        else int(inner.timesteps[t_index])
    return timestep, denoise_strength(float(inner.scheduler.alphas_cumprod[timestep]))


def set_denoise_ladder(stream, t_index: int, steps: int) -> List[int]:
    """One denoise setting over `steps` denoising steps; returns the schedule set.

    `set_denoise` above writes a one-rung list, which is right everywhere this
    harness measures SD-Turbo - it is distilled to one step. A base model that is
    not (issue #38) needs the rest of the ladder, and rendering a four-step arm at
    one step is not a weaker restyle, it is noise. At one step this is exactly what
    `set_denoise` does, so no committed measurement changes meaning.
    """
    from render_plan import t_index_ladder

    ladder = t_index_ladder(t_index, steps)
    stream.set_t_index_list(ladder)
    return ladder


def sweep_denoise(
    stream, case: CaseConfig, frames: Sequence,
    regions_per_frame: Sequence[Sequence[Box]], primitive: str,
    identity_probe: Optional[IdentityProbe] = None,
    log: Callable[[str], None] = print,
) -> List[DenoisePoint]:
    """Render a few frames at every rung of the ladder and record what each did.

    Cheap on purpose: the sweep selects a strength, it does not measure a latency.
    The `outside_change` column is the control - a selective primitive must leave
    the rest of the frame alone, and a rung where it did not is a compositing bug
    rather than a strength that worked.
    """
    control = statistics.fmean([
        mean_abs_diff(frame,
                      render_resample_only(stream, frame, regions_per_frame[index],
                                           primitive),
                      painted_mask(regions_per_frame[index], frame.shape[1],
                                   frame.shape[0]))
        for index, frame in enumerate(frames)])
    log(f"resize control {primitive}: {control:.1f}/255 inside the region before "
        f"anything is diffused")

    points: List[DenoisePoint] = []
    for rung in case.denoise_ladder:
        timestep, strength = set_denoise(stream, rung)
        inside, outside, rendered = [], [], []
        for index, frame in enumerate(frames):
            regions = regions_per_frame[index]
            output, _ = render(stream, frame, regions, primitive)
            mask = painted_mask(regions, frame.shape[1], frame.shape[0])
            inside.append(mean_abs_diff(frame, output, mask))
            outside.append(mean_abs_diff(frame, output, ~mask))
            rendered.append(output)
        hits, probed = identity_probe(rendered) if identity_probe else (None, None)
        point = DenoisePoint(
            t_index=rung, timestep=timestep, strength=strength,
            region_change=round(statistics.fmean(inside), 4),
            outside_change=round(statistics.fmean(outside), 4),
            frames=len(frames), resample_change=round(control, 4),
            identity_hits=hits, identity_frames=probed,
        )
        points.append(point)
        log(f"denoise {primitive} t_index {rung} (timestep {timestep}, strength "
            f"{strength:.2f}): region {point.region_change:.1f} "
            f"({point.net_region_change:.1f} net), outside "
            f"{point.outside_change:.1f}"
            + (f", identity {hits}/{probed}" if probed else ""))
    return points


# --- the detector, for the track and for the identity verdict --------------------------

def load_identity_detector(models_root: Path, log: Callable[[str], None] = print):
    """The open-vocabulary detector, or None when its weights are not cached.

    None rather than a download: issue #4's weights are hundreds of MB and fetching
    them is a decision someone takes with `--allow-download` on that benchmark, not
    a side effect of this one.
    """
    from bench.detector_runner import load_detector

    config = DETECTORS[PRIMARY_DETECTOR]
    weights = weights_path(config, models_root)
    if not weights.is_file():
        log(f"no {PRIMARY_DETECTOR} weights at {weights}; the identity verdict and "
            f"the track generator both need them")
        return None
    return load_detector(config, weights, models_root)


def set_detector_vocabulary(model, terms: Sequence[str], probe) -> None:
    """Change the vocabulary and re-warm, which is the mitigation spec 8.1 requires.

    `YOLOWorld.set_classes` drops the predictor, so the *next* detect costs ~108 ms
    more than a steady one. One throwaway detect here pays it off the frame path -
    and this run has a frame path, even if it is an offline one.
    """
    model.set_classes(list(terms))
    model.predict(probe, imgsz=(640, 640), device=0, conf=IDENTITY_CONF, verbose=False)


def labels_in(model, frame, conf: float = IDENTITY_CONF) -> Dict[str, float]:
    """Every label the detector returned for one frame, with its best confidence."""
    result = model.predict(frame, imgsz=(640, 640), device=0, conf=conf,
                           verbose=False)[0]
    best: Dict[str, float] = {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return best
    for class_index, confidence in zip(boxes.cls, boxes.conf):
        label = result.names[int(class_index)]
        best[label] = max(best.get(label, 0.0), float(confidence))
    return best


def identity_check(model, case: CaseConfig, frames: Sequence) -> IdentityCheck:
    """Ask the detector what the rendered subject now is.

    The Gate's identity question, answered by the same open-vocabulary detector the
    orchestrator uses rather than by how much the pixels moved. A human watching the
    clip is still the confirmation; this is the number that says whether it is worth
    watching.
    """
    from PIL import Image

    became = remained = 0
    for frame in frames:
        labels = labels_in(model, Image.fromarray(frame))
        became += int(case.becomes in labels)
        remained += int(case.target in labels)
    achieved = became >= IDENTITY_HIT_FRACTION * len(frames)
    return IdentityCheck(
        detector=PRIMARY_DETECTOR, asked_for=[case.target, case.becomes],
        frames_probed=len(frames), became=became, remained=remained,
        conf=IDENTITY_CONF, achieved=achieved,
        statement=(
            f"Asked for `{case.target}` and `{case.becomes}` at conf "
            f"{IDENTITY_CONF}, {PRIMARY_DETECTOR} read {became} of {len(frames)} "
            f"rendered frames as `{case.becomes}` and {remained} as still "
            f"`{case.target}`. "
            + ("The identity change reads as achieved." if achieved else
               "The identity change does not read as achieved.")
        ),
    )


def build_track(case: CaseConfig, models_dir: Optional[Path] = None,
                log: Callable[[str], None] = print) -> Path:
    """Detect `case.target` across the frames the case uses and commit the boxes.

    Run once, by hand, and the JSON is committed. Every comparison after that reads
    it, so two runs render the same regions - which is the issue's second trap, and
    the reason the flicker figures of two runs can be compared at all.
    """
    from PIL import Image

    models_root = resolve_models_dir() if models_dir is None else Path(models_dir)
    model = load_identity_detector(models_root, log)
    if model is None:
        raise SystemExit(
            f"bench: --write-track needs the {PRIMARY_DETECTOR} weights under "
            f"{models_root}. Run `python -m bench {PRIMARY_DETECTOR} "
            f"--allow-download` once."
        )
    path = clip_path(case.clip)
    frames, meta = read_clip(path, case.start_frame, case.frames)
    probe = Image.fromarray(frames[0])
    set_detector_vocabulary(model, [case.target], probe)

    detections: List[List[Box]] = []
    for frame in frames:
        result = model.predict(Image.fromarray(frame), imgsz=(640, 640), device=0,
                               conf=TRACK_CONF, verbose=False)[0]
        found = sorted(
            [(float(confidence), Box(*(int(round(float(v))) for v in box)))
             for confidence, box in zip(result.boxes.conf, result.boxes.xyxy)],
            key=lambda item: item[0], reverse=True)
        detections.append([box for _, box in found])

    smoothed = smooth_track(detections)
    track = Track(
        clip=case.clip, target=case.target, detector=PRIMARY_DETECTOR,
        conf=TRACK_CONF, width=meta["width"], height=meta["height"],
        fps=meta["fps"], frame_count=meta["total_frames"], generated_utc=utc_now(),
        boxes={case.start_frame + index: boxes
               for index, boxes in enumerate(smoothed)},
    )
    destination = track_path(case.clip)
    destination.write_text(
        json.dumps({**track.to_dict(),
                    "note": f"Generated by `python -m bench {case.name} "
                            f"--write-track`, then committed. Greedy IoU matching at "
                            f"{TRACK_IOU_THRESHOLD} with {TRACK_SMOOTHING} smoothing, "
                            f"so both primitives render the same regions and neither "
                            f"is charged for detector jitter."},
                   indent=2) + "\n",
        encoding="utf-8")
    counts = [len(boxes) for boxes in smoothed]
    log(f"track: {sum(counts)} boxes over {len(counts)} frames "
        f"({statistics.fmean(counts):.2f} per frame) -> {destination}")
    return destination


# --- the run ------------------------------------------------------------------------

def _arm(case: CaseConfig, primitive: str, points: List[DenoisePoint],
         per_frame_ms: List[float], calls: List[int], sources: Sequence,
         outputs: Sequence, masks: Sequence,
         identity: Optional[IdentityCheck]) -> PrimitiveArm:
    """One primitive's half of the record, assembled from what the pass produced."""
    requirement = required_denoise(points, case.kind)
    latency = LatencySummary.from_samples(per_frame_ms)
    inside = statistics.fmean(
        [mean_abs_diff(source, output, mask)
         for source, output, mask in zip(sources, outputs, masks)])
    outside = statistics.fmean(
        [mean_abs_diff(source, output, ~mask)
         for source, output, mask in zip(sources, outputs, masks)])
    expresses = requirement.met if case.becomes is None else (
        requirement.met and identity is not None and identity.achieved)
    return PrimitiveArm(
        primitive=primitive, spec_option=PRIMITIVES[primitive].spec_option,
        denoise=requirement, latency=latency, ms_per_frame=latency.mean_ms,
        calls_per_frame=round(statistics.fmean(calls), 4), calls_total=sum(calls),
        frames=len(per_frame_ms),
        flicker=flicker_score(sources, outputs, masks),
        region_change=round(inside, 4), outside_change=round(outside, 4),
        expresses=bool(expresses),
        cannot_express=PRIMITIVES[primitive].cannot_express,
        identity=identity,
    )


def _write_comparison_artefacts(
    frames: Sequence, outputs: Dict[str, List], primitives: Sequence[str],
    results_dir: Path, stem: str, fps: float, log: Callable[[str], None],
) -> Tuple[Dict[str, str], str, str]:
    """The files a human judges the comparison by; returns their names.

    One mp4 per arm, the source | A | B triptych they are read against, and one
    full-resolution still of it - the small-crop quality floor is only visible at
    native resolution, which the downscaled clip throws away. The Gate's manual
    verification step has to be pointed at a file, not at a number.
    """
    clip_files = {}
    for primitive in primitives:
        written = write_clip(
            [resized_panel(frame, COMPARISON_PANEL_WIDTH)
             for frame in outputs[primitive]],
            results_dir / f"{stem}-{primitive}.mp4", fps)
        clip_files[primitive] = written.name
    rendered = [outputs[primitive] for primitive in primitives]
    comparison = write_clip(triptych(frames, rendered),
                            results_dir / f"{stem}-triptych.mp4", fps).name
    middle = len(frames) // 2
    still = write_still(
        triptych(frames[middle:middle + 1],
                 [sequence[middle:middle + 1] for sequence in rendered],
                 panel_width=None)[0],
        results_dir / f"{stem}-triptych.jpg").name
    log(f"clips: {comparison}, {still}, {', '.join(clip_files.values())}")
    return clip_files, comparison, still


def run_case(
    case: CaseConfig,
    cooldown: bool = True,
    results_dir: Path = PRIMITIVE_RESULTS_DIR,
    threshold_c: float = DEFAULT_THRESHOLD_C,
    cap_s: float = DEFAULT_CAP_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    engines_root: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    primitives: Sequence[str] = tuple(PRIMITIVES),
    write_clips: bool = True,
    log: Callable[[str], None] = print,
) -> PrimitiveResult:
    """Measure both primitives on one case and write the result. Returns the record."""
    import torch

    # First, so a machine that cannot be fingerprinted fails before it spends
    # minutes loading an engine for a result that could never be written.
    fingerprint = capture_fingerprint()
    log(f"gpu: {fingerprint.gpu_name} | driver {fingerprint.driver_version} | "
        f"{regime_summary(fingerprint.clock_lock)}")

    path = clip_path(case.clip)
    track = load_track(case.clip)
    frames, meta = read_clip(path, case.start_frame, case.frames)
    regions_per_frame = [regions_for(track, case.start_frame + index, case,
                                     meta["width"], meta["height"])
                         for index in range(len(frames))]
    masks = [painted_mask(regions, meta["width"], meta["height"])
             for regions in regions_per_frame]
    log(f"clip: {case.clip} {meta['width']}x{meta['height']} @ {meta['fps']:.1f} fps, "
        f"{len(frames)} frames from {case.start_frame}, "
        f"{statistics.fmean([len(r) for r in regions_per_frame]):.2f} regions/frame")

    scenario = SCENARIOS[ENGINE_SCENARIO]
    log(f"building {scenario.name} - one engine for both primitives")
    stream = build_stream(scenario.replace(prompt=case.prompt),
                          engines_root=engines_root)

    models_root = resolve_models_dir() if models_dir is None else Path(models_dir)
    detector = load_identity_detector(models_root, log)
    if detector is not None:
        from PIL import Image

        # Both cases put a vocabulary in front of it, even the one that never asks a
        # question, so the CLIP text encoder is resident in both and the two cases'
        # VRAM and timing are comparable.
        vocabulary = [case.target] + ([case.becomes] if case.becomes else [])
        set_detector_vocabulary(detector, vocabulary, Image.fromarray(frames[0]))

    def identity_probe(rendered: Sequence) -> Tuple[Optional[int], Optional[int]]:
        if detector is None or not case.becomes:
            return None, None
        check = identity_check(detector, case, rendered)
        return check.became, check.frames_probed

    sweep_frames = frames[:case.sweep_frames]
    sweep_regions = regions_per_frame[:case.sweep_frames]
    points: Dict[str, List[DenoisePoint]] = {}
    for primitive in primitives:
        points[primitive] = sweep_denoise(stream, case, sweep_frames, sweep_regions,
                                          primitive, identity_probe, log)
    chosen = {primitive: required_denoise(points[primitive], case.kind).t_index
              for primitive in primitives}
    for primitive in primitives:
        log(f"denoise chosen for {primitive}: t_index {chosen[primitive]}")

    # Cooldown, then warm up, then time - the order `bench.runner` uses. The sweep
    # above is real work on the die, so the gate belongs after it.
    cooldown_record = cooldown_gate(cooldown, threshold_c, cap_s, poll_interval_s, log)
    for primitive in primitives:
        set_denoise(stream, chosen[primitive])
        for index in range(min(WARMUP_FRAMES, len(frames))):
            render(stream, frames[index], regions_per_frame[index], primitive)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started_utc = utc_now()
    outputs: Dict[str, List] = {primitive: [] for primitive in primitives}
    per_frame_ms: Dict[str, List[float]] = {primitive: [] for primitive in primitives}
    calls: Dict[str, List[int]] = {primitive: [] for primitive in primitives}
    with GpuSampler(sample_interval_s) as sampler:
        for index, frame in enumerate(frames):
            # Interleaved *per frame*, so the clock drift lands on both arms. The
            # denoise setting is changed outside the timed bracket; it is a runtime
            # update to a one-step schedule, not an engine swap.
            for primitive in primitives:
                set_denoise(stream, chosen[primitive])
                output, count, ms = timed_render(stream, frame,
                                                 regions_per_frame[index], primitive)
                outputs[primitive].append(output)
                per_frame_ms[primitive].append(ms)
                calls[primitive].append(count)
    finished_utc = utc_now()
    peak_vram_bytes = int(torch.cuda.max_memory_allocated())

    arms: List[PrimitiveArm] = []
    for primitive in primitives:
        identity = None
        if detector is not None and case.becomes:
            identity = identity_check(detector, case, outputs[primitive])
            log(f"identity {primitive}: {identity.statement}")
        arm = _arm(case, primitive, points[primitive], per_frame_ms[primitive],
                   calls[primitive], frames, outputs[primitive], masks, identity)
        arms.append(arm)
        log(f"{primitive}: {arm.ms_per_frame:.2f} ms/frame, flicker "
            f"{arm.flicker.mean_abs_diff}, expresses {arm.expresses}")

    results_dir = Path(results_dir)
    # The same spelling the record's own filename uses, so the clips written beside
    # it carry the same stem.
    timestamp = filename_timestamp(utc_now())
    stem = f"{case.name}-{timestamp}"
    clip_files, comparison, still = {}, "", ""
    if write_clips:
        clip_files, comparison, still = _write_comparison_artefacts(
            frames, outputs, primitives, results_dir, stem, meta["fps"], log)
    arms = [dataclasses.replace(
        arm, clip_file=clip_files.get(arm.primitive, ""),
        # One clock trace, two timings: each arm is normalised against its own raw
        # figure, so the estimate beside it is an estimate of *it*.
        clock_normalization=clock_normalization(fingerprint.clock_lock,
                                                sampler.samples,
                                                raw_ms_per_frame=arm.ms_per_frame))
        for arm in arms]

    rendered_regions = [region for regions in regions_per_frame for region in regions]
    # Achieved if *either* primitive managed it: the finding is about one-step
    # SD-Turbo, not about one primitive's compositing.
    achieved = case.becomes is None or any(arm.expresses for arm in arms)
    result = PrimitiveResult(
        case=case,
        clip=ClipRecord(name=case.clip, sha256=sha256_of(path), width=meta["width"],
                        height=meta["height"], fps=meta["fps"],
                        total_frames=meta["total_frames"],
                        start_frame=case.start_frame, frames_used=len(frames)),
        track=TrackRecord(
            detector=track.detector, target=track.target, conf=track.conf,
            region=case.region, max_objects=case.max_objects,
            objects_per_frame=round(
                statistics.fmean([len(r) for r in regions_per_frame]), 4),
            regions_rendered=len(rendered_regions),
            small_objects=small_object_summary(rendered_regions)),
        arms=arms,
        run=PrimitiveRunMetrics(
            started_utc=started_utc, finished_utc=finished_utc,
            warmup_reps=WARMUP_FRAMES, engine_scenario=ENGINE_SCENARIO,
            detector_resident=None if detector is None else PRIMARY_DETECTOR,
            mean_sm_clock_mhz=sampler.mean_sm_clock_mhz,
            max_temperature_c=sampler.max_temperature_c,
            peak_vram_bytes=peak_vram_bytes, gpu_samples=sampler.samples),
        cooldown=cooldown_record, hardware=fingerprint,
        comparison_clip=comparison, comparison_still=still,
        one_step_finding=one_step_finding(
            case, arms[0].denoise, achieved) if case.becomes else None,
    )

    written = write_primitive_result(result, results_dir=results_dir,
                                     timestamp=timestamp)
    append_primitive_readme_rows(result, results_dir / PRIMITIVE_README_NAME,
                                 filename=written.name)
    log(f"{case.name} -> {written.name}")
    return result
