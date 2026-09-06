"""The measuring half for detectors. The only detector module that touches the GPU.

Issue #4. torch and ultralytics are imported inside functions, never at module
scope, so `bench.detectors` and `bench.detector_results` stay importable - and
testable - in the merge gate's GPU-free tier, exactly as `bench.runner` does for
the diffusion half.

Three things about the numbers this produces:

- The detector is timed **with the diffusion engine resident**. A detector
  benchmarked alone says nothing about whether it fits, which is the issue's fourth
  trap; the engine is built and warmed first and kept alive across the timed region.
- One detect is the whole `predict` call - letterbox, forward pass and NMS - because
  that is what a frame loop would pay. ultralytics' own three-way split is recorded
  beside it so the model half can still be read off.
- The vocabulary comparison runs the two vocabularies in **alternating blocks** and
  times the first detect after each change **separately**. Both are forced by what
  the machine does: the SM clock drifts further inside a run than the effect being
  looked for, and `set_classes` makes exactly one detect expensive - averaging that
  detect into thirty ordinary ones would have reported a real 100 ms event as noise.

Downloads (weights, and the evidence photographs) happen here and only here, behind
an explicit flag. The worker enforces offline mode; nothing on the frame path
fetches anything.
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from bench.clocks import clock_normalization, regime_summary
from bench.cooldown import DEFAULT_CAP_S, DEFAULT_POLL_INTERVAL_S, DEFAULT_THRESHOLD_C
from bench.detectors import (
    CONCEPT_PROBES,
    DEFAULT_CADENCE,
    DEFAULT_DIFFUSION_SCENARIO,
    DESKTOP_CONTROL,
    TEXT_ENCODER,
    WEIGHTS_SUBDIR,
    ConceptProbe,
    DetectorConfig,
    FramePathVerdict,
    budget_verdict,
    frame_path_verdict,
    weights_path,
)
from bench.detector_results import (
    DETECTOR_README_NAME,
    ConceptEvidence,
    Detection,
    DetectorMetrics,
    DetectorResult,
    LatencySummary,
    VocabularyChange,
    VramRecord,
    append_detector_readme_row,
    write_detector_result,
)
from bench.fingerprint import capture_fingerprint, read_memory_used_mib, utc_now
from bench.paths import DETECTOR_RESULTS_DIR, resolve_models_dir
from bench.runner import DEFAULT_SAMPLE_INTERVAL_S, GpuSampler, cooldown_gate
from bench.scenarios import SCENARIOS

# Evidence photographs, cached under the shared models root beside the weights.
# Gitignored: the pictures are third-party, and the record identifies them by URL
# and SHA-256 instead.
IMAGE_CACHE_SUBDIR = "bench-images"
USER_AGENT = "ScreenDiffusion-bench (issue #4 detector evaluation)"

# How many alternating vocabulary swaps to time after the first one.
VOCABULARY_SWAPS = 6
# Blocks of detects alternating between the two vocabularies for the frame-path
# comparison. Six, interleaved rather than one arm after the other, because on this
# machine the clock moves further within a run than the effect being looked for.
FRAME_PATH_BLOCKS = 6
# At most this many boxes per concept reach the evidence: enough to show what was
# found, few enough that a result file stays readable.
MAX_DETECTIONS_RECORDED = 10


class DownloadRefused(SystemExit):
    """Something the run needs is not cached, and nobody asked for it to be fetched."""


def _require_file(path: Path, url: str, allow_download: bool, what: str,
                  log: Callable[[str], None] = print) -> Path:
    """`path`, fetching it from `url` first if it is missing and that was permitted."""
    if path.is_file():
        return path
    if not allow_download:
        raise DownloadRefused(
            f"bench: no cached {what} at {path}.\n"
            f"       It would be fetched from {url}\n"
            f"       Pass --allow-download if that is what you want. Nothing on the "
            f"frame path ever downloads anything; this is setup."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"fetching {what} -> {path}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())
    return path


def _cached_photo(probe: ConceptProbe, models_root: Path, allow_download: bool,
                  log: Callable[[str], None]) -> Tuple[Path, str]:
    """One evidence photograph on disk, with the hash that identifies it in the record."""
    path = _require_file(models_root / IMAGE_CACHE_SUBDIR / probe.image_name,
                         probe.image_url, allow_download,
                         f"evidence photograph for `{probe.concept}`", log)
    return path, sha256_of(path)


def sha256_of(path: Path) -> str:
    """The evidence photograph's identity, since the photograph itself is not committed."""
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def capture_desktop():
    """One frame of the real screen, and how it was captured.

    DXcam is what the app uses, so it is tried first; it needs a Desktop Duplication
    context and fails on some sessions (an RDP or headless one, notably), and mss is
    the fallback. Which one produced the frame is recorded, because "a desktop
    capture" is the claim the evidence rests on.
    """
    from PIL import Image

    try:
        import dxcam

        camera = dxcam.create(output_color="RGB")
        frame = camera.grab()
        del camera
        if frame is not None:
            return Image.fromarray(frame), "dxcam"
    except Exception as error:  # a session without Desktop Duplication
        print(f"dxcam unavailable ({error}); falling back to mss")

    import mss

    # `mss.MSS` on current versions, the lowercase alias on older ones.
    with getattr(mss, "MSS", mss.mss)() as capture:
        raw = capture.grab(capture.monitors[1])
    return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX"), "mss"


def compose_desktop(desktop, photo, coverage: float = 0.6):
    """The photograph shown on the desktop, the way a browser or viewer would show it.

    The product restyles what is on the screen, so the evidence has to come off a
    screen. Compositing is what makes that reproducible: a desktop that happens to
    have a dog on it is not a test anyone else can re-run.
    """
    frame = desktop.copy()
    width, height = frame.size
    scale = min(width * coverage / photo.width, height * coverage / photo.height)
    pasted = photo.resize((max(1, int(photo.width * scale)),
                           max(1, int(photo.height * scale))))
    frame.paste(pasted, ((width - pasted.width) // 2, (height - pasted.height) // 2))
    return frame


def _first_photo(images: Dict[str, Tuple[Path, str]]):
    """The photograph the timing frame is built from - the first probe's, whichever
    that is, so every run of a given registry times against the same composition."""
    from PIL import Image

    return Image.open(images[CONCEPT_PROBES[0].image_name][0]).convert("RGB")


def load_detector(config: DetectorConfig, weights: Path, models_root: Path):
    """The ultralytics model for `config`, on the GPU.

    `WEIGHTS_DIR` is redirected at the shared models root before anything can read
    it. Left alone, ultralytics resolves it to `<cwd>/weights` and drops a 338 MB
    CLIP checkpoint into whichever directory the harness was run from - which for
    this repo is the repo. `ultralytics.nn.text_model` binds the value at import
    time, so this has to happen before the first `set_classes`.
    """
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    import ultralytics
    from ultralytics import utils as ultralytics_utils

    ultralytics_utils.WEIGHTS_DIR = Path(models_root) / WEIGHTS_SUBDIR

    loader = getattr(ultralytics, config.loader)
    model = loader(str(weights))
    model.to("cuda")
    return model


def detect(model, image, config: DetectorConfig):
    """One detect, exactly as a frame loop would issue it.

    `imgsz=(n, n)` rather than `n`: ultralytics letterboxes to a rectangle when
    given a scalar, and the issue asks for a 640x640 input.
    """
    return model.predict(image, imgsz=(config.imgsz, config.imgsz), device=0,
                         conf=config.conf, verbose=False)[0]


def time_detects(model, image, config: DetectorConfig, reps: int) -> Tuple[List[float], dict]:
    """`reps` timed detects, and the ultralytics speed split from the last one."""
    import torch

    per_rep_ms: List[float] = []
    result = None
    for _ in range(reps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = detect(model, image, config)
        torch.cuda.synchronize()
        per_rep_ms.append((time.perf_counter() - started) * 1000.0)
    speed = {key: round(float(value), 4) for key, value in (result.speed or {}).items()}
    return per_rep_ms, speed


def set_vocabulary(model, terms: Sequence[str]) -> float:
    """Put `terms` in front of an open-vocabulary detector; return what it cost, in ms.

    Synchronised on both sides: the text encode runs on the GPU, and timing it
    without a synchronise would measure queue submission and report a cold-path cost
    of nearly nothing.
    """
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    model.set_classes(list(terms))
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def measure_vocabulary_change(model, config: DetectorConfig,
                              log: Callable[[str], None] = print) -> Tuple[VocabularyChange, float]:
    """What a vocabulary change costs, once the text encoder is already loaded.

    Returns the record and the median steady-state cost. The first change is timed
    separately and kept apart in the record: it fetches and loads CLIP, which happens
    once per process, and folding it into a mean would overstate the cost of a prompt
    edit by two orders of magnitude.
    """
    import statistics

    first_ms = set_vocabulary(model, config.vocabulary)
    log(f"vocabulary: first change {first_ms:.0f} ms (includes loading {TEXT_ENCODER})")

    swaps: List[float] = []
    for index in range(VOCABULARY_SWAPS):
        terms = config.swap_vocabulary if index % 2 == 0 else config.vocabulary
        swaps.append(set_vocabulary(model, terms))
    set_vocabulary(model, config.vocabulary)  # leave it on the vocabulary under test

    median = statistics.median(swaps)
    record = VocabularyChange(
        supported=True,
        terms=list(config.swap_vocabulary),
        first_change_ms=round(first_ms, 4),
        change_ms=[round(ms, 4) for ms in swaps],
        median_change_ms=round(median, 4),
        text_encoder=TEXT_ENCODER,
        note=("A vocabulary change re-encodes the terms with "
              f"{TEXT_ENCODER} and writes the embeddings into the detection head. "
              "It happens when the user edits the target field - the cold path - and "
              "never inside a frame. The first change also loads the text encoder."),
    )
    log(f"vocabulary: steady-state change {median:.1f} ms over {VOCABULARY_SWAPS} swaps")
    return record, median


def measure_frame_path(model, config: DetectorConfig, frame, change_ms: float,
                       log: Callable[[str], None] = print) -> FramePathVerdict:
    """Time the two vocabularies in alternating blocks, and the first detect after
    every change separately.

    Separately because they are two different costs and only one of them is small.
    Detecting against a changed vocabulary is the same work as before - the block
    comparison shows that - but the *first* detect after `set_classes` is not, because
    ultralytics drops the predictor and the next call rebuilds it. Folding that into
    the block would have averaged a 100+ ms event into thirty ordinary detects and
    reported the whole thing as noise.

    Alternating blocks rather than one arm then the other: under the 120 W limit the
    SM clock moves further inside a run than the effect being looked for, so both
    arms have to be exposed to the same drift. Each arm is pooled over all of its
    detects and the median taken, so one stalled detect cannot carry the verdict.
    """
    arms: Dict[int, List[float]] = {0: [], 1: []}
    blocks: List[float] = []
    firsts: List[float] = []
    reps = max(20, config.reps // FRAME_PATH_BLOCKS)
    for index in range(FRAME_PATH_BLOCKS):
        arm = index % 2
        set_vocabulary(model, config.vocabulary if arm == 0 else config.swap_vocabulary)
        first, _ = time_detects(model, frame, config, 1)
        firsts.extend(first)
        samples, _ = time_detects(model, frame, config, reps)
        arms[arm].extend(samples)
        blocks.append(LatencySummary.from_samples(samples).median_ms)
    set_vocabulary(model, config.vocabulary)
    detect(model, frame, config)  # re-warm, so the caller is not handed a cold predictor

    verdict = frame_path_verdict(
        LatencySummary.from_samples(arms[0]).median_ms,
        LatencySummary.from_samples(arms[1]).median_ms,
        change_ms, passes=blocks,
        first_detect_ms=LatencySummary.from_samples(firsts).median_ms,
    )
    log(f"frame path: {verdict.statement}")
    return verdict


def _all_detections(result) -> List[Detection]:
    """Every box the detector returned, strongest first."""
    boxes = getattr(result, "boxes", None)
    found = [] if boxes is None else [
        Detection(label=result.names[int(class_index)],
                  confidence=round(float(confidence), 4),
                  box_xyxy=[round(float(value), 1) for value in box])
        for class_index, confidence, box in zip(boxes.cls, boxes.conf, boxes.xyxy)
    ]
    found.sort(key=lambda detection: detection.confidence, reverse=True)
    return found


def _detections_from(result, wanted: str) -> Tuple[List[Detection], Optional[float], List[Detection]]:
    """The boxes labelled `wanted`, the best confidence among them, and the rest.

    The rest, because "found nothing" and "found it and called it something else" are
    different answers: YOLOv8n asked for `dog` returns no dog and a cat at 0.79, and a
    record that kept only the matching boxes would report that as a blank.
    """
    everything = _all_detections(result)
    matching = [detection for detection in everything if detection.label == wanted]
    others = [detection for detection in everything if detection.label != wanted]
    top = matching[0].confidence if matching else None
    return matching[:MAX_DETECTIONS_RECORDED], top, others[:3]


def gather_evidence(model, config: DetectorConfig, probes: Sequence[ConceptProbe],
                    images: Dict[str, Tuple[Path, str]], desktop, capture_backend: str,
                    log: Callable[[str], None] = print) -> Tuple[List[ConceptEvidence], ConceptEvidence]:
    """Ask the detector for each concept, on a desktop capture with the photo in it.

    A closed-vocabulary detector cannot be asked for `red mug` at all. It is asked
    for the nearest COCO class where there is one, and where there is none the probe
    is recorded unresolved with the substitution that was not available - which is
    the 80-noun cap, measured rather than asserted.
    """
    from PIL import Image

    frame_label = f"desktop capture ({capture_backend}) with the photo composited in"
    evidence: List[ConceptEvidence] = []
    for probe in probes:
        path, digest = images[probe.image_name]
        queried = probe.concept if config.open_vocabulary else probe.coco_equivalent
        if queried is None:
            evidence.append(ConceptEvidence(
                concept=probe.concept, kind=probe.kind, queried=None, frame=frame_label,
                image_name=probe.image_name, image_source=probe.image_url,
                image_sha256=digest, resolved=False, top_confidence=None, detections=[],
                note="Not expressible in this detector's vocabulary. " + probe.note,
            ))
            log(f"evidence: {probe.concept} - not expressible for {config.name}")
            continue

        frame = compose_desktop(desktop, Image.open(path).convert("RGB"))
        detections, top, others = _detections_from(detect(model, frame, config), queried)
        evidence.append(ConceptEvidence(
            concept=probe.concept, kind=probe.kind, queried=queried, frame=frame_label,
            image_name=probe.image_name, image_source=probe.image_url,
            image_sha256=digest, resolved=bool(detections), top_confidence=top,
            detections=detections, strongest_other=others,
            note=probe.note if queried == probe.concept else
            f"Asked for the nearest COCO class `{queried}` instead. {probe.note}",
        ))
        log(f"evidence: {probe.concept} -> {len(detections)} box(es), top {top}"
            + (f"; strongest other {others[0].label} {others[0].confidence}" if others else ""))

    control_detections = _all_detections(detect(model, desktop, config))
    control = ConceptEvidence(
        concept=DESKTOP_CONTROL, kind="control", queried=", ".join(
            config.vocabulary if config.open_vocabulary else ["the 80 COCO classes"]),
        frame=f"desktop capture ({capture_backend}), nothing composited in",
        image_name=DESKTOP_CONTROL, image_source=f"live screen via {capture_backend}",
        image_sha256="", resolved=bool(control_detections),
        top_confidence=max((d.confidence for d in control_detections), default=None),
        detections=control_detections[:MAX_DETECTIONS_RECORDED],
        note="The screen as it actually was, with nothing composited in. Not a "
             "concept probe: it is the control that separates a detector which "
             "found the composited photo from one that returns boxes regardless.",
    )
    log(f"evidence: raw desktop control -> {len(control_detections)} box(es)")
    return evidence, control


def build_diffusion(scenario_name: Optional[str], log: Callable[[str], None] = print):
    """The diffusion stream that has to be resident while the detector is timed.

    Warmed with two calls before the detector loads, so the engine's own device
    memory is allocated and the residency figure is a real one rather than a lazily
    -deferred one.
    """
    if not scenario_name:
        return None
    from PIL import Image

    from bench.runner import build_stream

    scenario = SCENARIOS[scenario_name]
    log(f"loading diffusion {scenario.name} so the detector is measured beside it")
    stream = build_stream(scenario)
    frame = Image.new("RGB", (scenario.width, scenario.height), (32, 96, 160))
    batch = stream.preprocess_image(frame)
    for _ in range(2):
        stream(image=batch)
    return stream


def run_detector(
    config: DetectorConfig,
    diffusion_scenario: Optional[str] = DEFAULT_DIFFUSION_SCENARIO,
    cadence: int = DEFAULT_CADENCE,
    cooldown: bool = True,
    results_dir: Path = DETECTOR_RESULTS_DIR,
    threshold_c: float = DEFAULT_THRESHOLD_C,
    cap_s: float = DEFAULT_CAP_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    models_dir: Optional[Path] = None,
    allow_download: bool = False,
    log: Callable[[str], None] = print,
) -> DetectorResult:
    """Measure `config` once and write its result. Returns the record that was written."""
    import torch

    # First, so a machine that cannot be fingerprinted fails before it spends
    # minutes loading models for a result that could never be written.
    fingerprint = capture_fingerprint()
    log(f"gpu: {fingerprint.gpu_name} | driver {fingerprint.driver_version} | "
        f"{regime_summary(fingerprint.clock_lock)}")

    models_root = resolve_models_dir() if models_dir is None else Path(models_dir)
    weights = _require_file(weights_path(config, models_root), config.weights_url,
                            allow_download, f"{config.name} weights", log)
    images = {probe.image_name: _cached_photo(probe, models_root, allow_download, log)
              for probe in CONCEPT_PROBES}

    # `stream` is held for the rest of this function on purpose: the engine has to
    # stay resident through the timed region, and dropping the reference would let it
    # be collected and turn the combined-VRAM figure into a detector-alone one.
    baseline_used = read_memory_used_mib()
    stream = build_diffusion(diffusion_scenario, log)
    diffusion_used = read_memory_used_mib() if stream is not None else None

    log(f"loading {config.name} from {weights.name}")
    model = load_detector(config, weights, models_root)

    if config.open_vocabulary:
        vocabulary_change, median_change_ms = measure_vocabulary_change(model, config, log)
    else:
        vocabulary_change = VocabularyChange.unsupported(
            "80 fixed COCO classes and no text encoder: this detector's vocabulary "
            "cannot be changed at all, which is the cap spec 8.1 exists to price."
        )
        median_change_ms = 0.0

    desktop, capture_backend = capture_desktop()
    log(f"captured the desktop via {capture_backend} at {desktop.size[0]}x{desktop.size[1]}")
    # Timed on a frame that contains something. An empty frame gives NMS nothing to
    # do, and the frame loop will not be handed empty frames.
    timing_source = compose_desktop(desktop, _first_photo(images))
    timing_frame = timing_source.resize((config.imgsz, config.imgsz))

    # Cooldown first, then warm up, then time - the order `bench.runner` uses. The
    # gate releases a cold GPU at a low clock, and the warmup is what carries the
    # clock back to where the timed region will find it.
    cooldown_record = cooldown_gate(cooldown, threshold_c, cap_s, poll_interval_s, log)
    for _ in range(config.warmup_reps):
        detect(model, timing_frame, config)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started_utc = utc_now()
    with GpuSampler(sample_interval_s) as sampler:
        per_rep_ms, speed = time_detects(model, timing_frame, config, config.reps)
    finished_utc = utc_now()
    torch_peak_bytes = int(torch.cuda.max_memory_allocated())
    combined_used = read_memory_used_mib()

    latency = LatencySummary.from_samples(per_rep_ms)

    frame_path = None
    if config.open_vocabulary:
        frame_path = measure_frame_path(model, config, timing_frame, median_change_ms, log)

    evidence, control = gather_evidence(model, config, CONCEPT_PROBES, images,
                                        desktop, capture_backend, log)

    run = DetectorMetrics(
        started_utc=started_utc, finished_utc=finished_utc,
        warmup_reps=config.warmup_reps, reps=config.reps, imgsz=config.imgsz,
        per_rep_ms=[round(ms, 4) for ms in per_rep_ms],
        latency=latency,
        detects_per_second=round(1000.0 / latency.mean_ms, 3),
        ultralytics_speed_ms=speed,
        mean_sm_clock_mhz=sampler.mean_sm_clock_mhz,
        max_temperature_c=sampler.max_temperature_c,
        gpu_samples=sampler.samples,
        timing_frame=f"desktop capture ({capture_backend}) with an evidence photo "
                     f"composited in, resized to {config.imgsz}x{config.imgsz}",
    )
    result = DetectorResult(
        detector=config, run=run, cooldown=cooldown_record, hardware=fingerprint,
        vram=VramRecord(diffusion_scenario=diffusion_scenario,
                        baseline_used_mib=baseline_used,
                        diffusion_used_mib=diffusion_used,
                        combined_used_mib=combined_used,
                        torch_peak_bytes=torch_peak_bytes),
        budget=budget_verdict(latency.mean_ms, cadence=cadence),
        vocabulary_change=vocabulary_change, frame_path=frame_path,
        evidence=evidence, desktop_control=control,
        clock_normalization=clock_normalization(fingerprint.clock_lock, sampler.samples,
                                                raw_ms_per_frame=latency.mean_ms),
    )

    results_dir = Path(results_dir)
    path = write_detector_result(result, results_dir=results_dir)
    append_detector_readme_row(result, results_dir / DETECTOR_README_NAME, filename=path.name)
    log(f"{config.name}: {latency.mean_ms:.2f} ms/detect -> {path.name}")
    log(result.budget.statement)
    return result
