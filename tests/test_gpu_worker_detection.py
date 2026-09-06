"""The worker's detector against the real one, on a committed clip. GPU tier.

Issue #7's Gate, the two items only a GPU can answer:

- with a plan targeting `person`, tracks appear with stable ids across at least 30
  consecutive frames of `bench/clips/people.mp4`;
- raising `detect_every_n` measurably lowers the amortised detector cost.

The clip is the one issue #17 committed, so this runs on the same pixels every
time. Weights are never fetched - a missing checkpoint is a skip with the command
that would cache it, exactly as the issue #4 run does.

Nothing here asserts an absolute latency, and the milliseconds it prints carry the
SM clock they were measured at. Spec 8.1's 14.32 ms per detect is a figure from a
run at boost; on a laptop sitting at its clock floor the same detector and the same
`bench.detector_runner` code measure eight times that, which is issue #13's whole
point. Both Gate items here are ratios - an id is the same id, half as many detects
cost less - and a ratio survives a clock the absolutes do not.
"""

import time

import pytest

from detection import EMPTY_TRACKS, Tracker, Tracks, amortised_ms, is_detect_frame
from detector_worker import BackgroundDetector, UltralyticsDetector

pytestmark = pytest.mark.gpu

CONCEPT = "person"
FRAMES = 36
STABLE_FRAMES = 30


@pytest.fixture(scope="module")
def models_root():
    from bench.detectors import DETECTORS, PRIMARY_DETECTOR, weights_path
    from bench.paths import resolve_models_dir

    root = resolve_models_dir()
    weights = weights_path(DETECTORS[PRIMARY_DETECTOR], root)
    if not weights.is_file():
        pytest.skip(f"no detector weights at {weights}. Run "
                    f"`python -m bench {PRIMARY_DETECTOR} --allow-download` once.")
    return root


@pytest.fixture(scope="module")
def clip():
    from bench.primitive_runner import read_clip
    from bench.primitives import clip_path

    frames, _ = read_clip(clip_path("people.mp4"), count=FRAMES)
    assert len(frames) >= STABLE_FRAMES
    return frames


@pytest.fixture(scope="module")
def detector(models_root):
    live = UltralyticsDetector(models_root)
    live.open()
    live.set_concepts((CONCEPT,))
    yield live
    live.close()


def clock_note() -> str:
    """The clock every millisecond printed below has to be read against."""
    from bench.fingerprint import read_clock_lock

    lock = read_clock_lock()
    return f"[{lock.state} clocks, SM {lock.current_sm_clock_mhz} MHz]"


def run_cadence(detector, frames, detect_every_n):
    """The frame loop's detection half, synchronously: ids per frame, ms per detect."""
    tracker = Tracker()
    tracks = EMPTY_TRACKS
    per_frame_ids = []
    detect_ms = []
    for index, frame in enumerate(frames):
        if is_detect_frame(index, detect_every_n):
            started = time.perf_counter()
            detections = detector.detect(frame)
            detect_ms.append((time.perf_counter() - started) * 1000.0)
            tracks = Tracks(tracks=tracker.update(
                [d for d in detections if d.concept == CONCEPT], index),
                frame_index=index, detector_ms=detect_ms[-1], ticks=len(detect_ms))
        per_frame_ids.append(tracks.ids)
    return per_frame_ids, detect_ms


def test_a_person_keeps_one_id_across_thirty_consecutive_frames(detector, clip):
    """The Gate's first item. The tracker bridges the two frames in three on which
    the detector never ran, so an id has to hold across ticks *and* between them."""
    per_frame_ids, detect_ms = run_cadence(detector, clip, detect_every_n=3)
    window = per_frame_ids[:STABLE_FRAMES]
    assert all(ids for ids in window), "some frame had no track at all"
    persistent = set(window[0]).intersection(*(set(ids) for ids in window[1:]))
    assert persistent, (
        f"no id survived {STABLE_FRAMES} frames; ids per frame were {window}")
    print(f"\nstable ids {sorted(persistent)} over {STABLE_FRAMES} frames; "
          f"{len(detect_ms)} detects, {sum(detect_ms) / len(detect_ms):.2f} ms each "
          f"{clock_note()}")


def test_the_detector_finds_the_people_the_clip_has(detector, clip):
    detections = detector.detect(clip[0])
    assert [d.concept for d in detections] == [CONCEPT] * len(detections)
    assert len(detections) >= 2, "people.mp4 has five subjects; this found under two"
    assert all(d.box.min_side > 0 for d in detections)


def test_the_frame_the_worker_actually_offers_is_a_cuda_tensor(detector, clip):
    """What the capture thread holds is a half-precision CUDA tensor in 0..1, and
    that - not a numpy frame - is what `offer` is given. The conversion happens on
    the detector's thread; this asserts it finds the same people."""
    import torch

    frame = clip[0]
    tensor = (torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
              .to(device="cuda", dtype=torch.float16) / 255.0)
    from_tensor = detector.detect(tensor)
    from_array = detector.detect(frame)
    assert len(from_tensor) == len(from_array)
    assert all(box.min_side > 0 for box in (d.box for d in from_tensor))


def test_raising_the_cadence_lowers_the_amortised_detector_cost(detector, clip):
    """The Gate's third item, measured rather than divided: the same clip, the same
    detector, twice, and the amortised figure is what spec 7.1 budgets."""
    _, dear = run_cadence(detector, clip, detect_every_n=3)
    _, cheap = run_cadence(detector, clip, detect_every_n=6)

    assert len(cheap) < len(dear)
    dear_ms = amortised_ms(sum(dear) / len(dear), 3)
    cheap_ms = amortised_ms(sum(cheap) / len(cheap), 6)
    assert cheap_ms < dear_ms
    assert sum(cheap) < sum(dear), "half the detects took at least as long in total"
    print(f"\namortised {dear_ms:.2f} ms/frame at every 3rd frame, "
          f"{cheap_ms:.2f} ms/frame at every 6th {clock_note()}")


def test_offering_a_frame_to_the_real_detector_does_not_block_the_caller(models_root,
                                                                        clip):
    """The issue's first trap against the real thing: a ~14 ms detect (and a much
    dearer first one, which loads the weights) must not be paid by the frame loop."""
    from render_plan import plan_from_fields

    plan = plan_from_fields(CONCEPT, "a charcoal drawing").plan
    detection = BackgroundDetector(UltralyticsDetector(models_root))
    detection.follow(plan)
    detection.start()
    try:
        worst_ms = 0.0
        for index, frame in enumerate(clip[:STABLE_FRAMES]):
            if not is_detect_frame(index, 3):
                continue
            started = time.perf_counter()
            detection.offer(frame, index)
            worst_ms = max(worst_ms, (time.perf_counter() - started) * 1000.0)
            time.sleep(0.005)
        assert detection.wait_for_tick(1, timeout=60.0), "the detector never published"
        assert worst_ms < 5.0, f"an offer cost the frame loop {worst_ms:.2f} ms"
        tracks = detection.tracks
        assert tracks.count >= 1
        assert tracks.concepts == (CONCEPT,)
        assert tracks.plan_version == plan.plan_version
        print(f"\nworst offer {worst_ms:.3f} ms; {tracks.ticks} ticks published, "
              f"newest {tracks.count} tracks at {tracks.detector_ms:.2f} ms")
    finally:
        detection.stop()
    assert detection.running is False
