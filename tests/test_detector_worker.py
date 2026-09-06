"""The worker's detector, driven by a fake one. Issue #7, GPU-free tier.

`BackgroundDetector` is the whole of the issue's first, second and fourth steps -
load, cadence, publish - and none of it needs a GPU to be true. The seam is the
detector object itself: anything with `open` / `set_concepts` / `detect` / `close`
will do, so the state machine is tested here against a fake and measured against
ultralytics in the GPU tier.

The trap these cover is the first one: a slow detector tick must not stall frame
output. `step()` is what the background thread calls, and calling it directly is
how the timing-free assertions get made.
"""

import threading
import time

import pytest

from detection import EMPTY_TRACKS, Box, Detection
from detector_worker import (
    DETECTOR_NAME,
    BackgroundDetector,
    DetectorUnavailable,
    concepts_of,
)
from render_plan import global_plan, plan_from_fields


def plan_for(target: str, style: str = "a painting"):
    result = plan_from_fields(target, style)
    assert result.plan is not None, result.reason
    return result.plan


def found(concept: str, x0: int = 0, confidence: float = 0.9) -> Detection:
    return Detection(box=Box(x0, 0, x0 + 100, 200), concept=concept,
                     confidence=confidence)


class FakeDetector:
    """Everything `BackgroundDetector` is allowed to know about a detector."""

    def __init__(self, per_call=(), open_error=None, detect_error=None, delay=0.0):
        self.per_call = list(per_call)
        self.open_error = open_error
        self.detect_error = detect_error
        self.delay = delay
        self.opened = 0
        self.closed = 0
        self.vocabularies = []
        self.frames = []
        # One-shot hooks, for the two things that can happen *during* a slow call:
        # the plan changing under a re-warm, and under a detect.
        self.during_set_concepts = None
        self.during_detect = None

    def _fire(self, name):
        hook = getattr(self, name)
        setattr(self, name, None)
        if hook is not None:
            hook()

    def open(self):
        if self.open_error:
            raise self.open_error
        self.opened += 1

    def set_concepts(self, concepts):
        self.vocabularies.append(tuple(concepts))
        self._fire("during_set_concepts")

    def detect(self, frame):
        self.frames.append(frame)
        self._fire("during_detect")
        if self.delay:
            time.sleep(self.delay)
        if self.detect_error:
            raise self.detect_error
        index = min(len(self.frames) - 1, len(self.per_call) - 1)
        return self.per_call[index] if self.per_call else ()

    def close(self):
        self.closed += 1


# --- what the plan asks to be found -----------------------------------------


def test_a_global_plan_asks_for_nothing():
    assert concepts_of(global_plan("a painting")) == ()


def test_a_targeted_plan_asks_for_its_concept():
    assert concepts_of(plan_for("person")) == ("person",)


def test_the_concepts_are_the_targets_in_order_without_repeats():
    from render_plan import validate_plan

    result = validate_plan({"targets": [{"id": "a", "concept": "person"},
                                        {"id": "b", "concept": "dog"},
                                        {"id": "c", "concept": "person"}]})
    assert concepts_of(result.plan) == ("person", "dog")


# --- loading, and not loading ------------------------------------------------


def test_a_run_with_no_target_never_loads_a_detector():
    """A `global` plan pays no VRAM and no milliseconds for a detector."""
    detector = FakeDetector()
    background = BackgroundDetector(detector)
    background.follow(global_plan("a painting"))
    background.offer(object(), frame_index=0)
    assert background.step() is False
    assert detector.opened == 0
    assert background.active is False


def test_following_a_targeted_plan_loads_the_detector_and_sets_its_vocabulary():
    detector = FakeDetector()
    background = BackgroundDetector(detector)
    background.follow(plan_for("person"))
    background.step()
    assert detector.opened == 1
    assert detector.vocabularies == [("person",)]
    assert background.active is True


def test_the_vocabulary_is_set_before_any_frame_is_detected():
    """Issue #4's finding: `set_classes` drops the predictor, so the re-warm is the
    detector's own business and it happens on the cold path, not on a frame."""
    detector = FakeDetector()
    background = BackgroundDetector(detector)
    background.follow(plan_for("person"))
    background.step()
    assert detector.vocabularies and detector.frames == []


# --- publishing tracks -------------------------------------------------------


def test_a_detected_frame_publishes_a_snapshot():
    detector = FakeDetector(per_call=[[found("person")]])
    background = BackgroundDetector(detector)
    plan = plan_for("person")
    background.follow(plan)
    background.step()
    background.offer("frame", frame_index=9)
    assert background.step() is True

    tracks = background.tracks
    assert tracks.count == 1
    assert tracks.frame_index == 9
    assert tracks.plan_version == plan.plan_version
    assert tracks.concepts == ("person",)
    assert tracks.detector_ms >= 0.0
    assert tracks.ticks == 1


def test_ids_are_stable_across_detector_ticks():
    """The Gate's first item, with the detector's part played by a fake."""
    detector = FakeDetector(per_call=[[found("person", x0=0)],
                                      [found("person", x0=4)],
                                      [found("person", x0=8)]])
    background = BackgroundDetector(detector)
    background.follow(plan_for("person"))
    background.step()
    seen = []
    for tick, frame_index in enumerate((0, 3, 6)):
        background.offer("frame", frame_index=frame_index)
        background.step()
        seen.append(background.tracks.ids)
    assert seen == [(0,), (0,), (0,)]


def test_boxes_the_plan_did_not_ask_for_are_not_tracked():
    """An open-vocabulary detector can answer with more than it was asked."""
    detector = FakeDetector(per_call=[[found("cat"), found("person", x0=300)]])
    background = BackgroundDetector(detector)
    background.follow(plan_for("person"))
    background.step()
    background.offer("frame", frame_index=0)
    background.step()
    assert [t.concept for t in background.tracks.tracks] == ["person"]


def test_only_the_newest_offered_frame_is_detected():
    """The capture deque sheds frames and so does this: the frame in hand wins."""
    detector = FakeDetector(per_call=[[]])
    background = BackgroundDetector(detector)
    background.follow(plan_for("person"))
    background.step()
    background.offer("stale", frame_index=0)
    background.offer("fresh", frame_index=3)
    background.step()
    assert detector.frames == ["fresh"]


def test_a_step_with_nothing_offered_does_nothing():
    detector = FakeDetector()
    background = BackgroundDetector(detector)
    background.follow(plan_for("person"))
    background.step()
    assert background.step() is False
    assert detector.frames == []


# --- a new plan --------------------------------------------------------------


def test_a_new_vocabulary_re_warms_the_detector_and_forgets_the_old_objects():
    detector = FakeDetector(per_call=[[found("person")], [found("dog")]])
    background = BackgroundDetector(detector)
    background.follow(plan_for("person"))
    background.step()
    background.offer("frame", frame_index=0)
    background.step()
    assert background.tracks.count == 1

    background.follow(plan_for("dog"))
    assert background.tracks is EMPTY_TRACKS, \
        "tracks about the old vocabulary must not survive into the new one"
    background.step()
    assert detector.vocabularies == [("person",), ("dog",)]

    background.offer("frame", frame_index=3)
    background.step()
    assert background.tracks.ids == (1,), "an id must not change what it means"


def test_a_plan_that_changes_nothing_the_detector_cares_about_costs_nothing():
    """A style edit is not a vocabulary change, and a re-warm is ~108 ms."""
    detector = FakeDetector()
    background = BackgroundDetector(detector)
    background.follow(plan_for("person", "a painting"))
    background.step()
    background.follow(plan_for("person", "a charcoal drawing"))
    background.step()
    assert detector.vocabularies == [("person",)]


def test_boxes_from_a_detect_the_plan_outlived_are_thrown_away():
    """`follow` runs on the frame loop and `step` on the detector's thread, so a
    plan can change while a detect is in flight. Those boxes are about the old
    vocabulary and must not be published under the new plan's version."""
    detector = FakeDetector(per_call=[[found("person")]])
    background = BackgroundDetector(detector)
    background.follow(plan_for("person"))
    background.step()
    detector.during_detect = lambda: background.follow(plan_for("dog"))

    background.offer("frame", frame_index=0)
    assert background.step() is False
    assert background.tracks is EMPTY_TRACKS


def test_a_vocabulary_changed_during_a_re_warm_is_not_lost():
    """The re-warm takes ~108 ms of real time, which is long enough to type in."""
    detector = FakeDetector()
    background = BackgroundDetector(detector)
    background.follow(plan_for("person"))
    detector.during_set_concepts = lambda: background.follow(plan_for("dog"))

    background.step()
    background.step()
    assert detector.vocabularies == [("person",), ("dog",)]


# --- failure -----------------------------------------------------------------


def test_weights_that_are_not_there_disable_detection_rather_than_the_worker():
    said = []
    detector = FakeDetector(open_error=DetectorUnavailable("no weights at X"))
    background = BackgroundDetector(detector, log=said.append)
    background.follow(plan_for("person"))
    assert background.step() is False
    assert background.failed is True
    assert background.tracks is EMPTY_TRACKS
    assert any("no weights at X" in line for line in said)


def test_a_detector_that_throws_is_disabled_once_and_not_retried():
    said = []
    detector = FakeDetector(detect_error=RuntimeError("CUDA said no"))
    background = BackgroundDetector(detector, log=said.append)
    background.follow(plan_for("person"))
    background.step()
    background.offer("frame", frame_index=0)
    background.step()
    background.offer("frame", frame_index=3)
    background.step()
    assert len(detector.frames) == 1, "a broken detector must not be asked every tick"
    assert len([line for line in said if "CUDA said no" in line]) == 1
    assert background.active is False


# --- off the frame path ------------------------------------------------------


def test_offering_a_frame_does_not_wait_for_a_slow_detector():
    """The issue's first trap, measured: a 200 ms detect must not cost the frame
    loop 200 ms. The bound is loose on purpose - it is testing that nothing is
    joined, not how fast this machine's locks are."""
    detector = FakeDetector(per_call=[[found("person")]], delay=0.2)
    background = BackgroundDetector(detector)
    background.follow(plan_for("person"))
    background.start()
    try:
        started = time.perf_counter()
        for frame_index in range(0, 30, 3):
            background.offer("frame", frame_index=frame_index)
        offered_ms = (time.perf_counter() - started) * 1000.0
        assert offered_ms < 50.0, f"offering ten frames blocked for {offered_ms:.1f} ms"
        assert background.wait_for_tick(1, timeout=5.0)
        assert background.tracks.count == 1
    finally:
        background.stop()
    assert detector.closed == 1


def test_stopping_ends_the_thread():
    background = BackgroundDetector(FakeDetector())
    background.start()
    background.stop()
    assert background.running is False
    assert threading.active_count() >= 1


# --- one detector, three registries ------------------------------------------


def test_the_worker_loads_the_detector_the_evaluation_chose_and_the_plan_validates_against():
    from bench.detectors import DETECTORS, PRIMARY_DETECTOR
    from detector_worker import DETECT_INPUT_SIZE, WEIGHTS_FILE
    from render_plan import ACTIVE_DETECTOR

    assert DETECTOR_NAME == PRIMARY_DETECTOR == ACTIVE_DETECTOR
    measured = DETECTORS[PRIMARY_DETECTOR]
    assert WEIGHTS_FILE == measured.weights
    assert DETECT_INPUT_SIZE == measured.imgsz, \
        "the worker must run the detector at the size it was measured at"


def test_the_weights_are_looked_for_under_the_shared_models_root():
    from detector_worker import UltralyticsDetector

    detector = UltralyticsDetector(models_root="C:/models")
    assert detector.weights.parent.name == "detectors"
    assert detector.weights.name.endswith(".pt")


def test_a_detector_asked_to_open_with_no_weights_says_where_it_looked(tmp_path):
    from detector_worker import UltralyticsDetector

    detector = UltralyticsDetector(models_root=tmp_path)
    with pytest.raises(DetectorUnavailable) as refused:
        detector.open()
    assert str(tmp_path) in str(refused.value)
    assert "--allow-download" in str(refused.value), \
        "the worker is offline; say how the weights get there instead"
