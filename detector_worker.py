"""The worker's detector: off the frame path, driven by the active plan.

Issue #7, spec 5.1 (C3). Two things live here.

`UltralyticsDetector` is the only shipped module that imports ultralytics, and it
does so inside its methods - so this file, like `detection.py` and
`render_plan.py`, imports in the GUI process and in the merge gate's GPU-free tier
without dragging torch in behind it.

`BackgroundDetector` is the state machine around it: which concepts to look for
(the plan's), when the model gets loaded (on the first plan that names one), what
happens to a frame while a detect is already running (it replaces the one waiting),
and what happens when the detector breaks (detection stops, the render loop does
not). It has a `step()` the caller drives and a `start()` that drives it from a
daemon thread; the worker uses the thread, and the tests use `step()`, which is why
none of the behaviour above needs a GPU or a sleep to be checked.

Nothing here downloads anything. The worker enforces offline mode, so weights that
are not cached are a refusal with the fetch command in it, not a fetch.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from detection import EMPTY_TRACKS, Box, Detection, Tracker, Tracks

# The detector issue #4 measured and chose, spelt again rather than imported from
# `bench.detectors`: that is the measurement harness and this is shipped code, the
# same split `render_plan.DETECTOR_VOCABULARIES` makes. A test holds all three
# registries - measured, validated against, loaded - to one name and one weights
# file, so the three cannot drift apart quietly.
DETECTOR_NAME = "yolo-world-s-640"
WEIGHTS_FILE = "yolov8s-worldv2.pt"
WEIGHTS_SUBDIR = "detectors"
# The ultralytics entry point. A YOLO-World has a text head and a `set_classes`.
DETECTOR_LOADER = "YOLOWorld"
# 640², which is what issue #4's 14.3 ms per detect was measured at. Running the
# shipped detector at another size would make that measurement about another thing.
DETECT_INPUT_SIZE = 640
# The bench asks at 0.05 to gather evidence of what a detector saw at all. A render
# wants boxes it can stand behind, so the worker uses ultralytics' own default.
DETECT_CONFIDENCE = 0.25

# How long the background thread sleeps between checks when nothing has been
# offered. It is woken by `offer` and `follow`, so this is only the ceiling on
# noticing a stop.
IDLE_WAKE_S = 0.1


class DetectorUnavailable(RuntimeError):
    """The detector cannot be loaded, and the reason a user can act on."""


def concepts_of(plan) -> Tuple[str, ...]:
    """What `plan` asks the detector to find, in order and without repeats.

    A plan with no target asks for nothing, and a detector is never loaded for it:
    `mode: "global"` is the whole frame restyled, which needs no boxes.
    """
    return tuple(dict.fromkeys(target.concept
                               for target in getattr(plan, "targets", ())))


def frame_to_array(frame):
    """Whatever the capture path is holding, as an HWC uint8 RGB array.

    The capture thread hands the frame loop a CUDA float tensor in 0..1, shaped
    (B, C, H, W); the bench hands clips over as numpy already. The conversion costs
    one small device-to-host copy and it happens **on the detector thread**, never
    on the frame path.

    torch is not imported to recognise a tensor - `detach` is enough of a
    signature, and importing torch here would put it in the GUI process.
    """
    import numpy as np

    if hasattr(frame, "detach"):
        tensor = frame.detach()
        if tensor.dim() == 4:
            tensor = tensor[-1]
        if tensor.dim() == 3 and tensor.shape[0] in (1, 3):  # CHW -> HWC
            tensor = tensor.permute(1, 2, 0)
        if tensor.dtype.is_floating_point:
            tensor = tensor.clamp(0.0, 1.0).mul(255.0).round()
        return tensor.to("cpu").numpy().astype(np.uint8)

    array = np.asarray(frame)
    if array.dtype == np.uint8:
        return array
    # A float frame is 0..1 from the capture path or already 0..255 from a decoder;
    # only the first needs scaling, and the array's own range is what says which.
    if array.max() <= 1.0:
        array = np.clip(array, 0.0, 1.0) * 255.0
    return array.astype(np.uint8)


class UltralyticsDetector:
    """YOLO-World, loaded once in the worker process and asked for boxes.

    Every ultralytics rule issue #4 measured is obeyed here and nowhere else:
    `WEIGHTS_DIR` is repointed before the import so a 338 MB CLIP checkpoint does
    not land in the repo, the model is moved to CUDA *before* the first
    `set_classes` (the text encoder caches the device it was built on), and every
    vocabulary change is followed by one throwaway detect, because `set_classes`
    drops the predictor and the next call rebuilds it for ~108 ms.
    """

    def __init__(self, models_root: Union[str, Path],
                 weights_file: str = WEIGHTS_FILE,
                 loader: str = DETECTOR_LOADER,
                 imgsz: int = DETECT_INPUT_SIZE,
                 conf: float = DETECT_CONFIDENCE) -> None:
        self.models_root = Path(models_root)
        self.weights = self.models_root / WEIGHTS_SUBDIR / weights_file
        self.loader = loader
        self.imgsz = imgsz
        self.conf = conf
        self._model = None

    def open(self) -> None:
        """Load the weights onto the GPU. Raises `DetectorUnavailable`, never fetches."""
        if not self.weights.is_file():
            raise DetectorUnavailable(
                f"no detector weights at {self.weights}. The worker is offline and "
                f"will not fetch them; run `python -m bench {DETECTOR_NAME} "
                f"--allow-download` once, or point SD_MODELS_DIR at a models root "
                f"that has them."
            )
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        import ultralytics
        from ultralytics import utils as ultralytics_utils

        # Before anything can read it: `ultralytics.nn.text_model` binds the value
        # at import time and drops the CLIP checkpoint into `<cwd>/weights` if it
        # is left alone - which for this app is wherever it was launched from.
        ultralytics_utils.WEIGHTS_DIR = self.models_root / WEIGHTS_SUBDIR

        model = getattr(ultralytics, self.loader)(str(self.weights))
        model.to("cuda")
        self._model = model

    def set_concepts(self, concepts: Sequence[str]) -> None:
        """Put `concepts` in front of the detector and pay the re-warm here.

        The throwaway detect is the cold-path half of issue #4's finding: without
        it the first frame after a prompt edit pays ~108 ms, which is three frames'
        worth of a 30 FPS budget.
        """
        import numpy as np

        if self._model is None:
            raise DetectorUnavailable("the detector was asked for concepts before it was loaded")
        self._model.set_classes(list(concepts))
        self._predict(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))

    def detect(self, frame) -> Tuple[Detection, ...]:
        """One detect, in the capture frame's own pixel space."""
        if self._model is None:
            raise DetectorUnavailable("the detector was asked to detect before it was loaded")
        return detections_from(self._predict(frame_to_array(frame)))

    def close(self) -> None:
        self._model = None

    def _predict(self, array):
        """`imgsz=(n, n)` rather than `n`: a scalar letterboxes to a rectangle."""
        return self._model.predict(array, imgsz=(self.imgsz, self.imgsz), device=0,
                                   conf=self.conf, verbose=False)[0]


def detections_from(result) -> Tuple[Detection, ...]:
    """An ultralytics result as `Detection`s, strongest first.

    Boxes are rounded to whole pixels here: everything downstream - overlap,
    smoothing, the mask the compositor will cut - is integer pixel work, and a box
    that carries four decimals it cannot act on invites someone to compare two of
    them for equality.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()
    found = [
        Detection(
            box=Box(*(int(round(float(value))) for value in box)),
            concept=result.names[int(class_index)],
            confidence=float(confidence),
        )
        for class_index, confidence, box in zip(boxes.cls, boxes.conf, boxes.xyxy)
    ]
    found.sort(key=lambda detection: detection.confidence, reverse=True)
    return tuple(found)


class BackgroundDetector:
    """Detection, off the frame path: a plan in, a `Tracks` snapshot out.

    The frame loop only ever does three things with one - `follow` a new plan
    (cold path), `offer` the newest frame on a detect tick, and read `tracks`. None
    of the three waits for a detect: `offer` replaces whatever frame was waiting
    and returns, so a detect that overruns costs the *detector* frames and costs
    the render loop nothing. That is the issue's first trap, and the reason the
    capture deque's shedding is mirrored here rather than fought.

    `detector` is anything with `open` / `set_concepts` / `detect` / `close`. In
    the worker that is `UltralyticsDetector`; in the GPU-free tier it is a fake,
    which is what makes the load, cadence, failure and identity behaviour testable
    without a CUDA device.
    """

    def __init__(self, detector, log: Callable[[str], None] = print,
                 tracker: Optional[Tracker] = None) -> None:
        self._detector = detector
        self._log = log
        self._tracker = tracker or Tracker()
        self._tracks: Tracks = EMPTY_TRACKS
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pending: Optional[Tuple[Any, int]] = None
        self._concepts: Tuple[str, ...] = ()
        self._plan_version = 0
        self._vocabulary_pending = False
        # Bumped by every vocabulary change. `step` carries the generation it
        # started under and drops what it produced if the plan moved on while it
        # was working - `follow` runs on the frame loop and `step` on this thread,
        # and a detect is ~15 ms of that gap while a re-warm is ~110 ms.
        self._generation = 0
        self._opened = False
        self._failed = False
        self._ticks = 0
        self._published = threading.Event()

    # --- what the frame loop reads ------------------------------------------

    @property
    def tracks(self) -> Tracks:
        """The newest published snapshot. One reference read, nothing allocated."""
        return self._tracks

    @property
    def ticks(self) -> int:
        return self._ticks

    @property
    def active(self) -> bool:
        """Is there anything to detect and something able to detect it?"""
        return bool(self._concepts) and not self._failed

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- what the frame loop calls ------------------------------------------

    def follow(self, plan) -> None:
        """Take the concepts from `plan`. Cold path: called when a plan changes.

        A new vocabulary invalidates the tracks that were about the old one, so
        they are dropped rather than re-labelled: track 2 meaning "the second
        person" must never quietly come to mean "the second dog". The tracker
        itself is reset on the detector's thread, where it is used - this method is
        called from the frame loop and touches nothing the detect is holding.
        """
        concepts = concepts_of(plan)
        with self._lock:
            self._plan_version = int(getattr(plan, "plan_version", 0))
            if concepts == self._concepts:
                return
            self._concepts = concepts
            self._generation += 1
            self._vocabulary_pending = bool(concepts)
            self._pending = None
            self._tracks = EMPTY_TRACKS
        self._wake.set()

    def offer(self, frame, frame_index: int) -> bool:
        """Hand over the newest frame. Returns whether anyone will look at it.

        Never blocks on a detect in flight; the frame waiting to be detected is
        simply replaced, which is the capture thread's own newest-frame-wins rule
        applied one stage later.
        """
        if not self.active:
            return False
        with self._lock:
            self._pending = (frame, frame_index)
        self._wake.set()
        return True

    # --- the work ------------------------------------------------------------

    def step(self) -> bool:
        """Do whatever is outstanding. Returns whether a snapshot was published.

        The background thread calls this in a loop and nothing else does in the
        worker - but it is public, and synchronous, because every rule above is
        then checkable without a thread or a sleep.
        """
        with self._lock:
            concepts, pending = self._concepts, self._pending
            vocabulary_pending, generation = self._vocabulary_pending, self._generation
            plan_version = self._plan_version
            self._pending = None
        if self._failed or not concepts:
            return False
        try:
            if not self._opened:
                self._detector.open()
                self._opened = True
                self._log(f"Detector loaded: {DETECTOR_NAME}")
            if vocabulary_pending:
                # The tracker first: the objects it holds are about the vocabulary
                # being replaced, and an id must not change what it means.
                self._tracker.reset()
                self._detector.set_concepts(concepts)
                self._log(f"Detector vocabulary: {', '.join(concepts)}")
                with self._lock:
                    # Only if nothing was asked for in the meantime. Clearing it
                    # unconditionally would swallow a plan that arrived during the
                    # re-warm, and the detector would keep answering the old one.
                    if self._generation == generation:
                        self._vocabulary_pending = False
            if pending is None:
                return False
            frame, frame_index = pending
            started = time.perf_counter()
            detections = self._detector.detect(frame)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if self._generation != generation:
                return False  # these boxes are about a plan nobody is rendering
            self._publish(detections, frame_index, concepts, plan_version, elapsed_ms)
            return True
        except Exception as error:
            # Detection stops; rendering does not. Said once, because a detector
            # that failed to load will fail again on every tick that follows.
            self._failed = True
            self._tracks = EMPTY_TRACKS
            self._log(f"Detection disabled: {error}")
            return False

    def _publish(self, detections: Sequence[Detection], frame_index: int,
                 concepts: Tuple[str, ...], plan_version: int,
                 elapsed_ms: float) -> None:
        """Build the whole snapshot, then swap it in with one assignment."""
        wanted = [d for d in detections if d.concept in concepts]
        self._ticks += 1
        self._tracks = Tracks(
            tracks=self._tracker.update(wanted, frame_index),
            frame_index=frame_index, plan_version=plan_version,
            concepts=concepts, detector_ms=elapsed_ms, ticks=self._ticks,
        )
        self._published.set()

    # --- the thread ----------------------------------------------------------

    def start(self) -> None:
        """Run `step` on a daemon thread until `stop`."""
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="detection", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)
        try:
            self._detector.close()
        except Exception:
            pass

    def wait_for_tick(self, ticks: int, timeout: float = 5.0) -> bool:
        """Block until at least `ticks` snapshots have been published.

        For a caller that wants to observe the thread - a test, or a measurement
        harness. The frame loop never waits for detection; that is the point of it.
        """
        deadline = time.perf_counter() + timeout
        while True:
            # Cleared *before* the count is read: a tick published in between would
            # otherwise have its flag thrown away and this would wait out the whole
            # timeout for a snapshot that had already arrived.
            self._published.clear()
            if self._ticks >= ticks:
                return True
            if not self._published.wait(max(0.0, deadline - time.perf_counter())):
                return self._ticks >= ticks

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(IDLE_WAKE_S)
            self._wake.clear()
            self.step()
