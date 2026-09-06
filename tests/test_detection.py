"""The tracker, the detect cadence, and the snapshot the frame loop reads.

Issue #7, spec 5.1 (C3, C4). Everything here is stdlib-pure and runs in the merge
gate's GPU-free tier: `detection.py` holds the half of the detector that decides
*identity*, and identity is exactly the half a GPU cannot help with.
"""

import pytest

from detection import (
    EMPTY_TRACKS,
    MAX_MISSES,
    TRACK_IOU_THRESHOLD,
    TRACK_SMOOTHING,
    Box,
    Detection,
    Tracker,
    Tracks,
    detection_status,
    fps_payload,
    is_detect_frame,
    iou,
)


def box(x0, y0, x1, y1) -> Box:
    return Box(x0, y0, x1, y1)


def person(x0, y0, x1, y1, confidence=0.9, concept="person") -> Detection:
    return Detection(box=box(x0, y0, x1, y1), concept=concept, confidence=confidence)


# --- overlap ----------------------------------------------------------------


def test_two_identical_boxes_overlap_completely():
    assert iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == 1.0


def test_boxes_that_do_not_touch_do_not_overlap():
    assert iou(box(0, 0, 10, 10), box(20, 20, 30, 30)) == 0.0


def test_a_half_shifted_box_overlaps_by_a_third():
    # 50 of 150 pixels of union.
    assert iou(box(0, 0, 10, 10), box(5, 0, 15, 10)) == pytest.approx(1.0 / 3.0)


def test_the_overlap_rule_is_the_one_the_committed_tracks_were_built_with():
    """`bench.primitives` has this function too, and cannot be imported from here -
    it is the harness and it imports numpy. So the two are held to one answer."""
    from bench.primitives import iou as measured_iou

    pairs = [((0, 0, 10, 10), (0, 0, 10, 10)), ((0, 0, 10, 10), (5, 0, 15, 10)),
             ((0, 0, 10, 10), (20, 20, 30, 30)), ((0, 0, 10, 10), (10, 0, 20, 10)),
             ((4, 4, 8, 9), (1, 2, 9, 7)), ((0, 0, 0, 0), (0, 0, 5, 5))]
    for a, b in pairs:
        assert iou(box(*a), box(*b)) == measured_iou(a, b)


# --- identity ---------------------------------------------------------------


def test_the_first_detection_starts_a_track():
    tracker = Tracker()
    tracks = tracker.update([person(0, 0, 10, 20)], frame_index=0)
    assert [t.track_id for t in tracks] == [0]
    assert tracks[0].concept == "person"
    assert tracks[0].hits == 1
    assert tracks[0].misses == 0


def test_an_object_that_moves_a_little_keeps_its_id():
    tracker = Tracker()
    tracker.update([person(0, 0, 100, 200)], frame_index=0)
    tracks = tracker.update([person(6, 0, 106, 200)], frame_index=3)
    assert [t.track_id for t in tracks] == [0]
    assert tracks[0].hits == 2


def test_an_object_that_teleports_is_a_new_object():
    """Below the IoU threshold there is nothing to claim identity with."""
    tracker = Tracker()
    tracker.update([person(0, 0, 100, 200)], frame_index=0)
    tracks = {t.track_id: t for t in
              tracker.update([person(500, 0, 600, 200)], frame_index=3)}
    assert tracks[1].box == box(500, 0, 600, 200)
    assert tracks[0].misses == 1, "the object that vanished is bridged, not claimed"


def test_two_objects_keep_their_own_ids_across_a_tick():
    tracker = Tracker()
    tracker.update([person(0, 0, 100, 200), person(400, 0, 500, 200, 0.8)],
                   frame_index=0)
    tracks = {t.track_id: t for t in
              tracker.update([person(404, 0, 504, 200, 0.8), person(4, 0, 104, 200)],
                             frame_index=3)}
    assert set(tracks) == {0, 1}
    assert tracks[0].box.x0 < 100 and tracks[1].box.x0 > 300


def test_a_box_cannot_claim_a_track_of_another_concept():
    """A dog standing where a person stood is a new object, not that person."""
    tracker = Tracker()
    tracker.update([person(0, 0, 100, 200)], frame_index=0)
    tracks = {t.track_id: t for t in
              tracker.update([person(0, 0, 100, 200, concept="dog")], frame_index=3)}
    assert tracks[1].concept == "dog"
    assert tracks[0].concept == "person" and tracks[0].misses == 1


def test_ids_are_never_reused_after_a_track_dies():
    tracker = Tracker(max_misses=0)
    tracker.update([person(0, 0, 100, 200)], frame_index=0)
    tracker.update([], frame_index=3)
    tracks = tracker.update([person(0, 0, 100, 200)], frame_index=6)
    assert [t.track_id for t in tracks] == [1], "a recycled id is a lie about identity"


def test_the_strongest_detection_claims_the_track_it_overlaps_most():
    tracker = Tracker()
    tracker.update([person(0, 0, 100, 200)], frame_index=0)
    tracks = {t.confidence: t.track_id for t in
              tracker.update([person(50, 0, 150, 200, 0.4),
                              person(2, 0, 102, 200, 0.95)], frame_index=3)}
    assert tracks[0.95] == 0, "the weaker box took the track from under the stronger"
    assert tracks[0.4] == 1


# --- bridging the gap -------------------------------------------------------


def test_a_track_the_detector_missed_survives_and_says_so():
    """The capture deque sheds frames and detectors blink; neither is an exit."""
    tracker = Tracker()
    tracker.update([person(0, 0, 100, 200)], frame_index=0)
    tracks = tracker.update([], frame_index=3)
    assert [t.track_id for t in tracks] == [0]
    assert tracks[0].misses == 1
    assert tracks[0].box == box(0, 0, 100, 200), "a missed track must not drift"


def test_a_track_missed_too_often_is_dropped():
    tracker = Tracker(max_misses=2)
    tracker.update([person(0, 0, 100, 200)], frame_index=0)
    for tick in range(1, 3):
        assert tracker.update([], frame_index=tick * 3)
    assert tracker.update([], frame_index=9) == ()


def test_a_reappearing_object_keeps_its_id_and_its_miss_count_resets():
    tracker = Tracker(max_misses=2)
    tracker.update([person(0, 0, 100, 200)], frame_index=0)
    tracker.update([], frame_index=3)
    tracks = tracker.update([person(0, 0, 100, 200)], frame_index=6)
    assert [t.track_id for t in tracks] == [0]
    assert tracks[0].misses == 0


# --- box smoothing ----------------------------------------------------------


def test_a_matched_box_is_an_ema_of_the_old_and_the_new():
    """Spec 8.5's box smoothing: the render follows the object, not the jitter."""
    tracker = Tracker(smoothing=0.5)
    tracker.update([person(0, 0, 100, 200)], frame_index=0)
    tracks = tracker.update([person(20, 0, 120, 200)], frame_index=3)
    assert tracks[0].box == box(10, 0, 110, 200)


def test_smoothing_of_one_follows_the_detector_exactly():
    tracker = Tracker(smoothing=1.0)
    tracker.update([person(0, 0, 100, 200)], frame_index=0)
    tracks = tracker.update([person(20, 0, 120, 200)], frame_index=3)
    assert tracks[0].box == box(20, 0, 120, 200)


def test_a_new_track_takes_the_detector_box_whole():
    """There is nothing to smooth against, and half a box is nobody's box."""
    tracker = Tracker(smoothing=0.1)
    tracks = tracker.update([person(20, 0, 120, 200)], frame_index=0)
    assert tracks[0].box == box(20, 0, 120, 200)


def test_the_defaults_are_the_ones_spec_8_5_asks_for():
    assert 0.0 < TRACK_SMOOTHING < 1.0
    assert 0.0 < TRACK_IOU_THRESHOLD < 1.0
    assert MAX_MISSES >= 1, "a tracker that drops on one miss does not bridge a gap"


# --- the cadence ------------------------------------------------------------


def test_detection_runs_every_nth_frame():
    assert [i for i in range(10) if is_detect_frame(i, 3)] == [0, 3, 6, 9]


def test_a_cadence_of_one_detects_on_every_frame():
    assert all(is_detect_frame(i, 1) for i in range(10))


def test_a_nonsense_cadence_still_detects_rather_than_dividing_by_zero():
    assert is_detect_frame(0, 0)
    assert is_detect_frame(7, -3)


def test_raising_the_cadence_lowers_the_amortised_cost():
    """The Gate's third item, in arithmetic: the same detect over more frames."""
    for ms in (3.0, 15.0):
        tracks = Tracks(detector_ms=ms)
        assert detection_status(tracks, 6)["amortised_ms"] < \
            detection_status(tracks, 2)["amortised_ms"]
    assert detection_status(EMPTY_TRACKS, detect_every_n=6)["detect_every_n"] == 6


# --- what the frame loop and the GUI read -----------------------------------


def test_the_empty_snapshot_is_a_snapshot_like_any_other():
    """"No tracks yet" must not be a second state the frame loop has to handle."""
    assert EMPTY_TRACKS.count == 0
    assert EMPTY_TRACKS.ids == ()
    assert EMPTY_TRACKS.tracks == ()


def test_a_snapshot_is_frozen():
    snapshot = Tracks(tracks=(), frame_index=4)
    with pytest.raises(Exception):
        snapshot.frame_index = 5


def test_the_fps_payload_carries_the_detection_count_and_the_detector_cost():
    """Step 5: the existing fps channel, so the GUI has one thing to read."""
    tracker = Tracker()
    tracks = tracker.update([person(0, 0, 100, 200)], frame_index=0)
    snapshot = Tracks(tracks=tracks, frame_index=0, detector_ms=14.5, ticks=1,
                      concepts=("person",))
    payload = fps_payload(30, snapshot, detect_every_n=3)
    assert payload["fps"] == 30
    assert payload["detections"] == 1
    assert payload["detector_ms"] == pytest.approx(14.5, abs=0.05)
    assert payload["amortised_ms"] == pytest.approx(14.5 / 3, abs=0.05)
    assert payload["concepts"] == ("person",)


def test_the_fps_payload_of_a_run_with_no_detector_says_nothing_about_one():
    payload = fps_payload(30, EMPTY_TRACKS, detect_every_n=3)
    assert payload["fps"] == 30
    assert "detections" not in payload, "a run with no target must not claim 0 objects"
