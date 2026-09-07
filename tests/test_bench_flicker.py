"""The flicker metric (issue #5, spec 8.5). GPU-free: it is arithmetic over arrays.

The Gate asks for a *pure function with unit tests*, and the reason is that the
metric is what ranks two rendering primitives against each other. A metric nobody
can check is an argument, not a measurement.

Everything here is numpy. `bench.flicker` never imports torch, cv2 or PIL - the
runner hands it arrays and takes a number back.
"""

import numpy as np
import pytest

from bench.flicker import (
    STATIC_THRESHOLD,
    FlickerScore,
    ResponseScore,
    flicker_score,
    response_score,
    static_pair_mask,
)


def frames(*values, size=4):
    """A sequence of flat grey frames, one per value - HxWx3 uint8-ranged floats."""
    return [np.full((size, size, 3), float(value)) for value in values]


# --- the static mask ------------------------------------------------------------

def test_a_pixel_that_did_not_move_in_the_source_is_static():
    before, after = frames(10, 10)
    assert static_pair_mask(before, after).all()


def test_a_pixel_that_moved_more_than_the_threshold_is_not_static():
    before, after = frames(10, 10 + STATIC_THRESHOLD + 1)
    assert not static_pair_mask(before, after).any()


def test_the_threshold_is_inclusive_so_sensor_noise_stays_static():
    before, after = frames(10, 10 + STATIC_THRESHOLD)
    assert static_pair_mask(before, after).all()


def test_one_moving_channel_is_enough_to_disqualify_a_pixel():
    """Static means static in every channel: a pixel that changed hue changed."""
    before = np.zeros((1, 2, 3))
    after = before.copy()
    after[0, 1, 0] = 255.0  # red channel of the second pixel only
    mask = static_pair_mask(before, after)
    assert mask.tolist() == [[True, False]]


# --- the score ------------------------------------------------------------------

def test_an_output_that_does_not_move_where_the_source_did_not_has_no_flicker():
    source = frames(10, 10, 10)
    output = frames(200, 200, 200)
    assert flicker_score(source, output).mean_abs_diff == 0.0


def test_the_score_is_the_mean_absolute_difference_between_consecutive_outputs():
    source = frames(10, 10, 10)  # entirely static, so the whole frame is scored
    output = frames(0, 6, 10)  # differences of 6 and 4
    assert flicker_score(source, output).mean_abs_diff == pytest.approx(5.0)


def test_pixels_that_moved_in_the_source_are_excluded():
    """The metric's whole point: a moving subject is not flicker, it is motion."""
    source = [np.zeros((1, 2, 3)), np.zeros((1, 2, 3))]
    source[1][0, 0] = 255.0  # the first pixel moves in the source
    output = [np.zeros((1, 2, 3)), np.zeros((1, 2, 3))]
    output[0][0, 0] = 255.0  # and the output disagrees wildly there
    output[1][0, 1] = 40.0  # while the static pixel moved by 40

    score = flicker_score(source, output)
    assert score.mean_abs_diff == pytest.approx(40.0)
    assert score.static_pixels == 1


def test_a_painted_mask_narrows_the_metric_to_what_the_primitive_rendered():
    """Untouched pass-through pixels are identical by construction, so leaving them
    in dilutes every primitive's score towards zero and ranks nothing."""
    source = [np.zeros((1, 2, 3)), np.zeros((1, 2, 3))]
    output = [np.zeros((1, 2, 3)), np.zeros((1, 2, 3))]
    output[1][0, 1] = 30.0  # only the second pixel was rendered, and it moved

    painted = [np.array([[False, True]]), np.array([[False, True]])]
    assert flicker_score(source, output, painted).mean_abs_diff == pytest.approx(30.0)
    # Unmasked, the identical pass-through pixel halves it.
    assert flicker_score(source, output).mean_abs_diff == pytest.approx(15.0)


def test_a_pixel_has_to_be_painted_in_both_frames_of_a_pair():
    """A box that moved off a pixel between two frames leaves a source/output edge
    there, which is compositing, not flicker."""
    source = [np.zeros((1, 2, 3)), np.zeros((1, 2, 3))]
    output = [np.zeros((1, 2, 3)), np.zeros((1, 2, 3))]
    output[1][0, 0] = 90.0
    painted = [np.array([[True, True]]), np.array([[False, True]])]

    score = flicker_score(source, output, painted)
    assert score.mean_abs_diff == 0.0, "only the second pixel is in both masks"
    assert score.static_pixels == 1


# --- the cases where there is no honest number -----------------------------------

def test_a_pair_with_no_static_painted_pixel_is_skipped_rather_than_scored_zero():
    source = [np.zeros((1, 1, 3)), np.full((1, 1, 3), 255.0)]
    output = frames(0, 255, size=1)

    score = flicker_score(source, output)
    assert score.mean_abs_diff is None
    assert score.pairs_scored == 0
    assert score.pairs == 1
    assert "no pixel" in score.note


def test_a_single_frame_has_no_consecutive_pair_at_all():
    score = flicker_score(frames(10), frames(200))
    assert score.mean_abs_diff is None
    assert score.pairs == 0
    assert "at least two frames" in score.note


def test_the_two_sequences_have_to_be_the_same_length():
    with pytest.raises(ValueError, match="same number of frames"):
        flicker_score(frames(1, 2, 3), frames(1, 2))


def test_the_frames_have_to_be_the_same_shape():
    with pytest.raises(ValueError, match="shape"):
        flicker_score([np.zeros((2, 2, 3))] * 2, [np.zeros((3, 3, 3))] * 2)


# --- what the record carries ------------------------------------------------------

def test_the_score_carries_what_it_was_computed_over():
    source = frames(10, 10, 10, size=2)
    output = frames(0, 4, 4, size=2)
    score = flicker_score(source, output)

    assert isinstance(score, FlickerScore)
    assert score.pairs == 2 and score.pairs_scored == 2
    assert score.static_pixels == 4  # 2x2, all static
    assert score.static_fraction == pytest.approx(1.0)
    assert score.per_pair_abs_diff == pytest.approx([4.0, 0.0])
    assert score.threshold == STATIC_THRESHOLD


def test_the_record_round_trips_to_json_safe_primitives():
    score = flicker_score(frames(10, 10), frames(0, 4))
    data = score.to_dict()
    assert data["mean_abs_diff"] == pytest.approx(4.0)
    assert isinstance(data["per_pair_abs_diff"], list)
    assert all(isinstance(value, float) for value in data["per_pair_abs_diff"])
    assert isinstance(data["static_pixels"], int)


def test_lower_is_steadier_which_is_the_direction_the_decision_reads():
    source = frames(10, 10, 10)
    steady = flicker_score(source, frames(0, 1, 2)).mean_abs_diff
    boiling = flicker_score(source, frames(0, 100, 200)).mean_abs_diff
    assert steady < boiling


# --- responsiveness: the other half of the EMA trade (issue #32) -------------


def test_responsiveness_scores_where_the_source_moved():
    """The mirror of flicker, on the same pairs and the same painted mask: what the
    output did where the subject *did* move."""
    source = [np.zeros((1, 2, 3)), np.zeros((1, 2, 3))]
    source[1][0, 0] = 255.0  # the first pixel moves in the source
    output = [np.zeros((1, 2, 3)), np.zeros((1, 2, 3))]
    output[1][0, 0] = 60.0  # and the render followed it by 60
    output[1][0, 1] = 40.0  # while the static pixel boiled by 40

    assert response_score(source, output).mean_abs_diff == pytest.approx(60.0)
    assert flicker_score(source, output).mean_abs_diff == pytest.approx(40.0)


def test_the_two_metrics_partition_the_painted_pixels_of_a_pair():
    """Same threshold, opposite sides of it - so neither can be improved by moving
    the line, which is what would make the trade unreadable."""
    source = [np.zeros((3, 3, 3)), np.zeros((3, 3, 3))]
    source[1][0, :] = 255.0
    output = [np.zeros((3, 3, 3)), np.full((3, 3, 3), 7.0)]
    flicker = flicker_score(source, output)
    response = response_score(source, output)
    assert flicker.static_pixels + response.moving_pixels == flicker.frame_pixels


def test_an_output_that_never_follows_the_source_has_no_response():
    """The failure an EMA is capable of: perfectly steady, and perfectly inert."""
    source = [np.zeros((1, 2, 3)), np.full((1, 2, 3), 255.0)]
    frozen = [np.full((1, 2, 3), 90.0), np.full((1, 2, 3), 90.0)]
    assert response_score(source, frozen).mean_abs_diff == 0.0


def test_responsiveness_is_none_when_nothing_moved_rather_than_zero():
    """A clip in which nothing moved says nothing about responsiveness, and 0.0
    would read as the inert output above."""
    score = response_score(frames(10, 10, 10), frames(0, 40, 80))
    assert score.mean_abs_diff is None
    assert "nothing to score" in score.note


def test_responsiveness_honours_the_painted_mask_too():
    source = [np.zeros((1, 2, 3)), np.full((1, 2, 3), 255.0)]
    output = [np.zeros((1, 2, 3)), np.zeros((1, 2, 3))]
    output[1][0, 1] = 30.0
    painted = [np.array([[False, True]]), np.array([[False, True]])]
    assert response_score(source, output, painted).mean_abs_diff == pytest.approx(30.0)


def test_the_response_record_round_trips_to_json_safe_primitives():
    import json

    score = response_score([np.zeros((1, 2, 3)), np.full((1, 2, 3), 255.0)],
                           [np.zeros((1, 2, 3)), np.full((1, 2, 3), 40.0)])
    assert isinstance(score, ResponseScore)
    assert json.loads(json.dumps(score.to_dict()))["mean_abs_diff"] == pytest.approx(40.0)


def test_higher_is_more_responsive_which_is_the_direction_the_trade_reads():
    assert "higher is more responsive" in response_score(
        [np.zeros((1, 2, 3)), np.full((1, 2, 3), 255.0)],
        [np.zeros((1, 2, 3)), np.full((1, 2, 3), 40.0)]).note

