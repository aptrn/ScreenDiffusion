"""The detector candidates and the arithmetic that judges them (issue #4).

Pure: a registry, a division by the detect cadence, and a comparison against the
spec 7.1 detection budget. None of it touches a GPU, so the merge gate can check the
reasoning that turns a measured millisecond figure into a verdict - which is the
half of the issue that a benchmark run cannot check for itself.
"""

from pathlib import Path

import pytest

from bench.detectors import (
    CONCEPT_PROBES,
    DEFAULT_CADENCE,
    DETECT_BUDGET_MAX_MS,
    DETECT_BUDGET_MIN_MS,
    DETECTORS,
    PRIMARY_DETECTOR,
    SPEED_FLOOR_DETECTOR,
    Candidate,
    amortised_ms,
    budget_verdict,
    frame_path_verdict,
    recommend,
    required_cadence,
    weights_path,
)


# --- the registry: what the issue says to measure, and no more ----------------

def test_yolo_world_is_the_primary_candidate():
    """Not one of five peers - the issue's Context makes it the thing to prove."""
    world = DETECTORS[PRIMARY_DETECTOR]
    assert world.open_vocabulary is True
    assert world.role == "candidate"
    assert world.imgsz == 640


def test_yolov8n_is_the_speed_floor_and_is_closed_vocabulary():
    floor = DETECTORS[SPEED_FLOOR_DETECTOR]
    assert floor.open_vocabulary is False
    assert floor.role == "speed floor"
    assert floor.imgsz == 640


def test_the_contingencies_are_not_registered_as_peers():
    """The issue's first trap: do not spend the run benchmarking all five."""
    assert set(DETECTORS) == {PRIMARY_DETECTOR, SPEED_FLOOR_DETECTOR}


def test_weights_live_under_the_shared_models_root():
    """Hundreds of MB of weights belong in the gitignored shared cache."""
    path = weights_path(DETECTORS[PRIMARY_DETECTOR], models_dir=Path("M"))
    assert path == Path("M") / "detectors" / DETECTORS[PRIMARY_DETECTOR].weights


def test_every_probe_the_issue_names_has_an_image_to_prove_it_on():
    """Step 4: `person`, an open-vocabulary concept, and a non-COCO animal."""
    concepts = {probe.concept for probe in CONCEPT_PROBES}
    assert {"person", "red mug", "dog"} <= concepts
    for probe in CONCEPT_PROBES:
        assert probe.image_url.startswith("https://")


def test_only_the_open_vocabulary_probe_is_beyond_a_coco_detector():
    """The 80-noun cap, stated as data rather than as prose."""
    beyond = {probe.concept for probe in CONCEPT_PROBES if probe.coco_equivalent is None}
    assert beyond == {"red mug"}


# --- amortising one detect over the detect cadence ---------------------------

def test_a_detect_amortises_over_the_frames_between_detects():
    assert amortised_ms(13.5, cadence=3) == pytest.approx(4.5)
    assert amortised_ms(13.5, cadence=1) == pytest.approx(13.5)


def test_a_cadence_below_one_is_refused_rather_than_dividing_by_zero():
    with pytest.raises(ValueError):
        amortised_ms(13.5, cadence=0)


def test_the_cadence_a_detector_would_need_to_fit():
    """Smallest whole number of frames per detect that lands inside the budget."""
    assert required_cadence(13.5, budget_max_ms=8.0) == 2
    assert required_cadence(24.0, budget_max_ms=8.0) == 3
    assert required_cadence(4.0, budget_max_ms=8.0) == 1


# --- the budget verdict ------------------------------------------------------

def test_a_detector_inside_the_budget_at_the_named_cadence_fits():
    verdict = budget_verdict(13.5, cadence=3)
    assert verdict.fits is True
    assert verdict.amortised_ms == pytest.approx(4.5)
    assert verdict.budget_min_ms == DETECT_BUDGET_MIN_MS
    assert verdict.budget_max_ms == DETECT_BUDGET_MAX_MS
    assert verdict.cadence_required == 2


def test_a_detector_outside_the_budget_says_so_and_names_the_cadence_it_needs():
    verdict = budget_verdict(60.0, cadence=3)
    assert verdict.fits is False
    assert verdict.cadence_required == 8
    assert "every 8th frame" in verdict.statement


def test_the_verdict_always_names_the_cadence_it_was_judged_at():
    """The issue's gate: a verdict without its cadence is not a verdict."""
    for ms in (2.0, 13.5, 60.0):
        assert f"every {DEFAULT_CADENCE}rd frame" in budget_verdict(ms, DEFAULT_CADENCE).statement


# --- does a vocabulary change touch the frame path? --------------------------

def test_a_vocabulary_change_off_the_frame_path_leaves_the_detect_latency_alone():
    verdict = frame_path_verdict(before_ms=13.40, after_ms=13.52, change_ms=66.0)
    assert verdict.unaffected is True
    assert verdict.free_on_frame_path is True
    assert verdict.delta_ms == pytest.approx(0.12)
    assert "cold path" in verdict.statement


def test_a_vocabulary_change_that_moves_the_detect_latency_is_flagged():
    verdict = frame_path_verdict(before_ms=13.4, after_ms=19.0, change_ms=66.0)
    assert verdict.unaffected is False
    assert "13.40" in verdict.statement and "19.00" in verdict.statement


def test_the_tolerance_is_a_fraction_of_the_measured_latency_not_an_absolute():
    """A 5% wobble on a 13 ms detect is noise; on a 130 ms one it is 6.5 ms."""
    assert frame_path_verdict(13.4, 14.0, 66.0, tolerance=0.05).unaffected is True
    assert frame_path_verdict(13.4, 14.5, 66.0, tolerance=0.05).unaffected is False


# --- the recommendation ------------------------------------------------------

def a_candidate(**overrides) -> Candidate:
    fields = dict(name="yolo-world-s-640", ms_per_detect=13.5, open_vocabulary=True,
                  concepts_resolved=3, concepts_probed=3)
    fields.update(overrides)
    return Candidate(**fields)


def test_the_open_vocabulary_candidate_wins_even_though_it_is_slower():
    """Step 5: rank, do not compare absolutes. A closed vocabulary caps the product
    at 80 nouns however fast it is, and the prompt compiler that used to bridge that
    gap is cut from v1."""
    floor = a_candidate(name="yolov8n-640", ms_per_detect=5.0, open_vocabulary=False,
                        concepts_resolved=2)
    outcome = recommend([a_candidate(), floor])

    assert outcome.name == "yolo-world-s-640"
    assert outcome.ranking == ("yolov8n-640", "yolo-world-s-640"), "ranking is by speed"
    assert outcome.fits is True
    assert "80" in outcome.reason


def test_an_open_vocabulary_candidate_that_is_also_the_fastest_is_not_told_it_is_slower():
    """The reason is prose that goes into spec 8.1, so it has to be true of the run
    that produced it - `It is not the fastest` is a claim, not a turn of phrase."""
    outcome = recommend([a_candidate(ms_per_detect=5.0),
                         a_candidate(name="yolov8n-640", ms_per_detect=9.0,
                                     open_vocabulary=False, concepts_resolved=2)])

    assert outcome.name == "yolo-world-s-640"
    assert outcome.ranking[0] == "yolo-world-s-640"
    assert "not the fastest" not in outcome.reason
    assert "80" in outcome.reason, "the vocabulary cap is still the criterion"


def test_an_open_vocabulary_candidate_that_needs_a_slower_cadence_still_wins_but_says_so():
    slow = a_candidate(ms_per_detect=40.0)
    outcome = recommend([slow, a_candidate(name="yolov8n-640", ms_per_detect=5.0,
                                           open_vocabulary=False, concepts_resolved=2)])

    assert outcome.name == "yolo-world-s-640"
    assert outcome.fits is False
    assert "every 5th frame" in outcome.reason


def test_a_candidate_that_cannot_resolve_what_was_asked_of_it_is_not_recommended():
    """Open vocabulary that resolves nothing is not open vocabulary in practice."""
    blind = a_candidate(concepts_resolved=1)
    floor = a_candidate(name="yolov8n-640", ms_per_detect=5.0, open_vocabulary=False,
                        concepts_resolved=2)
    outcome = recommend([blind, floor])

    assert outcome.name == "yolov8n-640"
    assert "did not resolve" in outcome.reason


def test_recommending_from_nothing_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        recommend([])


def test_the_two_figures_can_be_interleaved_so_a_drifting_clock_cancels():
    """The trap this walked into once: on this laptop the SM clock climbs for the
    first seconds of a run, so a plain before/after reported a 24% *speed-up* across
    a vocabulary change. The two vocabularies are therefore run in alternating blocks
    and each arm pooled over all of its detects, which puts the drift on both arms."""
    verdict = frame_path_verdict(13.4, 13.5, 66.0, passes=(14.0, 13.5, 12.8))

    assert verdict.passes == (14.0, 13.5, 12.8)
    assert "alternating blocks" in verdict.statement
    assert verdict.unaffected is True


def test_an_unbracketed_verdict_says_nothing_about_passes():
    assert frame_path_verdict(13.4, 13.5, 66.0).passes == ()


def test_the_first_detect_after_a_change_is_judged_apart_from_the_steady_state():
    """The finding this measurement turned up: `YOLOWorld.set_classes` drops the
    predictor, so the next `predict` rebuilds it and costs ~100 ms more than a steady
    detect. Steady state is untouched; a frame is not. Averaging the two would have
    reported the whole thing as noise."""
    verdict = frame_path_verdict(15.60, 15.80, 10.5, first_detect_ms=124.70)

    assert verdict.unaffected is True, "the steady state really is unaffected"
    assert verdict.free_on_frame_path is False, "but one frame pays for the change"
    assert verdict.first_detect_overhead_ms == pytest.approx(109.10)
    assert "first" in verdict.statement and "re-warm" in verdict.statement.lower()


def test_a_first_detect_inside_the_tolerance_leaves_the_claim_intact():
    verdict = frame_path_verdict(15.60, 15.80, 10.5, first_detect_ms=16.20)
    assert verdict.free_on_frame_path is True
    assert verdict.first_detect_overhead_ms == pytest.approx(0.60)


def test_a_verdict_with_no_first_detect_measurement_does_not_invent_one():
    verdict = frame_path_verdict(15.60, 15.80, 10.5)
    assert verdict.first_detect_ms is None
    assert verdict.first_detect_overhead_ms is None
    assert verdict.free_on_frame_path is True
