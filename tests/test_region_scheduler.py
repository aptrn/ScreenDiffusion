"""C5, the region scheduler (issue #8): which regions this frame renders.

Every rule the Gate names is here and none of it needs a GPU: the selection
policy, the round-robin bound - "with more tracks than slots, every track updates
within ceil(N/K) frames" - the minimum-size floor and its skipped count, and the
band arithmetic that turns a detected box into the region a target asked for.

The band arithmetic is also held against `bench.primitives.region_box`, which
computed the committed comparison's regions in issue #5. Two implementations of one
rule exist because the harness may import numpy and shipped code may not; a test is
what keeps them one rule.
"""

import math

import pytest

from detection import Box, Track, Tracks
from region_scheduler import (
    MIN_REGION_PX,
    EMPTY_SELECTION,
    RegionScheduler,
    dilate_box,
    region_box,
    selection_status,
    slots_for,
)
from render_plan import GLOBAL, INVERSE, SELECTIVE, validate_plan

FRAME_W, FRAME_H = 512, 512


def plan_of(**changes):
    """A validated plan with one `person` target, overridden as asked."""
    target = {"id": "t0", "concept": "person", "region": "lower_half",
              "box_scale": 1.0, "prompt": "wet denim"}
    target.update(changes.pop("target", {}))
    raw = {"source_prompt": "wet denim", "targets": [target]}
    raw.update(changes)
    result = validate_plan(raw)
    assert result.plan is not None, result.reason
    return result.plan


def tracks_of(*boxes, concept="person", start_id=0):
    return Tracks(
        tracks=tuple(Track(track_id=start_id + index, box=Box(*box), concept=concept,
                           confidence=0.9)
                     for index, box in enumerate(boxes)),
        ticks=1,
    )


def person(index, width=30, height=200):
    """A person-shaped box, `index` of them side by side and clear of each other.

    Narrow enough that a dozen of them fit inside the 512 px canvas: a box the
    frame clips to a sliver is skipped by the size floor, which is a different
    rule from the one most of these tests are about.
    """
    left = 5 + index * (width + 10)
    return Box(left, 100, left + width, 100 + height)


# --- what a region is -------------------------------------------------------


@pytest.mark.parametrize("region,expected", [
    ("full_box", (0, 300)),
    ("upper_third", (0, 100)),
    ("upper_half", (0, 150)),
    ("center", (100, 200)),
    ("lower_half", (150, 300)),
    ("lower_third", (200, 300)),
])
def test_a_region_names_a_horizontal_band_of_the_box(region, expected):
    banded = region_box(Box(10, 0, 70, 300), region)
    assert (banded.y0, banded.y1) == expected
    assert (banded.x0, banded.x1) == (10, 70), "a band never narrows the box"


def test_a_band_of_a_one_pixel_box_is_still_a_region():
    """Never empty: a region that rounded away would drop the object silently."""
    assert region_box(Box(0, 0, 4, 1), "lower_third").height >= 1


def test_the_bands_agree_with_the_harness_that_measured_them():
    """`bench.primitives.region_box` cut the regions issue #5's comparison rendered."""
    from bench.primitives import REGIONS as BENCH_REGIONS
    from bench.primitives import region_box as bench_region_box

    boxes = [Box(0, 0, 100, 300), Box(17, 5, 63, 148), Box(4, 4, 5, 5)]
    for region in BENCH_REGIONS:
        for box in boxes:
            assert tuple(region_box(box, region)) == tuple(bench_region_box(box, region))


def test_an_unknown_region_is_a_programming_error_not_a_default():
    with pytest.raises(ValueError):
        region_box(Box(0, 0, 10, 10), "upper_thrid")


def test_box_scale_dilates_around_the_centre():
    dilated = dilate_box(Box(100, 100, 200, 200), 1.2)
    assert tuple(dilated) == (90, 90, 210, 210)


def test_a_scale_of_one_changes_nothing():
    assert tuple(dilate_box(Box(3, 7, 11, 29), 1.0)) == (3, 7, 11, 29)


# --- selection --------------------------------------------------------------


def test_a_global_plan_selects_nothing_and_says_so():
    """`mode: global` is the whole frame; there is no region to schedule."""
    scheduler = RegionScheduler()
    selection = scheduler.select(tracks_of(person(0)), validate_plan({}).plan,
                                 FRAME_W, FRAME_H)
    assert selection.mode == GLOBAL
    assert selection.regions == ()
    assert selection.candidates == 0


def test_each_track_of_a_targeted_concept_becomes_one_region():
    scheduler = RegionScheduler()
    selection = scheduler.select(tracks_of(person(0), person(1)), plan_of(),
                                 FRAME_W, FRAME_H)
    assert selection.mode == SELECTIVE
    assert selection.count == 2
    assert selection.track_ids == (0, 1)
    # lower_half of a box 100..300 high
    assert [box.y0 for box in selection.boxes] == [200, 200]


def test_a_track_of_a_concept_no_target_asked_for_is_not_rendered():
    """The detector's vocabulary is the plan's, but a stale snapshot outlives a
    plan change by a tick - and those boxes are about a plan nobody is rendering."""
    scheduler = RegionScheduler()
    mixed = Tracks(tracks=tracks_of(person(0)).tracks
                   + tracks_of(person(1), concept="dog", start_id=9).tracks, ticks=1)
    selection = scheduler.select(mixed, plan_of(), FRAME_W, FRAME_H)
    assert selection.track_ids == (0,)


def test_a_region_is_clamped_to_the_frame():
    scheduler = RegionScheduler()
    off_frame = tracks_of(Box(-40, 400, 120, 900))
    selection = scheduler.select(off_frame, plan_of(target={"region": "full_box"}),
                                 FRAME_W, FRAME_H)
    box, = selection.boxes
    assert (box.x0, box.y0) == (0, 400)
    assert (box.x1, box.y1) == (120, FRAME_H)


def test_inverse_mode_still_selects_the_targets():
    """Which pixels are *protected* is the same question; the compositor inverts."""
    scheduler = RegionScheduler()
    selection = scheduler.select(tracks_of(person(0)), plan_of(mode=INVERSE),
                                 FRAME_W, FRAME_H)
    assert selection.mode == INVERSE
    assert selection.count == 1


def test_no_tracks_yet_is_a_selection_with_nothing_in_it():
    scheduler = RegionScheduler()
    selection = scheduler.select(Tracks(), plan_of(), FRAME_W, FRAME_H)
    assert selection.count == 0
    assert selection.candidates == 0
    assert selection.mode == SELECTIVE


def test_the_empty_selection_is_the_before_anything_state():
    assert EMPTY_SELECTION.count == 0
    assert EMPTY_SELECTION.boxes == ()


# --- the size floor ---------------------------------------------------------


def test_a_region_under_the_floor_is_skipped_and_counted():
    """The issue's first trap: too small to render is skipped, and *recorded*."""
    scheduler = RegionScheduler()
    tiny = tracks_of(Box(10, 10, 10 + MIN_REGION_PX - 1, 200))
    selection = scheduler.select(tiny, plan_of(target={"region": "full_box"}),
                                 FRAME_W, FRAME_H)
    assert selection.count == 0
    assert selection.skipped_small == 1
    assert scheduler.skipped_small_total == 1


def test_a_region_exactly_at_the_floor_is_rendered():
    scheduler = RegionScheduler()
    just_big_enough = tracks_of(Box(10, 10, 10 + MIN_REGION_PX, 200))
    selection = scheduler.select(just_big_enough,
                                 plan_of(target={"region": "full_box"}),
                                 FRAME_W, FRAME_H)
    assert selection.count == 1
    assert selection.skipped_small == 0


def test_a_skipped_region_does_not_consume_a_slot():
    """It is filtered before the slots are handed out, so a permanently tiny
    object cannot starve the round-robin of the objects that can be rendered."""
    scheduler = RegionScheduler()
    plan = plan_of(target={"region": "full_box", "max_instances": 1})
    mixed = tracks_of(Box(0, 0, 4, 4), person(1))
    selection = scheduler.select(mixed, plan, FRAME_W, FRAME_H)
    assert selection.skipped_small == 1
    assert selection.track_ids == (1,)


# --- K slots and the round robin --------------------------------------------


def test_k_comes_from_the_honoured_targets_max_instances():
    assert slots_for(plan_of(target={"max_instances": 3})) == 3


def test_at_most_k_regions_are_selected():
    scheduler = RegionScheduler()
    plan = plan_of(target={"max_instances": 2})
    selection = scheduler.select(tracks_of(*(person(i) for i in range(5))), plan,
                                 FRAME_W, FRAME_H)
    assert selection.count == 2
    assert selection.slots == 2
    assert selection.candidates == 5
    assert selection.deferred == 3


def test_every_track_is_updated_within_ceil_n_over_k_frames():
    """The Gate's second item, as arithmetic rather than as a promise."""
    for objects, slots in [(5, 2), (6, 6), (7, 3), (9, 4), (4, 1)]:
        scheduler = RegionScheduler()
        plan = plan_of(target={"max_instances": slots})
        tracks = tracks_of(*(person(index) for index in range(objects)))
        bound = math.ceil(objects / slots)
        served = set()
        for _ in range(bound):
            served.update(scheduler.select(tracks, plan, FRAME_W, FRAME_H).track_ids)
        assert served == set(range(objects)), (
            f"{objects} objects over {slots} slots left "
            f"{sorted(set(range(objects)) - served)} unrendered after {bound} frames")


def test_the_round_robin_keeps_going_round():
    """Frame ceil(N/K)+1 starts the next lap rather than stopping."""
    scheduler = RegionScheduler()
    plan = plan_of(target={"max_instances": 2})
    tracks = tracks_of(*(person(index) for index in range(5)))
    laps = [scheduler.select(tracks, plan, FRAME_W, FRAME_H).track_ids
            for _ in range(6)]
    assert laps[0] == (0, 1)
    assert laps[1] == (2, 3)
    assert laps[3] == (1, 2), f"the second lap did not resume where the first left off: {laps}"


def test_nothing_is_deferred_when_every_track_fits():
    scheduler = RegionScheduler()
    plan = plan_of(target={"max_instances": 6})
    for _ in range(3):
        selection = scheduler.select(tracks_of(*(person(i) for i in range(4))), plan,
                                     FRAME_W, FRAME_H)
        assert selection.track_ids == (0, 1, 2, 3)
        assert selection.deferred == 0


def test_a_track_that_appears_mid_rotation_is_not_skipped_forever():
    """Ids are monotonic, so the cursor is an id and not a position: a new object
    joins the rotation where its id says, and the lap still closes."""
    scheduler = RegionScheduler()
    plan = plan_of(target={"max_instances": 2})
    first = tracks_of(*(person(index) for index in range(4)))
    scheduler.select(first, plan, FRAME_W, FRAME_H)
    grown = tracks_of(*(person(index) for index in range(6)))
    served = set()
    for _ in range(3):
        served.update(scheduler.select(grown, plan, FRAME_W, FRAME_H).track_ids)
    assert served == {0, 1, 2, 3, 4, 5}


def test_the_selection_carries_the_plan_it_was_made_for():
    scheduler = RegionScheduler()
    plan = plan_of()
    selection = scheduler.select(tracks_of(person(0)), plan, FRAME_W, FRAME_H)
    assert selection.plan_version == plan.plan_version
    assert selection.regions[0].target_id == "t0"
    assert selection.regions[0].concept == "person"


# --- the readout ------------------------------------------------------------


def test_a_global_plan_says_nothing_about_regions():
    """"0 regions rendered" and "nothing was selective" read the same on a status
    line and are not the same thing - `detection.fps_payload`'s own rule."""
    assert selection_status(EMPTY_SELECTION) == {}


def test_the_status_carries_the_slots_the_waiting_and_the_skipped():
    scheduler = RegionScheduler()
    plan = plan_of(target={"max_instances": 2})
    tracks = tracks_of(Box(0, 0, 4, 4), *(person(index) for index in range(4)))
    status = scheduler.status(scheduler.select(tracks, plan, FRAME_W, FRAME_H))
    assert status["regions"] == 2
    assert status["slots"] == 2
    assert status["deferred"] == 2
    assert status["skipped_small"] == 1
    assert status["skipped_small_total"] == 1
