"""The seed policy without a device. Issue #32, spec 8.5, merge-gate tier.

`seeding.py` is on the frame loop's import path, so torch lives inside the methods
that touch a tensor and everything decidable without one is decidable here: which
policy a plan asks for, which seed a track gets, and where on the latent canvas a
region lands. The arithmetic on the noise itself needs a device and lives in
`tests/test_gpu_seeding.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import seeding
from detection import Box, Track, Tracks
from region_scheduler import RegionScheduler
from render_plan import FIXED, PER_TRACK, RANDOM, SEED_POLICIES, validate_plan
from seeding import (
    DEFAULT_BASE_SEED,
    LATENT_SCALE,
    NoiseField,
    latent_box,
    noise_tensor,
    seed_for_track,
)

W = H = 64
SOURCE = Path(seeding.__file__)


def plan_of(seed_policy=FIXED, concept="person"):
    result = validate_plan({"source_prompt": "wet denim",
                            "targets": [{"id": "t0", "concept": concept,
                                         "region": "full_box", "box_scale": 1.0,
                                         "seed_policy": seed_policy}]})
    assert result.plan is not None, result.reason
    return result.plan


def selection_of(*boxes, plan=None):
    tracks = Tracks(tracks=tuple(
        Track(track_id=index, box=Box(*box), concept="person", confidence=0.9)
        for index, box in enumerate(boxes)), ticks=1)
    return RegionScheduler().select(tracks, plan or plan_of(), W, H)


# --- the tier split ---------------------------------------------------------


def test_torch_is_imported_inside_the_functions_that_use_it():
    """The frame loop imports this module and so does the GUI process, which never
    touches a GPU; a torch import at module scope would take it out of the merge
    gate's tier the way it takes `wrapper.py` out."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    top_level = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {alias.name for node in top_level if isinstance(node, ast.Import)
             for alias in node.names}
    names |= {node.module for node in top_level if isinstance(node, ast.ImportFrom)}
    assert "torch" not in names


# --- the seed a track gets --------------------------------------------------


def test_a_track_gets_the_same_seed_every_time_it_is_asked_for():
    """The whole point: an object's noise is a function of its identity, so two
    frames holding the same track ask for the same field."""
    assert seed_for_track(7) == seed_for_track(7)


def test_two_tracks_get_different_seeds():
    seeds = {seed_for_track(track_id) for track_id in range(64)}
    assert len(seeds) == 64


def test_the_base_seed_moves_every_track_together():
    """So a run can be re-seeded as a whole - the engine's own `seed` box - without
    two tracks colliding."""
    assert seed_for_track(3, base_seed=1) != seed_for_track(3, base_seed=2)
    assert seed_for_track(3, base_seed=1) != seed_for_track(4, base_seed=1)


def test_the_seed_does_not_depend_on_this_process():
    """`hash()` is salted per process, so a seed built on one would give an object a
    different noise field on every launch - which is the failure this pins."""
    assert seed_for_track(11, base_seed=2) == 9199772856589933275


def test_a_seed_is_a_non_negative_int_torch_will_accept():
    for track_id in (0, 1, 999, 2 ** 31):
        seed = seed_for_track(track_id)
        assert isinstance(seed, int) and 0 <= seed < 2 ** 63


# --- where a region lands on the latent canvas ------------------------------


def test_a_region_maps_onto_the_latent_canvas_by_the_vae_scale():
    assert LATENT_SCALE == 8
    assert latent_box(Box(16, 24, 48, 64), 8, 8) == (2, 3, 6, 8)


def test_a_latent_box_is_clipped_to_the_canvas():
    assert latent_box(Box(-40, -40, 900, 900), 8, 8) == (0, 0, 8, 8)


def test_a_region_smaller_than_one_latent_cell_still_covers_one():
    """A sub-cell region rounds outwards rather than to nothing: the frame paints
    it, so the noise under it has to be the track's."""
    assert latent_box(Box(9, 9, 12, 12), 8, 8) == (1, 1, 2, 2)


def test_a_region_off_the_canvas_has_no_latent_box():
    assert latent_box(Box(200, 200, 300, 300), 8, 8) is None


# --- which policy a frame renders under -------------------------------------


@pytest.mark.parametrize("policy", SEED_POLICIES)
def test_the_field_follows_the_honoured_target_s_policy(policy):
    field = NoiseField()
    field.follow(plan_of(policy))
    assert field.policy == policy


def test_a_plan_with_no_target_leaves_the_prepared_field_alone():
    """No target is no selective render, and the field the engine was prepared with
    is what this app has always run on."""
    result = validate_plan({"source_prompt": "wet denim"})
    field = NoiseField()
    field.follow(result.plan)
    assert field.policy == FIXED


def test_a_policy_change_drops_the_cached_track_fields():
    """A track's field is drawn under one base seed and one policy; carrying it
    across a change would render the new policy through the old one's noise."""
    field = NoiseField()
    field._fields[3] = object()
    field.follow(plan_of(PER_TRACK))
    assert field._fields == {}


def test_following_the_same_policy_twice_keeps_the_cache():
    field = NoiseField()
    field.follow(plan_of(PER_TRACK))
    sentinel = object()
    field._fields[3] = sentinel
    field.follow(plan_of(PER_TRACK))
    assert field._fields[3] is sentinel


# --- what it does without an engine to write to -----------------------------


class _Pipeline:
    def __init__(self, init_noise=None):
        self.init_noise = init_noise


class _Wrapper:
    def __init__(self, stream):
        self.stream = stream


def test_the_noise_field_is_found_through_the_wrapper_or_on_it():
    """The frame loop holds a `StreamDiffusionWrapper` and the bench holds the same
    object; a test holds a stand-in. One accessor, so none of them is a special
    case."""
    pipeline = _Pipeline(init_noise="noise")
    assert noise_tensor(pipeline) == "noise"
    assert noise_tensor(_Wrapper(pipeline)) == "noise"


def test_an_engine_with_no_noise_field_is_not_an_error():
    """`apply` runs on the frame path. An engine that does not expose its noise is
    a policy that cannot be honoured, not a frame that fails to render."""
    field = NoiseField()
    field.follow(plan_of(PER_TRACK))
    assert field.apply(_Pipeline(None), selection_of((8, 8, 40, 40))) is False


def test_fixed_writes_nothing_at_all():
    """Today's behaviour is a field nobody touches, so the default policy costs the
    frame path not one tensor operation."""
    field = NoiseField()
    field.follow(plan_of(FIXED))
    assert field.apply(_Pipeline(None), selection_of((8, 8, 40, 40))) is False


def test_the_base_seed_is_the_engine_s_own():
    assert DEFAULT_BASE_SEED == 2


@pytest.mark.parametrize("policy", (PER_TRACK, RANDOM))
def test_a_policy_that_writes_says_so(policy):
    assert NoiseField(policy=policy).writes is True


def test_fixed_says_it_writes_nothing():
    assert NoiseField(policy=FIXED).writes is False
