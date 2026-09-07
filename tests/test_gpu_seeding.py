"""The seed policy on a device. Issue #32, spec 8.5, GPU tier.

`tests/test_seeding.py` holds everything decidable without one - the seed a track
gets, where a region lands on the latent canvas, which policy a plan asks for. What
needs a device is the noise itself, and three claims are made about it:

- **`fixed` is what the engine was prepared with**, so the default policy leaves
  the field untouched and the frame path pays nothing for it.
- **`per_track` pins a realisation to identity and moves it with the object.** The
  same track in two positions reads the same noise; two tracks read different noise;
  and nothing outside a rendered region moves at all.
- **A policy change is a runtime write.** Shape, dtype and device are preserved, so
  nothing TensorRT keys an engine on is touched - a seed edit cannot cost a rebuild.

No engine and no detector: the noise field is a plain tensor, so a stand-in with the
shape `prepare()` gives it says everything an engine would, in a second.
"""

import numpy as np
import pytest

from detection import Box, Track, Tracks
from region_scheduler import RegionScheduler
from render_plan import FIXED, PER_TRACK, RANDOM
from seeding import NoiseField, latent_box

pytestmark = pytest.mark.gpu

# The shipped canvas: 512x512 through a VAE that downscales by 8.
LATENT = 64
CANVAS = LATENT * 8


@pytest.fixture(scope="module")
def torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch


class Pipeline:
    """What `NoiseField` writes to: the one tensor `StreamDiffusion.prepare` draws.

    A stand-in rather than an engine, because the claim is about the tensor and the
    engine only owns it. `tests/test_gpu_selective_render.py` runs the real one.
    """

    def __init__(self, torch, dtype=None):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(2)
        self.init_noise = torch.randn(
            (1, 4, LATENT, LATENT), generator=generator, device="cuda",
            dtype=torch.float16 if dtype is None else dtype)


def selection_of(*boxes, ids=None):
    """A selection over `boxes`, with `ids` when a test needs two frames to hold the
    same track in different places."""
    track_ids = range(len(boxes)) if ids is None else ids
    tracks = Tracks(tracks=tuple(
        Track(track_id=track_id, box=Box(*box), concept="person", confidence=0.9)
        for track_id, box in zip(track_ids, boxes)), ticks=1)
    from render_plan import validate_plan

    plan = validate_plan({"source_prompt": "wet denim",
                          "targets": [{"id": "t0", "concept": "person",
                                       "region": "full_box", "box_scale": 1.0}]}).plan
    return RegionScheduler().select(tracks, plan, CANVAS, CANVAS)


def field_of(policy, torch):
    noise = NoiseField(policy=policy)
    return noise, Pipeline(torch)


def cells(box):
    return latent_box(Box(*box), LATENT, LATENT)


# --- fixed: the field the engine was prepared with ---------------------------


def test_fixed_leaves_the_prepared_field_exactly_as_it_was(torch):
    noise, pipeline = field_of(FIXED, torch)
    before = pipeline.init_noise.clone()
    assert noise.apply(pipeline, selection_of((64, 64, 320, 320))) is False
    assert torch.equal(pipeline.init_noise, before)


def test_a_policy_that_wrote_is_undone_when_the_plan_goes_back_to_fixed(torch):
    """A plan swap back to today's behaviour has to be today's behaviour, not
    whatever the last frame of the other policy left in the field."""
    noise, pipeline = field_of(PER_TRACK, torch)
    prepared = pipeline.init_noise.clone()
    noise.apply(pipeline, selection_of((64, 64, 320, 320)))
    assert not torch.equal(pipeline.init_noise, prepared)

    noise.policy = FIXED
    assert noise.apply(pipeline, selection_of((64, 64, 320, 320))) is True
    assert torch.equal(pipeline.init_noise, prepared)
    # And having restored it, it stops writing.
    assert noise.apply(pipeline, selection_of((64, 64, 320, 320))) is False


# --- per_track: one realisation per identity, moved with the object ----------


def test_per_track_writes_only_the_latent_cells_a_region_covers(torch):
    """Outside every region the field is the prepared one: those latents produce
    pixels the composite discards, and rewriting them would be noise nobody sees
    bought at the cost of a claim nobody can check."""
    noise, pipeline = field_of(PER_TRACK, torch)
    prepared = pipeline.init_noise.clone()
    box = (80, 80, 240, 240)
    noise.apply(pipeline, selection_of(box))

    x0, y0, x1, y1 = cells(box)
    written = pipeline.init_noise
    assert not torch.equal(written[..., y0:y1, x0:x1], prepared[..., y0:y1, x0:x1])
    outside = torch.ones((LATENT, LATENT), dtype=torch.bool, device="cuda")
    outside[y0:y1, x0:x1] = False
    assert torch.equal(written[..., outside], prepared[..., outside])


def test_one_track_reads_the_same_noise_wherever_it_has_moved_to(torch):
    """The whole of what a per-track seed can mean under the shipped primitive: the
    realisation is pinned to the identity and rolled to the object's centre, so an
    object that translates carries its noise with it."""
    noise, pipeline = field_of(PER_TRACK, torch)
    box, moved = (80, 80, 208, 208), (144, 144, 272, 272)

    noise.apply(pipeline, selection_of(box, ids=[7]))
    x0, y0, x1, y1 = cells(box)
    first = pipeline.init_noise[..., y0:y1, x0:x1].clone()

    noise.apply(pipeline, selection_of(moved, ids=[7]))
    mx0, my0, mx1, my1 = cells(moved)
    second = pipeline.init_noise[..., my0:my1, mx0:mx1]

    assert first.shape == second.shape
    assert torch.equal(first, second)


def test_two_tracks_in_the_same_place_read_different_noise(torch):
    """Identity, not position - otherwise the field is the canvas-pinned one under
    another name."""
    noise, pipeline = field_of(PER_TRACK, torch)
    box = (80, 80, 208, 208)
    x0, y0, x1, y1 = cells(box)

    noise.apply(pipeline, selection_of(box, ids=[1]))
    first = pipeline.init_noise[..., y0:y1, x0:x1].clone()
    noise.apply(pipeline, selection_of(box, ids=[2]))
    assert not torch.equal(first, pipeline.init_noise[..., y0:y1, x0:x1])


def test_the_same_track_and_the_same_place_is_the_same_field_twice(torch):
    """Between two detector ticks the frame loop reads one `Tracks` snapshot, so
    consecutive frames must produce identical noise - a redraw here would be
    exactly the boiling this lever exists to remove."""
    noise, pipeline = field_of(PER_TRACK, torch)
    selection = selection_of((80, 80, 208, 208), ids=[5])
    noise.apply(pipeline, selection)
    first = pipeline.init_noise.clone()
    noise.apply(pipeline, selection)
    assert torch.equal(pipeline.init_noise, first)


def test_a_frame_with_no_region_is_the_prepared_field(torch):
    noise, pipeline = field_of(PER_TRACK, torch)
    prepared = pipeline.init_noise.clone()
    noise.apply(pipeline, selection_of((80, 80, 208, 208)))
    noise.apply(pipeline, selection_of())
    assert torch.equal(pipeline.init_noise, prepared)


def test_a_track_field_is_drawn_once_and_kept(torch):
    noise, pipeline = field_of(PER_TRACK, torch)
    noise.apply(pipeline, selection_of((80, 80, 208, 208), ids=[3]))
    held = noise._fields[3]
    noise.apply(pipeline, selection_of((96, 96, 224, 224), ids=[3]))
    assert noise._fields[3] is held


# --- random: the control -----------------------------------------------------


def test_random_redraws_the_whole_field_every_frame(torch):
    """Every latent cell, not just the ones a region covers - which is what makes
    it the upper bound the other two policies are read against."""
    box = (80, 80, 208, 208)
    noise, pipeline = field_of(RANDOM, torch)
    noise.apply(pipeline, selection_of(box))
    first = pipeline.init_noise.clone()
    noise.apply(pipeline, selection_of(box))

    x0, y0, x1, y1 = cells(box)
    outside = torch.ones((LATENT, LATENT), dtype=torch.bool, device="cuda")
    outside[y0:y1, x0:x1] = False
    assert not torch.equal(pipeline.init_noise, first)
    assert not torch.equal(pipeline.init_noise[..., outside], first[..., outside])


def test_random_is_repeatable_across_two_runs_of_the_same_policy(torch):
    """A control that could not be re-run would not be one."""
    selection = selection_of((80, 80, 208, 208))
    outputs = []
    for _ in range(2):
        noise, pipeline = field_of(RANDOM, torch)
        noise.apply(pipeline, selection)
        outputs.append(pipeline.init_noise.clone())
    assert torch.equal(outputs[0], outputs[1])


# --- nothing here keys a TensorRT engine -------------------------------------


@pytest.mark.parametrize("policy", (PER_TRACK, RANDOM))
def test_a_written_field_keeps_its_shape_dtype_and_device(torch, policy):
    """The issue's fourth trap. TensorRT keys an engine on resolution, batch size,
    step count and fused LoRAs; the noise is a plain tensor `add_noise` reads in
    Python, and writing it in place cannot move any of them."""
    noise, pipeline = field_of(policy, torch)
    before = pipeline.init_noise
    noise.apply(pipeline, selection_of((80, 80, 208, 208)))
    after = pipeline.init_noise
    assert after is before, "the field was replaced rather than written in place"
    assert (after.shape, after.dtype, after.device) == (
        before.shape, before.dtype, before.device)


def test_the_noise_stays_the_dtype_the_engine_was_built_for(torch):
    """fp16 in, fp16 out: a field that came back as fp32 would be a different
    tensor to `add_noise` and a cast on the frame path."""
    noise = NoiseField(policy=PER_TRACK)
    pipeline = Pipeline(torch, dtype=torch.float16)
    noise.apply(pipeline, selection_of((80, 80, 208, 208)))
    assert pipeline.init_noise.dtype == torch.float16


def test_the_written_field_is_finite_and_of_the_expected_scale(torch):
    """It is standard normal noise the scheduler scales by `beta_prod_t_sqrt`; a
    field an order of magnitude off would not flicker, it would be a bad frame."""
    noise, pipeline = field_of(PER_TRACK, torch)
    noise.apply(pipeline, selection_of((80, 80, 400, 400)))
    written = pipeline.init_noise.float().cpu().numpy()
    assert np.isfinite(written).all()
    assert 0.5 < float(np.std(written)) < 2.0
