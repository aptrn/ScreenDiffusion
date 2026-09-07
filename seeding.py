"""The plan's `seed_policy`, applied to the engine's latent noise field.

Issue #32, spec 8.5. `seed_policy` has been in the Render Plan since issue #6 and
nothing read it; this is what reads it. Before anything else, what the field can
mean under the primitive that was actually chosen - the issue's third trap, answered
before it was built rather than after:

**StreamDiffusion draws one noise field, once.** `prepare()` fills `init_noise` from
a seeded generator and `encode_image` adds `init_noise[0]` to every frame's latent
for ever after; with one denoising step nothing ever overwrites it. So the shipped
path's noise is already *fixed and canvas-pinned*, and `fixed` is a name for what
this app has always done rather than a new setting. `random` - a fresh field every
frame - is the thing that does not ship, and it is here as the control: if the
flicker metric cannot tell it from `fixed`, the metric is not measuring noise.

**`per_track` cannot mean one call per track.** Issue #5 chose the full-frame masked
primitive: one diffusion call and one noise field covering all K regions, so a
per-track *strength*, *prompt* or *sampler* is not expressible without a second call,
and no amount of seeding makes it so. What *is* expressible is one field composed of
per-track patches: each track owns a canvas-sized noise realisation drawn from its
own id, and the frame pastes that realisation - rolled to the track's current centre,
so it moves with the object rather than staying pinned to the screen - into the
region the compositor is about to paint. Everything outside every region keeps the
prepared field, because those latents produce pixels the composite discards anyway.

That rolling is the whole of what `per_track` buys and the whole of what it costs.
An object that translates across a static background carries its noise with it,
which is the boiling this lever is aimed at; the static background *inside* its box
gets noise that moves, which is boiling this lever creates. Which one wins is a
measurement, not an argument - `bench/results/stability/` and spec 8.5.

Two constraints on the implementation:

- **A seed change must not rebuild an engine.** `init_noise` is a plain tensor read
  by `add_noise` in Python; TensorRT keys an engine on resolution, batch size, step
  count and fused LoRAs. Shape and dtype are preserved here, so a policy change is a
  runtime write - `tests/test_gpu_seeding.py` holds that.
- **torch is imported inside the methods that touch a tensor.** This module is on
  the frame loop's import path and therefore on the GUI process's, and it has to
  stay importable in the merge gate's GPU-free tier.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from detection import Box
from render_plan import DEFAULT_SEED_POLICY, FIXED, RANDOM

# The VAE's downscale: a 512x512 canvas is a 64x64 latent, and a region's noise is
# the latent cells its pixels land in.
LATENT_SCALE = 8

# The seed the app's own box defaults to (`StreamGUI.seed_var`, and StreamDiffusion's
# own default). A track's seed is derived from it, so re-seeding the run re-seeds
# every track together and two tracks still never collide.
DEFAULT_BASE_SEED = 2

# How many tracks' noise realisations to keep. Track ids are monotonic and never
# reused, so without a cap a long session accumulates one canvas-sized field per
# object that ever appeared. Each is small - 4x64x64 fp16 is 32 KiB at the shipped
# canvas - and the cap is what keeps that a constant rather than a leak.
MAX_CACHED_FIELDS = 64

LatentBox = Tuple[int, int, int, int]


def seed_for_track(track_id: int, base_seed: int = DEFAULT_BASE_SEED) -> int:
    """A stable, process-independent seed for one track under one base seed.

    Hashed rather than arithmetic on the id, so neighbouring ids do not give
    neighbouring - and visibly correlated - noise fields. blake2b rather than
    `hash()`, because `hash()` is salted per process: an object would get a
    different field on every launch, which is the one thing a *pinned* seed must
    not do.
    """
    digest = hashlib.blake2b(f"{int(base_seed)}:{int(track_id)}".encode("utf-8"),
                             digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2 ** 63)


def latent_box(box, width: int, height: int,
               scale: int = LATENT_SCALE) -> Optional[LatentBox]:
    """`box` as latent cells on a `width` x `height` latent canvas, or None.

    Rounds *outwards*: a region a few pixels across still paints a pixel, so the
    noise under it has to be that track's rather than the background field's. None
    when the box lies wholly off the canvas, which the caller reads as "nothing to
    write for this region" rather than as an error.
    """
    x0 = max(0, int(box.x0) // scale)
    y0 = max(0, int(box.y0) // scale)
    x1 = min(width, -(-int(box.x1) // scale))
    y1 = min(height, -(-int(box.y1) // scale))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


@dataclass(frozen=True)
class CanvasGeometry:
    """How a region in captured pixels lands on the engine's canvas (issue #39).

    The selection's boxes are in *capture* coordinates, and since the capture is no
    longer the diffusion canvas they have to be mapped before they can be read as
    latent cells. Two mappings, one expression: under `masked` the whole capture is
    squeezed onto the canvas, and under `crop` the crop box is - so the source
    rectangle is the crop or the frame, and everything else is the same scaling.

    `identity` is the shipped 512x512-onto-512x512 case, where the mapping is a
    no-op and `NoiseField` skips it entirely.
    """

    capture_width: int
    capture_height: int
    canvas_width: int
    canvas_height: int
    crop: Optional[Box] = None

    @property
    def identity(self) -> bool:
        return (self.crop is None and self.capture_width == self.canvas_width
                and self.capture_height == self.canvas_height)

    def region(self, box: Box) -> Box:
        """`box`, in captured pixels, as the canvas pixels the engine diffuses it at."""
        if self.identity:
            return Box(*box)
        box = Box(*box)
        source = (Box(0, 0, self.capture_width, self.capture_height)
                  if self.crop is None else Box(*self.crop))
        scale_x = self.canvas_width / max(1, source.width)
        scale_y = self.canvas_height / max(1, source.height)
        return Box(int(round((box.x0 - source.x0) * scale_x)),
                   int(round((box.y0 - source.y0) * scale_y)),
                   int(round((box.x1 - source.x0) * scale_x)),
                   int(round((box.y1 - source.y0) * scale_y)))


def noise_tensor(stream):
    """The engine's latent noise field, whichever object the caller is holding.

    The frame loop and the bench hold a `StreamDiffusionWrapper`, whose `.stream` is
    the `StreamDiffusion` that owns `init_noise`; a test holds the pipeline itself.
    One accessor so none of the three is a special case, and `None` when there is no
    such field - an engine that does not expose its noise is a policy that cannot be
    honoured, not a frame that fails to render.
    """
    inner = getattr(stream, "stream", stream)
    return getattr(inner, "init_noise", None)


class NoiseField:
    """The frame loop's end of `seed_policy`: a plan and a selection in, noise out.

    Held across frames, like the scheduler's rotation cursor and the compositor's
    alpha cache, because that is where the state is: the field the engine was
    prepared with, one noise realisation per live track, and the generator `random`
    draws from.
    """

    def __init__(self, policy: str = DEFAULT_SEED_POLICY,
                 base_seed: int = DEFAULT_BASE_SEED) -> None:
        self.policy = policy
        self.base_seed = int(base_seed)
        # The field the engine was prepared with, cloned on the first frame that
        # would overwrite it. It is what `per_track` composites onto and what
        # restores `fixed` after another policy has written.
        self._prepared = None
        self._fields: Dict[int, object] = {}
        self._generator = None
        self._dirty = False

    @property
    def writes(self) -> bool:
        """Does this policy touch the noise at all? `fixed` is the field as prepared."""
        return self.policy != FIXED

    def follow(self, plan) -> None:
        """Take the plan's policy. Called where the frame loop binds a changed plan.

        A track's realisation is drawn under one policy and one base seed, so a
        change drops them: carrying them over would render the new policy through
        the old one's noise.
        """
        policy = plan.effective_seed_policy
        if policy != self.policy:
            self.policy = policy
            self._fields.clear()

    def reset(self) -> None:
        """Forget everything held about an engine - after one is rebuilt or swapped."""
        self._prepared = None
        self._fields.clear()
        self._generator = None
        self._dirty = False

    def apply(self, stream, selection,
              geometry: Optional[CanvasGeometry] = None) -> bool:
        """Build this frame's noise field. True when the engine's noise was written.

        Runs on the frame path, before the diffusion call the field feeds, and does
        nothing at all under `fixed` until some other policy has dirtied the field -
        so the default costs the loop no tensor operation.

        `geometry` says how the selection's captured-pixel boxes land on the
        engine's canvas (issue #39). None is the identity, which is what the
        512x512 capture the app shipped with means and what every committed
        stability arm was measured under.
        """
        if not self.writes and not self._dirty:
            return False
        noise = noise_tensor(stream)
        if noise is None:
            return False
        if self._prepared is None:
            self._prepared = noise.detach().clone()
        if not self.writes:
            noise.copy_(self._prepared)
            self._dirty = False
            return True
        # `writes` has already excluded `fixed`, and `validate_plan` admits no
        # fourth policy, so these two are the whole vocabulary here.
        noise.copy_(self._random_like(noise) if self.policy == RANDOM
                    else self._per_track_field(noise, selection, geometry))
        self._dirty = True
        return True

    # --- the two policies that write ----------------------------------------

    def _random_like(self, noise):
        """A fresh field every frame - the boiling upper bound, and reproducible.

        Drawn from one generator advanced across the run rather than from the global
        RNG, so a run of the control is as repeatable as a run of the thing it is
        the control for.
        """
        import torch

        if self._generator is None:
            self._generator = torch.Generator(device=noise.device)
            self._generator.manual_seed(self.base_seed)
        return torch.randn(noise.shape, generator=self._generator,
                           device=noise.device, dtype=noise.dtype)

    def _per_track_field(self, noise, selection,
                         geometry: Optional[CanvasGeometry] = None):
        """The prepared field, with each region's cells taken from its track's own.

        The track's realisation is rolled to the track's current latent centre, so
        the pattern under an object is the same pattern wherever the object has
        moved to - which is the whole of what a per-track seed can mean when one
        noise field covers every region (see this module's docstring).

        Each region is read through `geometry` first: the boxes are in captured
        pixels and the latent canvas is the engine's, and since issue #39 those are
        two coordinate systems rather than one.
        """
        import torch

        field = self._prepared.clone()
        height, width = noise.shape[-2], noise.shape[-1]
        for region in selection.regions:
            box = (region.box if geometry is None or geometry.identity
                   else geometry.region(region.box))
            cells = latent_box(box, width, height)
            if cells is None:
                continue
            x0, y0, x1, y1 = cells
            rolled = torch.roll(
                self._track_field(region.track_id, noise),
                shifts=((y0 + y1) // 2, (x0 + x1) // 2), dims=(-2, -1))
            field[..., y0:y1, x0:x1] = rolled[..., y0:y1, x0:x1]
        return field

    def _track_field(self, track_id: int, noise):
        """One track's noise realisation, drawn once and kept for its lifetime.

        Canvas-sized rather than region-sized: a region that grows or shrinks
        between two frames would otherwise be a different *draw* rather than the
        same one seen through a different window, and the pinning would come apart
        exactly when the object moved towards the camera.
        """
        cached = self._fields.get(track_id)
        if cached is not None:
            return cached
        import torch

        generator = torch.Generator(device=noise.device)
        generator.manual_seed(seed_for_track(track_id, self.base_seed))
        field = torch.randn(noise.shape, generator=generator, device=noise.device,
                            dtype=noise.dtype)
        if len(self._fields) >= MAX_CACHED_FIELDS:
            # Ids are monotonic, so the smallest is the oldest object.
            del self._fields[min(self._fields)]
        self._fields[track_id] = field
        return field
