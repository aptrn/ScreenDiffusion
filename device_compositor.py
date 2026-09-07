"""C7 on the device: the same blend, run where the render already is.

Issue #31, spec 5.1 (C7) and 7.4. `compositor.py` said this was coming and named the
tension: the blend is numpy because the frame loop already has the capture on the
host for the detector, and because a compositor needing a CUDA device could not be
tested where the rest of this path is tested - "moving it onto the GPU is an M2
question, and the interface here, an alpha map and a blend, is the same either way."

This is that question answered, and the interface really is the same one:
`DeviceCompositor` *is* a `Compositor`. The actions, the feather, the alpha cache and
the numpy blend are inherited, not reimplemented, so everything the GPU-free tier
holds the reference implementation to is a rule about the shipped object too. What
this adds is `blend_device`, which takes the capture and the engine's output as the
device tensors they already are, blends them there, and pays **one** device-to-host
copy per frame - of uint8, after the mask, instead of float, before it.

Bit-identity is the whole safety net, so the arithmetic is not "equivalent", it is
the same operations in the same order:

- the capture becomes uint8 with `detector_worker.frame_to_array`'s own expression;
- the render becomes uint8 with streamdiffusion's own (`denormalize`, to float32,
  scale, round half to even);
- the blend is `source + (rendered - source) * alpha` in float32, written as three
  operations because that is what numpy does - a fused multiply-add would round once
  where numpy rounds twice, and a rounding that lands on a .5 boundary is a changed
  byte.

`tests/test_gpu_device_compositor.py` holds the two implementations to each other on
the same inputs; the end-to-end criterion - 48/48 frames bit-identical outside the
mask - is `bench/results/selective/` and `tests/test_gpu_selective_render.py`.

torch is imported inside each function that needs it. This module sits on the
frame loop's import path and therefore on the GUI process's, and it has to stay
importable in the merge gate's GPU-free tier.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from compositor import DEFAULT_FEATHER_PX, Compositor, alpha_bounds
from detection import Box

# Where a frame's blend ran. Recorded by the bench beside the milliseconds, because
# a composite figure measured on the host and one measured on the device are two
# designs as much as two numbers, and spec 7.4 compares them across machines.
HOST = "host"
DEVICE = "device"

Bounds = Tuple[int, int, int, int]


def _as_bhwc(tensor):
    """A capture or a render as `(B, H, W, C)`, detached and left where it is.

    Both arrive as `(B, C, H, W)`, or as one frame's `(C, H, W)` when the caller
    has already indexed the batch - `batch[i]` on the capture side, a
    `frame_buffer_size` of 1 on the wrapper's. The batch axis goes back on rather
    than being special-cased, so each conversion below is one expression about
    dtype and nothing about shape.
    """
    tensor = tensor.detach()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.permute(0, 2, 3, 1)


def capture_frames(capture):
    """The captured frames as uint8 HWC on the device, without leaving it.

    The device half of `detector_worker.frame_to_array`, operation for operation:
    clamp, scale, round, in the capture tensor's own dtype. The same expression is
    the point - this is what the output is required to stay bit-identical to, and
    an equivalent-looking conversion is not a guarantee, it is a hope.
    """
    import torch

    frames = _as_bhwc(capture)
    if frames.dtype.is_floating_point:
        frames = frames.clamp(0.0, 1.0).mul(255.0).round()
    return frames.to(torch.uint8)


def rendered_frames(rendered):
    """The engine's `output_type="pt"` output as uint8 HWC, still on the device.

    The device half of streamdiffusion's `pt_to_numpy` + `numpy_to_pil`: to float32,
    scale by 255, round half to even, cast. Everything those two do except the
    `.cpu()` in the middle of them, which is the round trip this issue removes.
    """
    import torch

    return _as_bhwc(rendered).float().mul(255.0).round().to(torch.uint8)


def resize_uint8(frames, width: int, height: int):
    """`frames`, uint8 BHWC on the device, at a new size. uint8 BHWC out.

    The capture and the diffusion canvas are two sizes since issue #39, so a
    render has to be grown back to the geometry it is composited onto - to the
    whole capture under `masked`, to the crop box under `crop`. Identity when the
    size already matches, which is the shipped 512x512-onto-512x512 case and the
    reason nothing this change adds costs that configuration a single operation.

    This is deliberately **not** under the host/device byte-identity rule the
    blend is under: there is no numpy resampler in `compositor.py` to hold it to,
    and none is wanted - the frame loop's other resize (the capture thread's) is
    a torch interpolate too. What the rule protects is the background, and the
    background is untouched by this: the blend still starts from a clone of the
    capture and writes only inside the alpha.
    """
    import torch

    if frames.shape[1] == height and frames.shape[2] == width:
        return frames
    planes = frames.permute(0, 3, 1, 2).to(torch.float32)
    shrinking = width * height < frames.shape[2] * frames.shape[1]
    resized = torch.nn.functional.interpolate(
        planes, size=(height, width), mode="bilinear", align_corners=False,
        antialias=shrinking)
    return resized.round().clamp(0.0, 255.0).to(torch.uint8).permute(0, 2, 3, 1)


def to_canvas(capture, width: int, height: int):
    """The capture as the engine's canvas: float BCHW in, float BCHW out.

    The other half of issue #39's split, on the way *in*. The capture thread hands
    the frame loop a frame at the capture geometry - 1920x1080, say - and the
    engine is 512x512 whatever the directory name claims (spec 7.2), so a frame
    that is not already the canvas is resized onto it here rather than by
    `preprocess_image`, which would want a host PIL image and a round trip.
    """
    import torch

    if capture.shape[-2] == height and capture.shape[-1] == width:
        return capture
    shrinking = width * height < capture.shape[-1] * capture.shape[-2]
    return torch.nn.functional.interpolate(
        capture, size=(height, width), mode="bilinear", align_corners=False,
        antialias=shrinking)


def crop_to_canvas(capture, box: Box, width: int, height: int):
    """One region of the capture, on the engine's whole canvas. `crop`, on the way in.

    This is the whole primitive in one line of slicing: the region is cut out and
    resized to 512x512 on its own, so an object occupying a tenth of the frame is
    diffused at 512 px rather than at 50. The aspect ratio is *not* preserved -
    the region is stretched onto the square canvas and squeezed back afterwards -
    because that is what `bench.primitive_runner`'s arm A does and what spec 8.2's
    option A means, and a shipped primitive measured against a different geometry
    than the one that was compared is not the one that was compared.
    """
    box = Box(*box)
    return to_canvas(capture[..., box.y0:box.y1, box.x0:box.x1], width, height)


def ema_blend_device(previous, current, coefficient: float):
    """`compositor.ema_blend`'s body, on tensors. uint8 in, uint8 out.

    The same interpolation written the same way round and the same round half to
    even, for the same reason the blend below is: this is held to the host
    implementation byte for byte, and an equivalent-looking expression is a hope
    rather than a guarantee.
    """
    import torch

    below = current.to(torch.float32)
    above = previous.to(torch.float32)
    return (below + (above - below) * float(coefficient)).round().to(torch.uint8)


def composite_device(source, rendered, weights, bounds: Bounds,
                     origin: Tuple[int, int] = (0, 0)):
    """`rendered` blended onto `source` through `weights`. uint8 in, uint8 out.

    `compositor.composite`'s body, on tensors: the same interpolation written the
    same way round (`source + (rendered - source) * alpha`, whose endpoints are
    exact in floating point), the same round half to even, and the same rule that
    only the bounding rectangle of a non-zero alpha is written at all. Outside it
    the output is not recomputed to the same value - it is the captured byte,
    copied.

    `origin` is the host body's, for the host body's reason: under `crop` the
    render covers the crop box and not the frame, and where it sits is one
    subtraction rather than a second blend.
    """
    import torch

    x0, y0, x1, y1 = bounds
    left, top = origin
    output = source.clone()
    below = source[:, y0:y1, x0:x1].to(torch.float32)
    above = rendered[:, y0 - top:y1 - top, x0 - left:x1 - left].to(torch.float32)
    output[:, y0:y1, x0:x1] = (
        (below + (above - below) * weights).round().to(torch.uint8))
    return output


class DeviceCompositor(Compositor):
    """The frame loop's C7 with the blend on the GPU.

    A `Compositor` in every way the merge gate tests one; `blend_device` is the
    addition. It also holds the alpha it last uploaded, for the reason the base
    class holds the alpha it last built: between two detector ticks the boxes are
    identical by construction, so without the cache two frames in three would copy
    a region-sized float map across PCIe for a map already sitting in VRAM.
    """

    def __init__(self, feather_px: int = DEFAULT_FEATHER_PX,
                 output_ema: float = 0.0) -> None:
        super().__init__(feather_px, output_ema)
        # The host array itself, not a hash of it: identity is only a safe key
        # while the object it identifies is held.
        self._alpha: Optional[np.ndarray] = None
        self._bounds: Optional[Bounds] = None
        self._weights = None
        self._device = None
        # The EMA's state stays where the frames it averages are. Its own
        # attribute rather than the base class's, because one holds a host array
        # and the other a device tensor, and a compositor asked for both blends
        # should not have them share a slot.
        self._previous_device = None

    def reset_ema(self) -> None:
        """Both histories - the device path's, and the host one it is checked against."""
        super().reset_ema()
        self._previous_device = None

    def smooth_device(self, rendered):
        """`smooth`, on the device: this frame's render averaged with the ones
        before it, in uint8, before the mask decides what reaches the screen.

        Held to `Compositor.smooth` byte for byte by
        `tests/test_gpu_device_compositor.py`, so the merge gate's tier goes on
        being where the EMA's rules are asserted.
        """
        previous = self._previous_device
        if (self.output_ema <= 0.0 or previous is None
                or previous.shape != rendered.shape):
            self._previous_device = rendered
            return rendered
        smoothed = ema_blend_device(previous, rendered, self.output_ema)
        self._previous_device = smoothed
        return smoothed

    def blend_device(self, capture, rendered, alpha: np.ndarray,
                     crop: Optional[Box] = None) -> List[np.ndarray]:
        """This frame's blend, on the device; the finished frames, on the host.

        `capture` is the capture thread's own tensor - at the *capture* geometry,
        which since issue #39 need not be the canvas - and `rendered` is what the
        engine returned under `output_type="pt"`, always at the canvas. The
        `.cpu()` at the end is the frame's **one** device-to-host copy, and it
        carries uint8 HWC - a quarter of the bytes the float round trip it replaces
        carried, and none of the PIL conversion that followed it.

        `crop` is the box the render covers under the `crop` primitive; None means
        it covers the whole capture. Either way the render is resized to the
        geometry it is composited onto, the EMA runs on the *rendered canvas*
        before that (which is what spec 8.5's safety argument is about), and the
        alpha decides every pixel that is written.
        """
        source = capture_frames(capture)
        weights, bounds = self._alpha_on(alpha, source.device)
        if bounds is None:
            # An alpha of nothing: the capture, exactly as `composite` returns it.
            return _to_host(source)
        smoothed = self.smooth_device(rendered_frames(rendered))
        height, width = ((source.shape[1], source.shape[2]) if crop is None
                         else (Box(*crop).height, Box(*crop).width))
        placed = resize_uint8(smoothed, width, height)
        origin = (0, 0) if crop is None else (Box(*crop).x0, Box(*crop).y0)
        return _to_host(
            composite_device(source, placed, weights, bounds, origin))

    def _alpha_on(self, alpha: np.ndarray, device):
        """The alpha's non-zero rectangle as a device tensor, uploaded once.

        Only the rectangle: it is the only part the blend reads, and outside it the
        weight is zero by construction rather than by arithmetic.
        """
        if alpha is not self._alpha or device != self._device:
            bounds = alpha_bounds(alpha)
            self._alpha, self._bounds, self._device = alpha, bounds, device
            self._weights = (None if bounds is None
                             else _upload_weights(alpha, bounds, device))
        return self._weights, self._bounds


def _upload_weights(alpha: np.ndarray, bounds: Bounds, device):
    """The alpha rectangle as `(h, w, 1)` float32 on `device`, ready to broadcast."""
    import torch

    x0, y0, x1, y1 = bounds
    patch = np.ascontiguousarray(alpha[y0:y1, x0:x1, None], dtype=np.float32)
    return torch.from_numpy(patch).to(device=device)


def _to_host(frames) -> List[np.ndarray]:
    """The one copy back. A list, because a batch is a list of frames upstream."""
    return list(frames.cpu().numpy())
