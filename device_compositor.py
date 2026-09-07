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


def composite_device(source, rendered, weights, bounds: Bounds):
    """`rendered` blended onto `source` through `weights`. uint8 in, uint8 out.

    `compositor.composite`'s body, on tensors: the same interpolation written the
    same way round (`source + (rendered - source) * alpha`, whose endpoints are
    exact in floating point), the same round half to even, and the same rule that
    only the bounding rectangle of a non-zero alpha is written at all. Outside it
    the output is not recomputed to the same value - it is the captured byte,
    copied.
    """
    import torch

    x0, y0, x1, y1 = bounds
    output = source.clone()
    below = source[:, y0:y1, x0:x1].to(torch.float32)
    above = rendered[:, y0:y1, x0:x1].to(torch.float32)
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

    def blend_device(self, capture, rendered, alpha: np.ndarray) -> List[np.ndarray]:
        """This frame's blend, on the device; the finished frames, on the host.

        `capture` is the tensor the engine was given and `rendered` is what it
        returned under `output_type="pt"`, so neither has been to the host yet.
        The `.cpu()` at the end is the frame's **one** device-to-host copy, and it
        carries uint8 HWC - a quarter of the bytes the float round trip it replaces
        carried, and none of the PIL conversion that followed it.
        """
        source = capture_frames(capture)
        weights, bounds = self._alpha_on(alpha, source.device)
        if bounds is None:
            # An alpha of nothing: the capture, exactly as `composite` returns it.
            return _to_host(source)
        smoothed = self.smooth_device(rendered_frames(rendered))
        return _to_host(composite_device(source, smoothed, weights, bounds))

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
