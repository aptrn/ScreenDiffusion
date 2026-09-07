"""The device composite against the numpy one, byte for byte. Issue #31, GPU tier.

This file is the design's entire safety net, and it is short on purpose. The claim
being made is not "the GPU blend looks the same" - it is that the two
implementations produce **identical bytes on the same inputs**, so the numpy one
can go on being the reference the merge gate holds and the device one can go on
being the one that ships.

Three seams, each checked as an equality rather than a tolerance:

- the blend itself, over random frames and every shape of alpha the compositor
  builds (feathered, hard-edged, overlapping, empty);
- the capture's conversion to uint8, against `detector_worker.frame_to_array` -
  the array the bit-identity criterion is stated against;
- the render's conversion to uint8, against streamdiffusion's own host path
  (`postprocess_image` to PIL), which is what the frame loop used to receive.

Plus the fourth trap, which no equality catches: **one** device-to-host copy per
frame. It is counted, because the prize here is a couple of milliseconds and two
stray syncs would eat it.

No engine and no detector: everything here is arithmetic on tensors, so it runs on
any CUDA device in a second.
"""

import numpy as np
import pytest

from compositor import Compositor, composite, feather_alpha
from detection import Box
from device_compositor import DeviceCompositor, capture_frames, rendered_frames
from detector_worker import frame_to_array

pytestmark = pytest.mark.gpu

W, H = 96, 64
REGION = Box(10, 8, 60, 50)


@pytest.fixture(scope="module")
def torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch


def frames(seed, count=1):
    generator = np.random.default_rng(seed)
    return generator.integers(0, 256, size=(count, H, W, 3), dtype=np.uint8)


ALPHAS = {
    "feathered": feather_alpha([REGION], W, H),
    "hard": feather_alpha([REGION], W, H, feather_px=0),
    "overlapping": feather_alpha([Box(4, 4, 40, 40), Box(30, 20, 90, 60)], W, H),
    "tiny": feather_alpha([Box(50, 30, 57, 37)], W, H),
    "empty": np.zeros((H, W), dtype=np.float32),
    "whole frame": feather_alpha([Box(0, 0, W, H)], W, H),
}


# --- the blend --------------------------------------------------------------


@pytest.mark.parametrize("shape", sorted(ALPHAS))
def test_the_device_blend_and_the_numpy_blend_agree_byte_for_byte(torch, shape):
    """The Gate's second item. Same inputs, same bytes - on every alpha the
    compositor can hand the frame loop, not just the ordinary one."""
    alpha = ALPHAS[shape]
    source, rendered = frames(1)[0], frames(2)[0]
    host = composite(source, rendered, alpha)

    device = DeviceCompositor().blend_device(
        _on_device(torch, source), _on_device(torch, rendered), alpha)[0]

    assert np.array_equal(device, host), (
        f"{int(np.count_nonzero(np.any(device != host, axis=-1)))} pixels differ "
        f"on the {shape} alpha")


def test_it_agrees_over_a_batch_of_frames(torch):
    """`frame_buffer_size > 1` blends a batch; each frame gets its own capture."""
    alpha = ALPHAS["feathered"]
    sources, rendered = frames(3, count=3), frames(4, count=3)
    device = DeviceCompositor().blend_device(
        _on_device(torch, sources), _on_device(torch, rendered), alpha)
    assert len(device) == 3
    for index, output in enumerate(device):
        assert np.array_equal(output, composite(sources[index], rendered[index],
                                                alpha))


def test_outside_the_alpha_the_device_path_returns_the_captured_bytes(torch):
    """The sharp criterion, stated on the device path's own output against the
    array `detector_worker.frame_to_array` would have produced from the capture."""
    alpha = ALPHAS["feathered"]
    capture = _on_device(torch, frames(5)[0])
    output = DeviceCompositor().blend_device(
        capture, _on_device(torch, frames(6)[0]), alpha)[0]
    outside = alpha == 0.0
    assert outside.any()
    assert np.array_equal(output[outside], frame_to_array(capture)[outside])


def test_an_empty_alpha_costs_no_blend_and_passes_the_capture_through(torch):
    capture = _on_device(torch, frames(7)[0])
    output = DeviceCompositor().blend_device(capture, None, ALPHAS["empty"])[0]
    assert np.array_equal(output, frame_to_array(capture))


# --- the two conversions the equality rests on -------------------------------


def test_the_capture_becomes_the_same_uint8_array_on_either_device(torch):
    """`capture_frames` is `frame_to_array`'s expression, so it has to produce
    `frame_to_array`'s bytes - including from the fp16 the engine is fed."""
    for dtype in (torch.float16, torch.float32):
        tensor = torch.rand((1, 3, H, W), device="cuda", dtype=dtype)
        assert np.array_equal(capture_frames(tensor)[-1].cpu().numpy(),
                              frame_to_array(tensor))


def test_the_render_becomes_the_same_uint8_array_as_the_host_postprocess(torch):
    """The frame loop used to receive a PIL image built on the host out of the
    same tensor. Same bytes, or the mask would be blending different pixels."""
    from streamdiffusion.image_utils import postprocess_image

    latent = (torch.rand((1, 3, H, W), device="cuda", dtype=torch.float16) * 2.4) - 1.2
    host = np.asarray(postprocess_image(latent.cpu(), output_type="pil")[0])
    device = rendered_frames(postprocess_image(latent, output_type="pt")[0])
    assert np.array_equal(device[0].cpu().numpy(), host)


def test_the_wrapper_leaves_a_pt_render_on_the_device(torch):
    """The feasibility this rests on: `postprocess_image` pays the `.cpu()` for
    every output type that is a host object, and for `pt` it pays nothing."""
    from wrapper import StreamDiffusionWrapper

    class OneFrame:
        frame_buffer_size = 1

    tensor = torch.rand((1, 3, H, W), device="cuda", dtype=torch.float16)
    kept = StreamDiffusionWrapper.postprocess_image(OneFrame(), tensor,
                                                    output_type="pt")
    assert kept.is_cuda, "the device path still brings the render home first"


# --- the fourth trap ---------------------------------------------------------


def test_one_device_to_host_copy_per_frame(torch, monkeypatch):
    """The prize is a couple of milliseconds; two extra syncs would spend it."""
    copies = []
    original = torch.Tensor.cpu
    monkeypatch.setattr(torch.Tensor, "cpu",
                        lambda self, *a, **k: (copies.append(tuple(self.shape)),
                                               original(self, *a, **k))[1])
    DeviceCompositor().blend_device(_on_device(torch, frames(8)[0]),
                                    _on_device(torch, frames(9)[0]),
                                    ALPHAS["feathered"])
    assert len(copies) == 1, f"{len(copies)} device-to-host copies: {copies}"


def test_the_alpha_is_uploaded_once_while_the_boxes_hold_still(torch):
    """Between two detector ticks the frame loop reads one alpha object; copying a
    region-sized float map per frame for it would be pure waste."""
    compositor = DeviceCompositor()
    capture = _on_device(torch, frames(10)[0])
    rendered = _on_device(torch, frames(11)[0])
    alpha = ALPHAS["feathered"]
    compositor.blend_device(capture, rendered, alpha)
    first = compositor._weights
    compositor.blend_device(capture, rendered, alpha)
    assert compositor._weights is first
    compositor.blend_device(capture, rendered, ALPHAS["overlapping"])
    assert compositor._weights is not first


# --- the output EMA, on either device (issue #32) ----------------------------


@pytest.mark.parametrize("coefficient", (0.0, 0.25, 0.5, 0.75, 0.9))
def test_the_device_ema_and_the_numpy_ema_agree_byte_for_byte(torch, coefficient):
    """The EMA is the second thing on this path that produces pixels, so it gets
    the same guarantee the blend does: identical bytes, not a tolerance."""
    sequence = frames(21, count=5)
    host = Compositor(output_ema=coefficient)
    device = DeviceCompositor(output_ema=coefficient)
    for frame in sequence:
        smoothed = device.smooth_device(
            torch.from_numpy(frame.copy()).to(device="cuda"))
        assert np.array_equal(smoothed.cpu().numpy(), host.smooth(frame))


def test_a_smoothed_frame_still_leaves_the_background_bit_identical(torch):
    """The issue's first trap, on the shipped path: an EMA on the *render* cannot
    reach outside the mask, because the blend it feeds still copies the captured
    byte wherever alpha is zero."""
    alpha = ALPHAS["feathered"]
    compositor = DeviceCompositor(output_ema=0.75)
    outside = alpha == 0.0
    for seed in range(30, 34):
        capture = _on_device(torch, frames(seed)[0])
        output = compositor.blend_device(
            capture, _on_device(torch, frames(seed + 100)[0]), alpha)[0]
        assert np.array_equal(output[outside], frame_to_array(capture)[outside])


def test_a_smoothed_frame_still_costs_one_device_to_host_copy(torch, monkeypatch):
    copies = []
    original = torch.Tensor.cpu
    monkeypatch.setattr(torch.Tensor, "cpu",
                        lambda self, *a, **k: (copies.append(tuple(self.shape)),
                                               original(self, *a, **k))[1])
    compositor = DeviceCompositor(output_ema=0.5)
    for seed in (40, 41):
        compositor.blend_device(_on_device(torch, frames(seed)[0]),
                                _on_device(torch, frames(seed + 1)[0]),
                                ALPHAS["feathered"])
    assert len(copies) == 2, f"{len(copies)} device-to-host copies: {copies}"


# --- how a frame reaches the two calls above ---------------------------------


def _on_device(torch, array):
    """uint8 HWC frames as either tensor the blend takes.

    The capture thread's frame and the engine's denormalized `pt` output have the
    same shape and the same range - (B, 3, H, W) fp16 in 0..1 - so one helper
    builds both, and the equality above is against the uint8 arrays it was built
    from rather than against a second conversion.
    """
    frames = array if array.ndim == 4 else array[None]
    tensor = torch.from_numpy(frames.copy()).to(device="cuda", dtype=torch.float16)
    return tensor.permute(0, 3, 1, 2) / 255.0
