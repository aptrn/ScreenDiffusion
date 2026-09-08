"""The unbatched denoising route, and why it had to be written rather than enabled.

Issue #46. `use_denoising_batch` off is the second of the two ways to pay for a
runtime step count, and it is the interesting one: with it off the step count stops
keying the engine at all. `wrapper.py` refused it outright for img2img, and the
refusal was right about upstream's implementation - `StreamDiffusion.predict_x0_batch`'s
unbatched branch overwrites `init_noise` with the current frame's noised latent,
which is harmless for txt2img and wrong for img2img, where `encode_image` reads
`init_noise[0]` on the next call.

`wrapper.unbatched_predict_x0` is the route as the batched path's own analogue. It
is pure arithmetic over the pipeline's attributes, so it is executed here against a
stand-in stream: no torch, no CUDA, no engine.
"""

from __future__ import annotations

import ast
from typing import List

import pytest
from sourceloader import ROOT, load_symbols


class Rung:
    """One entry of `sub_timesteps_tensor`, with the two calls the loop makes."""

    def __init__(self, timestep: int):
        self.timestep = timestep

    def view(self, _size):
        return self

    def repeat(self, frames):
        return [self.timestep] * frames


class Field:
    """A one-row noise field: `init_noise[0:1]` is a value, not a list of one."""

    def __init__(self, value: float):
        self.value = value

    def __getitem__(self, _item):
        return self.value


class FakeStream:
    """Everything `unbatched_predict_x0` reads, and a record of what it did."""

    def __init__(self, timesteps: List[int], do_add_noise: bool = True):
        self.sub_timesteps_tensor = [Rung(t) for t in timesteps]
        self.frame_bff_size = 1
        self.do_add_noise = do_add_noise
        # One row, as `prepare` draws it at `batch_size` 1 on the unbatched route.
        self.init_noise = Field(100.0)
        self.alpha_prod_t_sqrt = [0.5] * len(timesteps)
        self.beta_prod_t_sqrt = [0.25] * len(timesteps)
        self.calls: List[tuple] = []

    def unet_step(self, x_t_latent, t_list, idx):
        self.calls.append((x_t_latent, tuple(t_list), idx))
        return x_t_latent + 1.0, None


def predict():
    return load_symbols("wrapper.py", ["unbatched_predict_x0"])["unbatched_predict_x0"]


@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_every_rung_is_one_unet_call_on_the_frame_it_was_given(steps):
    """The batched route puts every rung in one batch and answers N-1 frames late;
    this one spends every rung on this frame, which is what it is for."""
    stream = FakeStream(list(range(steps)))

    predict()(stream, 1.0)

    assert len(stream.calls) == steps
    assert [call[2] for call in stream.calls] == list(range(steps))


def test_each_rung_is_denoised_at_its_own_scheduled_timestep():
    stream = FakeStream([399, 299, 199])

    predict()(stream, 1.0)

    assert [call[1] for call in stream.calls] == [(399,), (299,), (199,)]


def test_the_prepared_noise_field_is_never_overwritten():
    """Upstream's own branch does `self.init_noise = x_t_latent`, so `encode_image`
    noises the next frame with this frame's latent. That is the defect the
    `NotImplementedError` was standing in front of."""
    stream = FakeStream([399, 299])
    before = stream.init_noise

    predict()(stream, 1.0)

    assert stream.init_noise is before


def test_the_field_it_re_noises_with_is_the_prepared_one():
    """Not a fresh `torch.randn_like` per rung, which is what upstream draws: the
    app's whole steadiness story (spec 8.5) is one field, drawn once and pinned to
    the canvas, and a per-step redraw would be measuring a different lever."""
    stream = FakeStream([399, 299])

    predict()(stream, 1.0)

    # rung 0 gets the input; rung 1 gets alpha * (input + 1) + beta * init_noise[0]
    assert stream.calls[0][0] == pytest.approx(1.0)
    assert stream.calls[1][0] == pytest.approx(0.5 * 2.0 + 0.25 * 100.0)


def test_with_do_add_noise_off_only_the_scaling_is_applied():
    stream = FakeStream([399, 299], do_add_noise=False)

    predict()(stream, 1.0)

    assert stream.calls[1][0] == pytest.approx(0.5 * 2.0)


def test_the_last_rung_s_prediction_is_what_comes_back():
    stream = FakeStream([399, 299, 199])

    assert predict()(stream, 1.0) == pytest.approx(stream.calls[-1][0] + 1.0)


# --- the guard it replaced ----------------------------------------------------


def wrapper_tree() -> ast.Module:
    return ast.parse((ROOT / "wrapper.py").read_text(encoding="utf-8-sig"))


def test_img2img_no_longer_refuses_the_unbatched_route():
    text = ast.unparse(wrapper_tree())
    assert "img2img mode must use denoising batch" not in text


def test_the_override_is_installed_only_where_it_is_asked_for():
    """An instance attribute on this stream, so the shipped configuration reaches
    exactly the code it reached before."""
    text = ast.unparse(wrapper_tree())
    assert "enable_unbatched_img2img" in text
    assert "use_unbatched_img2img" in text
    assert "not use_denoising_batch" in text


def test_the_step_count_stops_keying_the_engine_on_that_route():
    """The reason the route is worth having at all, in the rule both the window and
    the harness read."""
    from engine_cache import STEP_LADDER, unet_batch_size

    assert {unet_batch_size(1, steps, False) for steps in STEP_LADDER} == {1}
    assert {unet_batch_size(1, steps, True) for steps in STEP_LADDER} == set(STEP_LADDER)
