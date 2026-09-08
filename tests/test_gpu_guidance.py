"""Guidance reaches the engine and changes the pixels. Issue #45, GPU tier.

`tests/test_bench_guidance.py` holds everything decidable without a device - the
arm vocabulary, the adherence arithmetic, the recommendation rule - and
`tests/test_guidance_wiring.py` reads the call that passes the settings along. What
needs a device is whether they *arrive*, and that is the one way this whole sweep
could have measured nothing at all: `wrapper.prepare` defaults to `guidance_scale`
1.2, so an arm that asked for 3.0 and did not pass it would have rendered at 1.2
and reported 3.0.

Four claims, all on the `none` accelerator - no engine, no compile, seconds.

- The scenario's `guidance_scale` and `delta` are what the pipeline holds.
- `cfg_type: none` overrides the scale to 1.0 whatever was asked for, which is why
  1.0 is the harness's spelling of "off" and why there is one control arm.
- The cfg type reaches `trt_unet_batch_size`, which is the number that keys an
  engine - the claim `engine_cache.unet_batch_size` mirrors in stdlib.
- Guidance changes the rendered frame. Not by how much - that is the sweep's job -
  but that it changes it at all, so a green table cannot come from a setting that
  went nowhere.
"""

import numpy as np
import pytest

from bench.guidance import ArmSpec, CASES, engine_keying
from bench.guidance_runner import arm_scenario
from engine_cache import CFG_FULL, CFG_INITIALIZE, CFG_NONE, CFG_SELF

pytestmark = pytest.mark.gpu

CASE = CASES["cfg-dog"]


def a_stream(spec: ArmSpec):
    from bench.runner import build_stream

    return build_stream(arm_scenario(CASE, spec))


@pytest.fixture(scope="module")
def guided():
    return a_stream(ArmSpec(CFG_FULL, 1.4, 1.0))


def test_the_scenario_s_guidance_and_delta_are_what_the_pipeline_holds(guided):
    """Not `prepare`'s own 1.2. An arm that asked for one scale and rendered at
    another would have produced a table of the right shape and the wrong numbers."""
    assert guided.stream.guidance_scale == pytest.approx(1.4)
    assert guided.stream.delta == pytest.approx(1.0)


def test_the_control_arm_is_forced_to_one_whatever_the_scale_asked_for():
    """`prepare`'s own rule under `cfg_type: none`, and the reason the sweep has
    exactly one control arm rather than a ladder of them."""
    stream = a_stream(ArmSpec(CFG_NONE, 3.0, 1.0))
    assert stream.stream.guidance_scale == pytest.approx(1.0)


@pytest.mark.parametrize("cfg_type", [CFG_NONE, CFG_SELF, CFG_INITIALIZE, CFG_FULL])
def test_the_unet_batch_is_the_one_engine_cache_predicts(cfg_type):
    """The stdlib mirror against the real pipeline. Getting this wrong is the
    window telling a user "cached" about a directory the build never writes."""
    spec = ArmSpec(cfg_type, 1.4, 1.0)
    stream = a_stream(spec)
    assert stream.stream.trt_unet_batch_size == engine_keying(CASE, spec).unet_batch


def test_guidance_changes_the_rendered_frame(guided):
    """That it changes it at all - how much is what the sweep measures. A green
    table from a setting that went nowhere is the failure this rules out."""
    import torch

    from bench.selective_runner import capture_tensor
    from detector_worker import frame_to_array

    frame = np.zeros((CASE.canvas, CASE.canvas, 3), dtype=np.uint8)
    frame[:, : CASE.canvas // 2] = (40, 110, 180)
    frame[:, CASE.canvas // 2:] = (200, 150, 60)
    control = a_stream(ArmSpec(CFG_NONE, 1.0, 1.0))

    outputs = []
    for stream in (control, guided):
        tensor = capture_tensor(frame, device=stream.device, dtype=stream.dtype)
        with torch.no_grad():
            outputs.append(np.asarray(frame_to_array(
                stream.img2img(tensor, output_type="pt"))))
    assert not np.array_equal(outputs[0], outputs[1])
