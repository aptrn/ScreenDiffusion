"""A plan swap against the real engine. GPU tier, issue #30.

The one item on criterion 3 that only a GPU can answer: applying a new plan to a
live TensorRT engine moves the schedule *value* and never the step *count*, so the
engine is not rebuilt and the loop keeps rendering through the one it has. The
committed records in `bench/results/swaps/` are that measured once; this is the
property, held for the next change to `apply_plan` or to the worker's frame loop.

`apply_plan` is the harness's copy of the worker's plan-changed branch, and
`tests/test_plan_swap_wiring.py` is what keeps the two the same three calls. What
is asserted here is what those three calls do to a real engine.

Short: two plans, three frames, no cooldown. It skips rather than builds - an
engine is ~5.0 GB and 15-25 minutes. No absolute latency is asserted (issue #13).
"""

import numpy as np
import pytest

from bench.cli import engine_dir_name
from bench.paths import resolve_engines_dir
from bench.plan_swap import CASES, STYLE_CASE, TARGET_CASE, rebuild_check
from bench.scenarios import SCENARIOS
from bench.selective import ENGINE_SCENARIO
from render_plan import ActivePlan, t_index_for_denoise

pytestmark = pytest.mark.gpu

CANVAS = 512


class _NoDetector:
    """Stands in for the detector thread: `apply_plan` only ever `follow`s it.

    The threaded detector is what `bench swap-target` measures; here it would only
    make the test slower and skippable on a machine with no weights, and the
    question is about the engine.
    """

    def __init__(self):
        self.followed = []

    def follow(self, plan):
        self.followed.append(plan.plan_version)


@pytest.fixture(scope="module")
def cached_engine():
    scenario = SCENARIOS[ENGINE_SCENARIO]
    root = resolve_engines_dir()
    engine = root / engine_dir_name(scenario) / "unet.engine"
    if not engine.is_file():
        pytest.skip(f"no cached engine for {ENGINE_SCENARIO} under {root}. Build one "
                    f"with `python -m bench {ENGINE_SCENARIO} --allow-engine-build`.")
    return root


@pytest.fixture(scope="module")
def stream(cached_engine):
    from bench.primitive_runner import set_denoise
    from bench.runner import build_stream

    before, _ = CASES[STYLE_CASE].plans()
    built = build_stream(SCENARIOS[ENGINE_SCENARIO].replace(
        prompt=before.effective_prompt), engines_root=cached_engine)
    set_denoise(built, t_index_for_denoise(before.effective_denoise))
    return built


@pytest.fixture
def frame(stream):
    from bench.selective_runner import capture_tensor

    pixels = np.full((CANVAS, CANVAS, 3), 64, dtype=np.uint8)
    return capture_tensor(pixels, device=stream.device, dtype=stream.dtype)


@pytest.mark.parametrize("case_name", [STYLE_CASE, TARGET_CASE])
def test_a_swap_moves_the_schedule_and_leaves_the_engine_where_it_was(
        stream, case_name):
    """Both kinds of swap, because both reach the engine through the same call -
    the vocabulary half happens on the detector's thread and changes nothing here.
    """
    from bench.plan_swap_runner import apply_plan, engine_identity

    before, after = CASES[case_name].plans()
    engine_id, unet_id = engine_identity(stream)
    t_index = [t_index_for_denoise(before.effective_denoise)]

    swapped = apply_plan(stream, _NoDetector(), after, t_index)

    engine_after, unet_after = engine_identity(stream)
    check = rebuild_check(engine_id, engine_after, unet_id, unet_after,
                          t_index, swapped)
    assert check.passed, check.statement
    assert check.schedule_moved, "the plan's denoise never reached the engine"
    assert swapped == [t_index_for_denoise(after.effective_denoise)]


def test_the_loop_keeps_rendering_through_the_engine_it_already_had(stream, frame):
    """The other half of "no stutter": the swap must not leave the engine in a
    state the next frame cannot be rendered from."""
    from bench.plan_swap_runner import apply_plan, engine_identity

    before, after = CASES[STYLE_CASE].plans()
    active = ActivePlan(before)
    active.submit(after)

    first = np.asarray(stream.img2img(frame))
    assert active.begin_frame().changed
    apply_plan(stream, _NoDetector(), after,
               [t_index_for_denoise(before.effective_denoise)])
    second = np.asarray(stream.img2img(frame))

    assert second.shape == first.shape
    # The same frame under a different prompt and a different strength is a
    # different render; identical output would mean the swap reached nothing.
    assert not np.array_equal(first, second)


def test_the_plan_the_frame_binds_is_the_one_the_swap_submitted(stream):
    """`submit` moves what the *next* frame picks up, so a plan arriving mid-frame
    cannot change the frame in flight - the worker's rule, on the real path."""
    before, after = CASES[TARGET_CASE].plans()
    active = ActivePlan(before)
    assert active.begin_frame().plan.plan_version == before.plan_version
    active.submit(after)
    assert active.frame_plan.plan_version == before.plan_version
    assert active.begin_frame().plan.plan_version == after.plan_version
