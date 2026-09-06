"""A global-mode plan renders exactly what `set_prompt` renders (issue #6, step 5).

The GPU tier's half of the Gate. `test_plan_control_messages.py` asserts the two
paths agree on the string; this asserts the pixels, through the real engine, using
the same `stream.img2img` call the worker's frame loop makes.

Skips rather than builds. Compiling an engine is ~5.0 GB and 15-25 minutes, which is
a decision someone takes with `--allow-engine-build`, not something a test does
behind their back.
"""

import numpy as np
import pytest
from render_plan import (
    GLOBAL,
    INITIAL_PLAN_VERSION,
    ActivePlan,
    global_plan,
    validate_plan,
)

from sourceloader import load_symbols

from bench.cli import engine_dir_name
from bench.paths import resolve_engines_dir
from bench.primitive_runner import ENGINE_SCENARIO
from bench.scenarios import SCENARIOS

pytestmark = pytest.mark.gpu

# Two prompts far enough apart that the same frame cannot render the same either way.
PROMPT = "a charcoal sketch of a city street"
OTHER_PROMPT = "a bright watercolour of a forest"


@pytest.fixture
def cached_engine():
    """The engine both paths render through, or a skip."""
    scenario = SCENARIOS[ENGINE_SCENARIO]
    root = resolve_engines_dir()
    engine = root / engine_dir_name(scenario) / "unet.engine"
    if not engine.is_file():
        pytest.skip(f"no cached engine for {ENGINE_SCENARIO} under {root}. Build one "
                    f"with `python -m bench {ENGINE_SCENARIO} --allow-engine-build`.")
    return root


@pytest.fixture
def control_transition():
    """The worker's own control-queue decision, executed out of the source file."""
    return load_symbols(
        "main_gpu_addon.py",
        ["T_INDEX_MIN", "T_INDEX_MAX", "_clamp_t_index", "_control_transition"],
        extra_globals={"validate_plan": validate_plan,
                       "INITIAL_PLAN_VERSION": INITIAL_PLAN_VERSION},
    )["_control_transition"]


def _test_frame(width: int, height: int):
    """A frame with structure in it - a flat colour renders to something a changed
    prompt barely moves, which would make the comparison pass on nothing."""
    from PIL import Image

    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    rgb = np.stack([np.broadcast_to(x, (height, width)),
                    np.broadcast_to(y, (height, width)),
                    np.full((height, width), 128.0)], axis=-1)
    rgb[height // 3:2 * height // 3, width // 4:3 * width // 4] = (240, 40, 40)
    return Image.fromarray(rgb.astype(np.uint8))


def test_a_global_plan_renders_what_set_prompt_renders(cached_engine, control_transition):
    from bench.runner import build_stream

    scenario = SCENARIOS[ENGINE_SCENARIO]
    stream = build_stream(scenario, engines_root=cached_engine)
    steps = list(scenario.t_index_list)
    batch = stream.preprocess_image(_test_frame(scenario.width, scenario.height))

    def render():
        return np.asarray(stream.img2img(batch))

    # The existing path, exactly as the worker drains it.
    delta = control_transition({"type": "set_prompt", "prompt": PROMPT}, steps)
    stream.stream.update_prompt(delta["prompt"])
    by_prompt = render()
    assert np.array_equal(by_prompt, render()), (
        "the engine is not reproducible on a repeated frame, so nothing below "
        "would mean anything"
    )

    # Move the embedding away, so the plan path cannot pass on a stale one.
    stream.stream.update_prompt(OTHER_PROMPT)
    assert not np.array_equal(by_prompt, render()), "the prompt changed nothing"

    # The plan path: validate, hold, swap at the frame boundary, render.
    active = ActivePlan(global_plan(OTHER_PROMPT))
    update = control_transition(
        {"type": "set_plan", "plan": {"mode": GLOBAL, "source_prompt": PROMPT}},
        steps,
        active.latest,
    )
    active.submit(update["plan"])
    frame = active.begin_frame()
    assert frame.changed
    stream.stream.update_prompt(frame.plan.effective_prompt)
    by_plan = render()

    assert np.array_equal(by_prompt, by_plan)


def test_a_plan_arriving_mid_frame_does_not_change_the_frame(cached_engine,
                                                             control_transition):
    """The Gate's atomicity item, against the real render: the plan is read once."""
    from bench.runner import build_stream

    scenario = SCENARIOS[ENGINE_SCENARIO]
    stream = build_stream(scenario, engines_root=cached_engine)
    batch = stream.preprocess_image(_test_frame(scenario.width, scenario.height))

    active = ActivePlan(global_plan(PROMPT))
    stream.stream.update_prompt(active.latest.effective_prompt)
    frame = active.begin_frame()

    # A control message lands while this frame is rendering.
    update = control_transition(
        {"type": "set_plan", "plan": {"mode": GLOBAL, "source_prompt": OTHER_PROMPT}},
        list(scenario.t_index_list),
        active.latest,
    )
    active.submit(update["plan"])

    assert active.frame_plan is frame.plan
    stream.stream.update_prompt(active.frame_plan.effective_prompt)
    mid_frame = np.asarray(stream.img2img(batch))

    stream.stream.update_prompt(PROMPT)
    assert np.array_equal(mid_frame, np.asarray(stream.img2img(batch)))
    assert active.begin_frame().plan.effective_prompt == OTHER_PROMPT
