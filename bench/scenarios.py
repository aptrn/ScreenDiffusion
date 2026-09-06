"""What gets measured. A scenario is a configuration, not a procedure.

The registry is the sweep spec 7.2 items 1-3 asks for: SD-Turbo img2img, TensorRT
vs `none`, batch 1 / 2 / 4 / 8, at 256 / 384 / 512 square. Naming every combination
costs nothing; *running* a TensorRT one that is not cached costs ~5.1 GB and several
minutes, which is why `bench.cli.engine_build_guard` makes that an explicit opt-in.

Measure the `none` accelerator first. It answers the question that matters - the
shape of the marginal-cost curve against batch size - at zero engine-build cost.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

DEFAULT_MODEL = "sd-turbo-fp16"
# One step at t=35, which is what the app builds by default.
DEFAULT_T_INDEX_LIST: Tuple[int, ...] = (35,)
DEFAULT_PROMPT = "a photograph of a city street, cinematic lighting"

ACCELERATIONS: Tuple[str, ...] = ("none", "tensorrt")
RESOLUTIONS: Tuple[int, ...] = (256, 384, 512)
BATCH_SIZES: Tuple[int, ...] = (1, 2, 4, 8)


@dataclass(frozen=True)
class ScenarioConfig:
    """One measurable configuration, serialised whole into the result file.

    Whole, because reproducing a number needs every knob that keyed the engine:
    resolution, batch size and step count each build a *different* TensorRT engine,
    so a result that recorded only the timing would not say what was timed.
    """

    name: str
    acceleration: str = "tensorrt"
    width: int = 512
    height: int = 512
    # `frame_buffer_size` in the app's vocabulary: how many frames one call diffuses.
    batch_size: int = 1
    t_index_list: List[int] = field(default_factory=lambda: list(DEFAULT_T_INDEX_LIST))
    model: str = DEFAULT_MODEL
    prompt: str = DEFAULT_PROMPT
    seed: int = 2
    mode: str = "img2img"
    use_tiny_vae: bool = True
    use_lcm_lora: bool = False
    use_denoising_batch: bool = True
    cfg_type: str = "none"
    do_add_noise: bool = True
    reps: int = 30
    warmup_reps: int = 5

    def replace(self, **changes) -> "ScenarioConfig":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["t_index_list"] = list(self.t_index_list)
        return data

    @property
    def steps(self) -> int:
        return len(self.t_index_list)

    @property
    def unet_batch_size(self) -> int:
        """The batch a TensorRT UNet engine is built for, which is what keys it."""
        return self.batch_size * self.steps if self.use_denoising_batch else self.batch_size


def scenario_name(acceleration: str, width: int, height: int, batch_size: int) -> str:
    return f"img2img-{acceleration}-{width}x{height}-b{batch_size}"


def _registry() -> Dict[str, ScenarioConfig]:
    scenarios: Dict[str, ScenarioConfig] = {}
    for acceleration in ACCELERATIONS:
        for size in RESOLUTIONS:
            for batch_size in BATCH_SIZES:
                name = scenario_name(acceleration, size, size, batch_size)
                scenarios[name] = ScenarioConfig(
                    name=name, acceleration=acceleration, width=size, height=size,
                    batch_size=batch_size,
                )
    return scenarios


SCENARIOS: Dict[str, ScenarioConfig] = _registry()
