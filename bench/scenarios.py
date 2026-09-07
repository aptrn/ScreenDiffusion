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
from typing import Dict, List, Optional, Tuple

DEFAULT_MODEL = "sd-turbo-fp16"
# The second base model this repo measures (issue #38). SD-Turbo is SD 2.1-based
# and can load no SD 1.5 LoRA, which is the whole reason to price 1.5: the style
# ecosystem is 1.5's. It is not a turbo model, so it needs LCM-LoRA and about four
# steps - `bench.models` is where that pairing is written down.
SD15_MODEL = "sd-v1-5-fp16"
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
    # A local LCM-LoRA file, when the base model needs one and one is staged. None
    # leaves the wrapper on its own default repo id, which needs the network.
    lcm_lora_id: Optional[str] = None
    # A style LoRA by *name* in `bench.models.STYLE_LORAS`, not by path: the path
    # is a machine's, and a scenario is serialised whole into a result that has to
    # still say what was measured on another one.
    style_lora: Optional[str] = None
    lora_scale: float = 1.0
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


def base_model_name(acceleration: str, base: str) -> str:
    """The 512x512 batch-1 cell of one base model - `...-b1-sd15` (issue #38).

    Only that cell, because a second base model is a second axis and the point of
    the sweep is the model rather than the batch curve, which spec 7.2 already has
    for SD-Turbo.
    """
    return f"{scenario_name(acceleration, 512, 512, 1)}-{base}"


SCENARIOS: Dict[str, ScenarioConfig] = _registry()

# The SD 1.5 cells, one per accelerator. `use_lcm_lora` is on because 1.5 is not a
# turbo model and the wrapper only fuses LCM-LoRA when the base is not one; the
# step count is not baked in here, because `--steps` is what moves it and issue
# #38's step 1 is what says which count is the working one.
for _acceleration in ACCELERATIONS:
    _name = base_model_name(_acceleration, "sd15")
    SCENARIOS[_name] = ScenarioConfig(name=_name, acceleration=_acceleration,
                                      model=SD15_MODEL, use_lcm_lora=True)
