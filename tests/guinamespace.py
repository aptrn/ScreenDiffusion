"""The module scope `main_gpu_addon.py`'s pure top-level helpers close over.

`sourceloader.load_symbols` executes those helpers out of the file without
importing it, and each one needs the handful of imports and module constants it
references. Two test modules ask for overlapping sets of them - the base-model
picker's and the style-LoRA picker's - so the stand-in scope is spelled once here
rather than drifting apart in two copies.

Everything in it is either the real object (`engine_cache`, `t_index_ladder`) or a
stub for something the GPU-free tier must not reach (the models and engines roots,
which would otherwise be read off this machine). Entries a given helper does not
reference are simply unused, which is why one scope can serve every caller.
"""

from __future__ import annotations

import typing
from pathlib import Path
from typing import Any, Dict

from sourceloader import load_symbols

import engine_cache
import render_plan


def _namespace() -> Dict[str, Any]:
    return dict(
        # Typing names. `load_symbols` compiles with `from __future__ import
        # annotations` in force, so these are only ever needed at runtime.
        Path=Path, Optional=object, List=list, Tuple=tuple, Union=object,
        Dict=dict, Mapping=object, NamedTuple=typing.NamedTuple,
        # The real rules the helpers are mirrors of.
        engine_cache=engine_cache,
        t_index_ladder=render_plan.t_index_ladder,
        ENGINE_BUILD_SIZE=engine_cache.ENGINE_BUILD_SIZE,
        ENGINE_BUILD_TIME=engine_cache.ENGINE_BUILD_TIME,
        DIFFUSION_CANVAS=512,
        MODEL_INDEX="model_index.json",
        LOCAL_MODEL_NAMES=("sd-turbo-fp16", "sd-turbo"),
        is_diffusers_dir=lambda path: (Path(path) / "model_index.json").is_file(),
        # Stubs: a test passes its own root explicitly, and neither resolver may
        # answer from the machine the tier happens to run on.
        resolve_models_dir=lambda **kwargs: Path("."),
        resolve_engines_dir=lambda: Path("."),
    )


def helpers(*names: str) -> Dict[str, Any]:
    """The named top-level definitions of `main_gpu_addon.py`, ready to call."""
    return load_symbols("main_gpu_addon.py", names, _namespace())
