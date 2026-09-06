"""What options C and D would actually cost, read out of the source (issue #5 step 6).

Spec 8.2 lists four primitives. A and B are implemented and measured; C - latent
-space masking - and D - ControlNet-conditioned - are assessed rather than built,
and the issue says to implement D only if A and B both fail. An assessment made of
prose rots the moment the dependency moves, so the two claims the spec's decision
section rests on are pinned here against the tree instead.

GPU-free: the source is parsed, never imported. `main_gpu_addon.py` primes the DLL
search path and `wrapper.py` imports torch.
"""

import ast
from pathlib import Path

from sourceloader import ROOT

APP = ROOT / "main_gpu_addon.py"
WRAPPER = ROOT / "wrapper.py"
SITE_PACKAGES = Path(ROOT) / ".venv" / "Lib" / "site-packages" / "streamdiffusion"


def worker_signature() -> ast.FunctionDef:
    tree = ast.parse(APP.read_text(encoding="utf-8-sig"), filename=str(APP))
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "image_generation_process")


def test_the_worker_still_accepts_the_controlnet_arguments():
    """The dead stub CLAUDE.md records. If it ever grows a body, D got cheaper."""
    names = {argument.arg for argument in worker_signature().args.args}
    assert {"controlnet_paths", "controlnet_scales"} <= names


def test_option_d_would_have_to_be_built_because_nothing_below_it_has_controlnet():
    """`controlnet_paths` is accepted by the worker and reaches nothing.

    The spec's D assessment turns on this: it is not a flag to wire up, it is a
    pipeline StreamDiffusion 0.1.1 does not have. If `streamdiffusion` ever ships one
    this test fails, and the assessment in 8.2 should be revisited rather than
    trusted.
    """
    body = ast.dump(ast.Module(body=worker_signature().body, type_ignores=[]))
    assert "controlnet_paths" not in body, (
        "the worker now does something with controlnet_paths - re-read spec 8.2's "
        "assessment of option D"
    )
    assert "controlnet" not in WRAPPER.read_text(encoding="utf-8-sig").lower()
    if SITE_PACKAGES.is_dir():
        sources = "".join(path.read_text(encoding="utf-8", errors="ignore").lower()
                          for path in SITE_PACKAGES.rglob("*.py"))
        assert "controlnet" not in sources, (
            "the pinned StreamDiffusion now mentions ControlNet; option D may no "
            "longer be a fork-and-build"
        )


def test_option_c_has_no_intermediate_latent_to_blend_on_a_one_step_schedule():
    """C blends masked and unmasked latents *per step*, and there is one step.

    `DEFAULT_T_INDEX_LIST` is a single rung, which is what the app builds and what
    every committed engine is keyed for. With one step the blend happens once, after
    the only UNet pass - which is option B with a coarser mask, not a third
    primitive. Adding steps is the engine rebuild spec 7.2 prices.
    """
    from bench.scenarios import DEFAULT_T_INDEX_LIST, SCENARIOS

    assert len(DEFAULT_T_INDEX_LIST) == 1
    assert all(scenario.steps == 1 for scenario in SCENARIOS.values())
