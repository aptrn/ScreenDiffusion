"""Where the harness looks for models and engines, and where it writes results.

`SD_MODELS_DIR` / `SD_ENGINES_DIR` when set, else `<repo root>/models` and
`<repo root>/engines`, always absolute - the same rule as `resolve_models_dir()` /
`resolve_engines_dir()` in `main_gpu_addon.py` and `_resolve_engine_dir()` in
`wrapper.py`, and for the same reason: a relative path re-anchors to the cwd, and a
worktree that resolves its own empty `engines/` rebuilds ~5.1 GB it already has.

This is a third copy of one rule, deliberately. `main_gpu_addon.py` imports the Tk
GUI stack at module scope and `wrapper.py` imports torch; the harness runs headless
and its CLI must stay importable without CUDA, so it can borrow neither.
`tests/test_bench_paths.py` runs all three against the same inputs and asserts they
still agree.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
# Detector results (issue #4) live one level down. `bench --marginal` reads every
# JSON in RESULTS_DIR as a diffusion cell, and a detector record has no batch size.
# The subdirectory name is named separately because `--results-dir` moves the root
# and the detector half has to follow it.
DETECTOR_RESULTS_SUBDIR = "detectors"
DETECTOR_RESULTS_DIR = RESULTS_DIR / DETECTOR_RESULTS_SUBDIR
# Rendering-primitive comparisons (issue #5) get their own directory for the same
# reason, and carry their side-by-side clips beside the JSON that names them.
PRIMITIVE_RESULTS_SUBDIR = "primitives"
PRIMITIVE_RESULTS_DIR = RESULTS_DIR / PRIMITIVE_RESULTS_SUBDIR
# End-to-end selective render runs (issue #8), same rule again: a record whose
# shape is not a diffusion cell does not sit where the marginal report reads.
SELECTIVE_RESULTS_SUBDIR = "selective"
SELECTIVE_RESULTS_DIR = RESULTS_DIR / SELECTIVE_RESULTS_SUBDIR
# The `detect_every_n` sweep (issue #23) is the same record shape as a selective
# run, and that is exactly why it needs its own directory: `--selective-report` and
# `--portability-report` reduce the selective directory to the newest run per
# (case, GPU), so an arm measured at another cadence sitting there would silently
# become the row spec 8.8 and 7.4 quote.
CADENCE_RESULTS_SUBDIR = "cadence"
CADENCE_RESULTS_DIR = RESULTS_DIR / CADENCE_RESULTS_SUBDIR
# The temporal-stability sweep (issue #32) is the selective record shape again,
# with `seed_policy` and `global.output_ema` swept instead of the cadence - so it
# needs its own directory for exactly the reason the cadence arms do.
STABILITY_RESULTS_SUBDIR = "stability"
STABILITY_RESULTS_DIR = RESULTS_DIR / STABILITY_RESULTS_SUBDIR
# Plan swaps (issue #30) are a third record shape again - two plans, an interval
# series and the two acceptance criteria they answer - so they get their own
# directory rather than sitting where the selective reports reduce and quote.
SWAP_RESULTS_SUBDIR = "swaps"
SWAP_RESULTS_DIR = RESULTS_DIR / SWAP_RESULTS_SUBDIR
# Capture-geometry comparisons (issue #39) are a fourth record shape - six arms in
# one record, one per (primitive, capture size), with a per-stage cost breakdown -
# so they get their own directory rather than sitting where another report reads.
CAPTURE_RESULTS_SUBDIR = "capture"
CAPTURE_RESULTS_DIR = RESULTS_DIR / CAPTURE_RESULTS_SUBDIR

# The step-count sweep (issue #38) writes plain diffusion records, and that is
# exactly why it cannot write them here: `bench --marginal` reads every JSON in
# RESULTS_DIR as a (resolution, batch) cell of spec 7.2's committed curve, so a
# 2-step arm at batch 1 would join that curve as a second batch-1 point.
STEPS_RESULTS_SUBDIR = "steps"
STEPS_RESULTS_DIR = RESULTS_DIR / STEPS_RESULTS_SUBDIR

# A selective run on another base model (issue #38) is the selective record shape
# again, swept on the model instead of the cadence - so it needs its own directory
# for exactly the reason the cadence and stability arms do. `base-models` rather
# than `models`, because `.gitignore` carries a bare `models/` for the multi-GB
# downloads and it matches at any depth: named the obvious way, every record in
# here would be silently untracked, which for a directory whose whole point is
# being committed is the worst kind of quiet.
MODEL_RESULTS_SUBDIR = "base-models"
MODEL_RESULTS_DIR = RESULTS_DIR / MODEL_RESULTS_SUBDIR

# Style-LoRA runs (issue #38) are a record shape of their own - one row per fused
# LoRA, with whether it loaded at all - so they get their own directory like every
# other shape does.
STYLE_RESULTS_SUBDIR = "styles"
STYLE_RESULTS_DIR = RESULTS_DIR / STYLE_RESULTS_SUBDIR

# The classifier-free-guidance sweep (issue #45) is a record shape of its own -
# one row per (cfg_type, guidance, delta), with an adherence score and the UNet
# batch each arm keys - so it gets its own directory like every other shape does.
GUIDANCE_RESULTS_SUBDIR = "guidance"
GUIDANCE_RESULTS_DIR = RESULTS_DIR / GUIDANCE_RESULTS_SUBDIR

SD_MODELS_DIR_ENV = "SD_MODELS_DIR"
SD_ENGINES_DIR_ENV = "SD_ENGINES_DIR"

_PathArg = Optional[Union[str, Path]]


def _unquoted_path(value: _PathArg) -> str:
    """`value` as a bare path string. A path pasted into a Windows env var keeps its quotes."""
    return "" if value is None else str(value).strip().strip('"').strip()


def _resolve_cache_dir(env_var: str, default_name: str, explicit: _PathArg = None,
                       base_dir: _PathArg = None,
                       environ: Optional[Mapping[str, str]] = None) -> Path:
    """Absolute cache root: `explicit`, else $env_var, else `<repo root>/<default_name>`."""
    environ = os.environ if environ is None else environ
    base = Path(REPO_ROOT if base_dir is None else base_dir)
    raw = _unquoted_path(explicit) or _unquoted_path(environ.get(env_var))
    candidate = Path(raw).expanduser() if raw else base / default_name
    if not candidate.is_absolute():
        candidate = base / candidate
    # normpath, not resolve(): collapse `..` and settle on one slash direction
    # without touching the filesystem or following symlinks.
    return Path(os.path.normpath(candidate))


def resolve_models_dir(explicit: _PathArg = None, base_dir: _PathArg = None,
                       environ: Optional[Mapping[str, str]] = None) -> Path:
    return _resolve_cache_dir(SD_MODELS_DIR_ENV, "models", explicit, base_dir, environ)


def resolve_engines_dir(explicit: _PathArg = None, base_dir: _PathArg = None,
                        environ: Optional[Mapping[str, str]] = None) -> Path:
    return _resolve_cache_dir(SD_ENGINES_DIR_ENV, "engines", explicit, base_dir, environ)


def resolve_model_path(model: str, models_dir: Optional[Path] = None) -> str:
    """A local directory under the models root when there is one, else `model` verbatim."""
    root = resolve_models_dir() if models_dir is None else Path(models_dir)
    local = root / model
    return str(local) if local.is_dir() else model
