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
