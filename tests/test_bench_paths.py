"""One cache-root rule, stated in three places (issue #2, issue #9).

`bench/paths.py` cannot import either of the other two - `main_gpu_addon.py` pulls in
the Tk GUI stack and `wrapper.py` imports torch, and the harness runs headless in the
GPU-free tier. So the copies are checked against each other here instead: same inputs,
same answers, or this fails.
"""

import os
from pathlib import Path

import pytest

from sourceloader import load_symbols

from bench.paths import resolve_engines_dir as bench_engines_dir
from bench.paths import resolve_models_dir as bench_models_dir

FAKE_ROOT = Path(r"C:\Program Files\Screen Diffusion")

_app = load_symbols(
    "main_gpu_addon.py",
    ["SD_MODELS_DIR_ENV", "SD_ENGINES_DIR_ENV", "_unquoted_path", "_resolve_cache_dir",
     "resolve_models_dir", "resolve_engines_dir"],
    extra_globals={"os": os, "Path": Path, "APP_ROOT": FAKE_ROOT},
)
_wrapper_engines_dir = load_symbols(
    "wrapper.py", ["SD_ENGINES_DIR_ENV", "_unquoted_path", "_resolve_engine_dir"],
    extra_globals={"os": os, "Path": Path, "REPO_ROOT": FAKE_ROOT},
)["_resolve_engine_dir"]

# The values that have broken a path resolver before: quoted, blank, relative, `..`,
# mixed slashes, UNC.
ENV_VALUES = [
    None,
    "",
    "   ",
    r"D:\shared\engines",
    r'"D:\shared caches\engines"',
    "engines",
    r"D:\shared\..\shared\engines",
    "D:/shared/engines",
    r"\\server\share\engines",
]


@pytest.mark.parametrize("value", ENV_VALUES)
def test_bench_resolves_engines_exactly_like_the_app_and_the_wrapper(value):
    environ = {} if value is None else {"SD_ENGINES_DIR": value}
    expected = _app["resolve_engines_dir"](base_dir=FAKE_ROOT, environ=environ)
    assert bench_engines_dir(base_dir=FAKE_ROOT, environ=environ) == expected
    assert _wrapper_engines_dir(None, base_dir=FAKE_ROOT, environ=environ) == expected


@pytest.mark.parametrize("value", ENV_VALUES)
def test_bench_resolves_models_exactly_like_the_app(value):
    environ = {} if value is None else {"SD_MODELS_DIR": value}
    assert bench_models_dir(base_dir=FAKE_ROOT, environ=environ) == (
        _app["resolve_models_dir"](base_dir=FAKE_ROOT, environ=environ)
    )


def test_the_default_does_not_follow_the_cwd(monkeypatch, tmp_path):
    before = bench_engines_dir(environ={})
    monkeypatch.chdir(tmp_path)
    assert bench_engines_dir(environ={}) == before
    assert before.is_absolute()
