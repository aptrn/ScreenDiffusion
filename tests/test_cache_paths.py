"""Shared `models/` and `engines/` roots (issue #9).

Both directories are gitignored, so a fresh worktree has neither. The resolution
rules under test: `SD_MODELS_DIR` / `SD_ENGINES_DIR` when set, else the app root's
own `models` / `engines`, and the answer is absolute either way - a relative path
handed to the wrapper silently re-anchors to the worker's cwd, which is the bug.

The app root here is a fake with a space in it: the repo path has none today, so
nothing else in the suite would catch a quoting mistake.
"""

import os
from pathlib import Path

from sourceloader import load_symbols

FAKE_APP_ROOT = Path(r"C:\Program Files\Screen Diffusion")

_symbols = load_symbols(
    "main_gpu_addon.py",
    [
        "SD_MODELS_DIR_ENV",
        "SD_ENGINES_DIR_ENV",
        "_resolve_cache_dir",
        "resolve_models_dir",
        "resolve_engines_dir",
        "_cache_paths_banner",
    ],
    extra_globals={"os": os, "Path": Path, "APP_ROOT": FAKE_APP_ROOT},
)
SD_MODELS_DIR_ENV = _symbols["SD_MODELS_DIR_ENV"]
SD_ENGINES_DIR_ENV = _symbols["SD_ENGINES_DIR_ENV"]
_resolve_cache_dir = _symbols["_resolve_cache_dir"]
resolve_models_dir = _symbols["resolve_models_dir"]
resolve_engines_dir = _symbols["resolve_engines_dir"]
_cache_paths_banner = _symbols["_cache_paths_banner"]


def test_the_variables_are_the_documented_names():
    assert SD_MODELS_DIR_ENV == "SD_MODELS_DIR"
    assert SD_ENGINES_DIR_ENV == "SD_ENGINES_DIR"


def test_unset_falls_back_to_the_app_root():
    assert resolve_models_dir(environ={}) == FAKE_APP_ROOT / "models"
    assert resolve_engines_dir(environ={}) == FAKE_APP_ROOT / "engines"


def test_the_default_does_not_follow_the_cwd(monkeypatch, tmp_path):
    """The whole point: the same process started elsewhere resolves the same root."""
    before = resolve_engines_dir(environ={})
    monkeypatch.chdir(tmp_path)
    assert resolve_engines_dir(environ={}) == before
    assert before.is_absolute()


def test_an_absolute_variable_wins(tmp_path):
    shared = tmp_path / "shared caches" / "engines"
    resolved = resolve_engines_dir(environ={SD_ENGINES_DIR_ENV: str(shared)})
    assert resolved == shared
    assert resolved.is_absolute()


def test_a_relative_variable_anchors_to_the_app_root_not_the_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    resolved = resolve_models_dir(environ={SD_MODELS_DIR_ENV: "shared/models"})
    assert resolved == FAKE_APP_ROOT / "shared" / "models"
    assert resolved.is_absolute()


def test_a_blank_variable_is_treated_as_unset():
    for blank in ("", "   ", '""'):
        assert resolve_models_dir(environ={SD_MODELS_DIR_ENV: blank}) == FAKE_APP_ROOT / "models"


def test_a_quoted_windows_path_is_unwrapped():
    quoted = r'"D:\shared caches\engines"'
    assert resolve_engines_dir(environ={SD_ENGINES_DIR_ENV: quoted}) == Path(r"D:\shared caches\engines")


def test_forward_and_back_slashes_land_on_the_same_place():
    forward = resolve_engines_dir(environ={SD_ENGINES_DIR_ENV: "D:/shared caches/engines"})
    backward = resolve_engines_dir(environ={SD_ENGINES_DIR_ENV: r"D:\shared caches\engines"})
    assert forward == backward


def test_dot_segments_are_normalised():
    resolved = resolve_engines_dir(environ={SD_ENGINES_DIR_ENV: r"D:\caches\..\caches\engines"})
    assert resolved == Path(r"D:\caches\engines")


def test_the_process_environment_is_the_default_source(monkeypatch, tmp_path):
    monkeypatch.setenv(SD_ENGINES_DIR_ENV, str(tmp_path / "from the process env"))
    assert resolve_engines_dir() == tmp_path / "from the process env"


def test_an_explicit_base_dir_overrides_the_app_root(tmp_path):
    assert resolve_models_dir(base_dir=tmp_path, environ={}) == tmp_path / "models"


def test_resolve_cache_dir_is_the_shared_rule():
    resolved = _resolve_cache_dir("SD_ANYTHING", "cache", base_dir=FAKE_APP_ROOT, environ={})
    assert resolved == FAKE_APP_ROOT / "cache"


def test_the_startup_banner_names_both_resolved_roots():
    models = Path(r"D:\shared caches\models")
    engines = Path(r"D:\shared caches\engines")
    banner = _cache_paths_banner(models, engines)
    assert str(models) in banner
    assert str(engines) in banner
