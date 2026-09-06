"""`_resolve_engine_dir` (wrapper.py) - the last line of defence for the engines root.

The worker hands the wrapper an already-absolute path, but the wrapper is also
constructed directly by scripts and benchmarks, so it resolves its own default the
same way. A relative `engine_dir` here re-anchors to cwd, which in a worktree means
rebuilding ~5.1 GB of TensorRT engines that already exist elsewhere.
"""

import os
from pathlib import Path

from sourceloader import load_symbols

FAKE_REPO_ROOT = Path(r"C:\Program Files\Screen Diffusion")

_symbols = load_symbols(
    "wrapper.py",
    ["SD_ENGINES_DIR_ENV", "_unquoted_path", "_resolve_engine_dir"],
    extra_globals={"os": os, "Path": Path, "REPO_ROOT": FAKE_REPO_ROOT},
)
SD_ENGINES_DIR_ENV = _symbols["SD_ENGINES_DIR_ENV"]
_resolve_engine_dir = _symbols["_resolve_engine_dir"]


def test_the_variable_is_the_one_main_gpu_addon_documents():
    assert SD_ENGINES_DIR_ENV == "SD_ENGINES_DIR"


def test_nothing_set_is_the_repo_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _resolve_engine_dir(None, environ={}) == FAKE_REPO_ROOT / "engines"


def test_the_variable_is_used_when_no_path_is_passed(tmp_path):
    shared = tmp_path / "shared caches" / "engines"
    assert _resolve_engine_dir(None, environ={SD_ENGINES_DIR_ENV: str(shared)}) == shared


def test_an_explicit_absolute_path_wins_over_the_variable(tmp_path):
    explicit = tmp_path / "explicit engines"
    resolved = _resolve_engine_dir(explicit, environ={SD_ENGINES_DIR_ENV: str(tmp_path / "ignored")})
    assert resolved == explicit


def test_a_quoted_windows_path_is_unwrapped():
    quoted = r'"D:\shared caches\engines"'
    assert _resolve_engine_dir(None, environ={SD_ENGINES_DIR_ENV: quoted}) == Path(r"D:\shared caches\engines")


def test_an_explicit_relative_path_becomes_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # "engines" was the old default and used to mean "whatever cwd happens to be".
    assert _resolve_engine_dir("engines", environ={}) == FAKE_REPO_ROOT / "engines"


def test_a_path_object_and_a_string_agree(tmp_path):
    shared = tmp_path / "shared caches" / "engines"
    assert _resolve_engine_dir(str(shared), environ={}) == _resolve_engine_dir(shared, environ={})


def test_a_blank_value_falls_through_to_the_default():
    assert _resolve_engine_dir("  ", environ={}) == FAKE_REPO_ROOT / "engines"
