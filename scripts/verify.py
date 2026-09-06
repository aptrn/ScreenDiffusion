#!/usr/bin/env python
"""Merge gate for the Sandcastle loop. Exit 0 = the PR branch may merge.

Deliberately GPU-free and dependency-free so it runs in seconds on any worktree
without syncing the multi-GB CUDA environment. It gets stronger on its own as
the test suite grows: once `tests/` exists and pytest is importable, the
non-GPU tier runs here too.

Run directly with any Python 3.9+:  python scripts/verify.py
"""

from __future__ import annotations

import compileall
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tracked_python_files() -> list[Path]:
    """Every .py file git knows about. Untracked scratch files are not the gate's problem."""
    out = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ROOT / line for line in out.splitlines() if line.strip()]


def check_syntax() -> bool:
    files = tracked_python_files()
    if not files:
        print("verify: no tracked Python files")
        return True
    ok = True
    for path in files:
        # quiet=1 still prints the traceback for a file that fails to compile
        if not compileall.compile_file(str(path), quiet=1, force=True):
            ok = False
    print(f"verify: syntax {'OK' if ok else 'FAILED'} ({len(files)} files)")
    return ok


def pytest_interpreter() -> str:
    """The uv-managed venv if it is there, else whatever is running this script.

    This script is meant to run on a bare interpreter, and the ambient `python`
    on PATH is usually not the project's - it has no pytest and no torch.
    """
    venv_python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(venv_python) if venv_python.exists() else sys.executable


def run_tests() -> bool:
    """Run the GPU-free tier if a suite exists. Absent suite is not a failure."""
    if not (ROOT / "tests").is_dir():
        print("verify: no tests/ directory - skipping (see the test-harness issue)")
        return True

    python = pytest_interpreter()
    if subprocess.run([python, "-c", "import pytest"], capture_output=True).returncode != 0:
        print(f"verify: pytest not importable by {python} - skipping tests (run `uv sync`)")
        return True

    result = subprocess.run(
        [python, "-m", "pytest", "-q", "-m", "not gpu"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode == 5:  # pytest's "no tests collected"
        print("verify: no non-GPU tests collected")
        return True

    passed = re.search(r"(\d+) passed", result.stdout)
    count = passed.group(1) if passed else "0"
    ok = result.returncode == 0
    print(f"verify: tests {'OK' if ok else 'FAILED'} ({count} passed, GPU tier deselected)")
    return ok


def main() -> int:
    passed = check_syntax()
    passed = run_tests() and passed
    print("verify: PASS" if passed else "verify: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
