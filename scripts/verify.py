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


def run_tests() -> bool:
    """Run the GPU-free tier if a suite exists. Absent suite is not a failure."""
    if not (ROOT / "tests").is_dir():
        print("verify: no tests/ directory - skipping (see the test-harness issue)")
        return True
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-m", "not gpu"],
        cwd=ROOT,
    )
    if result.returncode == 5:  # pytest's "no tests collected"
        print("verify: no non-GPU tests collected")
        return True
    return result.returncode == 0


def main() -> int:
    passed = check_syntax()
    passed = run_tests() and passed
    print("verify: PASS" if passed else "verify: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
