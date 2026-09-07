"""Read `StreamGUI`'s wiring out of `main_gpu_addon.py` without importing it.

`StreamGUI` cannot be instantiated in the GPU-free tier - importing
`main_gpu_addon.py` primes the Windows DLL search path and pulls in the whole
customtkinter stack - so the tests that check where a control is built, what it is
bound to and what it sends read the source instead.

`sourceloader.py` is the other half of the same trick and the one to reach for
first: it *executes* the pure top-level helpers, so their behaviour can be
asserted directly. This module only ever *looks* at the widget code, which is
what is left over once everything testable has been lifted out of it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

SOURCE = Path(__file__).resolve().parent.parent / "main_gpu_addon.py"
# utf-8-sig, for the same reason `sourceloader` uses it: the sources carry a BOM.
TREE = ast.parse(SOURCE.read_text(encoding="utf-8-sig"), filename=str(SOURCE))


def class_named(name: str) -> ast.ClassDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{SOURCE.name} defines no class {name}")


GUI = class_named("StreamGUI")


def gui_method(name: str) -> ast.FunctionDef:
    """The named method of `StreamGUI`, as source."""
    for node in GUI.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"StreamGUI defines no {name}")


def calls_named(node: ast.AST, name: str) -> List[ast.Call]:
    """Every call to `name` anywhere under `node`, by callee name or attribute."""
    return [call for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and getattr(call.func, "attr", getattr(call.func, "id", None)) == name]


def mentions(node: ast.AST, name: str) -> bool:
    """Does `name` appear under `node` as a bare name or an attribute?"""
    return any(getattr(n, "attr", getattr(n, "id", None)) == name
               for n in ast.walk(node))


def assignment_to(method: ast.FunctionDef, attribute: str) -> ast.Assign:
    """The statement that assigns `self.<attribute>` inside `method`."""
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Attribute) and t.attr == attribute for t in node.targets
        ):
            return node
    raise AssertionError(f"{method.name} assigns no self.{attribute}")
