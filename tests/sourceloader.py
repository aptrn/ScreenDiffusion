"""Execute named top-level definitions from a source file, and nothing else.

`main_gpu_addon.py` primes the Windows DLL search path and imports the whole GUI
stack at module scope, and `wrapper.py` imports torch. Neither can be imported in
the GPU-free tier. The helpers under test are pure, so we parse the file, keep
only the definitions we asked for, and execute those in a namespace we supply.

Line numbers survive the round trip, so a failure inside one of these functions
points at its real line in the real file.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parent.parent


def _defined_names(node: ast.stmt) -> set[str]:
    """The top-level names a statement binds. Anything else binds nothing we can select on."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


def load_symbols(
    filename: str,
    names: Iterable[str],
    extra_globals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return {name: object} for the named top-level definitions in `filename`.

    `extra_globals` stands in for the module scope the definitions close over -
    the imports they reference and any module constants not listed in `names`.
    """
    wanted = set(names)
    path = ROOT / filename
    # utf-8-sig: wrapper.py carries a BOM, which ast.parse rejects.
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    body = [node for node in tree.body if _defined_names(node) & wanted]
    missing = wanted - {name for node in body for name in _defined_names(node)}
    if missing:
        raise AssertionError(f"{filename} defines no top-level {sorted(missing)}")

    namespace: dict[str, Any] = dict(extra_globals or {})
    # compile() inherits this module's `from __future__ import annotations`, so the
    # extracted signatures keep their annotations as strings and we do not have to
    # supply the typing names a fragment would otherwise resolve at def time.
    code = compile(ast.Module(body=body, type_ignores=[]), str(path), "exec")
    exec(code, namespace)
    return {name: namespace[name] for name in wanted}
