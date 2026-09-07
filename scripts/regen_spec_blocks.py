"""Paste the generated measured blocks back into the spec, between their anchors.

The blocks in `docs/prompt-orchestrator-spec.md` are generated from the committed
results and pinned to a byte match by tests; this is the one-liner that does the
paste, so a regeneration is never a hand-copy that drops a digit.

    uv run python scripts/regen_spec_blocks.py [--check]
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.cli import (  # noqa: E402
    report_detectors,
    report_marginal,
    report_portability,
    report_primitives,
    report_selective,
)
from bench.paths import (  # noqa: E402
    DETECTOR_RESULTS_DIR,
    PRIMITIVE_RESULTS_DIR,
    RESULTS_DIR,
    SELECTIVE_RESULTS_DIR,
)

SPEC = ROOT / "docs/prompt-orchestrator-spec.md"

# anchor name -> (report function, results directory)
BLOCKS = {
    "MEASURED TABLE": (report_marginal, RESULTS_DIR),
    "DETECTOR TABLE": (report_detectors, DETECTOR_RESULTS_DIR),
    "PRIMITIVE DECISION": (report_primitives, PRIMITIVE_RESULTS_DIR),
    "SELECTIVE PATH": (report_selective, SELECTIVE_RESULTS_DIR),
    "DEPLOY HARDWARE": (report_portability, SELECTIVE_RESULTS_DIR),
}


def rendered(report, results_dir: Path) -> str:
    buffer = io.StringIO()
    report(results_dir, out=buffer)
    return buffer.getvalue().strip()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    text = SPEC.read_text(encoding="utf-8")
    stale = []
    for name, (report, results_dir) in BLOCKS.items():
        begin, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
        if begin not in text or end not in text:
            print(f"skipped {name}: no anchors in the spec")
            continue
        head, rest = text.split(begin, 1)
        current, tail = rest.split(end, 1)
        # Keep whatever padding the block already had between its anchors: the
        # spec's older blocks are written with a blank line either side, and
        # reflowing them here would put four unrelated sections in the diff.
        lead = current[:len(current) - len(current.lstrip("\n"))] or "\n"
        trail = current[len(current.rstrip("\n")):] or "\n"
        replacement = f"{begin}{lead}{rendered(report, results_dir)}{trail}{end}"
        if replacement != f"{begin}{current}{end}":
            stale.append(name)
        text = head + replacement + tail
    if "--check" in argv:
        print("stale:", ", ".join(stale) if stale else "none")
        return 1 if stale else 0
    SPEC.write_text(text, encoding="utf-8")
    print("regenerated:", ", ".join(stale) if stale else "nothing changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
