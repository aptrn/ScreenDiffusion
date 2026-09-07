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
from typing import Callable, Optional, Sequence, TextIO, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.cli import (  # noqa: E402
    report_cadence,
    report_detectors,
    report_marginal,
    report_portability,
    report_primitives,
    report_selective,
    report_stability,
    report_swap,
)
from bench.paths import (  # noqa: E402
    CADENCE_RESULTS_DIR,
    DETECTOR_RESULTS_DIR,
    PRIMITIVE_RESULTS_DIR,
    RESULTS_DIR,
    SELECTIVE_RESULTS_DIR,
    STABILITY_RESULTS_DIR,
    SWAP_RESULTS_DIR,
)

SPEC = ROOT / "docs/prompt-orchestrator-spec.md"

# One generated block: where its results live, and how to format them.
Report = Callable[[Path, TextIO], None]

# anchor name -> (report function, results directory)
BLOCKS = {
    "MEASURED TABLE": (report_marginal, RESULTS_DIR),
    "DETECTOR TABLE": (report_detectors, DETECTOR_RESULTS_DIR),
    "PRIMITIVE DECISION": (report_primitives, PRIMITIVE_RESULTS_DIR),
    "SELECTIVE PATH": (report_selective, SELECTIVE_RESULTS_DIR),
    "DEPLOY HARDWARE": (report_portability, SELECTIVE_RESULTS_DIR),
    "CADENCE SWEEP": (report_cadence, CADENCE_RESULTS_DIR),
    "STABILITY SWEEP": (report_stability, STABILITY_RESULTS_DIR),
    "PLAN SWAP": (report_swap, SWAP_RESULTS_DIR),
}


def rendered(report: Report, results_dir: Path) -> str:
    buffer = io.StringIO()
    report(results_dir, out=buffer)
    return buffer.getvalue().strip()


def padding(block: str) -> Tuple[str, str]:
    """The newlines a block already carries at each end, one newline at minimum.

    Kept as it was found: the spec's older blocks are written with a blank line
    either side, and reflowing them here would put four unrelated sections in the
    diff of a regeneration that touched one.
    """
    lead = block[:len(block) - len(block.lstrip("\n"))]
    trail = block[len(block.rstrip("\n")):]
    return lead or "\n", trail or "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
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
        lead, trail = padding(current)
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
