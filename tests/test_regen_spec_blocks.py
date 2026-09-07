"""The paste that keeps the spec's generated blocks honest.

The blocks themselves are byte-matched by the `tests/test_spec_*.py` modules; what
is asserted here is the mechanism `scripts/regen_spec_blocks.py` writes them with.
An anchor renamed on one side only would otherwise make the script print `skipped`
and exit zero - a regeneration that quietly regenerates nothing.

GPU-free: the script reads JSON, formats it, and rewrites Markdown.
"""

import importlib.util
from pathlib import Path

from sourceloader import ROOT

SCRIPT = Path(ROOT) / "scripts/regen_spec_blocks.py"


def load_script():
    """The script as a module. It is a script, not a package, so it has no import."""
    spec = importlib.util.spec_from_file_location("regen_spec_blocks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regen = load_script()


def test_every_block_it_regenerates_still_has_its_anchors_in_the_spec():
    text = regen.SPEC.read_text(encoding="utf-8")
    for name in regen.BLOCKS:
        assert f"<!-- BEGIN {name} -->" in text, name
        assert f"<!-- END {name} -->" in text, name


def test_the_six_generated_blocks_are_all_of_them():
    """One entry per spec block generated from a committed result: 7.2's marginal
    table, 8.1's detectors, 8.2's primitives, 8.8's selective path and its cadence
    sweep, 7.4's deploy hardware. A seventh block added without an entry here is a
    hand-copy waiting."""
    assert set(regen.BLOCKS) == {
        "MEASURED TABLE", "DETECTOR TABLE", "PRIMITIVE DECISION",
        "SELECTIVE PATH", "DEPLOY HARDWARE", "CADENCE SWEEP",
    }


def test_the_pasted_spec_is_what_the_committed_results_say():
    assert regen.main(["--check"]) == 0, (
        "docs/prompt-orchestrator-spec.md has drifted from bench/results/. "
        "Regenerate it with `uv run python scripts/regen_spec_blocks.py`.")


def test_checking_for_drift_does_not_rewrite_the_spec():
    before = regen.SPEC.read_bytes()
    regen.main(["--check"])
    assert regen.SPEC.read_bytes() == before


def test_the_blank_lines_a_block_already_had_are_kept():
    """The spec's older blocks sit inside a blank line either side. Reflowing them
    would put four unrelated sections in the diff of a one-block regeneration."""
    assert regen.padding("\n\nbody\n\n") == ("\n\n", "\n\n")
    assert regen.padding("\nbody\n") == ("\n", "\n")
    assert regen.padding("body") == ("\n", "\n")
