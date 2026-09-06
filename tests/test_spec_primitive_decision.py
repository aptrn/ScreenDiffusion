"""The decision block in spec 8.2 is the committed comparisons, not a transcription.

The same rule `tests/test_spec_measured_table.py` applies to 7.2 and
`tests/test_spec_detector_table.py` applies to 8.1: a table pasted into Markdown
drifts the moment a case is re-measured, silently, because nothing checks it. Run a
comparison again and the merge gate fails until 8.2 is regenerated with
`uv run python -m bench --primitive-report`.

The rest of this file is issue #5's Gate, asserted against what is on disk.

GPU-free: `bench.primitive_results` reads JSON and formats it.
"""

import io
from pathlib import Path

from sourceloader import ROOT

from bench.cli import report_primitives
from bench.paths import PRIMITIVE_RESULTS_DIR
from bench.primitives import (
    CASES,
    CROP,
    IDENTITY_CASE,
    MASKED,
    PRIMITIVES,
    RESTYLE_CASE,
    SMALL_OBJECT_PX,
)
from bench.primitive_results import (
    latest_per_case,
    load_primitive_results,
)
from bench.results import require_recordable

SPEC = Path(ROOT) / "docs/prompt-orchestrator-spec.md"
BEGIN = "<!-- BEGIN PRIMITIVE DECISION -->"
END = "<!-- END PRIMITIVE DECISION -->"


def spec_block() -> str:
    text = SPEC.read_text(encoding="utf-8")
    assert BEGIN in text and END in text, "8.2 lost its decision anchors"
    return text.split(BEGIN, 1)[1].split(END, 1)[0].strip()


def computed_block() -> str:
    buffer = io.StringIO()
    report_primitives(PRIMITIVE_RESULTS_DIR, out=buffer)
    return buffer.getvalue().strip()


def committed() -> dict:
    """The latest comparison per case, keyed by case name."""
    latest = latest_per_case(load_primitive_results(PRIMITIVE_RESULTS_DIR))
    return {result["case"]["name"]: result for result in latest.values()}


def arms_of(case_name: str) -> dict:
    return {arm["primitive"]: arm for arm in committed()[case_name]["arms"]}


def test_the_spec_block_is_what_the_committed_comparisons_say():
    assert spec_block() == computed_block(), (
        "spec 8.2 no longer matches bench/results/primitives/. Regenerate it with\n"
        "    uv run python -m bench --primitive-report\n"
        "and paste the output between the PRIMITIVE DECISION anchors."
    )


def test_the_spec_names_a_dated_decision_section():
    """The issue's step 7: a decision section, dated."""
    section = SPEC.read_text(encoding="utf-8").split("### 8.2", 1)[1].split("### 8.3", 1)[0]
    assert "#### Decision (2026-09-06)" in section
    assert "**Decision:" in section


def test_the_spec_assesses_c_and_d_rather_than_leaving_them_open():
    """Step 6. Implement D only if A and B both fail; they did not."""
    section = SPEC.read_text(encoding="utf-8").split("### 8.2", 1)[1].split("### 8.3", 1)[0]
    assert "C and D, assessed rather than built" in section
    assert "controlnet_paths" in section
    assert "not a third primitive on a one-step schedule" in section


def test_every_committed_comparison_could_have_reached_disk():
    """The door rules, re-applied to what is actually in the tree."""
    results = load_primitive_results(PRIMITIVE_RESULTS_DIR)
    assert results, "issue #5's deliverable is a committed comparison"
    for filename, result in results.items():
        require_recordable(result)
        assert result["kind"] == "primitive", filename


# --- the Gate ---------------------------------------------------------------------

def test_both_cases_have_a_committed_comparison():
    assert set(committed()) == set(CASES)


def test_both_primitives_have_ms_per_frame_and_a_flicker_figure_on_both_cases():
    """The first Gate item, on the same clip within a case."""
    for case_name in CASES:
        arms = arms_of(case_name)
        assert set(arms) == {CROP, MASKED}, case_name
        for primitive, arm in arms.items():
            assert arm["ms_per_frame"] > 0, (case_name, primitive)
            assert arm["flicker"]["mean_abs_diff"] is not None, (case_name, primitive)
            assert arm["flicker"]["pairs_scored"] > 0, (case_name, primitive)
        assert len({arm["frames"] for arm in arms.values()}) == 1, (
            "the two primitives have to be measured on the same frames"
        )


def test_the_denoise_strength_each_case_required_is_reported():
    """The second Gate item, with the rule that selected it."""
    for case_name in CASES:
        for primitive, arm in arms_of(case_name).items():
            denoise = arm["denoise"]
            assert denoise["rule"], (case_name, primitive)
            assert denoise["strength"] is not None
            assert 1 <= denoise["t_index"] <= 49


def test_each_primitive_states_what_it_cannot_express():
    """The third Gate item, and the half no benchmark produces."""
    block = spec_block()
    for key, config in PRIMITIVES.items():
        assert config.cannot_express in block, key
    assert "one prompt and one denoise per frame" in block.lower()
    assert "small-crop quality floor" in block.lower()


def test_the_decision_is_justified_on_cost_and_on_expressiveness():
    """The fourth Gate item, and the issue's first trap."""
    block = spec_block()
    assert "**Decision:" in block
    assert "ms/frame" in block
    assert "expresses" in block
    assert RESTYLE_CASE in block, "the priority case is what it is justified against"


def test_the_one_step_identity_finding_is_recorded_with_its_implication():
    """The fifth Gate item. Only fires if SD-Turbo could not do it - and if it could,
    the identity arms have to say so instead."""
    identity = committed()[IDENTITY_CASE]
    if identity["one_step_finding"]:
        assert "t_index_list" in identity["one_step_finding"]
        assert "engine" in identity["one_step_finding"]
        assert identity["one_step_finding"] in spec_block()
    else:
        assert any(arm["expresses"] for arm in identity["arms"]), (
            "no finding recorded, so some primitive must have achieved the change"
        )


def test_the_identity_case_is_judged_by_the_detector_not_by_a_change_figure():
    for primitive, arm in arms_of(IDENTITY_CASE).items():
        check = arm["identity"]
        assert check is not None, primitive
        assert check["asked_for"] == ["dog", "cat"]
        assert check["frames_probed"] == arm["frames"]
        assert check["statement"]


def test_a_small_object_case_is_included():
    """The last Gate item, and the gap issue #17 left open."""
    small = committed()[RESTYLE_CASE]["track"]["small_objects"]
    assert small["threshold_px"] == SMALL_OBJECT_PX
    assert small["min_side_under"] > 0, small["statement"]
    assert small["statement"] in spec_block()


def test_the_comparison_clips_a_human_has_to_watch_are_committed():
    """Manual verification is a Gate item; it needs a file to point at."""
    for case_name, result in committed().items():
        assert result["comparison_clip"], case_name
        assert (PRIMITIVE_RESULTS_DIR / result["comparison_clip"]).is_file()
        assert (PRIMITIVE_RESULTS_DIR / result["comparison_still"]).is_file()
        for arm in result["arms"]:
            assert (PRIMITIVE_RESULTS_DIR / arm["clip_file"]).is_file()


def test_the_comparison_did_not_touch_the_live_render_loop():
    """The issue's fourth trap: this is an offline comparison.

    The application must not have acquired a dependency on the harness. Checked over
    the imports rather than over prose, so a comment mentioning the benchmark does
    not fail it.
    """
    import ast

    for filename in ("main_gpu_addon.py", "wrapper.py"):
        tree = ast.parse((Path(ROOT) / filename).read_text(encoding="utf-8-sig"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "bench" not in imported, filename
