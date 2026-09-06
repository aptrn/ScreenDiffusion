"""The measured block in spec 8.1 is the committed detector results, not a transcription.

The same rule `tests/test_spec_measured_table.py` applies to 7.2, for the same
reason: a table pasted into Markdown drifts the moment a detector is re-measured,
silently, because nothing checks it. So 8.1 carries the report between two anchors
and this test regenerates it and demands a byte match. Measure a detector again and
the merge gate fails until 8.1 is regenerated with
`uv run python -m bench --detector-report`.

GPU-free: `bench.detector_results` reads JSON and formats it. It never imports torch.
"""

import io
from pathlib import Path

from sourceloader import ROOT

from bench.cli import report_detectors
from bench.detector_results import load_detector_results
from bench.detectors import DETECTORS, PRIMARY_DETECTOR, SPEED_FLOOR_DETECTOR
from bench.paths import DETECTOR_RESULTS_DIR

SPEC = Path(ROOT) / "docs/prompt-orchestrator-spec.md"
BEGIN = "<!-- BEGIN DETECTOR TABLE -->"
END = "<!-- END DETECTOR TABLE -->"


def spec_block() -> str:
    text = SPEC.read_text(encoding="utf-8")
    assert BEGIN in text and END in text, "8.1 lost its detector-table anchors"
    return text.split(BEGIN, 1)[1].split(END, 1)[0].strip()


def computed_block() -> str:
    buffer = io.StringIO()
    report_detectors(DETECTOR_RESULTS_DIR, out=buffer)
    return buffer.getvalue().strip()


def test_the_spec_block_is_what_the_committed_detector_results_say():
    assert spec_block() == computed_block(), (
        "spec 8.1 no longer matches bench/results/detectors/. Regenerate it with\n"
        "    uv run python -m bench --detector-report\n"
        "and paste the output between the DETECTOR TABLE anchors."
    )


def test_the_spec_names_the_gpu_the_numbers_came_from():
    """Issue #4 step 6, and section 7.4's rule: a detector table with no machine
    attached invites a laptop measurement to be read as a deploy-hardware one."""
    assert "RTX 3080" in spec_block()
    assert "640x640" in spec_block()


def test_the_spec_says_which_detector_was_chosen():
    """Step 5. A research agenda entry that still reads as five open options after
    the milestone has not recorded its own result."""
    section = SPEC.read_text(encoding="utf-8").split("### 8.1", 1)[1].split("### 8.2", 1)[0]
    assert "**Settled: YOLO-World.**" in section
    assert "Recommendation: yolo-world-s-640" in section


def test_the_contingencies_are_still_named_as_contingencies():
    """OWLv2 and Grounding DINO were not measured, and 8.1 must not imply they were."""
    section = SPEC.read_text(encoding="utf-8").split("### 8.1", 1)[1].split("### 8.2", 1)[0]
    for candidate in ("OWLv2", "Grounding DINO"):
        row = next(line for line in section.splitlines() if line.startswith(f"| {candidate}"))
        assert "not measured" in row, f"{candidate} was never benchmarked"


def test_both_the_candidate_and_the_floor_have_a_committed_result():
    """The gate: latency, VRAM and vocabulary-change cost for YOLO-World, plus the
    YOLOv8n baseline."""
    results = {result["detector"]["name"]: result
               for result in load_detector_results(DETECTOR_RESULTS_DIR).values()}
    assert set(results) == set(DETECTORS)

    candidate = results[PRIMARY_DETECTOR]
    assert candidate["run"]["latency"]["mean_ms"] > 0
    assert candidate["vram"]["combined_used_mib"] > 0
    assert candidate["vram"]["diffusion_scenario"], "measured beside a diffusion engine"
    assert candidate["vocabulary_change"]["median_change_ms"] > 0
    assert results[SPEED_FLOOR_DETECTOR]["vocabulary_change"]["supported"] is False


def test_the_candidate_resolved_every_concept_the_issue_names():
    """Step 4: `person`, an open-vocabulary concept, and a non-COCO animal, with the
    boxes to show for it."""
    results = {result["detector"]["name"]: result
               for result in load_detector_results(DETECTOR_RESULTS_DIR).values()}
    evidence = {item["concept"]: item for item in results[PRIMARY_DETECTOR]["evidence"]}

    assert set(evidence) == {"person", "red mug", "dog"}
    for concept, item in evidence.items():
        assert item["resolved"] is True, f"YOLO-World did not resolve {concept}"
        assert item["detections"], f"{concept} was resolved with no box to show for it"
        assert item["image_sha256"], "the evidence image is identified by hash"


def test_the_candidate_was_measured_beside_the_diffusion_engine():
    """The issue's fourth trap: a detector benchmarked alone says nothing."""
    results = load_detector_results(DETECTOR_RESULTS_DIR).values()
    for result in results:
        vram = result["vram"]
        assert vram["diffusion_scenario"] is not None, result["detector"]["name"]
        assert vram["combined_used_mib"] > vram["diffusion_used_mib"], (
            "the detector has to add something to a GPU that already holds the engine"
        )
        assert "deploy" in vram["note"], "and the figure has to be flagged non-portable"


def test_the_first_detect_after_a_vocabulary_change_is_recorded():
    """The finding: `set_classes` drops the predictor, so one frame pays ~100 ms.

    Pinned because it is the difference between "vocabulary changes are free" and
    "vocabulary changes are free once you re-warm", and the second is what the
    orchestrator has to implement.
    """
    results = {result["detector"]["name"]: result
               for result in load_detector_results(DETECTOR_RESULTS_DIR).values()}
    frame_path = results[PRIMARY_DETECTOR]["frame_path"]

    assert frame_path["unaffected"] is True, "steady-state detection is unaffected"
    assert frame_path["first_detect_ms"] > frame_path["before_ms"]
    assert frame_path["free_on_frame_path"] is False, (
        "if this ever comes back True on ultralytics, check whether set_classes still "
        "drops the predictor before softening 8.1"
    )
