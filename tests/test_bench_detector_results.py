"""Detector result records (issue #4): the shape on disk, and the doors to it.

The same two rules as `bench/results.py` - a result names its machine, and it names
the clock regime that produced it - applied to a different measurement. They are
imported rather than restated, which is the point of `require_recordable`.

The detector table lives in its own directory under `bench/results/`, so that
`bench --marginal` (which reads every JSON beside it as a diffusion cell) is not
handed a record it cannot parse.
"""

import dataclasses
import json

import pytest

from sourceloader import ROOT

from bench.clocks import UNLOCKED, clock_normalization
from bench.cooldown import REACHED, CooldownRecord
from bench.detectors import DETECTORS, PRIMARY_DETECTOR, SPEED_FLOOR_DETECTOR, budget_verdict, frame_path_verdict
from bench.detector_results import (
    DETECTOR_README_HEADER,
    ConceptEvidence,
    Detection,
    DetectorMetrics,
    DetectorResult,
    LatencySummary,
    VocabularyChange,
    VramRecord,
    append_detector_readme_row,
    format_detector_report,
    latest_per_detector,
    load_detector_results,
    write_detector_result,
)
from bench.paths import DETECTOR_RESULTS_DIR
from bench.results import FingerprintError

from test_bench_results import a_fingerprint, a_lock


def a_latency(mean=13.42) -> LatencySummary:
    return LatencySummary.from_samples([mean - 0.5, mean, mean + 0.5])


def a_vram(**overrides) -> VramRecord:
    fields = dict(diffusion_scenario="img2img-tensorrt-512x512-b1",
                  baseline_used_mib=900.0, diffusion_used_mib=4200.0,
                  combined_used_mib=5400.0, torch_peak_bytes=1337 * 1024 ** 2)
    fields.update(overrides)
    return VramRecord(**fields)


def an_evidence(**overrides) -> ConceptEvidence:
    fields = dict(
        concept="red mug", kind="open vocabulary", queried="red mug",
        frame="desktop capture with the photo composited in",
        image_name="red-mug.jpg", image_source="https://example.invalid/red-mug.jpg",
        image_sha256="0" * 64, resolved=True, top_confidence=0.967,
        detections=[Detection(label="red mug", confidence=0.967,
                              box_xyxy=[10.0, 20.0, 110.0, 140.0])],
        note="",
    )
    fields.update(overrides)
    return ConceptEvidence(**fields)


def a_detector_result(**overrides) -> DetectorResult:
    latency = a_latency()
    fields = dict(
        detector=DETECTORS[PRIMARY_DETECTOR].replace(reps=3, warmup_reps=1),
        run=DetectorMetrics(
            started_utc="2026-09-06T16:00:01Z", finished_utc="2026-09-06T16:00:31Z",
            warmup_reps=1, reps=3, imgsz=640, per_rep_ms=[12.92, 13.42, 13.92],
            latency=latency, detects_per_second=round(1000.0 / latency.mean_ms, 3),
            ultralytics_speed_ms={"preprocess": 1.1, "inference": 9.9, "postprocess": 0.7},
            mean_sm_clock_mhz=1600.0, max_temperature_c=70.0,
            gpu_samples=[[0.0, 1600.0, 70.0]],
        ),
        cooldown=CooldownRecord(enabled=True, outcome=REACHED, threshold_c=62.0,
                                cap_s=120.0, waited_s=5.0, final_temperature_c=61.0,
                                samples=[[0.0, 61.0]]),
        hardware=a_fingerprint(),
        vram=a_vram(),
        budget=budget_verdict(13.42, cadence=3),
        vocabulary_change=VocabularyChange(
            supported=True, terms=["cat", "blue chair", "laptop"],
            first_change_ms=14893.0, change_ms=[66.9, 65.1, 67.2],
            median_change_ms=66.9, text_encoder="clip:ViT-B/32",
            note="Cold path: the text encode happens when the user edits the prompt.",
        ),
        frame_path=frame_path_verdict(13.42, 13.50, 66.9),
        evidence=[an_evidence()],
        clock_normalization=clock_normalization(
            a_lock(), samples=[[0.0, 1600.0, 70.0], [0.5, 1600.0, 71.0]],
            raw_ms_per_frame=13.42,
        ),
    )
    fields.update(overrides)
    return DetectorResult(**fields)


# --- the record --------------------------------------------------------------

def test_a_latency_summary_is_computed_from_the_samples_not_asserted():
    summary = LatencySummary.from_samples([10.0, 12.0, 14.0, 100.0])
    assert summary.mean_ms == pytest.approx(34.0)
    assert summary.median_ms == pytest.approx(13.0)
    assert summary.max_ms == 100.0
    assert summary.p95_ms == 100.0, "nearest rank, the same as the diffusion harness"


def test_a_single_sample_still_summarises():
    """A 1-rep smoke run is legal; `stdev` of one sample is not."""
    summary = LatencySummary.from_samples([13.0])
    assert summary.stdev_ms == 0.0 and summary.mean_ms == 13.0


def test_the_record_carries_every_number_the_gate_asks_for():
    data = a_detector_result().to_dict()

    assert data["kind"] == "detector"
    assert data["detector"]["name"] == PRIMARY_DETECTOR
    assert data["detector"]["open_vocabulary"] is True
    assert data["run"]["latency"]["mean_ms"] == pytest.approx(13.42)
    assert data["run"]["imgsz"] == 640
    assert data["vram"]["torch_peak_mib"] == pytest.approx(1337.0)
    assert data["budget"]["fits"] is True and data["budget"]["cadence"] == 3
    assert data["vocabulary_change"]["median_change_ms"] == pytest.approx(66.9)
    assert data["frame_path"]["unaffected"] is True
    assert data["evidence"][0]["resolved"] is True
    assert data["hardware"]["gpu_name"].startswith("NVIDIA")
    json.dumps(data)  # plain types only


def test_the_combined_vram_names_what_else_was_resident():
    """The gate: combined peak VRAM *with the diffusion engine resident*."""
    vram = a_detector_result().to_dict()["vram"]
    assert vram["diffusion_scenario"] == "img2img-tensorrt-512x512-b1"
    assert vram["combined_used_mib"] == 5400.0
    assert vram["detector_delta_mib"] == pytest.approx(1200.0), (
        "what the detector added on top of the diffusion engine"
    )
    assert "deploy" in vram["note"], "spec 7.4: a laptop VRAM ceiling does not transfer"


def test_a_detector_that_cannot_change_its_vocabulary_records_that_rather_than_a_zero():
    change = VocabularyChange.unsupported(
        "80 fixed COCO classes and no text encoder: the vocabulary cannot change."
    )
    assert change.supported is False
    assert change.median_change_ms is None
    assert change.to_dict()["note"]


# --- the doors to disk -------------------------------------------------------

def test_a_detector_result_without_a_fingerprint_cannot_be_written(tmp_path):
    data = a_detector_result().to_dict()
    data["hardware"]["gpu_name"] = ""
    with pytest.raises(FingerprintError):
        write_detector_result(data, results_dir=tmp_path)
    assert list(tmp_path.glob("*.json")) == []


def test_a_detector_result_without_a_clock_regime_cannot_be_written(tmp_path):
    data = a_detector_result().to_dict()
    del data["hardware"]["clock_lock"]
    with pytest.raises(FingerprintError):
        write_detector_result(data, results_dir=tmp_path)
    assert list(tmp_path.glob("*.json")) == []


def test_writing_a_detector_result_names_the_detector_and_the_timestamp(tmp_path):
    path = write_detector_result(a_detector_result(), results_dir=tmp_path)
    assert path.name == f"{PRIMARY_DETECTOR}-20260906-160031Z.json"
    assert json.loads(path.read_text(encoding="utf-8"))["run"]["reps"] == 3


def test_the_readme_row_carries_the_verdict_not_only_the_milliseconds(tmp_path):
    readme = tmp_path / "README.md"
    append_detector_readme_row(a_detector_result(), readme, filename="x.json")
    text = readme.read_text(encoding="utf-8")

    assert DETECTOR_README_HEADER in text
    assert "RTX 3080 Laptop GPU" in text
    assert "| 13.42 |" in text
    assert "| 4.47 |" in text, "the amortised figure, which is what the budget judges"
    assert "| yes |" in text
    assert f"| {UNLOCKED} |" in text

    append_detector_readme_row(a_detector_result(), readme, filename="y.json")
    text = readme.read_text(encoding="utf-8")
    assert text.count(DETECTOR_README_HEADER) == 1, "the header is written once"


def test_the_readme_row_is_refused_with_the_result(tmp_path):
    readme = tmp_path / "README.md"
    data = a_detector_result().to_dict()
    data["hardware"]["driver_version"] = ""
    with pytest.raises(FingerprintError):
        append_detector_readme_row(data, readme, filename="x.json")
    assert not readme.exists()


# --- the report the spec carries ---------------------------------------------

def test_a_concept_resolved_as_the_wrong_thing_is_not_the_same_as_one_not_found():
    """YOLOv8n asked for `dog` returns no dog and a cat at 0.79. A record that kept
    only the matching boxes would report that as a blank, and the blank is the less
    interesting half of what happened."""
    missed = an_evidence(
        concept="dog", queried="dog", resolved=False, top_confidence=None, detections=[],
        strongest_other=[Detection(label="cat", confidence=0.79,
                                   box_xyxy=[0.0, 0.0, 10.0, 10.0])],
    )
    report = format_detector_report({"a.json": a_detector_result(evidence=[missed]).to_dict()})

    assert "| strongest other label |" in report
    assert "cat 0.79" in report


def test_the_report_names_both_detectors_and_ends_in_a_recommendation():
    floor = a_detector_result(
        detector=DETECTORS[SPEED_FLOOR_DETECTOR],
        budget=budget_verdict(5.0, cadence=3),
        vocabulary_change=VocabularyChange.unsupported("no text encoder"),
        frame_path=None,
        evidence=[an_evidence(resolved=False, queried="cup",
                              note="no COCO class expresses `red mug`")],
    )
    report = format_detector_report({"a.json": a_detector_result().to_dict(),
                                     "b.json": floor.to_dict()})

    assert PRIMARY_DETECTOR in report and SPEED_FLOOR_DETECTOR in report
    assert "Recommendation:" in report
    assert "640x640" in report, "the report says what input the numbers are for"
    assert "RTX 3080 Laptop GPU" in report, "and which GPU produced them"


def test_the_report_says_which_column_it_ranked_on():
    """The recommendation quotes the normalised figure and the table leads with the
    raw one, so the report has to say that rather than leave the reader to notice."""
    report = format_detector_report({"a.json": a_detector_result().to_dict()})
    assert "clock-normalised estimate" in report


def test_a_locked_run_is_ranked_on_its_raw_figure_and_says_nothing_about_bases():
    """Locked clocks produce no estimate, so there is no second figure to explain."""
    from bench.clocks import LOCKED

    locked = a_lock(state=LOCKED, applied_clock_mhz=1200.0)
    result = a_detector_result(
        hardware=a_fingerprint(clock_lock=locked),
        clock_normalization=clock_normalization(locked, [[0.0, 1200.0, 70.0]],
                                                raw_ms_per_frame=13.42),
    )
    report = format_detector_report({"a.json": result.to_dict()})
    assert "clock-normalised estimate" not in report


def test_the_report_is_stable_for_the_same_inputs():
    results = {"a.json": a_detector_result().to_dict()}
    assert format_detector_report(results) == format_detector_report(results)


def test_only_the_newest_run_of_each_detector_reaches_the_report():
    older = a_detector_result()
    newer_run = dataclasses.replace(older.run, finished_utc="2026-09-07T10:00:00Z")
    latest = latest_per_detector({"old.json": older.to_dict(),
                                  "new.json": a_detector_result(run=newer_run).to_dict()})
    assert list(latest) == ["new.json"]


# --- the committed tree ------------------------------------------------------

def test_every_committed_detector_result_is_recordable():
    """The rule applied to the tree: a committed result is the deliverable."""
    from bench.results import require_recordable

    committed = load_detector_results(DETECTOR_RESULTS_DIR)
    assert committed, "no detector result is committed yet"
    for filename, result in committed.items():
        require_recordable(result)
        assert result["kind"] == "detector", filename


def test_both_detectors_the_issue_names_are_measured():
    measured = {result["detector"]["name"]
                for result in load_detector_results(DETECTOR_RESULTS_DIR).values()}
    assert {PRIMARY_DETECTOR, SPEED_FLOOR_DETECTOR} <= measured


def test_every_committed_detector_result_has_a_row_in_its_readme():
    readme = (DETECTOR_RESULTS_DIR / "README.md").read_text(encoding="utf-8")
    assert DETECTOR_README_HEADER in readme
    for path in sorted(DETECTOR_RESULTS_DIR.glob("*.json")):
        assert path.name in readme, f"{path.name} was written but never listed"


def test_the_detector_results_do_not_land_in_the_diffusion_table():
    """`bench --marginal` reads every JSON beside it as a diffusion cell.

    A detector record has no `scenario.batch_size`, so one dropped into the parent
    directory would not merely be ignored - it would crash the marginal report.
    """
    from bench import marginal

    for name in marginal.load_results(ROOT / "bench" / "results"):
        assert "yolo" not in name
