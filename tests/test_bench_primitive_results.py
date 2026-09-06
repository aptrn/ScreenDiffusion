"""Primitive-comparison records (issue #5): the shape on disk, and the doors to it.

The same two rules as every other result - it names its machine, and it names the
clock regime that produced it - applied to a record that holds two timings instead
of one. They are imported rather than restated, which is the point of
`require_recordable`.

GPU-free: `bench.primitive_results` reads dicts and formats them.
"""

import json

import numpy as np
import pytest

from bench.clocks import LOCKED, UNLOCKED, ClockLock, clock_normalization
from bench.cooldown import REACHED, CooldownRecord
from bench.flicker import flicker_score
from bench.primitive_results import (
    PRIMITIVE_README_HEADER,
    ClipRecord,
    IdentityCheck,
    PrimitiveArm,
    PrimitiveResult,
    PrimitiveRunMetrics,
    TrackRecord,
    append_primitive_readme_rows,
    arm_normalised_cell,
    decision_from,
    format_primitive_report,
    latest_per_case,
    load_primitive_results,
    primitive_readme_rows,
    write_primitive_result,
)
from bench.primitives import (
    CASES,
    CROP,
    IDENTITY_CASE,
    MASKED,
    PRIMITIVES,
    RESTYLE_CASE,
    Box,
    DenoisePoint,
    required_denoise,
    small_object_summary,
)
from bench.results import FingerprintError

from test_bench_results import a_fingerprint, a_lock
from test_bench_detector_results import a_latency


def a_denoise(kind, t_index=40, region_change=12.0, hits=None):
    points = [DenoisePoint(t_index=rung, timestep=1000 - 20 * rung,
                           strength=1.0 - rung / 50,
                           region_change=region_change if rung <= t_index else 1.0,
                           outside_change=0.0, frames=4,
                           identity_hits=hits, identity_frames=4 if hits else None)
              for rung in (20, 30, t_index, 45)]
    return required_denoise(points, kind)


def a_flicker(level=1.5):
    source = [np.zeros((4, 4, 3)), np.zeros((4, 4, 3))]
    output = [np.zeros((4, 4, 3)), np.full((4, 4, 3), float(level))]
    return flicker_score(source, output)


def an_arm(primitive=MASKED, ms=32.0, expresses=True, kind="restyle", **overrides):
    fields = dict(
        primitive=primitive, spec_option=PRIMITIVES[primitive].spec_option,
        denoise=a_denoise(kind), latency=a_latency(ms), ms_per_frame=ms,
        calls_per_frame=1.0 if primitive == MASKED else 6.0,
        calls_total=48 if primitive == MASKED else 288, frames=48,
        flicker=a_flicker(), region_change=12.0, outside_change=0.0,
        expresses=expresses, cannot_express=PRIMITIVES[primitive].cannot_express,
        clock_normalization=clock_normalization(
            a_lock(state=UNLOCKED), [[0.0, 1500.0, 60.0], [0.5, 1500.0, 61.0]],
            raw_ms_per_frame=ms),
    )
    fields.update(overrides)
    return PrimitiveArm(**fields)


def a_primitive_result(case=RESTYLE_CASE, arms=None, **overrides):
    fields = dict(
        case=CASES[case].replace(frames=48),
        clip=ClipRecord(name=CASES[case].clip, sha256="a" * 64, width=1280,
                        height=720, fps=25.0, total_frames=377, start_frame=0,
                        frames_used=48),
        track=TrackRecord(detector="yolo-world-s-640", target=CASES[case].target,
                          conf=0.25, region=CASES[case].region, max_objects=6,
                          objects_per_frame=6.0, regions_rendered=288,
                          small_objects=small_object_summary([Box(0, 0, 45, 172)])),
        arms=arms if arms is not None else [an_arm(CROP, 190.0), an_arm(MASKED, 32.0)],
        run=PrimitiveRunMetrics(
            started_utc="2026-09-06T18:00:01Z", finished_utc="2026-09-06T18:01:31Z",
            warmup_reps=3, engine_scenario="img2img-tensorrt-512x512-b1",
            detector_resident="yolo-world-s-640", mean_sm_clock_mhz=1500.0,
            max_temperature_c=71.0, peak_vram_bytes=1024 ** 3,
            gpu_samples=[[0.0, 1500.0, 60.0]]),
        cooldown=CooldownRecord(enabled=True, outcome=REACHED, waited_s=12.0,
                                threshold_c=62.0,
                                cap_s=300.0, final_temperature_c=61.0, samples=[]),
        hardware=a_fingerprint(),
        comparison_clip="restyle-people-triptych.mp4",
        comparison_still="restyle-people-triptych.jpg",
    )
    fields.update(overrides)
    return PrimitiveResult(**fields)


# --- the record ---------------------------------------------------------------------

def test_the_record_serialises_to_json_safe_primitives():
    data = a_primitive_result().to_dict()
    assert json.loads(json.dumps(data))["kind"] == "primitive"
    assert data["schema_version"] >= 2
    assert len(data["arms"]) == 2


def test_one_record_holds_both_primitives_on_one_case():
    """The unit of the comparison: two arms measured under one clock trace."""
    data = a_primitive_result().to_dict()
    assert {arm["primitive"] for arm in data["arms"]} == {CROP, MASKED}
    assert data["case"]["name"] == RESTYLE_CASE


def test_every_arm_carries_what_that_primitive_cannot_express():
    for arm in a_primitive_result().to_dict()["arms"]:
        assert arm["cannot_express"], arm["primitive"]


def test_a_record_without_a_fingerprint_cannot_be_written(tmp_path):
    result = a_primitive_result(hardware=a_fingerprint(gpu_name=""))
    with pytest.raises(FingerprintError):
        write_primitive_result(result, tmp_path)
    assert list(tmp_path.glob("*.json")) == []


def test_a_record_without_a_clock_regime_cannot_be_written(tmp_path):
    data = a_primitive_result().to_dict()
    data["hardware"].pop("clock_lock")
    with pytest.raises(FingerprintError):
        write_primitive_result(data, tmp_path, timestamp="20260906-180131Z")


def test_a_written_record_is_named_for_its_case(tmp_path):
    path = write_primitive_result(a_primitive_result(), tmp_path)
    assert path.name.startswith(RESTYLE_CASE)
    assert json.loads(path.read_text(encoding="utf-8"))["case"]["name"] == RESTYLE_CASE


# --- the README table ------------------------------------------------------------------

def test_a_comparison_appends_one_row_per_primitive():
    rows = primitive_readme_rows(a_primitive_result().to_dict(), "a.json")
    assert len(rows) == 2
    assert all(row.count("|") == PRIMITIVE_README_HEADER.count("|") for row in rows)


def test_the_rows_and_the_header_agree_on_their_column_count(tmp_path):
    """A row is appended under whatever header the file already has."""
    append_primitive_readme_rows(a_primitive_result(), tmp_path / "README.md", "a.json")
    lines = (tmp_path / "README.md").read_text(encoding="utf-8").strip().splitlines()
    assert lines[-4] == PRIMITIVE_README_HEADER  # header, rule, and this run's two rows
    assert lines[-1].count("|") == PRIMITIVE_README_HEADER.count("|")


def test_the_readme_is_refused_a_record_that_cannot_reach_disk(tmp_path):
    with pytest.raises(FingerprintError):
        append_primitive_readme_rows(a_primitive_result(hardware=a_fingerprint(
            nvidia_smi_raw="")), tmp_path / "README.md", "a.json")
    assert not (tmp_path / "README.md").exists()


# --- the clock cell -------------------------------------------------------------------

def test_each_arm_is_normalised_against_its_own_raw_figure():
    """One normalisation on the record could only describe one of the two timings."""
    data = a_primitive_result().to_dict()
    cells = [arm_normalised_cell(data, arm) for arm in data["arms"]]
    assert cells[0] != cells[1]
    assert all("MHz" in cell for cell in cells)


def test_a_locked_run_says_raw_rather_than_offering_an_estimate():
    data = a_primitive_result(hardware=a_fingerprint(
        clock_lock=ClockLock(state=LOCKED, applied_clock_mhz=1200.0,
                             max_sm_clock_mhz=2100.0, current_sm_clock_mhz=1200.0,
                             evidence="Active"))).to_dict()
    assert arm_normalised_cell(data, data["arms"][0]) == "raw (clocks locked)"


# --- the report spec 8.2 carries ----------------------------------------------------------

def results_for_both_cases():
    return {
        "restyle.json": a_primitive_result(RESTYLE_CASE).to_dict(),
        "identity.json": a_primitive_result(
            IDENTITY_CASE,
            arms=[an_arm(CROP, 88.0, kind="identity", expresses=False,
                         identity=IdentityCheck(
                             detector="yolo-world-s-640", asked_for=["dog", "cat"],
                             frames_probed=48, became=4, remained=44, conf=0.25,
                             achieved=False, statement="read 4 of 48 as cat")),
                  an_arm(MASKED, 84.0, kind="identity", expresses=False)],
            one_step_finding="**One-step SD-Turbo did not perform it.** Implication.",
        ).to_dict(),
    }


def test_the_report_names_the_chosen_primitive_and_why():
    report = format_primitive_report(results_for_both_cases())
    assert "**Decision: masked.**" in report
    assert "cannot express" in report


def test_the_report_puts_the_priority_case_first():
    report = format_primitive_report(results_for_both_cases())
    assert report.index(RESTYLE_CASE) < report.index(IDENTITY_CASE)


def test_the_report_states_the_denoise_strength_each_case_needed():
    report = format_primitive_report(results_for_both_cases())
    assert "Denoise strength each case turned out to need" in report
    assert report.count("t_index") >= 2


def test_the_report_carries_the_one_step_finding_and_the_small_object_gap():
    report = format_primitive_report(results_for_both_cases())
    assert "One-step SD-Turbo did not perform it" in report
    assert "Small objects" in report


def test_the_report_points_a_human_at_the_clips():
    """The metric ranks cost and stability, not beauty - the issue's third trap."""
    report = format_primitive_report(results_for_both_cases())
    assert "restyle-people-triptych.mp4" in report
    assert "not beauty" in report


def test_the_decision_is_recomputed_from_the_records_not_quoted():
    results = results_for_both_cases()
    assert decision_from(list(results.values())).primitive == MASKED

    for arm in results["restyle.json"]["arms"]:
        arm["expresses"] = arm["primitive"] == CROP
    assert decision_from(list(results.values())).primitive == CROP


def test_an_empty_directory_says_so_rather_than_inventing_a_decision(tmp_path):
    assert "no primitive comparison" in format_primitive_report(
        load_primitive_results(tmp_path))


def test_a_case_measured_twice_reports_only_its_most_recent_run():
    older = a_primitive_result()
    newer = a_primitive_result()
    older_data, newer_data = older.to_dict(), newer.to_dict()
    older_data["run"]["finished_utc"] = "2026-09-05T10:00:00Z"
    kept = latest_per_case({"old.json": older_data, "new.json": newer_data})
    assert list(kept) == ["new.json"]
