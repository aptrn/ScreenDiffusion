"""One row per (case, GPU) - so measuring on a second machine does not erase the first.

Issue #25. The three generated blocks - spec 8.1 (detectors), 8.2 (primitives) and
8.8 (the selective path) - each reduced their committed JSON to the newest run *per
case name*. With one machine that was right. With two it silently deletes the
comparison spec 7.4 is built on: the first 4090 `selective-people` run would drop the
3080 row out of 8.8, and the evidence for the portability table would exist only as
JSON nobody reads.

Two properties, and the second is what makes this safe to land mid-flight:

- a directory holding one case from two GPUs renders two rows, each labelled;
- a single-GPU directory renders **byte-identically** to before, so no committed
  block churns and the byte-match tests keep passing untouched.

GPU-free: every record here is synthetic, built from the same dataclasses the
runners write.
"""

import pytest

from bench.detector_results import format_detector_report, latest_per_detector
from bench.primitive_results import format_primitive_report
from bench.primitive_results import latest_per_case as latest_primitive_per_case
from bench.results import UNKNOWN_GPU, gpu_of, latest_per
from bench.selective import format_selective_report
from bench.selective import latest_per_case as latest_selective_per_case

from test_bench_detector_results import a_detector_result
from test_bench_primitive_results import a_primitive_result
from test_bench_results import a_fingerprint
from test_bench_selective import a_selective_result

LAPTOP = "NVIDIA GeForce RTX 3080 Laptop GPU"
DESKTOP = "NVIDIA GeForce RTX 4090"


def on(gpu: str):
    """A fingerprint that differs from the default in nothing but the GPU."""
    return a_fingerprint(gpu_name=gpu)


# --- the reduction ------------------------------------------------------------


def a_named(name: str, gpu: str, finished: str) -> dict:
    return {"name": name, "hardware": {"gpu_name": gpu},
            "run": {"finished_utc": finished}}


def named(result: dict) -> str:
    return result["name"]


def test_the_same_case_from_two_gpus_keeps_both_rows():
    kept = latest_per(
        {"laptop.json": a_named("people", LAPTOP, "2026-09-06T20:00:00Z"),
         "desktop.json": a_named("people", DESKTOP, "2026-09-07T20:00:00Z")},
        named)
    assert sorted(kept) == ["desktop.json", "laptop.json"]


def test_the_same_case_measured_twice_on_one_gpu_still_keeps_only_the_newest():
    kept = latest_per(
        {"old.json": a_named("people", LAPTOP, "2026-09-06T20:00:00Z"),
         "new.json": a_named("people", LAPTOP, "2026-09-07T20:00:00Z")},
        named)
    assert list(kept) == ["new.json"]


def test_a_record_that_does_not_name_its_machine_is_labelled_rather_than_dropped():
    """The issue's second trap. A result predating the fingerprint has no
    `hardware.gpu_name`; reading that as `None` would neither crash nor be honest."""
    assert gpu_of({"hardware": {}}) == UNKNOWN_GPU
    assert gpu_of({"hardware": {"gpu_name": "  "}}) == UNKNOWN_GPU
    assert gpu_of({}) == UNKNOWN_GPU


def test_two_records_with_no_machine_do_not_delete_each_other():
    """They cannot be shown to come from the same machine, so neither supersedes
    the other. Grouping them together is exactly the deletion this issue is about."""
    kept = latest_per(
        {"a.json": {"name": "people", "run": {"finished_utc": "2026-09-06T20:00:00Z"}},
         "b.json": {"name": "people", "run": {"finished_utc": "2026-09-07T20:00:00Z"}}},
        named)
    assert sorted(kept) == ["a.json", "b.json"]


# --- the selective block (spec 8.8) -------------------------------------------


def two_gpu_selective() -> dict:
    return {
        "laptop.json": a_selective_result(hardware=on(LAPTOP)).to_dict(),
        "desktop.json": a_selective_result(hardware=on(DESKTOP)).to_dict(),
    }


def table_rows(report: str, first_cell: str) -> list:
    return [line for line in report.splitlines()
            if line.startswith(f"| {first_cell}")]


def test_a_selective_case_from_two_gpus_survives_as_two_labelled_rows():
    kept = latest_selective_per_case(two_gpu_selective())
    assert len(kept) == 2

    report = format_selective_report(two_gpu_selective())
    rows = table_rows(report, "selective-people")
    assert len(rows) == 2
    assert [gpu for gpu in (LAPTOP, DESKTOP) if any(gpu in row for row in rows)] == \
        [LAPTOP, DESKTOP]


def test_the_selective_table_grows_a_gpu_column_only_when_it_needs_one():
    assert "| GPU |" in format_selective_report(two_gpu_selective())
    assert "| GPU |" not in format_selective_report(
        {"laptop.json": a_selective_result(hardware=on(LAPTOP)).to_dict()})


def test_the_selective_preamble_names_every_machine_rather_than_the_first():
    """Step 3: the prose said "Measured on <one GPU> ... belong to this GPU"."""
    report = format_selective_report(two_gpu_selective())
    preamble = report.splitlines()[0]
    assert LAPTOP in preamble and DESKTOP in preamble
    assert "belong to this GPU" not in preamble


def test_each_machines_gate_lines_say_which_machine_they_are_from():
    report = format_selective_report(two_gpu_selective())
    assert f"The Gate, measured on {LAPTOP}:" in report
    assert f"The Gate, measured on {DESKTOP}:" in report


def test_one_machine_still_says_the_gate_was_measured_without_qualifying_it():
    report = format_selective_report(
        {"laptop.json": a_selective_result(hardware=on(LAPTOP)).to_dict()})
    assert "The Gate, measured:" in report


def test_the_selective_rows_are_deterministic_and_group_a_case_by_machine():
    """Two cases over two GPUs, fed in shuffled: the order must not depend on it."""
    from bench.selective import CASES, PRIORITY_CASE

    other = CASES[PRIORITY_CASE].replace(name="selective-dogs")
    records = {
        "d-desktop.json": a_selective_result(case=other, hardware=on(DESKTOP)).to_dict(),
        "p-laptop.json": a_selective_result(hardware=on(LAPTOP)).to_dict(),
        "d-laptop.json": a_selective_result(case=other, hardware=on(LAPTOP)).to_dict(),
        "p-desktop.json": a_selective_result(hardware=on(DESKTOP)).to_dict(),
    }
    report = format_selective_report(records)
    assert report == format_selective_report(dict(reversed(list(records.items()))))

    cases = [row.split(" | ")[0].lstrip("| ") for row in table_rows(report, "selective")]
    assert cases == ["selective-dogs", "selective-dogs",
                     "selective-people", "selective-people"]


# --- the primitive block (spec 8.2) -------------------------------------------


def two_gpu_primitive() -> dict:
    return {
        "laptop.json": a_primitive_result(hardware=on(LAPTOP)).to_dict(),
        "desktop.json": a_primitive_result(hardware=on(DESKTOP)).to_dict(),
    }


def test_a_primitive_case_from_two_gpus_survives_as_labelled_rows():
    assert len(latest_primitive_per_case(two_gpu_primitive())) == 2
    report = format_primitive_report(two_gpu_primitive())
    assert "| GPU |" in report
    assert LAPTOP in report and DESKTOP in report


def test_the_primitive_decision_is_taken_per_machine_not_across_them():
    """A cost comparison between two primitives measured on two different GPUs is
    not a comparison. Each machine decides for itself, and says which it is."""
    report = format_primitive_report(two_gpu_primitive())
    assert f"**Decision ({LAPTOP}): masked.**" in report
    assert f"**Decision ({DESKTOP}): masked.**" in report


def test_one_machine_still_states_the_decision_unqualified():
    report = format_primitive_report(
        {"laptop.json": a_primitive_result(hardware=on(LAPTOP)).to_dict()})
    assert "**Decision: masked.**" in report


# --- the detector block (spec 8.1) --------------------------------------------


def two_gpu_detector() -> dict:
    return {
        "laptop.json": a_detector_result(hardware=on(LAPTOP)).to_dict(),
        "desktop.json": a_detector_result(hardware=on(DESKTOP)).to_dict(),
    }


def test_a_detector_measured_on_two_gpus_survives_as_labelled_rows():
    assert len(latest_per_detector(two_gpu_detector())) == 2
    report = format_detector_report(two_gpu_detector())
    assert "| GPU |" in report
    assert LAPTOP in report and DESKTOP in report


def test_the_detector_recommendation_is_made_per_machine():
    """Ranking two detectors against each other only means something within one
    machine; the same detector listed twice would rank against itself."""
    report = format_detector_report(two_gpu_detector())
    assert f"**Recommendation ({LAPTOP}):" in report
    assert f"**Recommendation ({DESKTOP}):" in report


def test_one_machine_still_states_the_recommendation_unqualified():
    report = format_detector_report(
        {"laptop.json": a_detector_result(hardware=on(LAPTOP)).to_dict()})
    assert "**Recommendation: " in report


# --- no churn -----------------------------------------------------------------


@pytest.mark.parametrize("report, records", [
    (format_selective_report, {"a.json": a_selective_result().to_dict()}),
    (format_primitive_report, {"a.json": a_primitive_result().to_dict()}),
    (format_detector_report, {"a.json": a_detector_result().to_dict()}),
])
def test_a_single_machine_block_carries_no_gpu_column(report, records):
    """The no-churn property, stated once per block. The committed blocks are held
    to a byte match by the three spec tests; this says why they still match."""
    assert "| GPU |" not in report(records)
