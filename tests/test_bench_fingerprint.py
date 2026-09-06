"""The hardware fingerprint (issue #2, spec 7.4).

Every result carries one: it says which machine produced the number, and it is the
evidence the number was measured rather than invented. The parsing is pure, so the
laptop's `[N/A]` power limit is covered here rather than discovered mid-run.
"""

import pytest

from bench.fingerprint import (
    FINGERPRINT_FIELDS,
    SAMPLE_FIELDS,
    build_fingerprint,
    parse_csv_row,
    parse_number,
)


def test_the_queried_fields_are_the_ones_the_spec_names():
    assert set(FINGERPRINT_FIELDS) == {"name", "memory.total", "driver_version",
                                       "power.limit", "enforced.power.limit"}
    assert set(SAMPLE_FIELDS) == {"clocks.sm", "temperature.gpu"}


def test_a_csv_row_maps_onto_the_requested_fields():
    row = parse_csv_row(("name", "memory.total"), "NVIDIA GeForce RTX 3080 Laptop GPU, 16384\n")
    assert row == {"name": "NVIDIA GeForce RTX 3080 Laptop GPU", "memory.total": "16384"}


def test_only_the_first_gpu_is_read():
    row = parse_csv_row(("name",), "GPU A\nGPU B\n")
    assert row == {"name": "GPU A"}


def test_empty_output_is_an_empty_row_not_a_crash():
    assert parse_csv_row(("name",), "") == {}


def test_na_becomes_none_because_this_laptop_reports_no_power_limit():
    assert parse_number("[N/A]") is None
    assert parse_number("N/A") is None
    assert parse_number("") is None
    assert parse_number(None) is None
    assert parse_number("115.00") == pytest.approx(115.0)
    assert parse_number("16384 MiB") == pytest.approx(16384.0)


def test_a_fingerprint_carries_the_machine_and_its_evidence():
    fp = build_fingerprint(
        query_row={"name": "NVIDIA GeForce RTX 3080 Laptop GPU", "memory.total": "16384",
                   "driver_version": "595.79", "power.limit": "[N/A]",
                   "enforced.power.limit": "120.00"},
        nvidia_smi_raw="Fri Sep  6 14:00:00 2026\n+---- raw table ----+\n",
        captured_utc="2026-09-06T14:00:00Z",
        torch_version="2.7.0+cu128",
        cuda_version="12.8",
        hostname="dev-laptop",
    )
    assert fp.gpu_name == "NVIDIA GeForce RTX 3080 Laptop GPU"
    assert fp.total_vram_mib == pytest.approx(16384.0)
    assert fp.driver_version == "595.79"
    # This laptop reports no `power.limit` but does enforce one. Both are recorded:
    # unavailable is a fact about the machine, not a field to omit.
    assert fp.power_limit_w is None
    assert fp.enforced_power_limit_w == pytest.approx(120.0)
    assert "raw table" in fp.nvidia_smi_raw
    assert fp.nvidia_smi_captured_utc == "2026-09-06T14:00:00Z"
    assert {"power_limit_w", "enforced_power_limit_w"} <= set(fp.to_dict())
