"""Marginal cost per additional batch item (issue #3, spec 7.2 item 2 / 7.3).

The question the orchestrator design rests on: does the second crop in a batch cost
as much as the first, or much less? That is a division over numbers already in the
committed result files, so it is computed here rather than asserted in prose - a
reviewer can recompute it from the JSON, and this is the arithmetic they would use.
"""

import json

import pytest

from sourceloader import ROOT

from bench.marginal import (
    BatchPoint,
    curves_from_results,
    format_table,
    latest_result_per_scenario,
    point_from_result,
)


def a_result(batch_size=1, ms_per_frame=50.0, acceleration="none", size=512,
             finished="2026-09-06T12:00:00Z", name=None):
    name = name or f"img2img-{acceleration}-{size}x{size}-b{batch_size}"
    return {
        "scenario": {"name": name, "acceleration": acceleration, "width": size,
                     "height": size, "batch_size": batch_size, "t_index_list": [35]},
        "run": {"mean_ms_per_frame": ms_per_frame, "finished_utc": finished,
                "peak_vram_bytes": 2 * 1024 ** 3, "mean_sm_clock_mhz": 1600.0,
                "reps": 30},
        "cooldown": {"outcome": "reached"},
        "hardware": {"gpu_name": "NVIDIA GeForce RTX 3080 Laptop GPU"},
    }


def test_a_point_carries_the_per_call_cost_as_well_as_the_per_frame_one():
    """Per *call* is what the marginal question is about: one call renders the batch."""
    point = point_from_result(a_result(batch_size=4, ms_per_frame=20.0), source="x.json")
    assert isinstance(point, BatchPoint)
    assert point.batch_size == 4
    assert point.ms_per_frame == 20.0
    assert point.ms_per_call == 80.0
    assert point.source == "x.json"


def test_the_marginal_item_is_the_slope_between_two_measured_batches():
    """batch 1 -> 90 ms, batch 2 -> 100 ms: the second item cost 10 ms, not 90."""
    results = {"a.json": a_result(batch_size=1, ms_per_frame=90.0),
               "b.json": a_result(batch_size=2, ms_per_frame=50.0)}
    curve = curves_from_results(results)[0]
    assert [p.batch_size for p in curve.points] == [1, 2]
    assert curve.first_item_ms == 90.0
    step = curve.steps[0]
    assert (step.from_batch, step.to_batch) == (1, 2)
    assert step.marginal_ms_per_item == pytest.approx(10.0)
    assert step.fraction_of_first_item == pytest.approx(10.0 / 90.0)


def test_the_slope_spans_the_gap_when_batches_are_not_consecutive():
    """1 / 2 / 4 / 8 is the sweep, so a step covers 4 items, not one."""
    results = {"a.json": a_result(batch_size=4, ms_per_frame=25.0),   # 100 ms/call
               "b.json": a_result(batch_size=8, ms_per_frame=15.0)}   # 120 ms/call
    curve = curves_from_results(results)[0]
    assert curve.steps[0].marginal_ms_per_item == pytest.approx(5.0)


def test_a_curve_is_sublinear_when_extra_items_cost_less_than_the_first():
    cheap = curves_from_results({"a.json": a_result(batch_size=1, ms_per_frame=90.0),
                                 "b.json": a_result(batch_size=2, ms_per_frame=50.0)})[0]
    assert cheap.sublinear is True

    # 90 ms per item however many there are: batching buys nothing.
    linear = curves_from_results({"a.json": a_result(batch_size=1, ms_per_frame=90.0),
                                  "b.json": a_result(batch_size=2, ms_per_frame=90.0)})[0]
    assert linear.sublinear is False


def test_sublinearity_is_undefined_without_a_batch_of_one_to_compare_against():
    curve = curves_from_results({"a.json": a_result(batch_size=2, ms_per_frame=50.0)})[0]
    assert curve.steps == ()
    assert curve.sublinear is None


def test_curves_are_grouped_by_accelerator_and_resolution():
    results = {
        "a.json": a_result(batch_size=1, size=256),
        "b.json": a_result(batch_size=2, size=256),
        "c.json": a_result(batch_size=1, size=512),
        "d.json": a_result(batch_size=1, size=512, acceleration="tensorrt"),
    }
    grouped = {(c.acceleration, c.width) for c in curves_from_results(results)}
    assert grouped == {("none", 256), ("none", 512), ("tensorrt", 512)}


def test_a_re_run_supersedes_the_earlier_result_for_the_same_cell():
    """The sweep was run twice; the curve is the later run, not a mixture of both."""
    results = {
        "old.json": a_result(ms_per_frame=90.0, finished="2026-09-06T11:00:00Z"),
        "new.json": a_result(ms_per_frame=80.0, finished="2026-09-06T12:00:00Z"),
    }
    latest = latest_result_per_scenario(results)
    assert set(latest) == {"new.json"}
    assert latest["new.json"]["run"]["mean_ms_per_frame"] == 80.0


def test_the_table_is_markdown_a_spec_section_can_carry():
    results = {"a.json": a_result(batch_size=1, ms_per_frame=90.0),
               "b.json": a_result(batch_size=2, ms_per_frame=50.0)}
    table = format_table(curves_from_results(results))
    assert table.startswith("|")
    assert "| 512x512 |" in table
    assert "RTX 3080" in table


def test_the_committed_none_sweep_covers_every_cell_the_issue_asks_for():
    """The gate: a result file for every (resolution, batch) cell on `none`."""
    results = {path.name: json.loads(path.read_text(encoding="utf-8"))
               for path in (ROOT / "bench" / "results").glob("*.json")}
    cells = {(r["scenario"]["width"], r["scenario"]["batch_size"])
             for r in results.values() if r["scenario"]["acceleration"] == "none"}
    assert cells == {(res, batch) for res in (256, 384, 512) for batch in (1, 2, 4, 8)}


def test_a_single_measured_batch_says_so_rather_than_blaming_a_missing_batch_of_one():
    """The TensorRT confirmation runs are few; one of them is a curve of one point."""
    from bench.marginal import format_verdicts

    curve = curves_from_results({"a.json": a_result(batch_size=1)})[0]
    assert curve.first_item_ms == 50.0
    assert curve.sublinear is None
    assert "one batch size" in format_verdicts([curve])


def test_the_reference_clock_is_the_fastest_one_any_cell_actually_ran_at():
    """Normalising to a clock no cell reached would invent a number nothing measured."""
    from bench.marginal import reference_clock_mhz

    results = {"a.json": a_result(batch_size=1), "b.json": a_result(batch_size=2)}
    results["a.json"]["run"]["mean_sm_clock_mhz"] = 1785.0
    results["b.json"]["run"]["mean_sm_clock_mhz"] = 900.0
    assert reference_clock_mhz(curves_from_results(results)) == 1785.0


def test_normalising_undoes_the_power_limit_the_laptop_imposed_mid_run():
    """A cell that ran at half the clock did about half as much work per millisecond.

    Diffusion here is compute-bound, so time scales roughly with 1/clock. This is an
    estimate and labelled one - but without it the 512 curve reports the laptop's
    power limit as if it were the shape of the marginal-cost curve, which is the one
    conclusion spec 7.4 says *is* supposed to be portable.
    """
    from bench.marginal import reference_clock_mhz

    results = {"a.json": a_result(batch_size=1, ms_per_frame=50.0),
               "b.json": a_result(batch_size=2, ms_per_frame=50.0)}
    results["a.json"]["run"]["mean_sm_clock_mhz"] = 1600.0
    results["b.json"]["run"]["mean_sm_clock_mhz"] = 800.0

    raw = curves_from_results(results)[0]
    assert raw.sublinear is False, "at face value the second item cost as much as the first"

    normalised = raw.normalised_to(reference_clock_mhz([raw]))
    assert normalised.points[0].ms_per_call == pytest.approx(50.0)  # already at reference
    assert normalised.points[1].ms_per_call == pytest.approx(50.0)  # 100 ms at half clock
    assert normalised.steps[0].marginal_ms_per_item == pytest.approx(0.0)
    assert normalised.sublinear is True
    assert normalised.points[1].mean_sm_clock_mhz == 1600.0


def test_a_cell_with_no_clock_reading_is_left_alone_rather_than_guessed_at():
    results = {"a.json": a_result(batch_size=1, ms_per_frame=50.0)}
    results["a.json"]["run"]["mean_sm_clock_mhz"] = None
    point = curves_from_results(results)[0].normalised_to(1600.0).points[0]
    assert point.ms_per_call == 50.0
    assert point.mean_sm_clock_mhz is None
