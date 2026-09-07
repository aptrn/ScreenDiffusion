"""The step-count sweep (issue #38, step 1): what one more denoising step costs.

Every figure in this repo is SD-Turbo at one step, and the one committed per-module
split is a 256x256 laptop cell. The sweep replaces the estimate that split was used
to make; this module is its arithmetic and the block spec 7.2 carries.
"""

from __future__ import annotations

import pytest

from bench.scenarios import SCENARIOS
from bench.steps import (
    ESTIMATED_FOUR_STEP_FACTOR,
    STEPS_SWEPT,
    arm_name,
    format_steps_report,
    is_step_arm,
    module_share,
    step_arm,
    steps_of,
)


def a_result(steps=1, ms_per_frame=20.0, unet=16.0, vae_encode=2.0, vae_decode=2.0,
             finished="2026-09-07T12:00:00Z", acceleration="none", gpu=None):
    from render_plan import t_index_ladder

    return {
        "scenario": {"name": arm_name(f"img2img-{acceleration}-512x512-b1", steps),
                     "acceleration": acceleration, "width": 512, "height": 512,
                     "batch_size": 1, "model": "sd-turbo-fp16",
                     "t_index_list": t_index_ladder(35, steps)},
        "run": {"mean_ms_per_frame": ms_per_frame, "finished_utc": finished,
                "peak_vram_bytes": 2 * 1024 ** 3, "mean_sm_clock_mhz": 1600.0,
                "max_temperature_c": 50.0, "fps": 1000.0 / ms_per_frame, "reps": 30,
                "per_module_ms": {"unet": unet, "vae_encode": vae_encode,
                                  "vae_decode": vae_decode}},
        "cooldown": {"outcome": "reached"},
        "hardware": {"gpu_name": gpu or "NVIDIA GeForce RTX 4090"},
        "clock_normalization": {"regime": "unlocked", "ms_per_frame": ms_per_frame,
                                "basis_mhz": 2520.0},
    }


# --- naming an arm, and keeping it out of the batch curve --------------------


def test_an_arm_is_named_after_the_step_count_it_ran_at():
    assert arm_name("img2img-none-512x512-b1", 4) == "img2img-none-512x512-b1-s4"


def test_the_one_step_control_is_an_arm_too():
    """It has to be, or the control is a laptop row from another sweep.

    `bench --marginal` reads every JSON in `bench/results/` as a batch cell, so a
    1-step re-measurement written there would land in spec 7.2's committed curve
    as a second batch-1 point on another machine.
    """
    assert is_step_arm(arm_name("img2img-none-512x512-b1", 1))
    assert not is_step_arm("img2img-none-512x512-b1")


def test_an_arm_keeps_the_opening_index_and_spends_the_rest_after_it():
    """Only the count moves. A sweep that also moved the strength would measure two
    things at once."""
    base = SCENARIOS["img2img-none-512x512-b1"]
    one, four = step_arm(base, 1), step_arm(base, 4)
    assert one.t_index_list[0] == four.t_index_list[0] == base.t_index_list[0]
    assert one.steps == 1 and four.steps == 4


def test_an_arm_at_four_steps_keys_a_four_batch_unet():
    """`use_denoising_batch` puts the steps through the UNet as a batch, which is
    what a TensorRT engine is compiled for - and what makes a 4-step arm a build."""
    four = step_arm(SCENARIOS["img2img-tensorrt-512x512-b1"], 4)
    assert four.unet_batch_size == 4


def test_the_step_count_is_read_off_the_record_rather_than_its_name():
    assert steps_of(a_result(steps=4)) == 4


# --- what the table says -----------------------------------------------------


def test_the_module_share_is_the_submodule_over_the_measured_call():
    assert module_share(a_result(ms_per_frame=20.0, unet=16.0), "unet") == pytest.approx(0.8)


def test_a_missing_per_module_split_is_no_share_rather_than_zero():
    result = a_result()
    result["run"]["per_module_ms"] = None
    assert module_share(result, "unet") is None


def test_the_report_scales_every_arm_against_the_one_step_control():
    results = {"a.json": a_result(steps=1, ms_per_frame=20.0, unet=16.0),
               "b.json": a_result(steps=4, ms_per_frame=56.0, unet=64.0)}
    report = format_steps_report(results)
    assert "2.80x" in report, report


def test_the_report_names_the_estimate_it_replaces_and_says_whether_it_held():
    """The issue's Context estimated ~3.4x on the diffusion call from a 256x256
    laptop split. A sweep that did not answer that sentence has not replaced it."""
    results = {"a.json": a_result(steps=1, ms_per_frame=20.0),
               "b.json": a_result(steps=4, ms_per_frame=56.0)}
    report = format_steps_report(results)
    assert f"{ESTIMATED_FOUR_STEP_FACTOR:.1f}x" in report
    assert "2.80x" in report


def test_an_empty_directory_says_so_rather_than_drawing_an_empty_table():
    assert "no step-count sweep" in format_steps_report({})


def test_the_sweep_the_issue_asks_for_is_one_two_four():
    assert STEPS_SWEPT == (1, 2, 4)


def test_two_machines_grow_a_gpu_column_and_neither_row_is_deleted():
    results = {"a.json": a_result(steps=1, gpu="NVIDIA GeForce RTX 4090"),
               "b.json": a_result(steps=1, gpu="NVIDIA GeForce RTX 3080 Laptop GPU")}
    report = format_steps_report(results)
    assert "| GPU |" in report
    assert "RTX 4090" in report and "RTX 3080" in report


def test_an_arm_is_scaled_against_its_own_accelerator_not_another():
    """A TensorRT 1-step call against a `none` 1-step call is the accelerator's
    ratio, not the step count's - and printed in the `x 1 step` column it would
    read as the second."""
    results = {"a.json": a_result(steps=1, ms_per_frame=26.0, acceleration="none"),
               "b.json": a_result(steps=1, ms_per_frame=20.0,
                                  acceleration="tensorrt")}
    report = format_steps_report(results)
    assert "0.79x" not in report, report
    assert report.count("1.00x") == 2, report


def test_the_report_prices_the_unet_passes_as_well_as_the_call():
    """Four steps are four UNet passes but one batched call, and the whole reason
    the estimate was wrong is that the second is not four times the first."""
    results = {"a.json": a_result(steps=1, ms_per_frame=26.0, unet=16.0),
               "b.json": a_result(steps=4, ms_per_frame=48.0, unet=32.0)}
    report = format_steps_report(results)
    assert "2.00x the UNet" in report, report


# --- the second axis: the base model (issue #38, step 2) ---------------------


def sd15(steps=4, ms_per_frame=50.0, **overrides):
    result = a_result(steps=steps, ms_per_frame=ms_per_frame, **overrides)
    result["scenario"]["model"] = "sd-v1-5-fp16"
    result["scenario"]["use_lcm_lora"] = True
    result["scenario"]["name"] += "-sd15"
    return result


def test_an_arm_is_scaled_against_its_own_base_model():
    """SD 1.5 at four steps against SD-Turbo at one is the model *and* the count.
    In a column headed `x 1 step` it would read as the count alone, so with no
    one-step arm of its own model that cell has no figure to hold."""
    results = {"a.json": a_result(steps=1, ms_per_frame=26.0),
               "b.json": sd15(steps=4, ms_per_frame=50.0)}
    row = next(line for line in format_steps_report(results).splitlines()
               if "sd-v1-5-fp16" in line and line.startswith("|"))
    assert "x |" not in row and "x " not in row.split("50.00")[1], row


def test_a_cross_model_gap_is_only_claimed_where_both_models_were_measured():
    """"within 0% of each other" from two arms that were never both measured is a
    finding invented out of a missing record."""
    results = {"a.json": a_result(steps=1, ms_per_frame=26.0),
               "b.json": sd15(steps=4, ms_per_frame=50.0)}
    assert "of each other" not in format_steps_report(results)


def test_the_table_grows_a_model_column_once_two_models_are_in_it():
    one_model = {"a.json": a_result(steps=1)}
    assert "| model |" not in format_steps_report(one_model)
    two = {"a.json": a_result(steps=1), "b.json": sd15()}
    assert "| model |" in format_steps_report(two)


def test_the_report_prices_the_second_model_against_the_shipped_one():
    """The question issue #38 asks: what does moving to SD 1.5 cost per frame,
    at the step count each model actually needs?"""
    results = {"a.json": a_result(steps=1, ms_per_frame=26.0),
               "b.json": sd15(steps=4, ms_per_frame=52.0)}
    report = format_steps_report(results)
    assert "2.00x" in report
    assert "sd-v1-5-fp16" in report and "sd-turbo-fp16" in report
