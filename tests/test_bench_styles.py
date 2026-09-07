"""Style LoRAs on the SD 1.5 arm (issue #38, steps 3-5).

The Gate's sharp sentence: *a LoRA that loads and changes nothing is a failure, not
a pass*. So an arm carries two numbers, not one - whether the render did anything at
all against the source (net of the resize control, the way spec 8.2 measures it) and
whether the *LoRA* did anything against the same arm without it - and a verdict is
only a pass when both clear their thresholds.

The second trap is the other half: LoCon / LyCORIS convolution layers do not load on
this diffusers version, so which formats fail has to be recorded rather than inferred
from one that happened to work.
"""

from __future__ import annotations

import pytest

from bench.styles import (
    BASE_ARM,
    LORA_CHANGE_THRESHOLD,
    VISIBLE_CHANGE,
    StyleArm,
    delivery_recommendation,
    format_style_report,
    style_verdict,
)


def an_arm(style="loving-vincent", loaded=True, error=None, change_vs_source=30.0,
           control_change=3.0, change_vs_base=14.0, ms_per_frame=50.0,
           flicker=1.5, fmt="LoRA (kohya, linear only)"):
    return StyleArm(
        style=style, filename=f"{style}.safetensors", scale=0.9,
        source=f"hf/{style}", format=fmt, loaded=loaded, error=error,
        ms_per_frame=ms_per_frame, change_vs_source=change_vs_source,
        control_change=control_change, change_vs_base=change_vs_base,
        flicker=flicker, frames=24,
    )


def a_result(arms=None, gpu="NVIDIA GeForce RTX 4090",
             finished="2026-09-07T12:00:00Z"):
    arms = arms if arms is not None else [
        an_arm(style=BASE_ARM, change_vs_base=0.0),
        an_arm(),
    ]
    return {
        "case": {"name": "style-sd15", "clip": "people.mp4", "frames": 24,
                 "base_scenario": "img2img-none-512x512-b1-sd15", "steps": 4,
                 "prompt": "a painting", "denoise": 0.49, "canvas": 512},
        "clip": {"name": "people.mp4", "sha256": "abc", "width": 1280,
                 "height": 720, "fps": 24.0, "total_frames": 100,
                 "start_frame": 0, "frames_used": 24},
        "arms": [arm.to_dict() for arm in arms],
        "run": {"finished_utc": finished, "started_utc": finished},
        "cooldown": {"outcome": "reached"},
        "hardware": {"gpu_name": gpu},
        "clock_normalization": {"regime": "unlocked"},
        "comparison_still": "style-sd15-comparison.jpg",
    }


# --- the two numbers, and the verdict that needs both ------------------------


def test_a_lora_that_loads_and_changes_nothing_is_a_failure():
    verdict = style_verdict(an_arm(change_vs_base=0.4))
    assert not verdict.passed
    assert "did not change" in verdict.statement


def test_a_lora_that_changes_the_output_but_renders_nothing_is_also_a_failure():
    """Net of the control. A render that only resampled has not restyled anything,
    whatever the LoRA did to the weights."""
    verdict = style_verdict(an_arm(change_vs_source=4.0, control_change=3.0))
    assert not verdict.passed


def test_a_lora_that_loads_renders_and_changes_the_output_passes():
    verdict = style_verdict(an_arm())
    assert verdict.passed
    assert f"{LORA_CHANGE_THRESHOLD:.0f}" in verdict.statement


def test_a_lora_that_did_not_load_fails_carrying_its_reason():
    arm = an_arm(loaded=False, error="conv layers are unsupported",
                 change_vs_source=0.0, change_vs_base=0.0)
    verdict = style_verdict(arm)
    assert not verdict.passed
    assert "conv layers are unsupported" in verdict.statement


def test_the_net_change_subtracts_the_resize_control():
    arm = an_arm(change_vs_source=30.0, control_change=3.0)
    assert arm.net_change == pytest.approx(27.0)
    assert VISIBLE_CHANGE > 0


def test_the_base_arm_is_never_judged_as_a_style():
    """It has no LoRA to have changed anything, and scoring it against itself would
    disqualify the control every run."""
    assert style_verdict(an_arm(style=BASE_ARM, change_vs_base=0.0)) is None


# --- the block, and the delivery decision ------------------------------------


def test_the_report_says_which_formats_loaded_and_which_did_not():
    arms = [an_arm(style=BASE_ARM, change_vs_base=0.0),
            an_arm(style="loving-vincent"),
            an_arm(style="locon-probe", loaded=False, fmt="LoCon / LyCORIS",
                   error="conv2d weights are not supported",
                   change_vs_source=0.0, change_vs_base=0.0)]
    report = format_style_report({"a.json": a_result(arms)})
    assert "LoCon / LyCORIS" in report
    assert "conv2d weights are not supported" in report


def test_the_report_counts_how_many_styles_actually_worked():
    arms = [an_arm(style=BASE_ARM, change_vs_base=0.0),
            an_arm(style="a"), an_arm(style="b")]
    report = format_style_report({"a.json": a_result(arms)})
    assert "2 of 2" in report


def test_an_empty_directory_says_so():
    assert "no style-LoRA" in format_style_report({})


def a_step_record(name, ms, accel, style=None):
    return {
        "scenario": {"name": name, "acceleration": accel, "width": 512,
                     "height": 512, "batch_size": 1, "model": "sd-v1-5-fp16",
                     "style_lora": style, "t_index_list": [35, 40, 44, 49]},
        "run": {"mean_ms_per_frame": ms, "finished_utc": "2026-09-07T12:00:00Z",
                "peak_vram_bytes": 2 * 1024 ** 3, "mean_sm_clock_mhz": 2500.0},
        "cooldown": {"outcome": "reached"},
        "hardware": {"gpu_name": "NVIDIA GeForce RTX 4090"},
    }


def test_the_delivery_decision_prices_both_paths_from_the_committed_arms():
    """Step 4: pre-built engines per style, or a hot-swappable slower path. The
    answer is two milliseconds figures and one engine cost, not a preference."""
    records = {
        "a.json": a_step_record("x-tensorrt-sd15-s4", 30.0, "tensorrt"),
        "b.json": a_step_record("x-tensorrt-sd15-lv-s4", 31.0, "tensorrt",
                                style="loving-vincent"),
        "c.json": a_step_record("x-none-sd15-s4", 50.0, "none"),
        "d.json": a_step_record("x-none-sd15-lv-s4", 51.0, "none",
                                style="loving-vincent"),
    }
    recommendation = delivery_recommendation(records)
    assert recommendation is not None
    assert "31.00" in recommendation.statement
    assert "51.00" in recommendation.statement


def test_the_delivery_decision_is_absent_rather_than_guessed():
    assert delivery_recommendation({}) is None
