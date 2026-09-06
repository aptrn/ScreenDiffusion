"""The hardcoded priority-case plan, and the denoise the engine renders it at.

Issue #8, step 4. Two things live here.

The **plan**: `person` -> `lower_half` -> a subtle low-denoise change, which is the
v1 case the whole design is for and the one issue #5 measured the rendering
primitive against. It is a producer in `render_plan.py` beside `plan_from_fields`,
so the worker can be driven end to end before anything wires the GUI up.

The **mapping** from the plan's `denoise` to the engine's `t_index`. The plan says
how much of the frame to replace, on a 0-1 scale; the engine takes an index into a
50-step schedule that runs the other way. That arithmetic is pure and stdlib, and
it is checked against a measurement rather than against itself: issue #5's committed
comparison recorded the noise amplitude of every rung it swept, and this file
demands the same numbers out of `noise_amplitude`.
"""

import json
from pathlib import Path

import pytest

from render_plan import (
    PRIORITY_CONCEPT,
    PRIORITY_REGION,
    SCHEDULE_T_INDEX_RANGE,
    SELECTIVE,
    noise_amplitude,
    priority_case_plan,
    t_index_for_denoise,
    timestep_of,
)

RESULTS = Path(__file__).resolve().parent.parent / "bench" / "results" / "primitives"
LOW, HIGH = SCHEDULE_T_INDEX_RANGE


@pytest.fixture(scope="module")
def restyle_record():
    """The committed measurement of the priority case (issue #5)."""
    records = sorted(RESULTS.glob("restyle-people-*.json"))
    assert records, f"no committed restyle-people comparison under {RESULTS}"
    return json.loads(records[-1].read_text(encoding="utf-8"))


def masked_arm(record):
    arm, = [a for a in record["arms"] if a["primitive"] == "masked"]
    return arm


# --- the schedule -----------------------------------------------------------


def test_a_higher_index_is_less_denoise():
    """The gotcha the whole repo repeats, as an assertion."""
    assert noise_amplitude(20) > noise_amplitude(30) > noise_amplitude(45)


def test_the_timesteps_are_the_ones_the_measurement_recorded(restyle_record):
    for point in masked_arm(restyle_record)["denoise"]["points"]:
        assert timestep_of(point["t_index"]) == point["timestep"]


def test_the_noise_amplitudes_are_the_ones_the_measurement_recorded(restyle_record):
    """The mapping is pinned to the engine's own scheduler, through a run that
    read `alphas_cumprod` off it - not to a curve that looks about right."""
    for point in masked_arm(restyle_record)["denoise"]["points"]:
        computed = noise_amplitude(point["t_index"])
        assert abs(computed - point["strength"]) <= 2e-6, (
            f"t_index {point['t_index']}: computed {computed}, "
            f"measured {point['strength']}")


@pytest.mark.parametrize("t_index", [20, 25, 30, 35, 40, 45])
def test_a_denoise_taken_off_the_schedule_maps_back_to_its_own_index(t_index):
    assert t_index_for_denoise(noise_amplitude(t_index)) == t_index


def test_more_denoise_asks_for_a_lower_index():
    assert t_index_for_denoise(0.9) < t_index_for_denoise(0.4)


def test_a_denoise_outside_the_schedule_lands_on_its_end():
    assert t_index_for_denoise(1.0) == LOW
    assert t_index_for_denoise(0.0) == HIGH


@pytest.mark.parametrize("denoise", [0.0, 0.05, 0.33, 0.5, 0.77, 1.0])
def test_every_denoise_maps_into_the_usable_range(denoise):
    assert LOW <= t_index_for_denoise(denoise) <= HIGH


# --- the hardcoded plan -----------------------------------------------------


def test_the_priority_plan_is_the_case_the_issue_names():
    plan = priority_case_plan()
    assert plan.mode == SELECTIVE
    target, = plan.targets
    assert target.concept == PRIORITY_CONCEPT
    assert target.region == PRIORITY_REGION


def test_the_priority_plan_asks_for_the_denoise_the_comparison_measured(restyle_record):
    """Issue #5 swept this case and selected a strength for the masked primitive;
    the demo plan asks for that one, so the end-to-end run is not a fresh guess."""
    measured = masked_arm(restyle_record)["denoise"]["t_index"]
    assert t_index_for_denoise(priority_case_plan().effective_denoise) == measured


def test_the_priority_plan_carries_a_prompt_to_the_engine():
    plan = priority_case_plan()
    assert plan.effective_prompt
    assert plan.effective_prompt == plan.targets[0].prompt


def test_the_priority_plan_counts_its_version_up_from_the_one_it_replaces():
    assert priority_case_plan(previous_version=7).plan_version == 8


def test_the_priority_plan_is_a_validated_plan():
    """It goes through the same door every other plan does; a hardcoded plan that
    bypassed the validator could carry a field the worker does not honour."""
    from render_plan import validate_plan

    plan = priority_case_plan()
    revalidated = validate_plan(plan.to_dict(), previous_version=plan.plan_version - 1)
    assert revalidated.plan is not None, revalidated.reason
    assert revalidated.plan.to_dict() == plan.to_dict()
