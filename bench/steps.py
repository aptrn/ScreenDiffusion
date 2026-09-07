"""What one more denoising step costs. Issue #38 step 1, spec 7.2.

Every millisecond in this repo is SD-Turbo at **one** step, and the one committed
per-module split is a 256x256 `none` cell on the laptop - UNet 36.69 ms against
VAE-encode 4.25 and VAE-decode 4.66. Issue #38 reasoned from it that four steps
would cost about 3.4x the diffusion call, and opened by saying that is an estimate
and this sweep exists to replace it. So the sweep is the *first* thing to run: it
isolates the step axis from the model axis, needs no new download, and on the
`none` accelerator it needs no engine either. A model that needs four steps is
priced before anything is compiled for it.

Only the step count moves. The opening index is the one `DEFAULT_T_INDEX_LIST`
already carries, and `render_plan.t_index_ladder` spends the extra steps after it,
so an arm is not also a strength change.

**Arms get their own directory.** `bench --marginal` reads every JSON in
`bench/results/` as a (resolution, batch) cell of spec 7.2's committed curve, and
a 2-step arm at batch 1 would land in it as a second batch-1 point. That includes
the 1-step control, which is why `--steps 1` is an arm rather than a plain run:
the control has to be measured on this machine, in this session, beside the arms
it is the yardstick for.

GPU-free, like every other `bench.*` results module: the record is a plain
`BenchResult` written by `bench.runner`, and this is the arithmetic over it.
"""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from bench.results import (
    BYTES_PER_MIB,
    GpuColumn,
    OptionalColumn,
    distinct_gpus,
    format_number,
    gpu_of,
    latest_per,
    measured_on,
    normalised_cell,
    table_separator,
)
from bench.scenarios import ScenarioConfig

# The three the issue names. One is the control, two says whether the second step
# is cheaper than the first, four is what SD 1.5 + LCM-LoRA actually needs.
STEPS_SWEPT: Tuple[int, ...] = (1, 2, 4)

# The figure issue #38's Context reasoned to, from the 256x256 laptop split: the
# UNet is ~80% of the call there, so four passes would be ~3.4x the call. Written
# down so the report can say whether the measurement agreed with it, which is what
# "replacing the estimate" means.
ESTIMATED_FOUR_STEP_FACTOR = 3.4

# The one step every committed figure in this repo was measured at.
CONTROL_STEPS = 1

_ARM_SUFFIX = re.compile(r"-s(\d+)$")


def arm_name(scenario_name: str, steps: int) -> str:
    """`img2img-none-512x512-b1` at 4 steps -> `img2img-none-512x512-b1-s4`."""
    return f"{scenario_name}-s{int(steps)}"


def is_step_arm(scenario_name: str) -> bool:
    """Was this name made by `arm_name`? The one rule that routes a record."""
    return _ARM_SUFFIX.search(scenario_name) is not None


def step_arm(scenario: ScenarioConfig, steps: int) -> ScenarioConfig:
    """`scenario` at `steps` denoising steps, renamed after the count it ran at.

    The ladder comes from `render_plan`, the shipped module, so the arm denoises
    through the same indices a plan asking for the same opening strength would.
    """
    from render_plan import t_index_ladder

    ladder = t_index_ladder(scenario.t_index_list[0], steps)
    return scenario.replace(name=arm_name(scenario.name, steps), t_index_list=ladder)


# --- reading a committed arm -------------------------------------------------


def steps_of(result: Mapping) -> int:
    """How many steps this record was measured at, off the schedule it recorded.

    Read from `t_index_list` rather than from the name: the scenario is serialised
    whole into the result exactly so a figure can be re-read without its label.
    """
    return len(result["scenario"]["t_index_list"])


def ms_per_call(result: Mapping) -> float:
    """One diffusion call. Batch 1 throughout the sweep, so this is ms/frame."""
    return float(result["run"]["mean_ms_per_frame"]) * int(
        result["scenario"]["batch_size"])


def module_ms(result: Mapping, module: str) -> Optional[float]:
    """Milliseconds in one submodule per frame, or None without `--per-module`."""
    split = result["run"].get("per_module_ms") or {}
    value = split.get(module)
    return None if value is None else float(value)


def module_share(result: Mapping, module: str) -> Optional[float]:
    """That submodule as a fraction of the measured call.

    None rather than 0.0 when the arm was run without `--per-module`: a share of
    nothing is not a measurement of zero.
    """
    ms = module_ms(result, module)
    total = ms_per_call(result)
    if ms is None or not total:
        return None
    return ms / total


def latest_per_arm(results: Mapping[str, dict]) -> Dict[str, dict]:
    """The newest run of each arm per machine - the same rule every report uses."""
    return latest_per(results, lambda result: result["scenario"]["name"])


def accelerator_of(result: Mapping) -> str:
    return str(result["scenario"]["acceleration"])


def model_of(result: Mapping) -> str:
    return str(result["scenario"]["model"])


NO_STYLE = "-"


def style_of(result: Mapping) -> str:
    """The style LoRA fused into this arm, or `-` when none was (issue #38).

    Part of the configuration and not a footnote: a fused LoRA is compiled into
    the UNet weights, so an arm carrying one is a different engine measured at the
    same step count - two rows that look identical without this column.
    """
    return str(result["scenario"].get("style_lora") or NO_STYLE)


# What has to be equal before two arms differ only in their step count. A TensorRT
# call against a `none` one is the accelerator's ratio, an SD 1.5 call against an
# SD-Turbo one is the model's, and a fused-LoRA call against a bare one is the
# LoRA's; printed in a column headed `x 1 step`, any of them would read as the
# step count's.
Configuration = Tuple[str, str, str, str]


def configuration_of(result: Mapping) -> Configuration:
    return (gpu_of(result), accelerator_of(result), model_of(result),
            style_of(result))


def control_of(arms: Sequence[dict],
               configuration: Optional[Configuration] = None) -> Optional[dict]:
    """The one-step arm these are scaled against, or None if it was not measured."""
    for arm in arms:
        if steps_of(arm) != CONTROL_STEPS:
            continue
        if configuration is None or configuration_of(arm) == configuration:
            return arm
    return None


def in_configuration(arms: Sequence[dict],
                     configuration: Configuration) -> List[dict]:
    return [arm for arm in arms if configuration_of(arm) == configuration]


def factor_over_control(result: Mapping,
                        control: Optional[Mapping]) -> Optional[float]:
    """How many times the one-step call this arm cost."""
    if control is None:
        return None
    base = ms_per_call(control)
    return None if not base else ms_per_call(result) / base


# --- the block spec 7.2 carries ----------------------------------------------


REPORT_HEADER = ("| steps | accel | t_index list | ms/call | x 1 step |"
                 " UNet ms | VAE encode ms | VAE decode ms | UNet share |"
                 " peak VRAM (MiB) | SM clock (MHz) | cooldown |"
                 " ms/call at basis clock |")


def _row(result: dict, column: GpuColumn, models: OptionalColumn,
         styles: OptionalColumn, control: Optional[dict]) -> str:
    scenario, run = result["scenario"], result["run"]
    share = module_share(result, "unet")
    factor = factor_over_control(result, control)
    cells = styles.cells(models.cells([
        str(steps_of(result)),
        scenario["acceleration"],
        ",".join(str(index) for index in scenario["t_index_list"]),
        format_number(ms_per_call(result), 2),
        "-" if factor is None else f"{factor:.2f}x",
        format_number(module_ms(result, "unet"), 2),
        format_number(module_ms(result, "vae_encode"), 2),
        format_number(module_ms(result, "vae_decode"), 2),
        "-" if share is None else f"{share * 100:.0f}%",
        format_number(run["peak_vram_bytes"] / BYTES_PER_MIB, 0),
        format_number(run["mean_sm_clock_mhz"], 0),
        result["cooldown"]["outcome"],
        normalised_cell(result),
    ], result), result)
    return column.row(cells, result)


def _preamble(results: Sequence[dict], gpus: Sequence[str]) -> str:
    primary = results[0]
    scenario = primary["scenario"]
    counts = sorted({steps_of(result) for result in results})
    models = sorted({model_of(result) for result in results})
    return (
        f"Step count swept over {', '.join(str(n) for n in counts)} on "
        f"{', '.join(gpus)}, at {scenario['width']}x{scenario['height']} batch "
        f"{scenario['batch_size']} on "
        f"{', '.join(f'`{model}`' for model in models)}. Only the count moves: "
        f"every arm opens at schedule index {scenario['t_index_list'][0]} and "
        f"spends its extra steps after it (`render_plan.t_index_ladder`), so an "
        f"arm is not also a strength change. Batch 1 throughout, so ms/call is "
        f"ms/frame. The UNet, VAE-encode and VAE-decode columns come from "
        f"`--per-module`, whose extra synchronises perturb the total - they are "
        f"read as a split of the call, not as the call."
    )


def _finding(arms: Sequence[dict], configuration: Configuration) -> List[str]:
    """What the sweep replaced the estimate with, in one configuration."""
    arms = in_configuration(arms, configuration)
    control = control_of(arms, configuration)
    if control is None:
        return []
    lines = []
    share = module_share(control, "unet")
    if share is not None:
        lines.append(
            f"At one step the UNet is {share * 100:.0f}% of the "
            f"{ms_per_call(control):.2f} ms call "
            f"({module_ms(control, 'unet'):.2f} ms against VAE-encode "
            f"{module_ms(control, 'vae_encode'):.2f} and VAE-decode "
            f"{module_ms(control, 'vae_decode'):.2f}).")
    for arm in arms:
        steps = steps_of(arm)
        if steps == CONTROL_STEPS:
            continue
        factor = factor_over_control(arm, control)
        per_step = (ms_per_call(arm) - ms_per_call(control)) / (steps - CONTROL_STEPS)
        lines.append(
            f"{steps} steps cost **{factor:.2f}x** the one-step call "
            f"({ms_per_call(arm):.2f} ms against {ms_per_call(control):.2f}), "
            f"so each step after the first cost {per_step:.2f} ms.")
        unet, base_unet = module_ms(arm, "unet"), module_ms(control, "unet")
        if unet and base_unet:
            lines.append(
                f"That is {unet / base_unet:.2f}x the UNet for {steps}x the passes "
                f"({unet:.2f} ms against {base_unet:.2f}): `use_denoising_batch` "
                f"puts the steps through the UNet as one batch of {steps}, and a "
                f"batch of {steps} is not {steps} calls.")
    four = next((arm for arm in arms if steps_of(arm) == 4), None)
    factor = factor_over_control(four, control) if four is not None else None
    if factor is not None:
        verdict = ("below it - the estimate was pessimistic"
                   if factor < ESTIMATED_FOUR_STEP_FACTOR else
                   "at or above it - the estimate was not pessimistic enough")
        lines.append(
            f"Issue #38 estimated ~{ESTIMATED_FOUR_STEP_FACTOR:.1f}x for four "
            f"steps, from the 256x256 laptop per-module split. Measured here it is "
            f"{factor:.2f}x, {verdict}. That estimate is replaced by this table.")
    return lines


def _model_comparison(arms: Sequence[dict], gpu: str,
                      acceleration: str) -> Optional[str]:
    """What moving base model costs per frame, each at the count it needs.

    Issue #38's question, and the only comparison in this block that is *meant* to
    cross a configuration: SD-Turbo is distilled to one step and SD 1.5 is not, so
    comparing them at equal steps would compare one model against a crippled one.
    Each model is taken at its own working count - `bench.models.BASE_MODELS` -
    which is the same rule the issue's third trap states for denoise.
    """
    from bench.models import BASE_MODELS, DEFAULT_BASE

    def working(model: str, steps: int) -> Optional[dict]:
        # Bare arms only: a fused style LoRA is a different engine, and comparing
        # one against a bare arm of another model would price the LoRA as the model.
        return next((arm for arm in arms
                     if model_of(arm) == model and steps_of(arm) == steps
                     and style_of(arm) == NO_STYLE), None)

    shipped = BASE_MODELS[DEFAULT_BASE]
    baseline = working(shipped.model, shipped.steps)
    if baseline is None:
        return None
    lines = []
    for base in BASE_MODELS.values():
        if base.key == DEFAULT_BASE:
            continue
        arm = working(base.model, base.steps)
        if arm is None:
            continue
        sentence = (
            f"`{base.model}` ({base.architecture}) at its working "
            f"{base.steps} steps costs {ms_per_call(arm):.2f} ms/call against "
            f"`{shipped.model}`'s {ms_per_call(baseline):.2f} at "
            f"{shipped.steps} - **{ms_per_call(arm) / ms_per_call(baseline):.2f}x** "
            f"the diffusion call.")
        gap = _same_step_gap(arms, base, shipped)
        if gap is not None:
            sentence += (f" The step count is the whole of it: at {base.steps} "
                         f"steps the two models are within {abs(gap):.0f}% of "
                         f"each other.")
        lines.append(sentence)
    if not lines:
        return None
    machine = f"{gpu}, " if gpu else ""
    return f"**Base model, {machine}`{acceleration}`.** " + " ".join(lines)


def _same_step_gap(arms: Sequence[dict], base, shipped) -> Optional[float]:
    """How far apart the two models are at *one* step count, as a percentage.

    The control on the comparison above: it says whether the cost of the move is
    the model or the steps it needs, which are two different findings. None when
    only one of the two was measured at that count - a gap of zero from a missing
    record is a finding invented rather than measured.
    """
    at = {model: next((arm for arm in arms
                       if model_of(arm) == model and steps_of(arm) == base.steps
                       and style_of(arm) == NO_STYLE), None)
          for model in (base.model, shipped.model)}
    if not all(at.values()):
        return None
    return (ms_per_call(at[base.model]) / ms_per_call(at[shipped.model]) - 1) * 100


def format_steps_report(results: Mapping[str, dict]) -> str:
    """The measured step-count block spec 7.2 carries, from the committed arms."""
    ordered = sorted(latest_per_arm(results).values(),
                     key=lambda result: (steps_of(result), model_of(result),
                                         style_of(result), accelerator_of(result),
                                         gpu_of(result)))
    if not ordered:
        return "no step-count sweep committed yet (issue #38)"

    gpus = distinct_gpus(ordered)
    column = GpuColumn.for_gpus(gpus)
    models = OptionalColumn.when_varied(ordered, "model", model_of, index=1)
    styles = OptionalColumn.when_varied(ordered, "style LoRA", style_of, index=2)
    header = column.header(styles.header(models.header(REPORT_HEADER)))
    sections = [
        _preamble(ordered, gpus),
        "\n".join([header, table_separator(header)]
                  + [_row(result, column, models, styles,
                          control_of(ordered, configuration_of(result)))
                     for result in ordered]),
    ]
    uncontrolled = []
    for configuration in sorted({configuration_of(arm) for arm in ordered}):
        gpu, acceleration, model, style = configuration
        machine = "" if len(gpus) == 1 else f"{gpu}, "
        label = f"`{model}`, `{acceleration}`" if models.shown else f"`{acceleration}`"
        if styles.shown and style != NO_STYLE:
            label += f", `{style}` fused"
        lines = _finding(ordered, configuration)
        if lines:
            sections.append(f"**{machine}{label}.** " + " ".join(lines))
        else:
            uncontrolled.append(f"{machine}{label}")
    if uncontrolled:
        sections.append(
            f"No one-step arm in {len(uncontrolled)} configurations "
            f"({'; '.join(uncontrolled)}), so their `x 1 step` cell is empty - "
            f"which for a model that cannot render at one step is the honest "
            f"answer. What those arms are read against is the base-model "
            f"comparison below.")
    for gpu in gpus:
        arms = measured_on(ordered, gpu)
        for acceleration in sorted({accelerator_of(arm) for arm in arms}):
            comparison = _model_comparison(
                [arm for arm in arms if accelerator_of(arm) == acceleration],
                gpu if len(gpus) > 1 else "", acceleration)
            if comparison:
                sections.append(comparison)
    return "\n\n".join(sections)
