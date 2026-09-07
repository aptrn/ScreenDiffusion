"""The base model as a choice, and the style LoRAs that are the reason to make one.

Issue #38. Every number in this repo is SD-Turbo, which is SD 2.1-based; that came
with choosing SD-Turbo and was never a separate decision. The reason to want SD 1.5
is the LoRA ecosystem - SD-Turbo can load no SD 1.5 or SDXL LoRA at all - and live
testing found style control wanting on the axis this app can reach, so the model is
the lever that is left.

What lives here is the *naming*: which local folder a base model is, which LoRA file
a style is, and the one rule that turns a style name into the `lora_dict` the wrapper
fuses. Two readers need that rule to agree exactly, because they answer the same
question in two places:

- `bench.runner.build_stream`, which fuses the LoRA into the UNet, and
- `bench.cli.engine_dir_name`, which has to say whether *that* configuration is
  already compiled. It hashes the same dict through `engine_cache`, which is the
  app's own copy of `create_prefix`'s naming rule - the fused `lora_dict`'s sha1
  is part of the engine directory name, so every distinct style keys its own
  ~5 GB build.

A style is named rather than pathed in a `ScenarioConfig` because the path is a
machine's and the record has to still say what was measured on another one.

Stdlib only. The registry is read by the GPU-free tier and by the CLI's guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

from bench.cadence import background_passed
from bench.paths import resolve_models_dir
from bench.portability import (
    FRAME_BUDGET_MS,
    criterion_verdict,
    flicker,
    fps,
    ms_per_frame,
    ms_per_frame_with_detection,
    regions_per_frame,
)
from bench.results import (
    GpuColumn,
    distinct_gpus,
    format_number,
    gpu_of,
    latest_per,
    measured_on,
    sentence_case,
    table_separator,
)
from bench.scenarios import ScenarioConfig

# Where a downloaded LoRA is staged, under the shared models root. The app's own
# `LOCAL_LCM_LORA` points into the same directory, so a machine set up for the app
# is a machine set up for this.
LORAS_SUBDIR = "loras"


@dataclass(frozen=True)
class BaseModel:
    """One base model, and what it needs to be usable in real time.

    `steps` is not a preference: SD-Turbo is distilled to one step and SD 1.5 is
    not, so 1.5 needs LCM-LoRA and about four. Issue #38's step 1 measures what
    that costs before anything is compiled for it.
    """

    key: str
    model: str
    use_lcm_lora: bool
    steps: int
    architecture: str
    note: str


BASE_MODELS: Dict[str, BaseModel] = {
    "sd-turbo": BaseModel(
        key="sd-turbo", model="sd-turbo-fp16", use_lcm_lora=False, steps=1,
        architecture="SD 2.1",
        note="The shipped model. One step, no LCM-LoRA - it is distilled for "
             "this - and no SD 1.5 or SDXL LoRA will load on it."),
    "sd15": BaseModel(
        key="sd15", model="sd-v1-5-fp16", use_lcm_lora=True, steps=4,
        architecture="SD 1.5",
        note="The LoRA ecosystem's model. Not a turbo model, so it needs LCM-LoRA "
             "and about four steps to run in real time at all."),
}

# The base model each scenario in the registry is a cell of, keyed by the suffix
# `bench.scenarios.base_model_name` appends. The default suffix is no suffix.
DEFAULT_BASE = "sd-turbo"


@dataclass(frozen=True)
class StyleLora:
    """One style LoRA: the file, the scale it is fused at, and where it came from.

    `format` is what the file *is* rather than what its extension says. Issue #38's
    second trap: LoCon / LyCORIS convolution layers do not load on this diffusers
    version and will not load on SD 1.5 either, so which format a LoRA is has to be
    recorded rather than inferred from whether one that happened to work worked.
    """

    key: str
    filename: str
    scale: float
    source: str
    format: str
    note: str


STYLE_LORAS: Dict[str, StyleLora] = {
    "loving-vincent": StyleLora(
        key="loving-vincent", filename="style-loving-vincent.safetensors",
        scale=0.9, source="vislupus/SD1.5-LoRA-Loving-Vincent-Style",
        format="LoRA (kohya, linear only)",
        note="A painting style - the case a style LoRA exists for."),
    "illusion-pattern": StyleLora(
        key="illusion-pattern", filename="style-illusion-pattern.safetensors",
        scale=0.9, source="Norod78/SD15-IllusionDiffusionPattern-LoRA",
        format="LoRA (kohya, linear only)",
        note="A pattern style, trained on SD 1.5 itself rather than on a "
             "finetune of it - a second lineage, not a second checkpoint of one."),
    "locon-probe": StyleLora(
        key="locon-probe", filename="style-locon-probe.safetensors",
        scale=0.9, source="pmczip/SD1.5_LyCORIS_Models",
        format="LoCon / LyCORIS (198 conv keys)",
        note="Deliberately a format this diffusers version cannot fuse. It is "
             "here so that which formats load is measured rather than inferred "
             "from the ones that happened to - issue #38's second trap."),
}

# The local LCM-LoRA the app prefers when it has been staged, spelt the way
# `main_gpu_addon.LOCAL_LCM_LORA` spells it. Offline mode blocks the repo-id
# lookup, so a run that leans on the default repo id is a run that needs network.
LCM_LORA_FILENAME = "lcm-lora-sdv1-5.safetensors"


def loras_dir(models_root: Optional[Path] = None) -> Path:
    root = resolve_models_dir() if models_root is None else Path(models_root)
    return root / LORAS_SUBDIR


def lora_path(filename: str, models_root: Optional[Path] = None) -> Path:
    return loras_dir(models_root) / filename


def lcm_lora_path(models_root: Optional[Path] = None) -> Optional[str]:
    """The staged LCM-LoRA file, or None to leave the wrapper on its repo id."""
    path = lora_path(LCM_LORA_FILENAME, models_root)
    return str(path) if path.is_file() else None


def lora_dict_for(scenario: ScenarioConfig,
                  models_root: Optional[Path] = None) -> Optional[Dict[str, float]]:
    """The `lora_dict` this scenario fuses, resolved to this machine's paths.

    One function, two readers: the one that builds the stream and the one that
    works out whether the resulting engine is already on disk. A second spelling
    would let the guard say "cached" about a directory the build never writes.
    """
    if not scenario.style_lora:
        return None
    style = STYLE_LORAS[scenario.style_lora]
    return {str(lora_path(style.filename, models_root)): float(scenario.lora_scale)}


def style_arm_name(scenario_name: str, style: str) -> str:
    """`...-b1-sd15` fused with the Van Gogh style -> `...-b1-sd15-loving-vincent`."""
    return f"{scenario_name}-{style}"


def base_model_of(result: Mapping) -> str:
    """Which base model a selective arm rendered through, off its own case."""
    key = (result.get("case") or {}).get("base_model") or DEFAULT_BASE
    base = BASE_MODELS.get(str(key))
    return base.model if base is not None else str(key)


def steps_of(result: Mapping) -> int:
    """How many denoising steps that arm ran at - its base model's own count."""
    key = (result.get("case") or {}).get("base_model") or DEFAULT_BASE
    base = BASE_MODELS.get(str(key))
    return base.steps if base is not None else 1


MODEL_REPORT_HEADER = ("| base model | steps | regions/frame | ms/frame |"
                       " +detect | FPS | headroom (ms) | flicker |"
                       " background | 30 FPS |")


def _model_row(result: dict, column: GpuColumn) -> str:
    cost = ms_per_frame_with_detection(result)
    return column.row([
        f"`{base_model_of(result)}`",
        str(steps_of(result)),
        format_number(regions_per_frame(result), 2),
        format_number(ms_per_frame(result), 2),
        format_number(cost, 2),
        format_number(fps(result), 1),
        f"{FRAME_BUDGET_MS - cost:+.2f}",
        format_number(flicker(result), 2),
        "identical" if background_passed(result) else "CHANGED",
        "yes" if cost <= FRAME_BUDGET_MS else "**no**",
    ], result)


def format_model_report(results: Mapping[str, dict]) -> str:
    """The base-model comparison block spec 7.5 carries (issue #38, step 2).

    Both models through the *shipped* selective path over the same clip and the
    same plan, each at the step count it needs - which is the comparison the
    issue's third trap asks for and the one an equal-`t_index` table would not be.
    The verdict is `bench.portability`'s own, so "does 30 FPS survive the move" is
    decided by the function spec 7.4's verdict comes from rather than by a second
    reading of the same numbers.
    """
    reduced = latest_per(results, lambda result: result["case"]["name"])
    ordered = sorted(reduced.values(),
                     key=lambda result: (steps_of(result), base_model_of(result),
                                         gpu_of(result)))
    if not ordered:
        return "no base-model arm committed yet (issue #38)"

    primary = ordered[0]
    clip, plan = primary["clip"], primary["plan"]
    gpus = distinct_gpus(ordered)
    column = GpuColumn.for_gpus(gpus)
    header = column.header(MODEL_REPORT_HEADER)
    sections = [
        f"The shipped selective path on each base model, over the same "
        f"{clip['frames_used']} frames of `{clip['name']}` under the same "
        f"`{plan['concept']} / {plan['region']} / denoise {plan['denoise']}` plan, "
        f"on {', '.join(gpus)}. Each model runs at the step count it needs rather "
        f"than at a shared one: SD-Turbo is distilled to a single step and SD 1.5 "
        f"is not, so an equal-step table would compare one model against a "
        f"crippled one. The denoise is shared and *means* the same thing on both - "
        f"the two checkpoints carry the same `scaled_linear` beta schedule, so "
        f"`render_plan.t_index_for_denoise`'s ladder is the same ladder.",
        "\n".join([header, table_separator(header)]
                  + [_model_row(result, column) for result in ordered]),
    ]
    for gpu in gpus:
        for result in measured_on(ordered, gpu):
            verdict = criterion_verdict(result)
            sections.append(
                f"**`{base_model_of(result)}` at {steps_of(result)} step"
                f"{'' if steps_of(result) == 1 else 's'}: 30 FPS "
                f"{'MET' if verdict.met else 'NOT MET'}.** "
                f"{sentence_case(verdict.statement)}. "
                f"Background: {result['gate']['background']['statement']}.")
    unbroken = all(background_passed(result) for result in ordered)
    sections.append(
        "Every arm above left the background bit-identical to the capture."
        if unbroken else
        "**At least one arm changed a pixel outside the rendered regions.**")
    artefacts = [f"`{result['comparison_clip']}`" for result in ordered
                 if result.get("comparison_clip")]
    if artefacts:
        sections.append(f"Manual-verification artefacts, source | render: "
                        f"{', '.join(artefacts)}.")
    return "\n\n".join(sections)


def with_style(scenario: ScenarioConfig, style: str) -> ScenarioConfig:
    """`scenario` with one style LoRA fused, renamed after it.

    Renamed because the engine is a different engine: a fused LoRA is compiled into
    the UNet weights, so the arm cannot share a cache entry with the base one and
    must not share a result filename either.
    """
    if style not in STYLE_LORAS:
        raise KeyError(f"no such style LoRA: {style!r}. "
                       f"Known: {', '.join(sorted(STYLE_LORAS))}")
    return scenario.replace(name=style_arm_name(scenario.name, style),
                            style_lora=style)
