"""Do style LoRAs actually work on the SD 1.5 arm, and how should styles ship?

Issue #38 steps 3-5. The reason to price SD 1.5 at all is the LoRA ecosystem, so a
speed verdict on its own would answer the wrong question: a fast model that cannot
take the style is not the answer to the problem the issue exists to solve. Hence the
Gate's sharp sentence - **a LoRA that loads and changes nothing is a failure, not a
pass** - and hence two numbers per arm rather than one:

- `net_change`, the render against the source with the resize round trip subtracted.
  Spec 8.2's own criterion, because the round trip changes the pixels on its own and
  a strength that does nothing otherwise passes on its blur.
- `change_vs_base`, the arm against the *same arm without the LoRA*. This is the one
  that answers "did the LoRA do anything"; the first cannot, because a plain SD 1.5
  render already clears it.

And a third thing recorded rather than inferred: **which formats load**. LoCon /
LyCORIS carry convolution layers this diffusers version cannot fuse, so a run that
reported "LoRAs work" from one plain LoRA that happened to would be reporting a
coincidence. One arm of the case is deliberately a LoCon file.

The delivery question (step 4) is answered from the *step arms*, not from here: a
fused LoRA keys its own TensorRT engine, so what has to be compared is one style's
ms/call with an engine against the same style's ms/call without one, and both of
those are plain diffusion cells under `bench/results/steps/`.

GPU-free. `bench.style_runner` is the half that touches a GPU.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from bench import RESULT_SCHEMA_VERSION
from bench.clocks import ClockNormalization
from bench.contention import OccupancyRecord
from bench.cooldown import CooldownRecord
from bench.fingerprint import Fingerprint
from bench.primitive_results import ClipRecord
from bench.results import (
    GpuColumn,
    append_row,
    distinct_gpus,
    format_number,
    gpu_of,
    latest_per,
    load_records,
    measured_on,
    require_recordable,
    table_separator,
    timestamp_from,
    write_record,
)

RECORD_KIND = "style"

# The arm with no LoRA fused: the control every style arm is scored against.
BASE_ARM = "base"

# Mean absolute difference inside the frame, 0-255, below which a render has not
# visibly done anything. Spec 8.2's threshold, so "visible" means one thing in both
# records.
VISIBLE_CHANGE = 8.0

# How far a style arm has to sit from the same arm without the LoRA before the LoRA
# can be said to have changed the output. Lower than VISIBLE_CHANGE on purpose: this
# is the distance between two renders of the same model at the same strength, not
# the distance between a render and a photograph, and a style that moved every pixel
# by 2/255 has moved it visibly.
LORA_CHANGE_THRESHOLD = 4.0


@dataclass(frozen=True)
class StyleCase:
    """One clip, one base model, and the LoRAs to try on it."""

    name: str
    clip: str
    base_scenario: str
    steps: int
    styles: Tuple[str, ...]
    prompt: str
    denoise: float
    note: str
    frames: int = 24
    start_frame: int = 0
    canvas: int = 512
    warmup_frames: int = 3

    def replace(self, **changes) -> "StyleCase":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)


STYLE_CASE = "style-sd15"

CASES: Dict[str, StyleCase] = {
    STYLE_CASE: StyleCase(
        name=STYLE_CASE,
        clip="people.mp4",
        base_scenario="img2img-none-512x512-b1-sd15",
        steps=4,
        styles=("loving-vincent", "illusion-pattern", "locon-probe"),
        prompt="a painting",
        denoise=0.62,
        note="Do style LoRAs load on SD 1.5 + LCM-LoRA and visibly change the "
             "output? Rendered on the `none` accelerator, which is the "
             "hot-swappable path - under `tensorrt` each of these would be its "
             "own ~5 GB engine, which is the decision step 4 makes.",
    ),
}


@dataclass(frozen=True)
class StyleArm:
    """One arm: the base render, or one style LoRA fused into it.

    `loaded` and `error` are the format finding. A LoRA this diffusers version
    refuses is not a hole in the table - it is the measurement the issue's second
    trap asks for, and it carries the reason it refused.
    """

    style: str
    filename: str
    scale: float
    source: str
    format: str
    loaded: bool
    error: Optional[str]
    ms_per_frame: float
    change_vs_source: float
    control_change: float
    change_vs_base: float
    flicker: float
    frames: int

    @property
    def is_base(self) -> bool:
        return self.style == BASE_ARM

    @property
    def net_change(self) -> float:
        """The render against the source, net of the capture's own round trip."""
        return round(max(0.0, self.change_vs_source - self.control_change), 4)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["net_change"] = self.net_change
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> "StyleArm":
        fields = {name.name: data[name.name] for name in dataclasses.fields(cls)}
        return cls(**fields)


@dataclass(frozen=True)
class StyleVerdict:
    """Whether one style LoRA did the job, and in words."""

    style: str
    passed: bool
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def style_verdict(arm: StyleArm) -> Optional[StyleVerdict]:
    """The Gate's rule, executable. None for the base arm, which is the control.

    Three ways to fail and one way to pass, in the order a reader would ask them:
    it has to load, the render has to have done something, and the LoRA has to be
    the reason it looks different.
    """
    if arm.is_base:
        return None
    if not arm.loaded:
        return StyleVerdict(
            style=arm.style, passed=False,
            statement=(f"`{arm.style}` ({arm.format}) did not load: "
                       f"{arm.error or 'no reason recorded'}"))
    if arm.net_change < VISIBLE_CHANGE:
        return StyleVerdict(
            style=arm.style, passed=False,
            statement=(f"`{arm.style}` loaded, but the render changed only "
                       f"{arm.net_change:.1f}/255 net of the "
                       f"{arm.control_change:.2f}/255 the resize control costs, "
                       f"against a {VISIBLE_CHANGE:.0f}/255 threshold - nothing "
                       f"was restyled to judge the LoRA on"))
    if arm.change_vs_base < LORA_CHANGE_THRESHOLD:
        return StyleVerdict(
            style=arm.style, passed=False,
            statement=(f"`{arm.style}` loaded and did not change the output: "
                       f"{arm.change_vs_base:.1f}/255 against the same arm without "
                       f"it, under the {LORA_CHANGE_THRESHOLD:.0f}/255 threshold"))
    return StyleVerdict(
        style=arm.style, passed=True,
        statement=(f"`{arm.style}` loaded and moved the output "
                   f"{arm.change_vs_base:.1f}/255 against the same arm without it "
                   f"(threshold {LORA_CHANGE_THRESHOLD:.0f}), on a render that is "
                   f"itself {arm.net_change:.1f}/255 from the source net of the "
                   f"control"))


@dataclass(frozen=True)
class StyleResult:
    """One run: every arm over the same frames of one clip, and the artefact."""

    case: StyleCase
    clip: ClipRecord
    arms: Tuple[StyleArm, ...]
    started_utc: str
    finished_utc: str
    cooldown: CooldownRecord
    occupancy: Optional[OccupancyRecord]
    hardware: Fingerprint
    clock_normalization: Optional[ClockNormalization]
    comparison_still: str = ""
    comparison_clip: str = ""

    def to_dict(self) -> dict:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "kind": RECORD_KIND,
            "case": self.case.to_dict(),
            "clip": asdict(self.clip),
            "arms": [arm.to_dict() for arm in self.arms],
            "verdicts": [verdict.to_dict() for verdict in
                         (style_verdict(arm) for arm in self.arms)
                         if verdict is not None],
            # `run` so the shared readers - `latest_per`, `timestamp_from`,
            # `require_recordable` - find the finish time where they find it in
            # every other record shape.
            "run": {"started_utc": self.started_utc,
                    "finished_utc": self.finished_utc},
            "cooldown": self.cooldown.to_dict(),
            "occupancy": None if self.occupancy is None else self.occupancy.to_dict(),
            "hardware": self.hardware.to_dict(),
            "clock_normalization": (None if self.clock_normalization is None
                                    else self.clock_normalization.to_dict()),
            "comparison_still": self.comparison_still,
            "comparison_clip": self.comparison_clip,
        }


def write_style_result(result: StyleResult, results_dir: Path,
                       timestamp: Optional[str] = None) -> Path:
    data = result.to_dict()
    timestamp = timestamp_from(data) if timestamp is None else timestamp
    return write_record(data, results_dir,
                        f"{result.case.name}-{timestamp}.json")


def load_style_results(results_dir: Path) -> Dict[str, dict]:
    return load_records(Path(results_dir))


STYLE_README_NAME = "README.md"
STYLE_README_TITLE = "# Style LoRAs on SD 1.5"
STYLE_README_PREAMBLE_TEXT = (
    "Written by `uv run python -m bench style-sd15`, never by hand. Issue #38 steps\n"
    "3-5: whether a style LoRA loads on the SD 1.5 arm and whether it visibly changes\n"
    "the output, with the format it is in recorded either way.\n\n"
    "`net change` is the render against the source with the resize control\n"
    "subtracted; `vs base` is the arm against the same arm with no LoRA fused, which\n"
    "is the number that says the LoRA did anything.\n"
)
STYLE_README_HEADER = ("| finished (UTC) | case | GPU | arm | format | loaded |"
                       " ms/frame | net change | vs base | flicker | verdict |"
                       " file |")
STYLE_README_SEPARATOR = table_separator(STYLE_README_HEADER)


def append_style_readme_rows(result: StyleResult, readme_path: Path,
                             filename: str) -> None:
    """One row per arm - the base one too, because it is the yardstick."""
    data = result.to_dict()
    require_recordable(data)
    preamble = (f"{STYLE_README_TITLE}\n\n{STYLE_README_PREAMBLE_TEXT}\n"
                f"{STYLE_README_HEADER}\n{STYLE_README_SEPARATOR}\n")
    for arm in result.arms:
        verdict = style_verdict(arm)
        append_row(_readme_row(result, arm, verdict, filename), readme_path,
                   preamble)


def _readme_row(result: StyleResult, arm: StyleArm,
                verdict: Optional[StyleVerdict], filename: str) -> str:
    from bench.results import table_row

    return table_row([
        result.finished_utc,
        result.case.name,
        result.hardware.gpu_name,
        arm.style,
        arm.format,
        "yes" if arm.loaded else "NO",
        format_number(arm.ms_per_frame, 2),
        format_number(arm.net_change, 2),
        format_number(arm.change_vs_base, 2),
        format_number(arm.flicker, 2),
        "control" if verdict is None else ("pass" if verdict.passed else "FAIL"),
        f"[{filename}]({filename})",
    ])


# --- step 4: how a style is delivered ----------------------------------------


@dataclass(frozen=True)
class DeliveryRecommendation:
    """Pre-built TensorRT engines per style, or a hot-swappable slower path.

    Computed from the committed step arms rather than argued: the same style LoRA
    is measured with an engine and without one, and the difference between those
    two milliseconds figures is what the ~5 GB and the 15-25 minutes buy.
    """

    style: str
    tensorrt_ms: float
    torch_ms: float
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def _styled_arm(records: Mapping[str, dict], acceleration: str,
                style: Optional[str]) -> Optional[dict]:
    for record in records.values():
        scenario = record["scenario"]
        if (scenario["acceleration"] == acceleration
                and scenario.get("style_lora") == style):
            return record
    return None


def delivery_recommendation(step_records: Mapping[str, dict],
                            budget_ms: float = 33.33
                            ) -> Optional[DeliveryRecommendation]:
    """What a style costs on each path, from the committed diffusion cells.

    None when both halves were not measured: a recommendation from one of them is
    a preference wearing a number.
    """
    styled = [record for record in step_records.values()
              if record["scenario"].get("style_lora")]
    if not styled:
        return None
    style = styled[0]["scenario"]["style_lora"]
    engine = _styled_arm(step_records, "tensorrt", style)
    torch_path = _styled_arm(step_records, "none", style)
    if engine is None or torch_path is None:
        return None
    tensorrt_ms = float(engine["run"]["mean_ms_per_frame"])
    torch_ms = float(torch_path["run"]["mean_ms_per_frame"])
    fits = ("both paths fit the frame budget" if max(tensorrt_ms, torch_ms) <= budget_ms
            else ("only the pre-built engine fits the frame budget"
                  if tensorrt_ms <= budget_ms < torch_ms
                  else "neither path fits the frame budget on the diffusion call "
                       "alone"))
    verdict = ("**Ship a small fixed set of pre-built engines, one per style.** "
               if tensorrt_ms <= budget_ms
               else "**Neither path ships a real-time style at this step count.** ")
    return DeliveryRecommendation(
        style=style, tensorrt_ms=round(tensorrt_ms, 4), torch_ms=round(torch_ms, 4),
        statement=(
            f"{verdict}The same LoRA (`{style}`) fused into a TensorRT engine costs "
            f"{tensorrt_ms:.2f} ms/call against {torch_ms:.2f} ms without one - "
            f"{torch_ms / tensorrt_ms:.2f}x - and against a {budget_ms:.2f} ms "
            f"budget, {fits}. What the engine costs is that it *is* an engine: "
            f"each distinct style and scale keys its own ~5 GB, 15-25 minute build "
            f"(`create_prefix` puts the fused-LoRA fingerprint in the cache key), "
            f"so styles are a release-time set rather than something a user types. "
            f"The `none` path swaps a style for a model reload and no compile, "
            f"which is what LoRA *experimentation* needs."),
    )


# --- the block spec 8.10 carries ---------------------------------------------


REPORT_HEADER = ("| arm | format | loaded | ms/frame | net change |"
                 " vs base | flicker | verdict |")


def _row(result: dict, arm: StyleArm, column: GpuColumn) -> str:
    verdict = style_verdict(arm)
    return column.row([
        f"`{arm.style}`",
        arm.format,
        "yes" if arm.loaded else "**no**",
        format_number(arm.ms_per_frame, 2),
        format_number(arm.net_change, 2),
        format_number(arm.change_vs_base, 2),
        format_number(arm.flicker, 2),
        "control" if verdict is None else ("pass" if verdict.passed else "**FAIL**"),
    ], result)


def arms_of(result: Mapping) -> List[StyleArm]:
    return [StyleArm.from_dict(arm) for arm in result["arms"]]


def _preamble(result: Mapping, gpus: Sequence[str]) -> str:
    case, clip = result["case"], result["clip"]
    return (
        f"{len(result['arms']) - 1} style LoRAs on `{case['base_scenario']}` at "
        f"{case['steps']} steps, over {clip['frames_used']} frames of "
        f"`{clip['name']}` at {case['canvas']}x{case['canvas']}, on "
        f"{', '.join(gpus)}. Every arm renders the same frames at the same "
        f"denoise ({case['denoise']}) under the same prompt "
        f"(\"{case['prompt']}\"); only the fused LoRA moves. `net change` is the "
        f"render against the source with the resize control subtracted - spec "
        f"8.2's criterion - and `vs base` is the arm against the same arm with no "
        f"LoRA fused, which is the number that says the LoRA did anything."
    )


def _machine_section(result: dict) -> List[str]:
    arms = arms_of(result)
    verdicts = [verdict for verdict in (style_verdict(arm) for arm in arms)
                if verdict is not None]
    passed = [verdict for verdict in verdicts if verdict.passed]
    lines = [f"**{len(passed)} of {len(verdicts)} style LoRAs loaded and visibly "
             f"changed the output.**"]
    lines += [f"- {verdict.statement}." for verdict in verdicts]
    artefacts = [f"`{result[key]}`" for key in ("comparison_still",
                                                "comparison_clip")
                 if result.get(key)]
    if artefacts:
        lines.append(f"The artefact a human judges this by, source | base | one "
                     f"panel per style: {', '.join(artefacts)}.")
    return lines


def format_style_report(results: Mapping[str, dict],
                        step_records: Optional[Mapping[str, dict]] = None) -> str:
    """The measured block spec 8.10 carries, from the committed style runs."""
    reduced = latest_per(results, lambda result: result["case"]["name"])
    ordered = sorted(reduced.values(),
                     key=lambda result: (result["case"]["name"], gpu_of(result)))
    if not ordered:
        return "no style-LoRA run committed yet (issue #38)"

    gpus = distinct_gpus(ordered)
    column = GpuColumn.for_gpus(gpus)
    header = column.header(REPORT_HEADER)
    sections = [_preamble(ordered[0], gpus)]
    rows = [_row(result, arm, column)
            for result in ordered for arm in arms_of(result)]
    sections.append("\n".join([header, table_separator(header)] + rows))
    for gpu in gpus:
        for result in measured_on(ordered, gpu):
            machine = "" if len(gpus) == 1 else f"**{gpu}.** "
            lines = _machine_section(result)
            sections.append(machine + lines[0] + "\n\n" + "\n".join(lines[1:]))
    recommendation = delivery_recommendation(step_records or {})
    if recommendation is not None:
        sections.append(recommendation.statement)
    return "\n\n".join(sections)
