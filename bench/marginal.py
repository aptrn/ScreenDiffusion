"""Marginal cost per additional batch item, read back out of the committed results.

This is the question the orchestrator design rests on (spec 7.2 item 2, 7.3): a
Render Plan wants N crops diffused per frame, and that is only affordable if the
second crop costs materially less than the first. The answer is a division over
numbers already on disk, so it is computed here rather than asserted in prose - and
`python -m bench --marginal` prints the same table a spec section carries, from the
same files a reviewer would recompute it from.

Per *call* is the unit that matters. One call diffuses the whole batch, so the slope
between two measured batch sizes is what an extra item actually costs; the per-frame
figure the README leads with is that slope already averaged over the batch, which
hides it.

Nothing here touches the GPU or reads a scenario definition: a result file is
self-describing, which is the point of serialising the scenario whole into it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True)
class BatchPoint:
    """One measured (resolution, batch) cell."""

    batch_size: int
    ms_per_call: float
    ms_per_frame: float
    peak_vram_mib: float
    mean_sm_clock_mhz: Optional[float]
    cooldown: str
    source: str


@dataclass(frozen=True)
class MarginalStep:
    """What the items between two measured batch sizes cost, each."""

    from_batch: int
    to_batch: int
    marginal_ms_per_item: float
    fraction_of_first_item: Optional[float]


@dataclass(frozen=True)
class Curve:
    """The batch curve for one accelerator at one resolution, on one GPU."""

    acceleration: str
    width: int
    height: int
    gpu_name: str
    points: Tuple[BatchPoint, ...]

    @property
    def first_item_ms(self) -> Optional[float]:
        """A batch of one - the yardstick every marginal item is judged against."""
        for point in self.points:
            if point.batch_size == 1:
                return point.ms_per_call
        return None

    @property
    def steps(self) -> Tuple[MarginalStep, ...]:
        first = self.first_item_ms
        steps = []
        for earlier, later in zip(self.points, self.points[1:]):
            span = later.batch_size - earlier.batch_size
            marginal = (later.ms_per_call - earlier.ms_per_call) / span
            steps.append(MarginalStep(
                from_batch=earlier.batch_size, to_batch=later.batch_size,
                marginal_ms_per_item=marginal,
                fraction_of_first_item=None if not first else marginal / first,
            ))
        return tuple(steps)

    @property
    def sublinear(self) -> Optional[bool]:
        """Does every extra item cost less than the first one did?

        None, not False, when the curve has no batch of one: without that yardstick
        the question has no answer, and reporting a missing measurement as a
        negative result would misstate what was measured.
        """
        if self.first_item_ms is None or not self.steps:
            return None
        return all(step.marginal_ms_per_item < self.first_item_ms for step in self.steps)

    def normalised_to(self, reference_mhz: float) -> "Curve":
        """The same curve with each cell rescaled to what it would cost at one clock.

        The RTX 3080 laptop holds a 120 W limit, and a longer call sinks deeper into
        it: across this sweep the mean SM clock ranges from 1785 MHz down to 787 MHz,
        and a comparison of raw milliseconds across those cells is partly a
        comparison of clocks. Diffusion here is compute-bound, so `ms * f / f_ref` is
        a first-order correction - an estimate, and reported as one, but the
        alternative is reporting the power limit as if it were the shape of the
        marginal-cost curve, which is precisely the conclusion spec 7.4 says has to
        be portable.

        A cell with no clock reading is passed through untouched rather than guessed.
        """
        scaled = tuple(
            point if point.mean_sm_clock_mhz is None else replace(
                point,
                ms_per_call=point.ms_per_call * point.mean_sm_clock_mhz / reference_mhz,
                ms_per_frame=point.ms_per_frame * point.mean_sm_clock_mhz / reference_mhz,
                mean_sm_clock_mhz=reference_mhz,
            )
            for point in self.points
        )
        return replace(self, points=scaled)


def point_from_result(result: dict, source: str) -> BatchPoint:
    scenario, run = result["scenario"], result["run"]
    batch_size = int(scenario["batch_size"])
    return BatchPoint(
        batch_size=batch_size,
        ms_per_call=float(run["mean_ms_per_frame"]) * batch_size,
        ms_per_frame=float(run["mean_ms_per_frame"]),
        peak_vram_mib=round(float(run["peak_vram_bytes"]) / BYTES_PER_MIB, 1),
        mean_sm_clock_mhz=run.get("mean_sm_clock_mhz"),
        cooldown=result.get("cooldown", {}).get("outcome", "unknown"),
        source=source,
    )


def latest_result_per_scenario(results: Mapping[str, dict]) -> Dict[str, dict]:
    """One result per scenario: the most recently finished run of each.

    A cell measured twice is two honest records, and both stay on disk. A curve
    built from a mixture of them would compare a cold run against a hot one.
    """
    newest: Dict[str, Tuple[str, str]] = {}
    for filename, result in results.items():
        name = result["scenario"]["name"]
        finished = str(result["run"]["finished_utc"])
        if name not in newest or finished > newest[name][1]:
            newest[name] = (filename, finished)
    return {filename: results[filename] for filename, _ in newest.values()}


def curves_from_results(results: Mapping[str, dict]) -> List[Curve]:
    """The batch curves in `results`, one per (accelerator, resolution, GPU)."""
    grouped: Dict[Tuple[str, int, int, str], List[BatchPoint]] = {}
    for filename, result in sorted(latest_result_per_scenario(results).items()):
        scenario = result["scenario"]
        key = (scenario["acceleration"], int(scenario["width"]), int(scenario["height"]),
               result.get("hardware", {}).get("gpu_name", "unknown"))
        grouped.setdefault(key, []).append(point_from_result(result, source=filename))
    return [
        Curve(acceleration=acceleration, width=width, height=height, gpu_name=gpu_name,
              points=tuple(sorted(points, key=lambda point: point.batch_size)))
        for (acceleration, width, height, gpu_name), points in sorted(grouped.items())
    ]


def reference_clock_mhz(curves: List[Curve]) -> float:
    """The fastest mean SM clock any cell actually ran at.

    Chosen from the data rather than fixed, so the normalised table never claims a
    clock nothing on this machine reached.
    """
    clocks = [point.mean_sm_clock_mhz for curve in curves for point in curve.points
              if point.mean_sm_clock_mhz is not None]
    if not clocks:
        raise ValueError("no cell recorded an SM clock, so there is nothing to normalise to")
    return max(clocks)


def load_results(results_dir: Path) -> Dict[str, dict]:
    return {path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(Path(results_dir).glob("*.json"))}


TABLE_HEADER = ("| GPU | accel | res | batch | ms/call | ms/frame | marginal ms/item |"
                " x first item | peak VRAM (MiB) | SM clock (MHz) | cooldown |")
# Derived from the header, so a new column cannot leave a mismatched rule behind.
TABLE_SEPARATOR = "|" + "---|" * (TABLE_HEADER.count("|") - 1)


def format_table(curves: List[Curve]) -> str:
    """The measured table spec 7.2 carries: one row per cell, marginal cost included."""
    rows = [TABLE_HEADER, TABLE_SEPARATOR]
    for curve in curves:
        steps = {step.to_batch: step for step in curve.steps}
        for point in curve.points:
            step = steps.get(point.batch_size)
            rows.append("| " + " | ".join([
                curve.gpu_name,
                curve.acceleration,
                f"{curve.width}x{curve.height}",
                str(point.batch_size),
                f"{point.ms_per_call:.1f}",
                f"{point.ms_per_frame:.2f}",
                "-" if step is None else f"{step.marginal_ms_per_item:.1f}",
                ("-" if step is None or step.fraction_of_first_item is None
                 else f"{step.fraction_of_first_item:.2f}"),
                f"{point.peak_vram_mib:.0f}",
                "-" if point.mean_sm_clock_mhz is None else f"{point.mean_sm_clock_mhz:.0f}",
                point.cooldown,
            ]) + " |")
    return "\n".join(rows)


def format_verdicts(curves: List[Curve]) -> str:
    """One line per curve: whether the marginal cost is sublinear, and by how much."""
    lines = []
    for curve in curves:
        label = f"{curve.acceleration} {curve.width}x{curve.height}"
        if curve.sublinear is None:
            reason = ("only one batch size measured" if len(curve.points) < 2
                      else "no batch-1 measurement to compare against")
            lines.append(f"{label}: sublinearity undetermined - {reason}")
            continue
        fractions = [step.fraction_of_first_item for step in curve.steps
                     if step.fraction_of_first_item is not None]
        verdict = "sublinear" if curve.sublinear else "NOT sublinear"
        lines.append(
            f"{label}: {verdict} - first item {curve.first_item_ms:.1f} ms, "
            f"extra items {min(fractions):.0%}-{max(fractions):.0%} of it"
        )
    return "\n".join(lines)
