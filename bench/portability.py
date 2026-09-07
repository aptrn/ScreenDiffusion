"""Which conclusions survived the move from the dev laptop to deploy hardware.

Issue #24, spec 7.4 and acceptance criterion 2. Every figure in this repo was
measured on one RTX 3080 laptop at a 120 W limit, and 7.4 has always *asserted*
which of them travel: curve shapes and rankings yes, absolute milliseconds and the
30 FPS verdict no. This module computes that table from two committed
`selective-people` runs instead - one per GPU - so each row is arithmetic over the
two records rather than a claim about them.

Three rules the report obeys, each of them one of the issue's traps made executable:

- **Region count drives cost.** Two selective runs at different regions/frame are
  not a hardware comparison, they are a comparison of how much frame was rendered.
  `comparability` gates the whole block: an incomparable pair prints the reason and
  no table at all, because a table is what someone would quote.
- **The criterion is judged on what a frame costs**, not on the frame path alone.
  The detector runs beside the diffusion, so `ms_per_frame_with_detection` is the
  figure, and the verdict carries the region count and the clock regime it was
  measured at - a verdict with neither beside it is not one.
- **A desktop does not throttle the way the laptop does.** That is a difference to
  record, so the clock the card actually held under load is a row rather than
  something the report normalises away.

GPU-free, like every other `bench.*` results module: it reads JSON and formats it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from bench.results import format_number, latest_per

# Acceptance criterion 2 (spec 11): the frame rate v1 has to reach, and what it
# leaves for one frame.
TARGET_FPS = 30.0
FRAME_BUDGET_MS = 1000.0 / TARGET_FPS

# The deploy targets spec 7.4 names, matched on the part of the nvidia-smi product
# name that identifies the die. `3090 ti` and not `3090`: a plain 3090 is a
# different card, and this table is about the two v1 deploys to.
DEPLOY_GPU_MARKERS: Tuple[str, ...] = ("3090 ti", "4090")

# How far apart two runs' regions/frame may be and still be a hardware comparison.
REGIONS_TOLERANCE = 0.05
# How far apart any other pair of figures may be and still count as the same
# answer on both machines. The same 5%: a conclusion that moves less than the
# run-to-run noise carried, and one that moves more did not.
AGREEMENT_TOLERANCE = 0.05

BYTES_PER_MIB = 1024 * 1024


def is_deploy_gpu(gpu_name: str) -> bool:
    """Is this one of the cards v1 deploys to?"""
    name = (gpu_name or "").lower()
    return any(marker in name for marker in DEPLOY_GPU_MARKERS)


def latest_per_gpu(results: Mapping[str, dict]) -> Dict[str, dict]:
    """One run per GPU: the most recently finished on each machine.

    Keyed by GPU rather than by case, which is what `bench.selective` keys by. Two
    runs on one card are two honest records and both stay on disk; the comparison
    wants the newest of each, or it compares a first attempt against a settled one.
    """
    return latest_per(results, lambda result: result["hardware"]["gpu_name"])


def _finished(result: dict) -> str:
    return str(result["run"]["finished_utc"])


def split_by_role(results: Mapping[str, dict]) -> Tuple[List[dict], List[dict]]:
    """`(dev, deploy)` - one run per GPU, newest first within each role.

    The split is read off the fingerprint, so a record cannot be filed under a
    machine it was not measured on.
    """
    runs = sorted(latest_per_gpu(results).values(), key=_finished, reverse=True)
    return ([run for run in runs if not is_deploy_gpu(run["hardware"]["gpu_name"])],
            [run for run in runs if is_deploy_gpu(run["hardware"]["gpu_name"])])


# --- reading one record ------------------------------------------------------


def gpu_of(result: dict) -> str:
    return result["hardware"]["gpu_name"]


def regions_per_frame(result: dict) -> float:
    return float(result["regions"]["regions_per_frame"])


def calls_per_frame(result: dict) -> float:
    run = result["run"]
    return run["diffusion_calls"] / max(1, run["frames"])


def clock_regime(result: dict) -> str:
    return str(result["hardware"]["clock_lock"]["state"])


def power_limit_w(result: dict) -> Optional[float]:
    hardware = result["hardware"]
    return hardware.get("enforced_power_limit_w") or hardware.get("power_limit_w")


def _nested_ms(result: dict, key: str) -> Optional[float]:
    block = result["run"].get(key)
    return None if block is None else float(block["mean_ms"])


def clock_held_fraction(result: dict) -> Optional[float]:
    """Mean SM clock under load as a fraction of the card's own maximum.

    The laptop's answer and a desktop's are the throttling difference the issue
    asks to be recorded rather than normalised away.
    """
    mean = result["run"].get("mean_sm_clock_mhz")
    ceiling = result["hardware"]["clock_lock"].get("max_sm_clock_mhz")
    if not mean or not ceiling:
        return None
    return float(mean) / float(ceiling)


def _agree(left: Optional[float], right: Optional[float],
           tolerance: float = AGREEMENT_TOLERANCE) -> Optional[bool]:
    """Do two figures say the same thing? `None` when one of them is missing.

    Relative to the larger of the two, so the question is the same one at 4 ms and
    at 400 ms.
    """
    if left is None or right is None:
        return None
    largest = max(abs(left), abs(right))
    if largest == 0.0:
        return True
    return abs(left - right) <= tolerance * largest


def _ratio(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None or right == 0:
        return None
    return left / right


# --- the third trap ----------------------------------------------------------


@dataclass(frozen=True)
class Comparability:
    """Whether these two runs rendered enough of the same thing to be compared."""

    baseline_regions: float
    deploy_regions: float
    tolerance: float
    comparable: bool
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def comparability(baseline: dict, deploy: dict,
                  tolerance: float = REGIONS_TOLERANCE) -> Comparability:
    """Same clip, same plan, same amount of frame rendered - or no comparison."""
    left, right = regions_per_frame(baseline), regions_per_frame(deploy)
    comparable = bool(_agree(left, right, tolerance))
    statement = (
        f"{gpu_of(baseline)} rendered {left:.2f} regions/frame and "
        f"{gpu_of(deploy)} rendered {right:.2f}, "
        + ("the same selection to within "
           f"{tolerance * 100:.0f}%" if comparable else
           f"further apart than the {tolerance * 100:.0f}% that makes a "
           f"millisecond figure a hardware figure")
    )
    return Comparability(baseline_regions=round(left, 4),
                         deploy_regions=round(right, 4), tolerance=tolerance,
                         comparable=comparable, statement=statement)


# --- acceptance criterion 2 --------------------------------------------------


@dataclass(frozen=True)
class Spread:
    """Every run on one card, so a verdict can say whether it is inside the noise.

    `decisive` is the question that matters when the criterion is close: did every
    run on this machine land on the same side of the target? One run 2% short of
    30 FPS and one 2% over are not a verdict, they are a spread - and the report
    has to say which of the two it is holding.
    """

    gpu: str
    runs: int
    lowest_fps: float
    highest_fps: float
    target_fps: float
    decisive: bool
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def fps_spread(results: Mapping[str, dict], gpu_name: str,
               target_fps: float = TARGET_FPS) -> Spread:
    """The FPS every committed run on `gpu_name` reached - all of them, not the newest."""
    rates = sorted(float(result["run"]["fps"]) for result in results.values()
                   if gpu_of(result) == gpu_name)
    if not rates:
        return Spread(gpu=gpu_name, runs=0, lowest_fps=0.0, highest_fps=0.0,
                      target_fps=target_fps, decisive=False,
                      statement=f"no committed run on {gpu_name}")
    decisive = (rates[0] >= target_fps) == (rates[-1] >= target_fps)
    side = ("clear of" if rates[0] >= target_fps else "short of") \
        if decisive else "either side of"
    statement = (
        f"{len(rates)} committed run{'' if len(rates) == 1 else 's'} on {gpu_name} "
        f"span {rates[0]:.1f}-{rates[-1]:.1f} FPS, {side} the "
        f"{target_fps:.0f} FPS target"
        + ("" if decisive else
           " - the verdict is inside the run-to-run spread, not outside it")
    )
    return Spread(gpu=gpu_name, runs=len(rates), lowest_fps=round(rates[0], 4),
                  highest_fps=round(rates[-1], 4), target_fps=target_fps,
                  decisive=decisive, statement=statement)


@dataclass(frozen=True)
class CriterionVerdict:
    """Does this run meet the 30 FPS criterion, and at what?"""

    gpu: str
    fps: float
    ms_per_frame: float
    ms_per_frame_with_detection: float
    regions_per_frame: float
    target_fps: float
    budget_ms: float
    over_budget_x: float
    met: bool
    clock_regime: str
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def criterion_verdict(result: dict,
                      target_fps: float = TARGET_FPS) -> CriterionVerdict:
    """The verdict, with the region count and the clock regime attached to it.

    Judged on `ms_per_frame_with_detection`. The frame path alone is not what a
    frame costs - the detector shares the SMs with the UNet - and quoting the
    cheaper figure is how a budget goes missing.
    """
    run = result["run"]
    budget = 1000.0 / target_fps
    cost = float(run["ms_per_frame_with_detection"])
    regions = regions_per_frame(result)
    over = cost / budget if budget else 0.0
    met = cost <= budget
    margin = (f"{budget - cost:.2f} ms to spare" if met
              else f"{over:.2f}x the budget")
    statement = (
        f"{run['fps']:.1f} FPS at {regions:.2f} regions/frame on {gpu_of(result)} - "
        f"{cost:.2f} ms per frame with detection amortised against the "
        f"{budget:.2f} ms a {target_fps:.0f} FPS budget allows, {margin}; "
        f"clocks {clock_regime(result)}"
    )
    return CriterionVerdict(
        gpu=gpu_of(result), fps=float(run["fps"]),
        ms_per_frame=float(run["ms_per_frame"]),
        ms_per_frame_with_detection=cost, regions_per_frame=round(regions, 4),
        target_fps=target_fps, budget_ms=budget, over_budget_x=round(over, 4),
        met=met, clock_regime=clock_regime(result), statement=statement,
    )


# --- what carried ------------------------------------------------------------


@dataclass(frozen=True)
class PortabilityRow:
    """One of spec 7.4's conclusions, answered by the two records rather than asserted."""

    conclusion: str
    carried: Optional[bool]
    evidence: str

    @property
    def verdict(self) -> str:
        return {True: "Yes", False: "**No**", None: "not measured"}[self.carried]

    def to_dict(self) -> dict:
        return asdict(self)


def _pair(baseline: dict, deploy: dict, value, digits: int = 2,
          unit: str = "") -> str:
    """`<dev figure> dev -> <deploy figure> deploy`, the evidence shape every row uses."""
    suffix = f" {unit}" if unit else ""
    return (f"{format_number(value(baseline), digits)}{suffix} dev -> "
            f"{format_number(value(deploy), digits)}{suffix} deploy")


def portability_rows(baseline: dict, deploy: dict) -> List[PortabilityRow]:
    """Spec 7.4's table, computed. Each `carried` is a comparison, not a claim."""
    both_identical = (baseline["gate"]["background"]["passed"]
                      and deploy["gate"]["background"]["passed"])
    background = (f"{baseline['gate']['background']['identical_frames']}/"
                  f"{baseline['gate']['background']['frames']} frames dev -> "
                  f"{deploy['gate']['background']['identical_frames']}/"
                  f"{deploy['gate']['background']['frames']} frames deploy")
    coverage = (f"worst gap {baseline['gate']['coverage']['worst_gap_frames']} of "
                f"{baseline['gate']['coverage']['bound_frames']} allowed dev -> "
                f"{deploy['gate']['coverage']['worst_gap_frames']} of "
                f"{deploy['gate']['coverage']['bound_frames']} deploy")
    held = clock_held_fraction(baseline), clock_held_fraction(deploy)
    return [
        PortabilityRow(
            "Non-target pixels stay bit-identical to the capture",
            both_identical, background),
        PortabilityRow(
            "How much of the frame the scheduler picks: regions and calls per frame",
            _agree(regions_per_frame(baseline), regions_per_frame(deploy),
                   REGIONS_TOLERANCE)
            and _agree(calls_per_frame(baseline), calls_per_frame(deploy)),
            f"{_pair(baseline, deploy, regions_per_frame)} regions/frame, "
            f"{_pair(baseline, deploy, calls_per_frame)} calls/frame"),
        PortabilityRow(
            "The ceil(N/K) round-robin bound",
            baseline["gate"]["coverage"]["passed"]
            and deploy["gate"]["coverage"]["passed"], coverage),
        PortabilityRow(
            "Flicker over pixels static in the source",
            _agree(baseline["flicker"]["mean_abs_diff"],
                   deploy["flicker"]["mean_abs_diff"]),
            _pair(baseline, deploy, lambda r: r["flicker"]["mean_abs_diff"])),
        PortabilityRow(
            "Absolute ms/frame on the frame path",
            _agree(baseline["run"]["ms_per_frame"], deploy["run"]["ms_per_frame"]),
            _pair(baseline, deploy, lambda r: r["run"]["ms_per_frame"], unit="ms")),
        PortabilityRow(
            f"Whether the {TARGET_FPS:.0f} FPS criterion is met",
            criterion_verdict(baseline).met == criterion_verdict(deploy).met,
            _pair(baseline, deploy, lambda r: r["run"]["fps"], digits=1,
                  unit="FPS")),
        PortabilityRow(
            "What one detect costs beside the diffusion",
            _agree(_nested_ms(baseline, "detect"), _nested_ms(deploy, "detect")),
            _pair(baseline, deploy, lambda r: _nested_ms(r, "detect"), unit="ms")),
        PortabilityRow(
            "Peak VRAM the path allocates",
            _agree(baseline["run"]["peak_vram_bytes"],
                   deploy["run"]["peak_vram_bytes"]),
            _pair(baseline, deploy,
                  lambda r: r["run"]["peak_vram_bytes"] / BYTES_PER_MIB, digits=0,
                  unit="MiB")),
        PortabilityRow(
            "The clock the card holds under load, against its own maximum",
            _agree(*held),
            "-" if None in held else
            f"{held[0] * 100:.0f}% of maximum dev "
            f"({power_limit_w(baseline):.0f} W limit) -> "
            f"{held[1] * 100:.0f}% deploy ({power_limit_w(deploy):.0f} W)"),
    ]


# --- the block spec 7.4 carries ---------------------------------------------


# What the side-by-side table prints, in order: the label, how to read it off a
# record, and how many places it is worth to. One list rather than a list of keys
# and a dict of readers, so a row cannot be labelled here and read somewhere else.
MEASURES: Tuple[Tuple[str, Callable[[dict], Optional[float]], int], ...] = (
    ("regions/frame", regions_per_frame, 2),
    ("diffusion calls/frame", calls_per_frame, 2),
    ("ms/frame, frame path", lambda r: r["run"]["ms_per_frame"], 2),
    ("ms/frame, with detection",
     lambda r: r["run"]["ms_per_frame_with_detection"], 2),
    ("FPS", lambda r: r["run"]["fps"], 1),
    ("ms/detect", lambda r: _nested_ms(r, "detect"), 2),
    ("composite ms/frame", lambda r: _nested_ms(r, "composite"), 2),
    ("flicker (static px)", lambda r: r["flicker"]["mean_abs_diff"], 2),
    ("peak VRAM (MiB)", lambda r: r["run"]["peak_vram_bytes"] / BYTES_PER_MIB, 0),
    ("mean SM clock (MHz)", lambda r: r["run"].get("mean_sm_clock_mhz"), 0),
)


def _measure_table(baseline: dict, deploy: dict) -> str:
    header = (f"| measure | {gpu_of(baseline)} (dev) | "
              f"{gpu_of(deploy)} (deploy) | deploy / dev |")
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for label, read, digits in MEASURES:
        left, right = read(baseline), read(deploy)
        ratio = _ratio(right, left)
        lines.append("| " + " | ".join([
            label, format_number(left, digits), format_number(right, digits),
            "-" if ratio is None else f"{ratio:.2f}x",
        ]) + " |")
    return "\n".join(lines)


def _rows_table(rows: Sequence[PortabilityRow]) -> str:
    header = "| Conclusion | Carried? | Evidence |"
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    lines += [f"| {row.conclusion} | {row.verdict} | {row.evidence} |"
              for row in rows]
    return "\n".join(lines)


def _preamble(baseline: dict, deploy: dict) -> str:
    case, clip = deploy["case"], deploy["clip"]
    return (
        f"`{case['name']}` measured on {gpu_of(deploy)} "
        f"({clock_regime(deploy)} clocks, {power_limit_w(deploy):.0f} W) against the "
        f"{gpu_of(baseline)} baseline of {_finished(baseline)[:10]} "
        f"({clock_regime(baseline)} clocks, {power_limit_w(baseline):.0f} W): the "
        f"same {deploy['run']['engine_scenario']} engine rebuilt for this "
        f"architecture, the same {clip['frames_used']} frames of `{clip['name']}` at "
        f"the app's {case['canvas']}x{case['canvas']} capture canvas, the same "
        f"`{deploy['plan']['concept']} / {deploy['plan']['region']} / "
        f"t_index {deploy['plan']['t_index']}` plan."
    )


def format_portability_report(results: Mapping[str, dict],
                              target_fps: float = TARGET_FPS) -> str:
    """The measured block spec 7.4 carries, from the committed selective runs.

    Generated rather than transcribed, for the reason 7.2, 8.1, 8.2 and 8.8 are: a
    table pasted into Markdown drifts the moment a case is re-measured and nothing
    notices. An incomparable pair prints no table at all - a table is the part
    someone quotes, and one built across two region counts would be quoted wrongly.
    """
    dev, deploy = split_by_role(results)
    if not deploy:
        return ("no deploy-hardware run committed yet: spec 7.4's absolute rows are "
                "still unanswered (issue #24)")
    if not dev:
        return ("no dev-hardware baseline to compare the deploy run against: "
                "spec 7.4's table is a comparison, not a single number")

    baseline, card = dev[0], deploy[0]
    check = comparability(baseline, card)
    if not check.comparable:
        return (f"{_preamble(baseline, card)}\n\nThe two runs are **not comparable**, "
                f"so no table is drawn: {check.statement}. Region count drives cost; "
                f"re-run both arms at one region count.")

    verdict = criterion_verdict(card, target_fps)
    spread = fps_spread(results, gpu_of(card), target_fps)
    return "\n\n".join([
        _preamble(baseline, card),
        _measure_table(baseline, card),
        f"**Acceptance criterion 2 ({target_fps:.0f} FPS): "
        f"{'MET' if verdict.met else 'NOT MET'}"
        f"{'' if spread.decisive else ', but not decisively'}.** "
        f"{verdict.statement}. {spread.statement}.",
        "What carried across the move, measured:",
        _rows_table(portability_rows(baseline, card)),
        f"Comparable because {check.statement}.",
    ])
