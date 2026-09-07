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

from bench.results import BYTES_PER_MIB, format_number, latest_per
from bench.selective import DEVICE_COMPOSITE, HOST_COMPOSITE

# How one figure is read off one record. The side-by-side table and the evidence
# cells share these, so a measure is read the same way wherever it appears.
Reader = Callable[[dict], Optional[float]]

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


def is_deploy_gpu(gpu_name: str) -> bool:
    """Is this one of the cards v1 deploys to?"""
    name = (gpu_name or "").lower()
    return any(marker in name for marker in DEPLOY_GPU_MARKERS)


# --- reading one record ------------------------------------------------------


def gpu_of(result: dict) -> str:
    return result["hardware"]["gpu_name"]


def _finished(result: dict) -> str:
    return str(result["run"]["finished_utc"])


def regions_per_frame(result: dict) -> float:
    return float(result["regions"]["regions_per_frame"])


def calls_per_frame(result: dict) -> float:
    run = result["run"]
    return run["diffusion_calls"] / max(1, run["frames"])


def ms_per_frame(result: dict) -> float:
    return float(result["run"]["ms_per_frame"])


def ms_per_frame_with_detection(result: dict) -> float:
    return float(result["run"]["ms_per_frame_with_detection"])


def fps(result: dict) -> float:
    return float(result["run"]["fps"])


def flicker(result: dict) -> float:
    return float(result["flicker"]["mean_abs_diff"])


def peak_vram_mib(result: dict) -> float:
    return result["run"]["peak_vram_bytes"] / BYTES_PER_MIB


def mean_sm_clock_mhz(result: dict) -> Optional[float]:
    return result["run"].get("mean_sm_clock_mhz")


def clock_regime(result: dict) -> str:
    return str(result["hardware"]["clock_lock"]["state"])


def power_limit_w(result: dict) -> Optional[float]:
    hardware = result["hardware"]
    return hardware.get("enforced_power_limit_w") or hardware.get("power_limit_w")


def _nested_ms(result: dict, key: str) -> Optional[float]:
    block = result["run"].get(key)
    return None if block is None else float(block["mean_ms"])


def detect_ms(result: dict) -> Optional[float]:
    """What one detect cost, beside the diffusion rather than behind it."""
    return _nested_ms(result, "detect")


def composite_ms(result: dict) -> Optional[float]:
    return _nested_ms(result, "composite")


def composite_path(result: dict) -> str:
    """Which implementation of C7 produced that figure (issue #31).

    A record with no field ran on the host: the device path post-dates every one of
    them. An answer, not a gap.
    """
    return str(result["run"].get("composite_path", HOST_COMPOSITE))


# How each recorded composite path reads in a sentence.
COMPOSITE_PATHS = {HOST_COMPOSITE: "host (numpy)", DEVICE_COMPOSITE: "device (torch)"}


def _where(path: str) -> str:
    """A recorded composite path, in words. An unknown one speaks for itself."""
    return COMPOSITE_PATHS.get(path, path)


def clock_held_fraction(result: dict) -> Optional[float]:
    """Mean SM clock under load as a fraction of the card's own maximum.

    The laptop's answer and a desktop's are the throttling difference the issue
    asks to be recorded rather than normalised away.
    """
    mean = mean_sm_clock_mhz(result)
    ceiling = result["hardware"]["clock_lock"].get("max_sm_clock_mhz")
    if not mean or not ceiling:
        return None
    return float(mean) / float(ceiling)


# --- comparing two figures ---------------------------------------------------


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


# --- which machine is which --------------------------------------------------


def latest_per_gpu(results: Mapping[str, dict]) -> Dict[str, dict]:
    """One run per GPU: the most recently finished on each machine.

    Keyed by GPU rather than by case, which is what `bench.selective` keys by. Two
    runs on one card are two honest records and both stay on disk; the comparison
    wants the newest of each, or it compares a first attempt against a settled one.
    """
    return latest_per(results, gpu_of)


def split_by_role(results: Mapping[str, dict]) -> Tuple[List[dict], List[dict]]:
    """`(dev, deploy)` - one run per GPU, newest first within each role.

    The split is read off the fingerprint, so a record cannot be filed under a
    machine it was not measured on.
    """
    runs = sorted(latest_per_gpu(results).values(), key=_finished, reverse=True)
    return ([run for run in runs if not is_deploy_gpu(gpu_of(run))],
            [run for run in runs if is_deploy_gpu(gpu_of(run))])


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
    dev_regions = regions_per_frame(baseline)
    deploy_regions = regions_per_frame(deploy)
    comparable = bool(_agree(dev_regions, deploy_regions, tolerance))
    if comparable:
        judgement = f"the same selection to within {tolerance * 100:.0f}%"
    else:
        judgement = (f"further apart than the {tolerance * 100:.0f}% that makes a "
                     f"millisecond figure a hardware figure")
    statement = (f"{gpu_of(baseline)} rendered {dev_regions:.2f} regions/frame and "
                 f"{gpu_of(deploy)} rendered {deploy_regions:.2f}, {judgement}")
    return Comparability(baseline_regions=round(dev_regions, 4),
                         deploy_regions=round(deploy_regions, 4),
                         tolerance=tolerance, comparable=comparable,
                         statement=statement)


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
               target_fps: float = TARGET_FPS,
               composite: Optional[str] = None) -> Spread:
    """The FPS every committed run on `gpu_name` reached - all of them, not the newest.

    `composite` narrows that to the runs that blended the way the run being judged
    did. A spread answers "is this verdict inside the run-to-run noise", and runs of
    a design this one replaced are history rather than noise - but they are also the
    "before" the change is read against, so the ones left out are counted out loud
    instead of vanishing (issue #31).
    """
    def blended_the_same_way(result: dict) -> bool:
        return composite is None or composite_path(result) == composite

    on_card = [result for result in results.values() if gpu_of(result) == gpu_name]
    counted = [result for result in on_card if blended_the_same_way(result)]
    excluded = [result for result in on_card if not blended_the_same_way(result)]
    rates = sorted(fps(result) for result in counted)
    if not rates:
        return Spread(gpu=gpu_name, runs=0, lowest_fps=0.0, highest_fps=0.0,
                      target_fps=target_fps, decisive=False,
                      statement=f"no committed run on {gpu_name}")
    slowest, fastest = rates[0], rates[-1]
    decisive = (slowest >= target_fps) == (fastest >= target_fps)
    if decisive:
        side = "clear of" if slowest >= target_fps else "short of"
        caveat = ""
    else:
        side = "either side of"
        caveat = " - the verdict is inside the run-to-run spread, not outside it"
    statement = (
        f"{len(rates)} committed run{'' if len(rates) == 1 else 's'} on {gpu_name} "
        f"span {slowest:.1f}-{fastest:.1f} FPS, {side} the "
        f"{target_fps:.0f} FPS target{caveat}{_excluded_phrase(excluded)}"
    )
    return Spread(gpu=gpu_name, runs=len(rates), lowest_fps=round(slowest, 4),
                  highest_fps=round(fastest, 4), target_fps=target_fps,
                  decisive=decisive, statement=statement)


def _excluded_phrase(dropped: Sequence[dict]) -> str:
    """What was left out of the spread, and where it blended. Empty when nothing was."""
    if not dropped:
        return ""
    paths = ", ".join(sorted({_where(composite_path(run)) for run in dropped}))
    return (f" ({len(dropped)} earlier run{'' if len(dropped) == 1 else 's'} on the "
            f"card blended on the {paths} and "
            f"{'is' if len(dropped) == 1 else 'are'} not in this spread)")


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
    budget = 1000.0 / target_fps
    cost = ms_per_frame_with_detection(result)
    regions = regions_per_frame(result)
    over = cost / budget if budget else 0.0
    met = cost <= budget
    margin = (f"{budget - cost:.2f} ms to spare" if met
              else f"{over:.2f}x the budget")
    statement = (
        f"{fps(result):.1f} FPS at {regions:.2f} regions/frame on {gpu_of(result)} - "
        f"{cost:.2f} ms per frame with detection amortised against the "
        f"{budget:.2f} ms a {target_fps:.0f} FPS budget allows, {margin}; "
        f"clocks {clock_regime(result)}"
    )
    return CriterionVerdict(
        gpu=gpu_of(result), fps=fps(result), ms_per_frame=ms_per_frame(result),
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


def _pair(baseline: dict, deploy: dict, value: Reader, digits: int = 2,
          unit: str = "") -> str:
    """`<dev figure> dev -> <deploy figure> deploy`, the evidence shape every row uses."""
    suffix = f" {unit}" if unit else ""
    return (f"{format_number(value(baseline), digits)}{suffix} dev -> "
            f"{format_number(value(deploy), digits)}{suffix} deploy")


def portability_rows(baseline: dict, deploy: dict) -> List[PortabilityRow]:
    """Spec 7.4's table, computed. Each `carried` is a comparison, not a claim."""
    dev_background, deploy_background = (baseline["gate"]["background"],
                                         deploy["gate"]["background"])
    background = (f"{dev_background['identical_frames']}/"
                  f"{dev_background['frames']} frames dev -> "
                  f"{deploy_background['identical_frames']}/"
                  f"{deploy_background['frames']} frames deploy")
    dev_coverage, deploy_coverage = (baseline["gate"]["coverage"],
                                     deploy["gate"]["coverage"])
    coverage = (f"worst gap {dev_coverage['worst_gap_frames']} of "
                f"{dev_coverage['bound_frames']} allowed dev -> "
                f"{deploy_coverage['worst_gap_frames']} of "
                f"{deploy_coverage['bound_frames']} deploy")
    dev_held, deploy_held = clock_held_fraction(baseline), clock_held_fraction(deploy)
    # Both halves of the selection, or the row has not carried: the same regions
    # rendered in a different number of calls is a different render.
    selection = (_agree(regions_per_frame(baseline), regions_per_frame(deploy),
                        REGIONS_TOLERANCE)
                 and _agree(calls_per_frame(baseline), calls_per_frame(deploy)))

    def compared(conclusion: str, read: Reader, digits: int = 2,
                 unit: str = "") -> PortabilityRow:
        """One row off a single reader: the same figure decides it and evidences it."""
        return PortabilityRow(conclusion, _agree(read(baseline), read(deploy)),
                              _pair(baseline, deploy, read, digits, unit))

    return [
        PortabilityRow(
            "Non-target pixels stay bit-identical to the capture",
            dev_background["passed"] and deploy_background["passed"], background),
        PortabilityRow(
            "How much of the frame the scheduler picks: regions and calls per frame",
            selection,
            f"{_pair(baseline, deploy, regions_per_frame)} regions/frame, "
            f"{_pair(baseline, deploy, calls_per_frame)} calls/frame"),
        PortabilityRow(
            "The ceil(N/K) round-robin bound",
            dev_coverage["passed"] and deploy_coverage["passed"], coverage),
        compared("Flicker over pixels static in the source", flicker),
        compared("Absolute ms/frame on the frame path", ms_per_frame, unit="ms"),
        # Not a `compared` row: the question is whether the two runs land on the
        # same side of the criterion, which two figures 2x apart can still do.
        PortabilityRow(
            f"Whether the {TARGET_FPS:.0f} FPS criterion is met",
            criterion_verdict(baseline).met == criterion_verdict(deploy).met,
            _pair(baseline, deploy, fps, digits=1, unit="FPS")),
        compared("What one detect costs beside the diffusion", detect_ms, unit="ms"),
        compared("Peak VRAM the path allocates", peak_vram_mib, digits=0,
                 unit="MiB"),
        PortabilityRow(
            "The clock the card holds under load, against its own maximum",
            _agree(dev_held, deploy_held),
            "-" if dev_held is None or deploy_held is None else
            f"{dev_held * 100:.0f}% of maximum dev "
            f"({power_limit_w(baseline):.0f} W limit) -> "
            f"{deploy_held * 100:.0f}% deploy ({power_limit_w(deploy):.0f} W)"),
    ]


# --- the block spec 7.4 carries ---------------------------------------------


# What the side-by-side table prints, in order: the label, how to read it off a
# record, and how many places it is worth to. One list rather than a list of keys
# and a dict of readers, so a row cannot be labelled here and read somewhere else.
MEASURES: Tuple[Tuple[str, Reader, int], ...] = (
    ("regions/frame", regions_per_frame, 2),
    ("diffusion calls/frame", calls_per_frame, 2),
    ("ms/frame, frame path", ms_per_frame, 2),
    ("ms/frame, with detection", ms_per_frame_with_detection, 2),
    ("FPS", fps, 1),
    ("ms/detect", detect_ms, 2),
    ("composite ms/frame", composite_ms, 2),
    ("flicker (static px)", flicker, 2),
    ("peak VRAM (MiB)", peak_vram_mib, 0),
    ("mean SM clock (MHz)", mean_sm_clock_mhz, 0),
)


def composite_note(baseline: dict, deploy: dict) -> str:
    """Whether the `composite ms/frame` row above is a hardware comparison at all.

    It is the row issue #31 exists because of - 4.03 ms against 2.75, host code that
    a faster GPU did not shrink - and the moment one machine's run blends on the
    device it stops being two cards and becomes two designs. Said out loud rather
    than left for a reader to notice, for the reason `comparability` refuses a table
    across two region counts.
    """
    dev, deployed = composite_path(baseline), composite_path(deploy)
    if dev == deployed:
        return f"The composite ran on the {_where(dev)} on both machines."
    return (f"**The `composite ms/frame` row is not a hardware ratio**: the blend ran "
            f"on the {_where(dev)} on {gpu_of(baseline)} and on the {_where(deployed)} "
            f"on {gpu_of(deploy)} (issue #31), so those two figures are two designs "
            f"as much as two cards.")


def _table(header: str, cells: Sequence[Sequence[str]]) -> str:
    """One Markdown table: header, separator, rows.

    The separator is derived from the header, so a column added to the header cannot
    leave a separator of the wrong width behind - which renders the whole table as
    plain text. A function rather than the module-level `HEADER`/`SEPARATOR` pair the
    other report tables use, because this block's header names the two GPUs and so is
    only known once there are two records to name.
    """
    separator = "|" + "---|" * (header.count("|") - 1)
    return "\n".join([header, separator]
                     + ["| " + " | ".join(row) + " |" for row in cells])


def _measure_table(baseline: dict, deploy: dict) -> str:
    rows = []
    for label, read, digits in MEASURES:
        left, right = read(baseline), read(deploy)
        ratio = _ratio(right, left)
        rows.append([label, format_number(left, digits),
                     format_number(right, digits),
                     "-" if ratio is None else f"{ratio:.2f}x"])
    return _table(f"| measure | {gpu_of(baseline)} (dev) | "
                  f"{gpu_of(deploy)} (deploy) | deploy / dev |", rows)


def _rows_table(rows: Sequence[PortabilityRow]) -> str:
    return _table("| Conclusion | Carried? | Evidence |",
                  [[row.conclusion, row.verdict, row.evidence] for row in rows])


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
    dev_runs, deploy_runs = split_by_role(results)
    if not deploy_runs:
        return ("no deploy-hardware run committed yet: spec 7.4's absolute rows are "
                "still unanswered (issue #24)")
    if not dev_runs:
        return ("no dev-hardware baseline to compare the deploy run against: "
                "spec 7.4's table is a comparison, not a single number")

    baseline, deploy = dev_runs[0], deploy_runs[0]
    check = comparability(baseline, deploy)
    if not check.comparable:
        return (f"{_preamble(baseline, deploy)}\n\nThe two runs are **not "
                f"comparable**, so no table is drawn: {check.statement}. Region "
                f"count drives cost; re-run both arms at one region count.")

    verdict = criterion_verdict(deploy, target_fps)
    spread = fps_spread(results, gpu_of(deploy), target_fps,
                        composite=composite_path(deploy))
    return "\n\n".join([
        _preamble(baseline, deploy),
        _measure_table(baseline, deploy),
        composite_note(baseline, deploy),
        f"**Acceptance criterion 2 ({target_fps:.0f} FPS): "
        f"{'MET' if verdict.met else 'NOT MET'}"
        f"{'' if spread.decisive else ', but not decisively'}.** "
        f"{verdict.statement}. {spread.statement}.",
        "What carried across the move, measured:",
        _rows_table(portability_rows(baseline, deploy)),
        f"Comparable because {check.statement}.",
    ])
