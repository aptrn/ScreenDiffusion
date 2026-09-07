"""What `detect_every_n` buys, and what it spends. Issue #23, spec 8.8.

The frame budget has three levers on it and only one of them is free of a quality
cost. A smaller engine changes what the model can see - a 45 px region at 256² is a
handful of latent pixels - and moving the composite to the device is real work.
Raising the detector cadence changes no pixel the diffusion produces: it detects
less often and renders the same regions from slightly older boxes.

So the sweep is worth running whether or not there is a gap to close, and its
report has to say both halves of one trade in one table: the milliseconds a cadence
gives back (`bench.selective`'s `amortised_detect_ms`) and the freshness it spends
(`StalenessSummary`). A table with only the first would recommend detecting once a
second.

Three rules, each of them one of the issue's Gate items made executable:

- **A configuration that broke bit-identity is disqualified, not the criterion.**
  It cannot be recommended however cheap it is, and the recommendation names it.
- **The recommendation is the *freshest* setting that fits**, not the fastest.
  Staleness is what a cadence costs, so having cleared the budget there is nothing
  left to buy by spending more of it.
- **The baseline gap is taken, not re-derived.** Step 1 of the issue says to read
  #24's committed baseline; `format_cadence_report` reads those very records
  through `bench.portability`, so the sentence about whether an optimisation was
  needed at all is arithmetic over the same JSON spec 7.4 quotes.

GPU-free, like every other `bench.*` results module: it reads JSON and formats it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from bench.portability import (
    FRAME_BUDGET_MS,
    TARGET_FPS,
    composite_path,
    criterion_verdict,
    detect_ms,
    fps_spread,
    ms_per_frame_with_detection,
    regions_per_frame,
    split_by_role,
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

# How far apart the arms' regions/frame may be and still be one sweep. Looser than
# spec 7.4's 5%, deliberately: there, a region-count difference between two machines
# is a confound, and here it is *downstream* of the variable being swept - a staler
# tracker keeps or loses a marginal object, so the count moves a few percent because
# the cadence moved. What it costs is bounded: with the masked primitive one
# diffusion call covers every region, so the only cost that scales with the count is
# the host composite, ~2.7 ms of a ~28 ms frame. A 15% swing in regions is therefore
# ~0.4 ms, under the run-to-run spread - and past that the arms are rendering
# materially different amounts of frame and the table would be read wrongly.
REGIONS_TOLERANCE = 0.15

# How much of the 33.33 ms frame a recommendation insists is still free. The
# measurement is one process on an otherwise idle card; the app also runs a capture
# thread, a GUI process and a real screen, so a setting that fits with nothing to
# spare fits only in the benchmark. Ten percent is 3.33 ms - about one composite.
HEADROOM_FRACTION = 0.10


def cadence_of(result: Mapping) -> int:
    """The cadence an arm actually ran at, off the plan it rendered.

    The plan, not the case name: `validate_plan` clamps `detect_every_n` into
    1..30, and what a run measured is what the plan carried, not what was typed.
    """
    return int(result["plan"]["detect_every_n"])


def staleness_of(result: Mapping) -> dict:
    """The arm's staleness block, or an empty one for a record predating it."""
    return result.get("staleness") or {}


def amortised_detect_ms(result: Mapping) -> Optional[float]:
    value = result["run"].get("amortised_detect_ms")
    return None if value is None else float(value)


def background_passed(result: Mapping) -> bool:
    return bool(result["gate"]["background"]["passed"])


def headroom_ms(result: Mapping, budget_ms: float = FRAME_BUDGET_MS) -> float:
    """What a frame at this cadence left of the budget. Negative is over it."""
    return budget_ms - ms_per_frame_with_detection(result)


def latest_per_cadence(results: Mapping[str, dict]) -> Dict[str, dict]:
    """One arm per cadence *per machine*: the newest run of each.

    Per machine for the reason every other report is (issue #25): a deploy-card
    sweep is not a re-measurement of the laptop's, it is the other half of a
    portability claim, and a reduction keyed on the cadence alone would delete one
    of them.
    """
    return latest_per(results, lambda result: f"n{cadence_of(result)}")


# --- is this a sweep at all? -------------------------------------------------


@dataclass(frozen=True)
class Comparability:
    """Whether these arms rendered enough of the same thing to be one sweep.

    Spec 7.4's third trap in this section's terms: region count drives cost, so
    arms that rendered different amounts of frame differ for a reason that is not
    the cadence, and a table across them would be quoted as if they did not.
    """

    lowest_regions: float
    highest_regions: float
    tolerance: float
    comparable: bool
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def comparability(results: Sequence[dict],
                  tolerance: float = REGIONS_TOLERANCE) -> Comparability:
    regions = sorted(regions_per_frame(result) for result in results)
    lowest, highest = (regions[0], regions[-1]) if regions else (0.0, 0.0)
    spread = 0.0 if highest == 0 else (highest - lowest) / highest
    comparable = spread <= tolerance
    if comparable:
        judgement = (f"every arm rendered {lowest:.2f}-{highest:.2f} regions/frame, "
                     f"the same selection to within {tolerance * 100:.0f}%")
    else:
        judgement = (f"the arms rendered {lowest:.2f}-{highest:.2f} regions/frame, "
                     f"further apart than the {tolerance * 100:.0f}% that makes the "
                     f"difference between them a cadence difference")
    return Comparability(lowest_regions=round(lowest, 4),
                         highest_regions=round(highest, 4), tolerance=tolerance,
                         comparable=comparable, statement=judgement)


# --- which cadence to ship ---------------------------------------------------


@dataclass(frozen=True)
class Recommendation:
    """One machine's answer: the cadence to run at, and why that one."""

    gpu: str
    detect_every_n: int
    arms: int
    disqualified: Tuple[int, ...]
    ms_per_frame_with_detection: float
    budget_ms: float
    headroom_ms: float
    headroom_required_ms: float
    meets_budget: bool
    has_headroom: bool
    staleness: str
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def _reason(meets_budget: bool, has_headroom: bool, required: float,
            budget: float) -> str:
    """Why this arm and not another - the rule, in the words it was applied in."""
    if has_headroom:
        return (f"the freshest cadence measured that leaves at least "
                f"{required:.2f} ms of the {budget:.2f} ms budget free")
    if meets_budget:
        return (f"the freshest cadence measured that fits the {budget:.2f} ms "
                f"budget, though without the {required:.2f} ms of headroom a "
                f"machine that is also capturing a screen and driving a GUI wants")
    return (f"no cadence measured fits the {budget:.2f} ms budget; this is the "
            f"cheapest of them, and the gap is what the other levers have to close")


def _cadence_list(cadences: Sequence[int]) -> str:
    return ", ".join(f"detect_every_n {cadence}" for cadence in cadences)


def _choose(qualified: Sequence[dict], every_arm: Sequence[dict], budget_ms: float,
            required: float) -> dict:
    """The freshest arm that clears the budget by `required`, else that clears it
    at all, else the cheapest measured - which is not a recommendation and says so.

    `qualified` is in cadence order, so the first match is the freshest one, and
    `every_arm` is what is left to name when nothing qualified.
    """
    for wanted in (required, 0.0):
        for result in qualified:
            if headroom_ms(result, budget_ms) >= wanted:
                return result
    return min(qualified or every_arm, key=ms_per_frame_with_detection)


def recommend_cadence(results: Sequence[dict], budget_ms: float = FRAME_BUDGET_MS,
                      headroom_fraction: float = HEADROOM_FRACTION
                      ) -> Optional[Recommendation]:
    """The cadence to ship at, out of one machine's arms. `None` with no arms.

    Freshest-that-fits rather than fastest: raising the cadence costs staleness and
    nothing else, so past the budget there is nothing further to buy with it.
    """
    if not results:
        return None
    disqualified = tuple(sorted(cadence_of(result) for result in results
                                if not background_passed(result)))
    qualified = sorted((result for result in results if background_passed(result)),
                       key=cadence_of)
    required = budget_ms * headroom_fraction
    chosen = _choose(qualified, results, budget_ms, required)
    cadence, gpu = cadence_of(chosen), gpu_of(chosen)
    cost = ms_per_frame_with_detection(chosen)
    headroom = headroom_ms(chosen, budget_ms)
    meets_budget, has_headroom = headroom >= 0.0, headroom >= required
    refused = ("" if not disqualified else
               f"; {_cadence_list(disqualified)} disqualified for leaving "
               f"non-target pixels other than bit-identical")
    return Recommendation(
        gpu=gpu, detect_every_n=cadence, arms=len(results),
        disqualified=disqualified, ms_per_frame_with_detection=cost,
        budget_ms=round(budget_ms, 4), headroom_ms=round(headroom, 4),
        headroom_required_ms=round(required, 4), meets_budget=meets_budget,
        has_headroom=has_headroom,
        staleness=staleness_of(chosen).get("statement", ""),
        statement=(
            f"`detect_every_n: {cadence}` on {gpu} - "
            f"{cost:.2f} ms per frame with detection "
            f"amortised, {headroom:+.2f} ms against the {budget_ms:.2f} ms a "
            f"{TARGET_FPS:.0f} FPS budget allows: "
            f"{_reason(meets_budget, has_headroom, required, budget_ms)}"
            f"{refused}"),
    )


# --- is the rule deciding, or is the noise? ----------------------------------


@dataclass(frozen=True)
class RepeatSpread:
    """How far apart two runs of one cadence were, and whether that settles it.

    The recommendation turns on a 3.33 ms line. Issue #24 asked the same question
    of its 1 ms margin and answered it by running the case five times: a verdict
    inside the run-to-run spread is a spread, not a verdict. So every arm here was
    measured twice, and `decisive` says whether any of them sits close enough to
    the line that the other run of the *same* arm would have put it on the other
    side.
    """

    cadences: int
    repeats: int
    worst_ms: float
    closest_margin_ms: Optional[float]
    decisive: bool
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def _by_cadence(results: Sequence[dict]) -> Dict[int, List[float]]:
    costs: Dict[int, List[float]] = {}
    for result in results:
        costs.setdefault(cadence_of(result), []).append(
            ms_per_frame_with_detection(result))
    return costs


def repeat_spread(results: Sequence[dict], budget_ms: float = FRAME_BUDGET_MS,
                  headroom_fraction: float = HEADROOM_FRACTION) -> RepeatSpread:
    """Every committed run of each cadence - all of them, not the newest."""
    costs = _by_cadence(results)
    spreads = [max(runs) - min(runs) for runs in costs.values() if len(runs) > 1]
    worst = max(spreads) if spreads else 0.0
    line = budget_ms * (1.0 - headroom_fraction)
    margins = [abs(cost - line) for runs in costs.values() for cost in runs]
    closest = min(margins) if margins else None
    repeats = sum(len(runs) - 1 for runs in costs.values())
    decisive = bool(spreads) and closest is not None and closest > worst
    if not spreads:
        statement = (f"each of the {len(costs)} cadences was measured once, so "
                     f"there is no run-to-run spread to judge the recommendation "
                     f"against")
    elif decisive:
        statement = (f"each cadence was measured twice and the two runs of one "
                     f"never differed by more than {worst:.2f} ms, against "
                     f"{closest:.2f} ms from the nearest arm to the headroom line - "
                     f"so the recommendation is outside the run-to-run spread")
    else:
        statement = (f"each cadence was measured twice and the two runs of one "
                     f"differed by up to {worst:.2f} ms, which is not less than the "
                     f"{closest:.2f} ms from the nearest arm to the headroom line - "
                     f"so that arm's side of the line is inside the run-to-run "
                     f"spread, not outside it")
    return RepeatSpread(cadences=len(costs), repeats=repeats,
                        worst_ms=round(worst, 4),
                        closest_margin_ms=(None if closest is None
                                           else round(closest, 4)),
                        decisive=decisive, statement=statement)


# --- step 1: the gap this sweep was contingent on ----------------------------


# What there is to say when nothing on disk was measured on a deploy card. The
# sweep still reports its arms; what it cannot report is whether the unmodified
# path needed them.
NO_BASELINE = (
    "Step 1 has no deploy-hardware baseline to read: nothing in "
    "`bench/results/selective/` was measured on a 3090 Ti or a 4090, so "
    "whether the unmodified path has a gap is unanswered (issue #24)."
)


def baseline_statement(baseline: Optional[Mapping[str, dict]]) -> str:
    """Whether the unmodified 512x512 path had a gap, from #24's own records.

    Read rather than re-derived, which is the issue's first Gate item and its first
    trap: the decision not to build a 384² engine rests on this sentence, so it is
    computed from the committed baseline JSON by the same function spec 7.4's
    verdict comes from.
    """
    _, deploy = split_by_role(baseline or {})
    if not deploy:
        return NO_BASELINE
    verdict = criterion_verdict(deploy[0])
    # The spread of the design the verdict is about, not of every run the card ever
    # produced: since issue #31 the same case has been measured on both sides of a
    # composite that moved to the device, and one span across the two is not noise.
    spread = fps_spread(baseline, verdict.gpu,
                        composite=composite_path(deploy[0]))
    if verdict.met:
        finding = ("so **there is no gap to close by lowering the resolution, and "
                   "no lower-resolution engine was built** - a 384x384 or 256x256 "
                   "engine would trade what the model can see for speed nobody "
                   "needs")
    else:
        short_by = verdict.ms_per_frame_with_detection - verdict.budget_ms
        finding = (f"so the unmodified path is short of the budget by "
                   f"{short_by:.2f} ms, which is the gap the other levers have "
                   f"to close")
    return (f"Step 1, taken from issue #24's committed baseline rather than "
            f"re-derived: {verdict.statement}. {spread.statement} - {finding}.")


# --- the block spec 8.8 carries ----------------------------------------------


REPORT_HEADER = ("| detect_every_n | regions/frame | ms/detect |"
                 " amortised ms/frame | ms/frame | +detect | FPS |"
                 " headroom (ms) | box age (frames) | refresh IoU |"
                 " ids/objects | flicker | background |")


def _row(result: dict, column: GpuColumn, budget_ms: float) -> str:
    stale = staleness_of(result)
    run = result["run"]
    ids = (f"{stale['distinct_track_ids']}/{stale['max_concurrent_tracks']}"
           if stale else "-")
    return column.row([
        str(cadence_of(result)),
        format_number(regions_per_frame(result), 2),
        format_number(detect_ms(result), 2),
        format_number(amortised_detect_ms(result), 2),
        format_number(run["ms_per_frame"], 2),
        format_number(ms_per_frame_with_detection(result), 2),
        format_number(run["fps"], 1),
        f"{headroom_ms(result, budget_ms):+.2f}",
        format_number(stale.get("mean_age_frames"), 2),
        format_number(stale.get("mean_refresh_iou"), 2),
        ids,
        format_number(result["flicker"]["mean_abs_diff"], 2),
        "identical" if background_passed(result) else "CHANGED",
    ], result)


def _preamble(results: Sequence[dict], gpus: Sequence[str]) -> str:
    primary = results[0]
    case, clip, run = primary["case"], primary["clip"], primary["run"]
    cadences = sorted({cadence_of(result) for result in results})
    return (
        f"`detect_every_n` swept over {', '.join(str(n) for n in cadences)} on "
        f"{', '.join(gpus)}, through the shipped selective path: the same "
        f"{run['engine_scenario']} engine, the same {clip['frames_used']} frames of "
        f"`{clip['name']}` at the app's {case['canvas']}x{case['canvas']} capture "
        f"canvas, the same `{primary['plan']['concept']} / "
        f"{primary['plan']['region']} / t_index {primary['plan']['t_index']}` plan. "
        f"Only the cadence moves, so no pixel the diffusion produces changes - what "
        f"a higher cadence spends is the freshness of the boxes, which is the right "
        f"half of the table."
    )


def _recommended_arm(results: Sequence[dict],
                     recommendation: Recommendation) -> Optional[dict]:
    """The record the recommendation named, so the report can point at its clip."""
    for result in results:
        if cadence_of(result) == recommendation.detect_every_n:
            return result
    return None


def _machine_sections(reduced: Sequence[dict], every_run: Sequence[dict],
                      gpus: Sequence[str], budget_ms: float) -> List[str]:
    """The recommendation, what it costs, and how repeatable it is - per machine.

    A recommendation is a claim about one machine's milliseconds (issue #25), and
    fed two machines' arms at once it would rank a 4090 arm against a laptop one.
    `every_run` is all the committed runs rather than the newest per cadence,
    because the spread that says whether the rule decided anything is a fact about
    the repeats, not about the row.
    """
    sections = []
    for gpu in gpus:
        arms = measured_on(reduced, gpu)
        recommendation = recommend_cadence(arms, budget_ms)
        if recommendation is None:
            continue
        sections.append(f"**Recommended: {recommendation.statement}.**")
        if recommendation.staleness:
            sections.append(
                f"What that costs: {sentence_case(recommendation.staleness)}.")
        spread = repeat_spread(measured_on(every_run, gpu), budget_ms)
        sections.append(f"{sentence_case(spread.statement)}.")
        chosen = _recommended_arm(arms, recommendation)
        if chosen and chosen.get("comparison_clip"):
            sections.append(
                f"Manual verification artefact at the recommended cadence: "
                f"`{chosen['comparison_clip']}` (source | selective render) and "
                f"`{chosen['comparison_still']}`.")
    return sections


def format_cadence_report(results: Mapping[str, dict],
                          baseline: Optional[Mapping[str, dict]] = None,
                          budget_ms: float = FRAME_BUDGET_MS) -> str:
    """The measured block spec 8.8 carries for the cadence sweep.

    Generated from the committed JSON rather than transcribed, for the reason every
    other block in the spec is: a table pasted into Markdown drifts the moment an
    arm is re-measured and nothing notices.
    """
    ordered = sorted(latest_per_cadence(results).values(),
                     key=lambda result: (cadence_of(result), gpu_of(result)))
    if not ordered:
        return "no cadence sweep committed yet (issue #23)"

    check = comparability(ordered)
    if not check.comparable:
        return (f"{baseline_statement(baseline)}\n\nThe arms are **not comparable**, "
                f"so no table is drawn: {check.statement}. Region count drives cost; "
                f"re-run the sweep over one clip and one plan.")

    gpus = distinct_gpus(ordered)
    column = GpuColumn.for_gpus(gpus)
    header = column.header(REPORT_HEADER)
    sections = [
        baseline_statement(baseline),
        _preamble(ordered, gpus),
        "\n".join([header, table_separator(header)]
                  + [_row(result, column, budget_ms) for result in ordered]),
    ]
    sections += _machine_sections(ordered, list(results.values()), gpus,
                                  budget_ms)
    sections.append(
        f"Every arm above left the background bit-identical to the capture; "
        f"{check.statement}.")
    return "\n\n".join(sections)

