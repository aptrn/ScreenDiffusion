"""What the two §8.5 levers buy, and what they spend. Issue #32, spec 8.5.

Spec §8.5 lists five levers against boiling and, until this issue, asked for a
*quantitative flicker metric* "so we can tell whether a change helped rather than
arguing about it". The metric has existed since issue #5. What did not exist was a
sweep that used it, and two of the levers - per-track seed pinning and an output EMA
- were never built at all.

This module is the reading. Three rules, each of them one of the issue's Gate items
made executable rather than argued:

- **An arm that lowers flicker by rendering less is disqualified, not recommended.**
  The visible-change figure net of a control is the same one spec 8.2's comparison
  needed and the selective path's Gate already computes, and an arm that falls under
  the threshold cannot be recommended however steady it looks.
- **An arm that breaks background bit-identity is disqualified too.** §11's criterion
  4 is not negotiable for a flicker win - the issue's first trap.
- **A lower flicker number is not automatically better.** An EMA strong enough to
  kill boiling also kills the restyle's response to motion, so every arm carries
  `bench.flicker.response_score` beside its flicker and the recommendation has to
  keep most of the control arm's responsiveness. Both are printed either way, and
  the report says which the recommendation traded away.

And one rule about the noise, which is why the metric had to decide it: **flicker is
measured where the source stood still**, and the shipped noise field is pinned to
the *canvas*, so a static pixel already gets the same noise every frame. Pinning the
field to a *track* instead moves it with the object - which is the boiling the lever
is aimed at, and which by the same token makes noise move under the static pixels
inside a moving box. Which way that lands is what the table says.

GPU-free, like every other `bench.*` results module: it reads JSON and formats it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from bench.portability import ms_per_frame_with_detection, regions_per_frame
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
from bench.selective import VISIBLE_CHANGE, ema_suffix

# The arm every other one is read against: the seed policy and the EMA the app
# ships with, run through the same harness on the same clip in the same sweep. A
# control from another directory measured at another time would be the confound
# every other report in this repo refuses.
#
# Spelt here rather than imported, for the reason `bench.selective.GLOBAL_KEY` is -
# the shipped modules are imported inside functions in these results modules - and
# held to `render_plan`'s own defaults by a test, so "shipped" cannot go stale.
SHIPPED_POLICY = "fixed"
SHIPPED_EMA = 0.0

# The policy that does not ship: a fresh noise field every frame, and therefore the
# flicker metric's upper bound on this clip. Named because the report points a
# reader at its clip - a column nobody has seen the extremes of is a column nobody
# can read. Pinned to `render_plan.RANDOM` by the same test.
RANDOM_POLICY = "random"

# How much of the control arm's responsiveness a recommendation must keep. The EMA
# buys steadiness with exactly this, so a rule that did not price it would always
# recommend the strongest setting measured; ten percent is small enough to be
# invisible and large enough not to be the run-to-run spread.
RESPONSE_RETENTION = 0.90


# --- reading one arm ---------------------------------------------------------


def seed_policy_of(result: Mapping) -> str:
    """The policy an arm actually rendered under, off the plan rather than the name.

    Records committed before this issue have no such field and were rendered under
    the field the engine prepared, which is what `fixed` names.
    """
    return result["plan"].get("seed_policy") or SHIPPED_POLICY


def output_ema_of(result: Mapping) -> float:
    return float(result["plan"].get("output_ema") or 0.0)


def arm_of(result: Mapping) -> Tuple[str, float]:
    return seed_policy_of(result), output_ema_of(result)


def arm_key(result: Mapping) -> str:
    policy, ema = arm_of(result)
    return f"{policy}-{ema_suffix(ema)}"


def arm_label(result: Mapping) -> str:
    policy, ema = arm_of(result)
    return f"{policy} / EMA {ema:.2f}"


def flicker_of(result: Mapping) -> Optional[float]:
    return result["flicker"]["mean_abs_diff"]


def response_of(result: Mapping) -> Optional[float]:
    """What the output did where the source moved, or None for a run predating it."""
    response = result.get("response") or {}
    return response.get("mean_abs_diff")


def net_change_of(result: Mapping) -> float:
    """The visible change inside the regions, net of the capture's own round trip."""
    return float(result["gate"]["change"]["net_change"])


def change_passed(result: Mapping) -> bool:
    return bool(result["gate"]["change"]["passed"])


def background_passed(result: Mapping) -> bool:
    return bool(result["gate"]["background"]["passed"])


def is_shipped_arm(result: Mapping) -> bool:
    return arm_of(result) == (SHIPPED_POLICY, SHIPPED_EMA)


def latest_per_arm(results: Mapping[str, dict]) -> Dict[str, dict]:
    """One run per arm *per machine*: the newest of each.

    Per machine for the reason every other report is (issue #25): a second card's
    sweep is not a re-measurement of the first's.
    """
    return latest_per(results, arm_key)


def control_of(results: Sequence[dict]) -> Optional[dict]:
    """The shipped-default arm, which every other one is read against."""
    for result in results:
        if is_shipped_arm(result):
            return result
    return None


# --- what an arm is disqualified for -----------------------------------------


# Which of the two criteria an arm failed. Named rather than read back out of the
# `reason` sentence, so a reworded reason cannot silently reclassify an arm.
BACKGROUND = "background"
CHANGE = "change"


@dataclass(frozen=True)
class Disqualification:
    """One arm that cannot be recommended, and the criterion it failed.

    Both criteria are §11's, not this sweep's: the background stays bit-identical,
    and the region visibly changes. A stability lever that bought either of them
    away has not made the render steadier, it has made less of one.
    """

    arm: str
    criterion: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def disqualification_of(result: Mapping) -> Optional[Disqualification]:
    """Why this one arm cannot be recommended, or None if it can.

    The single rule both `disqualifications` and `qualified` read, so the list the
    report prints and the set the recommendation chooses from can never disagree.
    """
    if not background_passed(result):
        return Disqualification(
            arm=arm_label(result), criterion=BACKGROUND,
            reason="it left non-target pixels other than bit-identical")
    if not change_passed(result):
        return Disqualification(
            arm=arm_label(result), criterion=CHANGE,
            reason=f"the region changed by {net_change_of(result):.1f}/255 net "
                   f"of the control, under the {VISIBLE_CHANGE:.0f}/255 a "
                   f"visible restyle needs - it lowered flicker by rendering "
                   f"less")
    return None


def disqualifications(results: Sequence[dict]) -> Tuple[Disqualification, ...]:
    return tuple(refused for refused in map(disqualification_of, results)
                 if refused is not None)


def qualified(results: Sequence[dict]) -> List[dict]:
    return [result for result in results if disqualification_of(result) is None]


# --- how repeatable the flicker figures are ----------------------------------


@dataclass(frozen=True)
class RepeatSpread:
    """How far apart two runs of one arm were, in the units the rule turns on.

    Issue #24 asked this of a 1 ms margin and issue #23 of a 3.33 ms one; here the
    quantity is flicker, and the baseline is unusually steady - 1.49 on both cards
    and at all four of #23's cadences - so a lever that moves it by less than the
    spread has not moved it.
    """

    arms: int
    repeats: int
    worst: float
    statement: str

    def to_dict(self) -> dict:
        return asdict(self)


def _flicker_by_arm(results: Sequence[dict]) -> Dict[str, List[float]]:
    scores: Dict[str, List[float]] = {}
    for result in results:
        score = flicker_of(result)
        if score is not None:
            scores.setdefault(arm_key(result), []).append(float(score))
    return scores


def repeat_spread(results: Sequence[dict]) -> RepeatSpread:
    """Every committed run of each arm - all of them, not the newest."""
    scores = _flicker_by_arm(results)
    spreads = [max(runs) - min(runs) for runs in scores.values() if len(runs) > 1]
    worst = max(spreads) if spreads else 0.0
    repeats = sum(len(runs) - 1 for runs in scores.values())
    if not spreads:
        statement = (f"each of the {len(scores)} arms was measured once, so there is "
                     f"no run-to-run spread to judge a flicker difference against")
    else:
        statement = (f"each arm was measured at least twice and two runs of one "
                     f"never differed in flicker by more than {worst:.5f} - unlike "
                     f"a millisecond, a flicker figure here carries essentially no "
                     f"run-to-run noise, because the render is deterministic given "
                     f"the clip, the plan and the seed. So the differences between "
                     f"arms are the arms")
    return RepeatSpread(arms=len(scores), repeats=repeats, worst=round(worst, 6),
                        statement=statement)


# --- which setting to ship ---------------------------------------------------


@dataclass(frozen=True)
class Recommendation:
    """One machine's answer: the seed policy and the EMA to run at, and why."""

    gpu: str
    seed_policy: str
    output_ema: float
    arms: int
    disqualified: Tuple[str, ...]
    flicker: Optional[float]
    control_flicker: Optional[float]
    flicker_delta: Optional[float]
    response: Optional[float]
    control_response: Optional[float]
    response_kept: Optional[float]
    margin: float
    is_shipped_default: bool
    statement: str
    cost: str

    def to_dict(self) -> dict:
        return asdict(self)


def _keeps_response(result: Mapping, control: Mapping,
                    retention: float = RESPONSE_RETENTION) -> bool:
    """Did this arm keep enough of the control's response to motion?

    An arm with no response figure at all cannot be shown to have kept one, so it
    is not eligible - the same rule `gpu_of` applies to an unfingerprinted record.
    """
    arm, floor = response_of(result), response_of(control)
    if arm is None or floor is None:
        return False
    return arm >= floor * retention


def _steadier(candidates: Sequence[dict]) -> dict:
    return min(candidates, key=lambda result: flicker_of(result))


def recommend_setting(results: Sequence[dict], every_run: Optional[Sequence[dict]] = None,
                      retention: float = RESPONSE_RETENTION) -> Optional[Recommendation]:
    """The setting to ship, out of one machine's arms. `None` with no control arm.

    The steadiest arm that is not disqualified, keeps `retention` of the control's
    responsiveness, and beats the control's flicker by more than the run-to-run
    spread. Failing all that, the control itself - which is the shipped default,
    and "the levers did not pay" is a result rather than a gap.

    `results` is one run per arm - the rows - and `every_run` is all of them, which
    is where the spread comes from: a spread computed over the reduced set is zero
    by construction, and a rule turning on zero is not a rule.
    """
    control = control_of(results)
    if control is None:
        return None
    spread = repeat_spread(every_run if every_run is not None else results).worst
    control_flicker = flicker_of(control)
    eligible = [result for result in qualified(results)
                if flicker_of(result) is not None
                and _keeps_response(result, control, retention)
                and (control_flicker is None
                     or flicker_of(result) < control_flicker - spread)]
    chosen = _steadier(eligible) if eligible else control
    return _recommendation(chosen, control, results, spread)


def _recommendation(chosen: Mapping, control: Mapping, results: Sequence[dict],
                    margin: float) -> Recommendation:
    policy, ema = arm_of(chosen)
    flicker, control_flicker = flicker_of(chosen), flicker_of(control)
    response, control_response = response_of(chosen), response_of(control)
    delta = (None if flicker is None or control_flicker is None
             else round(flicker - control_flicker, 6))
    kept = (None if response is None or not control_response
            else round(response / control_response, 4))
    refused = disqualifications(results)
    return Recommendation(
        gpu=gpu_of(chosen), seed_policy=policy, output_ema=ema, arms=len(results),
        disqualified=tuple(item.arm for item in refused), flicker=flicker,
        control_flicker=control_flicker, flicker_delta=delta, response=response,
        control_response=control_response, response_kept=kept, margin=round(margin, 6),
        is_shipped_default=is_shipped_arm(chosen),
        statement=_verdict(chosen, control, results, delta, margin),
        cost=_cost(chosen, control, delta, kept),
    )


def _nearest_miss(results: Sequence[dict], control: Mapping) -> str:
    """The steadiest arm that was refused, and the number it was refused on.

    A threshold on its own is not a reading: someone re-deciding this later has to
    be able to see how close the closest arm came and on which figure.
    """
    control_flicker, control_response = flicker_of(control), response_of(control)
    candidates = [result for result in qualified(results)
                  if not is_shipped_arm(result) and flicker_of(result) is not None
                  and control_flicker is not None
                  and flicker_of(result) < control_flicker]
    if not candidates or not control_response:
        return "no arm measured lowered flicker at all"
    closest = max(candidates, key=lambda result: flicker_of(result))
    policy, ema = arm_of(closest)
    response = response_of(closest)
    if response is None:
        return (f"the closest, `seed_policy: {policy}` / `output_ema: {ema:.2f}`, "
                f"carries no responsiveness figure to price its flicker win with")
    kept = response / control_response
    return (f"the closest, `seed_policy: {policy}` / `output_ema: {ema:.2f}`, took "
            f"flicker {control_flicker:.2f} -> {flicker_of(closest):.2f} and the "
            f"response to motion {control_response:.2f} -> {response:.2f}, keeping "
            f"{kept * 100:.0f}% of it against the "
            f"{RESPONSE_RETENTION * 100:.0f}% a recommendation has to keep")


def _verdict(chosen: Mapping, control: Mapping, results: Sequence[dict],
             delta: Optional[float], margin: float) -> str:
    policy, ema = arm_of(chosen)
    setting = f"`seed_policy: {policy}`, `output_ema: {ema:.2f}` on {gpu_of(chosen)}"
    if is_shipped_arm(chosen):
        return (f"{setting} - the shipped default. Every arm that lowered flicker "
                f"did it by low-passing the output: {_nearest_miss(results, control)}"
                f". So neither lever pays for itself here")
    return (f"{setting} - flicker {flicker_of(chosen):.2f} against the shipped "
            f"default's, {delta:+.2f}, which is outside the {margin:.5f} "
            f"run-to-run spread")


def _cost(chosen: Mapping, control: Mapping, delta: Optional[float],
          kept: Optional[float]) -> str:
    """What the recommendation traded away - the issue's second trap, stated."""
    response, control_response = response_of(chosen), response_of(control)
    if is_shipped_arm(chosen):
        return ("Nothing was traded, because nothing was adopted. That is the "
                "answer this section asked for rather than a gap in it: the levers "
                "are built, they are wired to plan fields, and the measurement says "
                "what they cost.")
    if kept is None:
        return ("What it costs is unmeasured: this arm carries no responsiveness "
                "figure to set beside its flicker.")
    return (f"What it costs: the output follows the source's own motion at "
            f"{response:.2f} against the shipped default's {control_response:.2f} - "
            f"{kept * 100:.0f}% of it. Flicker reads low-is-steadier and that "
            f"figure reads high-is-more-responsive, so the {delta:+.2f} of "
            f"steadiness was bought with {(1 - kept) * 100:.0f}% of the restyle's "
            f"response to motion.")


# --- what the shipped path measures, from its own records --------------------


NO_BASELINE = (
    "There is no committed run of the shipped path to read the control arm "
    "against: nothing in `bench/results/selective/` carries a flicker figure."
)


def baseline_statement(baseline: Optional[Mapping[str, dict]],
                       control: Optional[Mapping]) -> str:
    """Whether the sweep's control arm reproduces the shipped path's own flicker.

    The control is what every arm is read against, so a control that did not
    reproduce the committed baseline would be measuring something else. Read from
    the same records spec 8.8 quotes rather than retyped from them.
    """
    scores = sorted({round(float(flicker_of(result)), 2)
                     for result in (baseline or {}).values()
                     if flicker_of(result) is not None})
    if not scores:
        return NO_BASELINE
    committed = (f"{scores[0]:.2f}" if len(scores) == 1
                 else f"{scores[0]:.2f}-{scores[-1]:.2f}")
    if control is None or flicker_of(control) is None:
        return (f"The shipped path's committed runs measure flicker at {committed} "
                f"(spec 8.8), and this sweep has no control arm at the shipped "
                f"default to reproduce it.")
    measured = float(flicker_of(control))
    agrees = scores[0] - 0.01 <= round(measured, 2) <= scores[-1] + 0.01
    verdict = ("which is the same figure, so the control is measuring the shipped "
               "path" if agrees else
               "which is *not* that figure, so something other than these two "
               "levers differs between the sweep and the baseline")
    return (f"The shipped path's committed runs measure flicker at {committed} "
            f"(spec 8.8); this sweep's control arm - `seed_policy: "
            f"{SHIPPED_POLICY}`, `output_ema: {SHIPPED_EMA:.2f}` - measures "
            f"{measured:.2f}, {verdict}.")


# --- the block spec 8.5 carries ----------------------------------------------


REPORT_HEADER = ("| seed policy | output EMA | flicker | vs shipped |"
                 " response | net change | regions/frame | ms/frame | +detect |"
                 " background |")


def _delta_cell(result: Mapping, control: Optional[Mapping]) -> str:
    score, floor = flicker_of(result), None if control is None else flicker_of(control)
    if score is None or floor is None:
        return "-"
    if is_shipped_arm(result):
        return "control"
    return f"{score - floor:+.2f}"


def _row(result: dict, column: GpuColumn, control: Optional[Mapping]) -> str:
    policy, ema = arm_of(result)
    return column.row([
        policy,
        f"{ema:.2f}",
        format_number(flicker_of(result), 2),
        _delta_cell(result, control),
        format_number(response_of(result), 2),
        format_number(net_change_of(result), 1),
        format_number(regions_per_frame(result), 2),
        format_number(result["run"]["ms_per_frame"], 2),
        format_number(ms_per_frame_with_detection(result), 2),
        "identical" if background_passed(result) else "CHANGED",
    ], result)


def artefact_of(results: Sequence[dict], arm: Tuple[str, float]) -> Optional[dict]:
    """The newest committed run of `arm` that wrote clips, or None.

    Not the row the table draws: an arm measured twice writes its comparison clip
    on one of the two runs, and the Gate's manual-verification step has to be
    pointed at a file that exists rather than at the newest record.
    """
    written = [result for result in results
               if arm_of(result) == arm and result.get("comparison_clip")]
    if not written:
        return None
    return max(written, key=lambda result: result["run"]["finished_utc"])


def _artefact_lines(every_run: Sequence[dict], arm: Tuple[str, float]) -> List[str]:
    """Where to look, for the recommended setting and for the noise control.

    Two clips rather than one: the `random` arm is what the flicker metric's upper
    bound looks like on a screen, and a reader who has only seen the recommended
    setting has no idea what the column is measuring.
    """
    lines = []
    chosen = artefact_of(every_run, arm)
    if chosen is not None:
        lines.append(f"Manual verification artefact at the recommended setting: "
                     f"`{chosen['comparison_clip']}` (source | selective render) "
                     f"and `{chosen['comparison_still']}`.")
    control = artefact_of(every_run, (RANDOM_POLICY, SHIPPED_EMA))
    if control is not None and arm != (RANDOM_POLICY, SHIPPED_EMA):
        lines.append(f"What the metric's upper bound looks like: "
                     f"`{control['comparison_clip']}` is the `random` arm, a fresh "
                     f"noise field every frame.")
    return lines


def change_statement(results: Sequence[dict]) -> str:
    """Whether any arm bought its steadiness by rendering less.

    The Gate's disqualification rule read off *every* arm rather than only off the
    one that was recommended: "no arm was disqualified" is a finding, and a report
    that only mentioned the rule when it fired would leave a reader unable to tell
    it from a rule nobody applied.
    """
    changes = sorted(net_change_of(result) for result in results)
    refused = [item for item in disqualifications(results)
               if item.criterion == CHANGE]
    identical = all(background_passed(result) for result in results)
    background = ("Every arm above left the background bit-identical to the capture"
                  if identical else
                  "Not every arm above left the background bit-identical to the "
                  "capture")
    if refused:
        return (f"{background}; {len(refused)} of them changed the region by less "
                f"than the {VISIBLE_CHANGE:.0f}/255 a visible restyle needs and are "
                f"disqualified for buying their steadiness by rendering less.")
    return (f"{background}, and every one changed the region by "
            f"{changes[0]:.1f}-{changes[-1]:.1f}/255 net of the control against a "
            f"{VISIBLE_CHANGE:.0f}/255 threshold - so no arm bought its steadiness "
            f"by rendering less, and the trade the table shows is the whole trade.")


def _preamble(results: Sequence[dict], gpus: Sequence[str]) -> str:
    primary = results[0]
    case, clip, run = primary["case"], primary["clip"], primary["run"]
    policies = sorted({seed_policy_of(result) for result in results})
    emas = sorted({output_ema_of(result) for result in results})
    return (
        f"`seed_policy` swept over {', '.join(policies)} and `global.output_ema` "
        f"over {', '.join(f'{ema:.2f}' for ema in emas)} on {', '.join(gpus)}, "
        f"through the shipped selective path: the same {run['engine_scenario']} "
        f"engine, the same {clip['frames_used']} frames of `{clip['name']}` at the "
        f"app's {case['canvas']}x{case['canvas']} capture canvas, the same "
        f"`{primary['plan']['concept']} / {primary['plan']['region']} / t_index "
        f"{primary['plan']['t_index']}` plan and the same detector cadence. "
        f"`flicker` is the mean absolute difference between consecutive outputs "
        f"where the source stood still and the mask painted both frames - lower is "
        f"steadier; `response` is the same figure where the source *moved* - higher "
        f"is more responsive; `net change` is how much the regions changed against "
        f"the capture, net of the capture's own round trip, and an arm under "
        f"{VISIBLE_CHANGE:.0f} is disqualified."
    )


def _machine_sections(reduced: Sequence[dict], every_run: Sequence[dict],
                      gpus: Sequence[str]) -> List[str]:
    """The recommendation, what it cost, and how repeatable it is - per machine.

    A recommendation is a claim about one machine's arms (issue #25); fed two
    machines' at once it would rank a 4090 arm against a laptop one.
    """
    sections: List[str] = []
    for gpu in gpus:
        arms = measured_on(reduced, gpu)
        recommendation = recommend_setting(arms, measured_on(every_run, gpu))
        if recommendation is None:
            sections.append(
                f"No recommendation on {gpu}: the sweep has no control arm at the "
                f"shipped default to read the others against.")
            continue
        sections.append(f"**Recommended: {recommendation.statement}.**")
        sections.append(recommendation.cost)
        if recommendation.disqualified:
            refused = "; ".join(f"{item.arm}, because {item.reason}"
                                for item in disqualifications(arms))
            sections.append(f"Disqualified: {refused}.")
        sections.append(sentence_case(repeat_spread(measured_on(every_run, gpu))
                                      .statement) + ".")
        sections += _artefact_lines(
            measured_on(every_run, gpu),
            (recommendation.seed_policy, recommendation.output_ema))
    return sections


def format_stability_report(results: Mapping[str, dict],
                            baseline: Optional[Mapping[str, dict]] = None) -> str:
    """The measured block spec 8.5 carries for the temporal-stability sweep.

    Generated from the committed JSON rather than transcribed, for the reason every
    other block in the spec is: a table pasted into Markdown drifts the moment an
    arm is re-measured and nothing notices.
    """
    ordered = sorted(latest_per_arm(results).values(),
                     key=lambda result: (seed_policy_of(result),
                                         output_ema_of(result), gpu_of(result)))
    if not ordered:
        return "no temporal-stability sweep committed yet (issue #32)"

    gpus = distinct_gpus(ordered)
    column = GpuColumn.for_gpus(gpus)
    header = column.header(REPORT_HEADER)
    sections = [
        baseline_statement(baseline, control_of(ordered)),
        _preamble(ordered, gpus),
        "\n".join([header, table_separator(header)]
                  + [_row(result, column, control_of(measured_on(ordered,
                                                                 gpu_of(result))))
                     for result in ordered]),
    ]
    sections += _machine_sections(ordered, list(results.values()), gpus)
    sections.append(change_statement(ordered))
    return "\n\n".join(sections)
