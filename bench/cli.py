"""`uv run python -m bench <scenario>`.

Parsing, listing and the engine-build guard are all GPU-free; `bench.runner` is
imported only once a run is actually about to happen, so `--help` and `--list` work
on a machine with no CUDA device and the merge gate can exercise them.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence, TextIO, Tuple, Union

from bench import marginal
from bench.cadence import format_cadence_report
from bench.capture import (
    CASES as CAPTURE_CASES,
    CaptureCase,
    format_capture_report,
    load_capture_results,
)
from bench.clocks import ClockLock, regime_summary
from bench.contention import (
    OccupancyRecord,
    measure_occupancy_now,
    occupancy_summary,
)
from bench.cooldown import DEFAULT_CAP_S, DEFAULT_POLL_INTERVAL_S, DEFAULT_THRESHOLD_C
from bench.detector_results import format_detector_report, load_detector_results
from bench.detectors import (
    DEFAULT_CADENCE,
    DEFAULT_DIFFUSION_SCENARIO,
    DETECTORS,
    DetectorConfig,
)
from bench.disk import DiskRecord, Usage, read_disk, require_free_space
from bench.fingerprint import read_clock_lock
from bench.guidance import (
    CASES as GUIDANCE_CASES,
    GuidanceCase,
    format_guidance_report,
    load_guidance_results,
)
from bench.models import (
    BASE_MODELS,
    STYLE_LORAS,
    format_model_report,
    lora_dict_for,
    with_style,
)
from bench.quality import (
    CASES as QUALITY_CASES,
    QualityCase,
    format_quality_report,
    load_quality_results,
)
from bench.paths import (
    CADENCE_RESULTS_SUBDIR,
    CAPTURE_RESULTS_SUBDIR,
    DETECTOR_RESULTS_SUBDIR,
    GUIDANCE_RESULTS_SUBDIR,
    MODEL_RESULTS_SUBDIR,
    PRIMITIVE_RESULTS_SUBDIR,
    QUALITY_RESULTS_SUBDIR,
    RESULTS_DIR,
    SELECTIVE_RESULTS_DIR,
    SELECTIVE_RESULTS_SUBDIR,
    STABILITY_RESULTS_SUBDIR,
    STEPS_RESULTS_DIR,
    STEPS_RESULTS_SUBDIR,
    STYLE_RESULTS_SUBDIR,
    SWAP_RESULTS_SUBDIR,
    resolve_engines_dir,
)
from bench.plan_swap import (
    CASES as SWAP_CASES,
    SwapCase,
    format_swap_report,
    load_swap_results,
)
from bench.portability import format_portability_report
from bench.primitive_results import format_primitive_report, load_primitive_results
from bench.primitives import CASES, CaseConfig
from bench.results import load_records
from bench.scenarios import SCENARIOS, ScenarioConfig
from bench.selective import (
    CASES as SELECTIVE_CASES,
    SelectiveCase,
    ema_suffix,
    format_selective_report,
    load_selective_results,
    readme_preamble,
    results_subdir,
)
from bench.stability import format_stability_report
from bench.steps import format_steps_report, is_step_arm, step_arm
from bench.styles import (
    CASES as STYLE_CASES,
    StyleCase,
    format_style_report,
    load_style_results,
)
# Two shipped stdlib modules, imported for the reason every `bench.*` import here
# is stdlib: the CLI has to stay loadable without CUDA. `engine_cache` is the app's
# own answer to "which engine does this configuration need, and is it built"; the
# seed vocabulary is the app's own, so `--seed-policy` cannot drift from what a
# plan accepts.
from engine_cache import UNET_ENGINE, engine_dir_name as engine_dir_for
from render_plan import SEED_POLICIES

# What the one positional slot can name: a diffusion scenario, a detector, a
# rendering-primitive case (issue #5), an end-to-end selective render (issue #8), a
# plan swap (issue #30), a capture-geometry comparison (issue #39), a style-LoRA
# comparison (issue #38) or a classifier-free-guidance sweep (issue #45). One slot
# for all eight - a run measures one thing, the names cannot collide, and someone
# holding a name should not have to know which flag it belongs behind.
Target = Union[ScenarioConfig, DetectorConfig, CaseConfig, SelectiveCase, SwapCase,
              CaptureCase, StyleCase, GuidanceCase]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Measure one ScreenDiffusion configuration and write a tracked, "
                    "machine-readable result under bench/results/.",
        epilog="Results are written by a run and never by hand: if a run did not "
               "happen, there is no record.",
    )
    parser.add_argument("scenario", nargs="?",
                        help="scenario, detector, primitive-case, selective-case, "
                             "swap-case or capture-case name; --list shows them all")
    parser.add_argument("--list", action="store_true",
                        help="list the scenarios, detectors, primitive cases, "
                             "selective cases, swap cases and capture cases, "
                             "and exit")
    parser.add_argument("--marginal", action="store_true",
                        help="report the marginal cost per additional batch item from "
                             "the committed results, and exit")
    parser.add_argument("--detector-report", action="store_true",
                        help="report the measured detector table spec 8.1 carries, "
                             "from the committed detector results, and exit")
    parser.add_argument("--primitive-report", action="store_true",
                        help="report the rendering-primitive decision block spec 8.2 "
                             "carries, from the committed comparisons, and exit")
    parser.add_argument("--selective-report", action="store_true",
                        help="report the selective render block spec 8.8 carries, "
                             "from the committed end-to-end runs, and exit")
    parser.add_argument("--portability-report", action="store_true",
                        help="report the dev-vs-deploy block spec 7.4 carries, from "
                             "the committed selective runs on each GPU, and exit")
    parser.add_argument("--swap-report", action="store_true",
                        help="report the plan-swap block spec 8.9 carries - "
                             "acceptance criteria 1 and 3 - from the committed "
                             "swaps, and exit")
    parser.add_argument("--capture-report", action="store_true",
                        help="report the capture-geometry block spec 8.2 carries - "
                             "crop against masked at K=1 - from the committed "
                             "runs, and exit")
    parser.add_argument("--cadence-report", action="store_true",
                        help="report the detect_every_n sweep spec 8.8 carries, from "
                             "the committed arms, and exit")
    parser.add_argument("--steps-report", action="store_true",
                        help="report the step-count block spec 7.2 carries - what "
                             "one more denoising step costs - from the committed "
                             "arms, and exit")
    parser.add_argument("--model-report", action="store_true",
                        help="report the base-model block spec 7.5 carries - the "
                             "shipped path on SD-Turbo against SD 1.5 + LCM-LoRA "
                             "- from the committed arms, and exit")
    parser.add_argument("--guidance-report", action="store_true",
                        help="report the classifier-free-guidance block spec 8.11 "
                             "carries - whether turning CFG on makes the prompt "
                             "land, and what it costs - and exit")
    parser.add_argument("--quality-report", action="store_true",
                        help="print the step-count block spec 8.12 carries: what "
                             "a denoising step buys, what a cached-engine swap "
                             "costs, and which of the two routes ships (issue #46)")
    parser.add_argument("--style-report", action="store_true",
                        help="report the style-LoRA block spec 8.10 carries - "
                             "which LoRAs load on SD 1.5, which visibly change "
                             "the output, and how styles should ship - and exit")
    parser.add_argument("--stability-report", action="store_true",
                        help="report the temporal-stability sweep spec 8.5 carries - "
                             "per-track seeds and the output EMA - from the "
                             "committed arms, and exit")
    parser.add_argument("--reps", type=int, help="timed reps (default: the scenario's)")
    parser.add_argument("--warmup", type=int, dest="warmup_reps",
                        help="warmup reps before timing (default: the scenario's)")
    parser.add_argument("--prompt", help="override the scenario's prompt")

    cooldown = parser.add_argument_group("cooldown gate")
    cooldown.add_argument("--no-cooldown", dest="cooldown", action="store_false",
                          help="start hot; the result records the skip")
    cooldown.set_defaults(cooldown=True)
    cooldown.add_argument("--cooldown-threshold", type=float, default=DEFAULT_THRESHOLD_C,
                          metavar="C", help=f"degrees C (default {DEFAULT_THRESHOLD_C:.0f})")
    cooldown.add_argument("--cooldown-cap", type=float, default=DEFAULT_CAP_S, metavar="S",
                          help=f"give up waiting after this many seconds "
                               f"(default {DEFAULT_CAP_S:.0f}); the outcome is recorded")
    cooldown.add_argument("--cooldown-poll", type=float, default=DEFAULT_POLL_INTERVAL_S,
                          metavar="S", help="seconds between temperature readings")

    parser.add_argument("--require-locked-clocks", action="store_true",
                        help="refuse to run unless someone has already locked the GPU "
                             "clocks; for a run that decides something (issue #13)")
    parser.add_argument("--require-idle-gpu", action="store_true",
                        help="refuse to run unless nothing else is drawing on the GPU; "
                             "for a run that decides something (issue #33)")
    parser.add_argument("--per-module", action="store_true",
                        help="also time UNet / VAE encode / VAE decode, in a separate "
                             "pass (the extra synchronises perturb the total)")
    detector = parser.add_argument_group("detectors (issue #4)")
    detector.add_argument("--with-diffusion", metavar="SCENARIO",
                          default=DEFAULT_DIFFUSION_SCENARIO,
                          help="keep this diffusion scenario resident while the "
                               "detector is timed (default %(default)s)")
    detector.add_argument("--no-diffusion", dest="with_diffusion", action="store_const",
                          const=None,
                          help="time the detector alone; the result then says nothing "
                               "about whether it fits beside the diffusion engine")
    detector.add_argument("--detect-cadence", type=int, default=DEFAULT_CADENCE,
                          metavar="N",
                          help="one detect every N frames, which is what the budget "
                               "verdict amortises over (default %(default)s)")
    detector.add_argument("--allow-download", action="store_true",
                          help="permit fetching detector weights and the evidence "
                               "photographs (hundreds of MB) into the shared models root")

    primitive = parser.add_argument_group("rendering primitives (issue #5)")
    primitive.add_argument("--frames", type=int, metavar="N",
                           help="consecutive clip frames to render per primitive "
                                "(default: the case's)")
    primitive.add_argument("--write-track", action="store_true",
                           help="regenerate this case's committed box track by "
                                "running the detector over the clip, and exit. The "
                                "track is committed so that two comparisons render "
                                "the same regions")
    primitive.add_argument("--no-clips", dest="write_clips", action="store_false",
                           help="skip the side-by-side clips; the Gate's manual "
                                "verification step needs them, so this is for "
                                "development only")
    primitive.set_defaults(write_clips=True)

    selective = parser.add_argument_group("selective render path (issues #8, #23, #32)")
    selective.add_argument("--detect-every-n", type=int, metavar="N",
                           help="run this selective case at one detector cadence "
                                "instead of the plan's, as one arm of the sweep. "
                                "The arm is named `<case>-nN` and is written under "
                                "bench/results/cadence/, never beside the baselines")
    selective.add_argument("--seed-policy", choices=SEED_POLICIES, metavar="POLICY",
                           help="run this selective case under one seed policy "
                                f"({', '.join(SEED_POLICIES)}) instead of the "
                                "plan's, as one arm of the temporal-stability "
                                "sweep, under bench/results/stability/")
    selective.add_argument("--output-ema", type=float, metavar="E",
                           help="run this selective case with the compositor's "
                                "output EMA at E (0.0-0.9) instead of the plan's, "
                                "as one arm of the same sweep")

    model = parser.add_argument_group("base model and step count (issue #38)")
    model.add_argument("--steps", type=int, metavar="N",
                       help="run this diffusion scenario at N denoising steps "
                            "instead of the registry's one, as one arm of the "
                            "step-count sweep. The arm is named `<scenario>-sN` "
                            "and is written under bench/results/steps/, never "
                            "beside the batch curve spec 7.2 quotes - including "
                            "at N=1, which is the sweep's own control")
    model.add_argument("--base-model", choices=sorted(BASE_MODELS), metavar="NAME",
                       help="render this selective case through another base "
                            f"model ({', '.join(sorted(BASE_MODELS))}) at the "
                            "step count it needs. The arm is named "
                            "`<case>-<NAME>` and is written under "
                            "bench/results/base-models/, never beside the baselines")
    model.add_argument("--style-lora", choices=sorted(STYLE_LORAS), metavar="NAME",
                       help="fuse this style LoRA into the arm "
                            f"({', '.join(sorted(STYLE_LORAS))}). Under "
                            "`tensorrt` that keys its own ~5 GB engine, which is "
                            "the design question issue #38 asks")

    parser.add_argument("--allow-engine-build", action="store_true",
                        help="permit compiling a TensorRT engine that is not cached "
                             "(~5.1 GB and several minutes)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help=argparse.SUPPRESS)
    return parser


def _rep_overrides(args: argparse.Namespace) -> dict:
    """The overrides that mean the same thing to a scenario and to a detector."""
    changes = {}
    if args.reps is not None:
        changes["reps"] = args.reps
    if args.warmup_reps is not None:
        changes["warmup_reps"] = args.warmup_reps
    return changes


def _selective_case(case: SelectiveCase, args: argparse.Namespace) -> SelectiveCase:
    """One selective case with its overrides applied.

    An override renames the arm after the setting it ran at, so a filename on disk
    says what it measured and two arms cannot overwrite each other's record. The
    name carries what was *asked* for; the record's `plan` block carries what the
    validator allowed, and those can differ by a clamp.
    """
    if args.frames:
        case = case.replace(frames=args.frames)
    if args.detect_every_n is not None:
        case = case.replace(name=f"{case.name}-n{args.detect_every_n}",
                            detect_every_n=args.detect_every_n)
    if args.seed_policy is not None:
        case = case.replace(name=f"{case.name}-{args.seed_policy}",
                            seed_policy=args.seed_policy)
    if args.output_ema is not None:
        case = case.replace(name=f"{case.name}-{ema_suffix(args.output_ema)}",
                            output_ema=args.output_ema)
    if args.base_model is not None:
        case = case.replace(name=f"{case.name}-{args.base_model}",
                            base_model=args.base_model)
    return case


def _swap_case(case: SwapCase, args: argparse.Namespace) -> SwapCase:
    """One swap case with its overrides applied.

    `--frames` is a development shortcut, and it moves the swap with it: a run
    shorter than `swap_frame` would submit the new instruction after the last
    frame and measure nothing at all.
    """
    if not args.frames:
        return case
    return case.replace(frames=args.frames,
                        swap_frame=min(case.swap_frame, args.frames // 2))


def resolve_target(args: argparse.Namespace) -> Tuple[str, Target]:
    """The scenario, detector or case `args.scenario` names, with overrides applied.

    One positional slot for all nine registries. A run measures one thing, the
    kinds of name cannot collide, and someone holding a name should not have to know
    which of eight flags it belongs behind.

    An unmodified name resolves to the registry's own object, so a caller can tell a
    plain run from an overridden one by identity.
    """
    if args.scenario in SCENARIOS:
        changes = _rep_overrides(args)
        if args.prompt:
            changes["prompt"] = args.prompt
        scenario = SCENARIOS[args.scenario]
        scenario = scenario.replace(**changes) if changes else scenario
        if args.style_lora:
            scenario = with_style(scenario, args.style_lora)
        return "scenario", (scenario if args.steps is None
                            else step_arm(scenario, args.steps))
    if args.scenario in DETECTORS:
        changes = _rep_overrides(args)
        detector = DETECTORS[args.scenario]
        return "detector", (detector.replace(**changes) if changes else detector)
    if args.scenario in CASES:
        case = CASES[args.scenario]
        return "primitive", (case.replace(frames=args.frames) if args.frames else case)
    if args.scenario in SELECTIVE_CASES:
        return "selective", _selective_case(SELECTIVE_CASES[args.scenario], args)
    if args.scenario in SWAP_CASES:
        return "swap", _swap_case(SWAP_CASES[args.scenario], args)
    if args.scenario in CAPTURE_CASES:
        case = CAPTURE_CASES[args.scenario]
        return "capture", (case.replace(frames=args.frames) if args.frames else case)
    if args.scenario in STYLE_CASES:
        case = STYLE_CASES[args.scenario]
        return "style", (case.replace(frames=args.frames) if args.frames else case)
    if args.scenario in QUALITY_CASES:
        case = QUALITY_CASES[args.scenario]
        return "quality", (case.replace(frames=args.frames) if args.frames else case)
    if args.scenario in GUIDANCE_CASES:
        case = GUIDANCE_CASES[args.scenario]
        return "guidance", (case.replace(frames=args.frames) if args.frames else case)
    raise SystemExit(
        f"bench: unknown scenario, detector or case {args.scenario!r}. "
        f"Run `python -m bench --list`."
    )


def scenario_results_dir(scenario: ScenarioConfig, results_dir: Path) -> Path:
    """Where a diffusion record belongs: with the batch curve, or with the arms.

    One rule in one place, the same rule `bench.selective.results_subdir` applies
    to a swept selective arm - and here it has teeth in both directions, because a
    step arm in `bench/results/` would be read as a batch cell and change the
    committed curve spec 7.2 quotes.
    """
    if is_step_arm(scenario.name):
        return results_dir / STEPS_RESULTS_SUBDIR
    return results_dir


def engine_dir_name(scenario: ScenarioConfig) -> str:
    """Where this scenario's UNet engine would be, by the app's own naming rule.

    `engine_cache` is that rule, shared with `StreamGUI` rather than mirrored here
    a second time: the guard and the window have to agree about what is built.
    """
    return engine_dir_for(
        scenario.model, use_lcm_lora=scenario.use_lcm_lora,
        use_tiny_vae=scenario.use_tiny_vae, unet_batch=scenario.unet_batch_size,
        width=scenario.width, height=scenario.height,
        lora_dict=lora_dict_for(scenario), mode=scenario.mode,
    )


def engine_build_guard(scenario: ScenarioConfig, engines_root: Optional[Path] = None,
                       allow_build: bool = False,
                       usage: Usage = shutil.disk_usage) -> Optional[DiskRecord]:
    """Two gates on compiling an engine, and the disk reading every TensorRT run carries.

    Spec 7.2: run the sweep on the `none` accelerator first and confirm only the two
    or three configurations the curve says are interesting. So an uncached engine
    needs `--allow-engine-build` - ~5.1 GB and several minutes is not a surprise
    anyone should get from a typo - and even then the volume has to have room for it.

    Reading and refusing are deliberately not the same decision. Every TensorRT run
    records its headroom, because issue #3's gate asks for the free-disk check to be
    *recorded* on each confirmation run and a reader months later cannot otherwise
    tell "the volume had room" from "nobody looked". Only a run that is about to
    compile something refuses to continue without it.

    Returns None for the `none` accelerator, which has no engine and so nothing to
    say about the engines volume.
    """
    if scenario.acceleration != "tensorrt":
        return None
    root = resolve_engines_dir() if engines_root is None else Path(engines_root)
    record = read_disk(root, usage=usage)
    if (root / engine_dir_name(scenario) / UNET_ENGINE).is_file():
        return record
    if not allow_build:
        raise SystemExit(
            f"bench: no cached TensorRT engine for {scenario.name} under {root}.\n"
            f"       Building one costs ~5.1 GB and several minutes. Pass "
            f"--allow-engine-build if that is what you want."
        )
    require_free_space(record)
    return record


LOCK_INSTRUCTIONS = (
    "       Lock them from an *elevated* shell, then re-run:\n"
    "           nvidia-smi --lock-gpu-clocks=<min>,<max>\n"
    "           ... run the sweep ...\n"
    "           nvidia-smi --reset-gpu-clocks\n"
    "       This harness never locks them itself: it is not elevated, and a failed\n"
    "       attempt must not be mistaken for a lock."
)


def clock_lock_guard(require_locked: bool,
                     read: Callable[[], ClockLock] = read_clock_lock) -> ClockLock:
    """The clock regime this run will be measured under; refuse it if it must be locked.

    Issue #13 step 4. `unknown` fails the same way `unlocked` does - the harness
    cannot see a lock, so for a run that decides something it must assume there is
    none. The regime that lands in the *result* is read again by the runner at run
    time; this reading is the gate, and it happens before an engine build so a
    refusal costs nothing.
    """
    lock = read()
    if require_locked and not lock.locked:
        raise SystemExit(
            f"bench: --require-locked-clocks, but the GPU clocks are {lock.state}.\n"
            f"       {regime_summary(lock)} | {lock.evidence}\n"
            f"{LOCK_INSTRUCTIONS}"
        )
    return lock


IDLE_INSTRUCTIONS = (
    "       Close whatever else is drawing on the card and re-run. The harness\n"
    "       cannot tell you which process it is - `nvidia-smi` reports per-process\n"
    "       memory but not per-process SM time on consumer cards - so this is the\n"
    "       one door that needs a human to look at the machine."
)


def idle_gpu_guard(require_idle: bool,
                   measure: Callable[[], OccupancyRecord] = measure_occupancy_now,
                   ) -> Optional[OccupancyRecord]:
    """Whether anything else is using the GPU; refuse the run if something is.

    Issue #33, and the same shape as `clock_lock_guard`: it happens before an
    engine build so a refusal costs nothing, and `unknown` refuses for the reason
    `unknown` refuses there - a gate that cannot see the card has not seen an empty
    one. Unlike the clock gate it *samples*, which costs a couple of seconds, so it
    only samples when it was asked to; the record that lands in the result file is
    taken again by the runner.
    """
    if not require_idle:
        return None
    record = measure()
    if not record.clear:
        raise SystemExit(
            f"bench: --require-idle-gpu, but the GPU is {record.outcome}.\n"
            f"       {occupancy_summary(record)}\n"
            f"{IDLE_INSTRUCTIONS}"
        )
    return record


def report_marginal(results_dir: Path, out: TextIO = sys.stdout) -> None:
    """The batch curves in `results_dir`, raw and normalised to one SM clock.

    Both tables, because both are needed to read the result honestly: the raw one is
    what the machine did, and the normalised one is what it would have done had the
    120 W limit not lowered the clock under a longer call. Spec 7.4 wants the curve
    *shape* to be the portable conclusion, and at 512 the raw shape is partly the
    power limit.
    """
    curves = marginal.curves_from_results(marginal.load_results(results_dir))
    if not curves:
        out.write(f"no results under {results_dir}\n")
        return
    reference = marginal.reference_clock_mhz(curves)
    normalised = [curve.normalised_to(reference) for curve in curves]

    sections = [
        "As measured",
        marginal.format_table(curves),
        marginal.format_verdicts(curves),
        f"Normalised to {reference:.0f} MHz (estimate: ms x clock / reference)",
        marginal.format_table(normalised),
        marginal.format_verdicts(normalised),
    ]
    out.write("\n\n".join(sections) + "\n")


def report_detectors(results_dir: Path, out: TextIO = sys.stdout) -> None:
    """The measured detector block spec 8.1 carries, from the committed results."""
    out.write(format_detector_report(load_detector_results(results_dir)) + "\n")


def report_primitives(results_dir: Path, out: TextIO = sys.stdout) -> None:
    """The decision block spec 8.2 carries, from the committed comparisons."""
    out.write(format_primitive_report(load_primitive_results(results_dir)) + "\n")


def report_selective(results_dir: Path, out: TextIO = sys.stdout) -> None:
    """The selective-path block spec 8.8 carries, from the committed runs."""
    out.write(format_selective_report(load_selective_results(results_dir)) + "\n")


def report_portability(results_dir: Path, out: TextIO = sys.stdout) -> None:
    """The dev-vs-deploy block spec 7.4 carries (issue #24).

    Reads the same selective records `--selective-report` does, and asks a different
    question of them: not "does the path work" but "which of its numbers survived
    the move to the hardware this ships on".
    """
    out.write(format_portability_report(load_selective_results(results_dir)) + "\n")


def report_swap(results_dir: Path, out: TextIO = sys.stdout) -> None:
    """The plan-swap block spec 8.9 carries (issue #30).

    Acceptance criteria 1 and 3, from the committed swaps: how long a typed
    instruction takes to reach the screen, and what swapping it cost the output
    stream and the engine.
    """
    out.write(format_swap_report(load_swap_results(results_dir)) + "\n")


def report_capture(results_dir: Path, out: TextIO = sys.stdout) -> None:
    """The capture-geometry block spec 8.2 carries (issue #39).

    Crop against masked at K=1 at every capture geometry, the cost of the larger
    capture broken out per stage, and the 30 FPS verdict per arm.
    """
    out.write(format_capture_report(load_capture_results(results_dir)) + "\n")


def report_cadence(results_dir: Path, out: TextIO = sys.stdout,
                   baseline_dir: Path = SELECTIVE_RESULTS_DIR) -> None:
    """The `detect_every_n` sweep block spec 8.8 carries (issue #23).

    Two directories, because the block answers two questions with one set of
    arithmetic: `baseline_dir` holds issue #24's unmodified runs, which say whether
    there was a gap to close at all, and `results_dir` holds the arms that say what
    the one quality-free lever buys.
    """
    out.write(format_cadence_report(load_selective_results(results_dir),
                                    baseline=load_selective_results(baseline_dir))
              + "\n")


def report_steps(results_dir: Path, out: TextIO = sys.stdout) -> None:
    """The step-count block spec 7.2 carries (issue #38).

    What one more denoising step costs, per module, on the model already on disk -
    the measurement that sizes a four-step base model before anything is compiled
    for one.
    """
    out.write(format_steps_report(load_records(results_dir)) + "\n")


def report_models(results_dir: Path, out: TextIO = sys.stdout) -> None:
    """The base-model block spec 7.5 carries (issue #38, step 2).

    The same selective records the other reports read, from the directory the
    base-model arms land in - never the baselines', because an arm rendered
    through another checkpoint at another step count must not become the row
    spec 8.8 and 7.4 quote for the shipped path.
    """
    out.write(format_model_report(load_selective_results(results_dir)) + "\n")


def report_styles(results_dir: Path, out: TextIO = sys.stdout,
                  step_dir: Path = STEPS_RESULTS_DIR) -> None:
    """The style-LoRA block spec 8.10 carries (issue #38).

    Two directories, like the cadence and stability reports: the arms say whether
    a style LoRA loads and does anything, and the step arms beside them say what
    that style costs with a TensorRT engine and without one - which is the
    delivery question step 4 asks and the arms themselves cannot answer.
    """
    out.write(format_style_report(load_style_results(results_dir),
                                  step_records=load_records(step_dir)) + "\n")


def report_guidance(results_dir: Path, out: TextIO = sys.stdout) -> None:
    """The classifier-free-guidance block spec 8.11 carries (issue #45).

    Whether the weak prompt adherence is the checkpoint or a setting that has
    never been on, from the committed sweep: an adherence number per arm, the
    drift it costs, and which arms are a ~5 GB build rather than a setting.
    """
    out.write(format_guidance_report(load_guidance_results(results_dir)) + "\n")


def report_quality(results_dir: Path, out: TextIO = sys.stdout) -> None:
    """The step-count block spec 8.12 carries (issue #46).

    What a second, fourth and eighth denoising step buys, what switching to an
    already-built engine costs, and which of the two ways of paying for a runtime
    step count the committed arms recommend.
    """
    out.write(format_quality_report(load_quality_results(results_dir)) + "\n")


def report_stability(results_dir: Path, out: TextIO = sys.stdout,
                     baseline_dir: Path = SELECTIVE_RESULTS_DIR) -> None:
    """The temporal-stability block spec 8.5 carries (issue #32).

    Two directories, like the cadence report and for the same reason: `results_dir`
    holds the swept arms and `baseline_dir` holds the shipped path's own runs, and
    the block is about the distance between them.
    """
    out.write(format_stability_report(load_selective_results(results_dir),
                                      baseline=load_selective_results(baseline_dir))
              + "\n")


def list_targets(out: TextIO = sys.stdout) -> None:
    """Every registry, one name per line - whatever the positional slot accepts."""
    for name, scenario in SCENARIOS.items():
        out.write(f"{name}\t{scenario.acceleration}\t{scenario.width}x{scenario.height}"
                  f"\tbatch {scenario.batch_size}\t{scenario.steps} step(s)\n")
    for name, config in DETECTORS.items():
        vocabulary = "open vocabulary" if config.open_vocabulary else "80 COCO classes"
        out.write(f"{name}\tdetector\t{config.imgsz}x{config.imgsz}\t{vocabulary}"
                  f"\t{config.role}\n")
    for name, case in CASES.items():
        priority = "priority case" if case.priority else "eventual case"
        out.write(f"{name}\tprimitive case\t{case.clip}\t{case.region}"
                  f"\t{case.frames} frames\t{priority}\n")
    for name, case in SELECTIVE_CASES.items():
        out.write(f"{name}\tselective case\t{case.clip}"
                  f"\t{case.canvas}x{case.canvas} canvas\t{case.frames} frames"
                  f"\tthe shipped path end to end\n")
    for name, case in SWAP_CASES.items():
        out.write(f"{name}\tswap case\t{case.clip}\t{case.before.target} -> "
                  f"{case.after.target}\t{case.frames} frames, swap on "
                  f"{case.swap_frame}\tacceptance criteria 1 and 3\n")
    for name, case in STYLE_CASES.items():
        out.write(f"{name}\tstyle case\t{case.clip}\t{case.base_scenario}"
                  f"\t{len(case.styles)} LoRAs at {case.steps} steps"
                  f"\tdo style LoRAs work on SD 1.5\n")
    for name, case in CAPTURE_CASES.items():
        geometries = ", ".join(f"{w}x{h}" for w, h in case.geometries)
        out.write(f"{name}\tcapture case\t{case.clip}\t{geometries}"
                  f"\t{'/'.join(case.primitives)} at K={case.max_instances}"
                  f"\tcrop against masked, per stage\n")
    for name, case in QUALITY_CASES.items():
        out.write(f"{name}\tstep-quality case\t{case.clip}\t{case.base_scenario}"
                  f"\t{len(case.specs())} step arms over both routes"
                  f"\twhat a denoising step buys and how it is paid for\n")
    for name, case in GUIDANCE_CASES.items():
        out.write(f"{name}\tguidance case\t{case.clip}\t{case.base_scenario}"
                  f"\t{len(case.specs())} cfg arms at {case.steps} step(s)"
                  f"\tdoes turning CFG on make the prompt land\n")


def run_detector_target(args: argparse.Namespace, config: DetectorConfig) -> int:
    """Measure one detector (issue #4). Imported late, like the diffusion runner.

    The diffusion scenario the detector is measured beside goes through the same
    engine-build guard a diffusion run does: `--with-diffusion` names a TensorRT
    configuration, and one that is not cached is still ~5.1 GB and several minutes.
    """
    if args.with_diffusion:
        if args.with_diffusion not in SCENARIOS:
            raise SystemExit(
                f"bench: --with-diffusion names no scenario: {args.with_diffusion!r}. "
                f"Run `python -m bench --list`."
            )
        engine_build_guard(SCENARIOS[args.with_diffusion],
                           allow_build=args.allow_engine_build)

    from bench.detector_runner import run_detector  # imports torch, like run_scenario

    run_detector(
        config,
        diffusion_scenario=args.with_diffusion,
        cadence=args.detect_cadence,
        cooldown=args.cooldown,
        results_dir=args.results_dir / DETECTOR_RESULTS_SUBDIR,
        threshold_c=args.cooldown_threshold,
        cap_s=args.cooldown_cap,
        poll_interval_s=args.cooldown_poll,
        allow_download=args.allow_download,
    )
    return 0


def run_primitive_target(args: argparse.Namespace, case: CaseConfig) -> int:
    """Compare both rendering primitives on one case (issue #5). Imported late.

    The comparison renders through the one cached TensorRT engine both primitives
    share, so it goes through the same engine-build guard a diffusion run does.
    `--write-track` is the other half of the same target: it regenerates the
    committed boxes, which is a detector run rather than a diffusion one, and it
    exits before the engine guard because it needs no engine.
    """
    # Late, like `run_scenario`: torch and cv2 live inside this module's functions,
    # but importing it pulls in the whole measuring half, which --help does not need.
    from bench.primitive_runner import ENGINE_SCENARIO, build_track, run_case

    if args.write_track:
        build_track(case)
        return 0

    engine_build_guard(SCENARIOS[ENGINE_SCENARIO], allow_build=args.allow_engine_build)

    run_case(
        case,
        cooldown=args.cooldown,
        results_dir=args.results_dir / PRIMITIVE_RESULTS_SUBDIR,
        threshold_c=args.cooldown_threshold,
        cap_s=args.cooldown_cap,
        poll_interval_s=args.cooldown_poll,
        write_clips=args.write_clips,
    )
    return 0


def run_selective_target(args: argparse.Namespace, case: SelectiveCase) -> int:
    """Drive the shipped selective path over one clip (issue #8). Imported late.

    Rendered through the same cached engine both primitives were compared on, so it
    passes the same engine-build guard a diffusion run does.

    Where the record lands, and under which heading, is `bench.selective`'s
    decision rather than this function's: an arm of the cadence sweep (issue #23)
    must not sit beside the baselines that spec 8.8 and 7.4 quote.
    """
    from bench.selective import engine_scenario_for
    from bench.selective_runner import run_selective

    engine_build_guard(engine_scenario_for(case),
                       allow_build=args.allow_engine_build)
    run_selective(
        case,
        cooldown=args.cooldown,
        results_dir=args.results_dir / results_subdir(case),
        readme_preamble=readme_preamble(case),
        threshold_c=args.cooldown_threshold,
        cap_s=args.cooldown_cap,
        poll_interval_s=args.cooldown_poll,
        write_clips=args.write_clips,
    )
    return 0


def run_swap_target(args: argparse.Namespace, case: SwapCase) -> int:
    """Swap one instruction for another mid-clip (issue #30). Imported late.

    Rendered through the same cached engine every other case is, so it passes the
    same engine-build guard - and the record lands in `bench/results/swaps/`,
    because the selective reports read every JSON beside them as a selective run.
    """
    from bench.plan_swap_runner import run_swap
    from bench.selective import ENGINE_SCENARIO

    engine_build_guard(SCENARIOS[ENGINE_SCENARIO], allow_build=args.allow_engine_build)
    run_swap(
        case,
        cooldown=args.cooldown,
        results_dir=args.results_dir / SWAP_RESULTS_SUBDIR,
        threshold_c=args.cooldown_threshold,
        cap_s=args.cooldown_cap,
        poll_interval_s=args.cooldown_poll,
        write_clips=args.write_clips,
    )
    return 0


def run_capture_target(args: argparse.Namespace, case: CaptureCase) -> int:
    """Render one case at every capture geometry under both primitives (issue #39).

    Rendered through the same cached 512x512 engine every other case is - the
    canvas does not move, which is the whole premise - so it passes the same
    engine-build guard, and the record lands in `bench/results/capture/`.
    """
    from bench.capture import ENGINE_SCENARIO
    from bench.capture_runner import run_capture

    engine_build_guard(SCENARIOS[ENGINE_SCENARIO], allow_build=args.allow_engine_build)
    run_capture(
        case,
        cooldown=args.cooldown,
        results_dir=args.results_dir / CAPTURE_RESULTS_SUBDIR,
        threshold_c=args.cooldown_threshold,
        cap_s=args.cooldown_cap,
        poll_interval_s=args.cooldown_poll,
        write_clips=args.write_clips,
    )
    return 0


def run_style_target(args: argparse.Namespace, case: StyleCase) -> int:
    """Try every style LoRA on the SD 1.5 arm (issue #38, steps 3-5). Imported late.

    The case's own base scenario decides whether an engine is needed at all, and
    every arm is guarded separately because a fused LoRA keys its own: the
    committed case runs on `none`, which is the hot-swappable path and the one a
    LoRA comparison belongs on, because under `tensorrt` each style would be its
    own ~5 GB build before it could be looked at once.
    """
    from bench.style_runner import arm_scenario, run_style

    for style in (None, *case.styles):
        engine_build_guard(arm_scenario(case, style),
                           allow_build=args.allow_engine_build)
    run_style(
        case,
        cooldown=args.cooldown,
        results_dir=args.results_dir / STYLE_RESULTS_SUBDIR,
        threshold_c=args.cooldown_threshold,
        cap_s=args.cooldown_cap,
        poll_interval_s=args.cooldown_poll,
        write_clips=args.write_clips,
    )
    return 0


def run_guidance_target(args: argparse.Namespace, case: GuidanceCase) -> int:
    """Sweep classifier-free guidance over one case (issue #45). Imported late.

    No engine guard: the arms run on the `none` accelerator on purpose. Two of the
    four cfg types key a different UNet batch, so sweeping them under `tensorrt`
    would be several ~5 GB builds spent to find out whether the axis does anything
    at all - and what the record carries instead is `engine_keying`, which names
    the engine each arm *would* need and whether this machine already has it.
    """
    from bench.guidance_runner import run_guidance

    run_guidance(
        case,
        cooldown=args.cooldown,
        results_dir=args.results_dir / GUIDANCE_RESULTS_SUBDIR,
        threshold_c=args.cooldown_threshold,
        cap_s=args.cooldown_cap,
        poll_interval_s=args.cooldown_poll,
        write_clips=args.write_clips,
    )
    return 0


def run_quality_target(args: argparse.Namespace, case: QualityCase) -> int:
    """Sweep the step count over both routes (issue #46). Imported late.

    Every arm goes through the engine-build guard, because on this case they are
    TensorRT arms and half of them key a batch the machine may never have compiled
    - which is exactly the cost the recommendation weighs. The guard is asked once
    per arm rather than once for the case: a run that would compile three engines
    should say so three times before it starts, not after the first one.
    """
    from bench.quality_runner import arm_scenario, run_quality

    from render_plan import t_index_for_denoise, t_index_ladder

    opening = t_index_for_denoise(case.denoise)
    for spec in case.specs():
        engine_build_guard(arm_scenario(case, spec, t_index_ladder(opening,
                                                                  spec.steps)),
                           allow_build=args.allow_engine_build)

    run_quality(
        case,
        cooldown=args.cooldown,
        results_dir=args.results_dir / QUALITY_RESULTS_SUBDIR,
        threshold_c=args.cooldown_threshold,
        cap_s=args.cooldown_cap,
        poll_interval_s=args.cooldown_poll,
        write_clips=args.write_clips,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        list_targets()
        return 0
    if args.marginal:
        report_marginal(args.results_dir)
        return 0
    if args.detector_report:
        report_detectors(args.results_dir / DETECTOR_RESULTS_SUBDIR)
        return 0
    if args.primitive_report:
        report_primitives(args.results_dir / PRIMITIVE_RESULTS_SUBDIR)
        return 0
    if args.selective_report:
        report_selective(args.results_dir / SELECTIVE_RESULTS_SUBDIR)
        return 0
    if args.portability_report:
        report_portability(args.results_dir / SELECTIVE_RESULTS_SUBDIR)
        return 0
    if args.swap_report:
        report_swap(args.results_dir / SWAP_RESULTS_SUBDIR)
        return 0
    if args.capture_report:
        report_capture(args.results_dir / CAPTURE_RESULTS_SUBDIR)
        return 0
    if args.cadence_report:
        report_cadence(args.results_dir / CADENCE_RESULTS_SUBDIR,
                       baseline_dir=args.results_dir / SELECTIVE_RESULTS_SUBDIR)
        return 0
    if args.steps_report:
        report_steps(args.results_dir / STEPS_RESULTS_SUBDIR)
        return 0
    if args.model_report:
        report_models(args.results_dir / MODEL_RESULTS_SUBDIR)
        return 0
    if args.style_report:
        report_styles(args.results_dir / STYLE_RESULTS_SUBDIR,
                      step_dir=args.results_dir / STEPS_RESULTS_SUBDIR)
        return 0
    if args.guidance_report:
        report_guidance(args.results_dir / GUIDANCE_RESULTS_SUBDIR)
        return 0
    if args.quality_report:
        report_quality(args.results_dir / QUALITY_RESULTS_SUBDIR)
        return 0
    if args.stability_report:
        report_stability(args.results_dir / STABILITY_RESULTS_SUBDIR,
                         baseline_dir=args.results_dir / SELECTIVE_RESULTS_SUBDIR)
        return 0
    if not args.scenario:
        parser.print_usage()
        print("bench: name a scenario, a detector or a case, or pass --list",
              file=sys.stderr)
        return 2

    kind, target = resolve_target(args)
    clock_lock_guard(args.require_locked_clocks)
    idle_gpu_guard(args.require_idle_gpu)
    if kind == "detector":
        return run_detector_target(args, target)
    if kind == "primitive":
        return run_primitive_target(args, target)
    if kind == "selective":
        return run_selective_target(args, target)
    if kind == "swap":
        return run_swap_target(args, target)
    if kind == "capture":
        return run_capture_target(args, target)
    if kind == "style":
        return run_style_target(args, target)
    if kind == "guidance":
        return run_guidance_target(args, target)
    if kind == "quality":
        return run_quality_target(args, target)

    scenario = target
    disk = engine_build_guard(scenario, allow_build=args.allow_engine_build)

    from bench.runner import run_scenario  # imports torch - kept off the --help path

    run_scenario(
        scenario,
        cooldown=args.cooldown,
        per_module=args.per_module,
        results_dir=scenario_results_dir(scenario, args.results_dir),
        threshold_c=args.cooldown_threshold,
        cap_s=args.cooldown_cap,
        poll_interval_s=args.cooldown_poll,
        disk=disk,
    )
    return 0
