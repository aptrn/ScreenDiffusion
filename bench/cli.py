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
from bench.clocks import ClockLock, regime_summary
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
from bench.paths import (
    CADENCE_RESULTS_SUBDIR,
    DETECTOR_RESULTS_SUBDIR,
    PRIMITIVE_RESULTS_SUBDIR,
    RESULTS_DIR,
    SELECTIVE_RESULTS_DIR,
    SELECTIVE_RESULTS_SUBDIR,
    resolve_engines_dir,
)
from bench.portability import format_portability_report
from bench.primitive_results import format_primitive_report, load_primitive_results
from bench.primitives import CASES, CaseConfig
from bench.scenarios import SCENARIOS, ScenarioConfig
from bench.selective import (
    CADENCE_README_PREAMBLE,
    CASES as SELECTIVE_CASES,
    SELECTIVE_README_PREAMBLE,
    SelectiveCase,
    format_selective_report,
    load_selective_results,
    results_subdir,
)

# The directory name `create_prefix()` in wrapper.py builds for a UNet engine, with
# `--lora-none` because no scenario fuses a LoRA. Mirrored here so the guard can
# answer "is this configuration already built?" without loading torch.
ENGINE_DIR_TEMPLATE = ("{model}--lcm_lora-{lcm}--tiny_vae-{tiny}--max_batch-{batch}"
                       "--min_batch-{batch}--res-{width}x{height}--lora-none--mode-{mode}")

# What the one positional slot can name: a diffusion scenario, a detector, a
# rendering-primitive case (issue #5) or an end-to-end selective render (issue #8).
# One slot for all four - a run measures one thing, the names cannot collide, and
# someone holding a name should not have to know which flag it belongs behind.
Target = Union[ScenarioConfig, DetectorConfig, CaseConfig, SelectiveCase]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Measure one ScreenDiffusion configuration and write a tracked, "
                    "machine-readable result under bench/results/.",
        epilog="Results are written by a run and never by hand: if a run did not "
               "happen, there is no record.",
    )
    parser.add_argument("scenario", nargs="?",
                        help="scenario, detector, primitive-case or selective-case "
                             "name; --list shows them all")
    parser.add_argument("--list", action="store_true",
                        help="list the scenarios, detectors, primitive cases and "
                             "selective cases, and exit")
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
    parser.add_argument("--cadence-report", action="store_true",
                        help="report the detect_every_n sweep spec 8.8 carries, from "
                             "the committed arms, and exit")
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

    selective = parser.add_argument_group("selective render path (issues #8, #23)")
    selective.add_argument("--detect-every-n", type=int, metavar="N",
                           help="run this selective case at one detector cadence "
                                "instead of the plan's, as one arm of the sweep. "
                                "The arm is named `<case>-nN` and is written under "
                                "bench/results/cadence/, never beside the baselines")

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

    A cadence override renames the arm after the cadence it ran at, so a filename
    on disk says what it measured and two arms cannot overwrite each other's
    record. The name carries what was *asked* for; `plan.detect_every_n` in the
    record carries what the validator allowed, and those can differ by a clamp.
    """
    if args.frames:
        case = case.replace(frames=args.frames)
    if args.detect_every_n is not None:
        case = case.replace(name=f"{case.name}-n{args.detect_every_n}",
                            detect_every_n=args.detect_every_n)
    return case


def resolve_target(args: argparse.Namespace) -> Tuple[str, Target]:
    """The scenario, detector or case `args.scenario` names, with overrides applied.

    One positional slot for all three registries. A run measures one thing, the
    kinds of name cannot collide, and someone holding a name should not have to know
    which of three flags it belongs behind.

    An unmodified name resolves to the registry's own object, so a caller can tell a
    plain run from an overridden one by identity.
    """
    if args.scenario in SCENARIOS:
        changes = _rep_overrides(args)
        if args.prompt:
            changes["prompt"] = args.prompt
        scenario = SCENARIOS[args.scenario]
        return "scenario", (scenario.replace(**changes) if changes else scenario)
    if args.scenario in DETECTORS:
        changes = _rep_overrides(args)
        detector = DETECTORS[args.scenario]
        return "detector", (detector.replace(**changes) if changes else detector)
    if args.scenario in CASES:
        case = CASES[args.scenario]
        return "primitive", (case.replace(frames=args.frames) if args.frames else case)
    if args.scenario in SELECTIVE_CASES:
        return "selective", _selective_case(SELECTIVE_CASES[args.scenario], args)
    raise SystemExit(
        f"bench: unknown scenario, detector or case {args.scenario!r}. "
        f"Run `python -m bench --list`."
    )


def engine_dir_name(scenario: ScenarioConfig) -> str:
    return ENGINE_DIR_TEMPLATE.format(
        model=scenario.model, lcm=scenario.use_lcm_lora, tiny=scenario.use_tiny_vae,
        batch=scenario.unet_batch_size, width=scenario.width, height=scenario.height,
        mode=scenario.mode,
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
    if (root / engine_dir_name(scenario) / "unet.engine").is_file():
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

    Where the record lands is `results_subdir`'s decision, not this function's: an
    arm of the cadence sweep (issue #23) must not sit beside the baselines that
    spec 8.8 and 7.4 quote.
    """
    from bench.selective import ENGINE_SCENARIO
    from bench.selective_runner import run_selective

    engine_build_guard(SCENARIOS[ENGINE_SCENARIO], allow_build=args.allow_engine_build)
    run_selective(
        case,
        cooldown=args.cooldown,
        results_dir=args.results_dir / results_subdir(case),
        readme_preamble=(SELECTIVE_README_PREAMBLE if case.detect_every_n is None
                         else CADENCE_README_PREAMBLE),
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
    if args.cadence_report:
        report_cadence(args.results_dir / CADENCE_RESULTS_SUBDIR,
                       baseline_dir=args.results_dir / SELECTIVE_RESULTS_SUBDIR)
        return 0
    if not args.scenario:
        parser.print_usage()
        print("bench: name a scenario, a detector or a case, or pass --list",
              file=sys.stderr)
        return 2

    kind, target = resolve_target(args)
    clock_lock_guard(args.require_locked_clocks)
    if kind == "detector":
        return run_detector_target(args, target)
    if kind == "primitive":
        return run_primitive_target(args, target)
    if kind == "selective":
        return run_selective_target(args, target)

    scenario = target
    disk = engine_build_guard(scenario, allow_build=args.allow_engine_build)

    from bench.runner import run_scenario  # imports torch - kept off the --help path

    run_scenario(
        scenario,
        cooldown=args.cooldown,
        per_module=args.per_module,
        results_dir=args.results_dir,
        threshold_c=args.cooldown_threshold,
        cap_s=args.cooldown_cap,
        poll_interval_s=args.cooldown_poll,
        disk=disk,
    )
    return 0
