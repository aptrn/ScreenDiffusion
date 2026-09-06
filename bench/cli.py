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
from typing import Optional, Sequence, TextIO

from bench import marginal
from bench.cooldown import DEFAULT_CAP_S, DEFAULT_POLL_INTERVAL_S, DEFAULT_THRESHOLD_C
from bench.disk import DiskRecord, Usage, read_disk, require_free_space
from bench.paths import RESULTS_DIR, resolve_engines_dir
from bench.scenarios import SCENARIOS, ScenarioConfig

# The directory name `create_prefix()` in wrapper.py builds for a UNet engine, with
# `--lora-none` because no scenario fuses a LoRA. Mirrored here so the guard can
# answer "is this configuration already built?" without loading torch.
ENGINE_DIR_TEMPLATE = ("{model}--lcm_lora-{lcm}--tiny_vae-{tiny}--max_batch-{batch}"
                       "--min_batch-{batch}--res-{width}x{height}--lora-none--mode-{mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Measure one ScreenDiffusion configuration and write a tracked, "
                    "machine-readable result under bench/results/.",
        epilog="Results are written by a run and never by hand: if a run did not "
               "happen, there is no record.",
    )
    parser.add_argument("scenario", nargs="?", help="scenario name; --list shows them all")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    parser.add_argument("--marginal", action="store_true",
                        help="report the marginal cost per additional batch item from "
                             "the committed results, and exit")
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

    parser.add_argument("--per-module", action="store_true",
                        help="also time UNet / VAE encode / VAE decode, in a separate "
                             "pass (the extra synchronises perturb the total)")
    parser.add_argument("--allow-engine-build", action="store_true",
                        help="permit compiling a TensorRT engine that is not cached "
                             "(~5.1 GB and several minutes)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help=argparse.SUPPRESS)
    return parser


def resolve_scenario(args: argparse.Namespace) -> ScenarioConfig:
    """The named scenario with the command line's overrides applied."""
    if args.scenario not in SCENARIOS:
        raise SystemExit(
            f"bench: unknown scenario {args.scenario!r}. Run `python -m bench --list`."
        )
    scenario = SCENARIOS[args.scenario]
    changes = {}
    if args.reps is not None:
        changes["reps"] = args.reps
    if args.warmup_reps is not None:
        changes["warmup_reps"] = args.warmup_reps
    if args.prompt:
        changes["prompt"] = args.prompt
    return scenario.replace(**changes) if changes else scenario


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


def list_scenarios(out: TextIO = sys.stdout) -> None:
    for name, scenario in SCENARIOS.items():
        out.write(f"{name}\t{scenario.acceleration}\t{scenario.width}x{scenario.height}"
                  f"\tbatch {scenario.batch_size}\t{scenario.steps} step(s)\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        list_scenarios()
        return 0
    if args.marginal:
        report_marginal(args.results_dir)
        return 0
    if not args.scenario:
        parser.print_usage()
        print("bench: name a scenario, or pass --list", file=sys.stderr)
        return 2

    scenario = resolve_scenario(args)
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
