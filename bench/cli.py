"""`uv run python -m bench <scenario>`.

Parsing, listing and the engine-build guard are all GPU-free; `bench.runner` is
imported only once a run is actually about to happen, so `--help` and `--list` work
on a machine with no CUDA device and the merge gate can exercise them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from bench.cooldown import DEFAULT_CAP_S, DEFAULT_POLL_INTERVAL_S, DEFAULT_THRESHOLD_C
from bench.paths import RESULTS_DIR, resolve_engines_dir
from bench.scenarios import SCENARIOS, ScenarioConfig

# The prefix wrapper.py builds an engine directory name from. Mirrored here so the
# guard can answer "is this configuration already built?" without loading torch.
ENGINE_PREFIX = ("{model}--lcm_lora-{lcm}--tiny_vae-{tiny}--max_batch-{batch}"
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
    try:
        scenario = SCENARIOS[args.scenario]
    except KeyError:
        raise SystemExit(
            f"bench: unknown scenario {args.scenario!r}. Run `python -m bench --list`."
        )
    changes = {}
    if args.reps is not None:
        changes["reps"] = args.reps
    if args.warmup_reps is not None:
        changes["warmup_reps"] = args.warmup_reps
    if getattr(args, "prompt", None):
        changes["prompt"] = args.prompt
    return scenario.replace(**changes) if changes else scenario


def engine_dir_name(scenario: ScenarioConfig) -> str:
    return ENGINE_PREFIX.format(
        model=scenario.model, lcm=scenario.use_lcm_lora, tiny=scenario.use_tiny_vae,
        batch=scenario.unet_batch_size, width=scenario.width, height=scenario.height,
        mode=scenario.mode,
    )


def engine_build_guard(scenario: ScenarioConfig, engines_root: Optional[Path] = None,
                       allow_build: bool = False) -> None:
    """Refuse to silently spend ~5.1 GB and several minutes compiling an engine.

    Spec 7.2: run the sweep on the `none` accelerator first and confirm only the two
    or three configurations the curve says are interesting. An accidental TensorRT
    run across the registry would fill the disk.
    """
    if scenario.acceleration != "tensorrt" or allow_build:
        return
    root = resolve_engines_dir() if engines_root is None else Path(engines_root)
    if (root / engine_dir_name(scenario) / "unet.engine").is_file():
        return
    raise SystemExit(
        f"bench: no cached TensorRT engine for {scenario.name} under {root}.\n"
        f"       Building one costs ~5.1 GB and several minutes. Pass "
        f"--allow-engine-build if that is what you want."
    )


def list_scenarios(out=sys.stdout) -> None:
    for name, scenario in SCENARIOS.items():
        out.write(f"{name}\t{scenario.acceleration}\t{scenario.width}x{scenario.height}"
                  f"\tbatch {scenario.batch_size}\t{scenario.steps} step(s)\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        list_scenarios()
        return 0
    if not args.scenario:
        parser.print_usage()
        print("bench: name a scenario, or pass --list", file=sys.stderr)
        return 2

    scenario = resolve_scenario(args)
    engine_build_guard(scenario, allow_build=args.allow_engine_build)

    from bench.runner import run_scenario  # imports torch - kept off the --help path

    run_scenario(
        scenario,
        cooldown=args.cooldown,
        per_module=args.per_module,
        results_dir=args.results_dir,
        threshold_c=args.cooldown_threshold,
        cap_s=args.cooldown_cap,
        poll_interval_s=args.cooldown_poll,
    )
    return 0
