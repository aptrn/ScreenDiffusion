"""The cooldown gate: measure a cool GPU, or say plainly that you could not.

A GPU benchmarked while hot measures the thermal throttle instead of the change
under test. The convention this project inherited is to wait for the die to fall
below ~62 degC before each timed rep - but a thermally-limited laptop under
sustained load may never get there, so the wait is capped and its outcome is
recorded either way.

The decision is a pure function of one temperature reading and the elapsed wait
(`cooldown_verdict`); the polling loop around it takes its clock, its sleep and
its thermometer as arguments. Both are therefore testable without a GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

# ~62 degC: below this the die is not throttling on this class of hardware.
DEFAULT_THRESHOLD_C = 62.0
# Two minutes. Past that a laptop is very unlikely to reach the threshold at all.
DEFAULT_CAP_S = 120.0
DEFAULT_POLL_INTERVAL_S = 2.0

# Verdicts. REACHED / CAPPED / SKIPPED are terminal and land in the result file;
# WAITING only ever exists inside the loop.
REACHED = "reached"
CAPPED = "capped"
SKIPPED = "skipped"
WAITING = "waiting"


def cooldown_verdict(
    temperature_c: Optional[float],
    elapsed_s: float,
    threshold_c: float = DEFAULT_THRESHOLD_C,
    cap_s: float = DEFAULT_CAP_S,
) -> str:
    """Whether to start the run now, give up waiting, or keep polling.

    Temperature is checked before the clock: a cold GPU is what the gate is for,
    so reaching the threshold late still counts as reaching it. An unreadable
    temperature (`None`) can never be `REACHED` - it falls through to the cap,
    and the run is then recorded as one that never proved it was cool.
    """
    if temperature_c is not None and temperature_c <= threshold_c:
        return REACHED
    if elapsed_s >= cap_s:
        return CAPPED
    return WAITING


@dataclass(frozen=True)
class CooldownRecord:
    """What the gate did, as it lands in the result file.

    `enabled` is False only for `--no-cooldown`; the outcome is never absent, so a
    comparison can always tell a cool run from a throttled one from a skipped one.
    """

    enabled: bool
    outcome: str
    threshold_c: float = DEFAULT_THRESHOLD_C
    cap_s: float = DEFAULT_CAP_S
    waited_s: float = 0.0
    final_temperature_c: Optional[float] = None
    # [[elapsed_s, temperature_c], ...] - the whole descent, so a run that only
    # just made it under the threshold is visible rather than implied.
    samples: List[List[Optional[float]]] = field(default_factory=list)

    @property
    def reached(self) -> bool:
        return self.outcome == REACHED

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "outcome": self.outcome,
            "reached": self.reached,
            "threshold_c": self.threshold_c,
            "cap_s": self.cap_s,
            "waited_s": self.waited_s,
            "final_temperature_c": self.final_temperature_c,
            "samples": [list(sample) for sample in self.samples],
        }


def skipped_cooldown(
    threshold_c: float = DEFAULT_THRESHOLD_C, cap_s: float = DEFAULT_CAP_S
) -> CooldownRecord:
    """The record for `--no-cooldown`. Skipping the wait does not skip the record."""
    return CooldownRecord(enabled=False, outcome=SKIPPED, threshold_c=threshold_c, cap_s=cap_s)


def wait_for_cooldown(
    read_temperature: Callable[[], Optional[float]],
    threshold_c: float = DEFAULT_THRESHOLD_C,
    cap_s: float = DEFAULT_CAP_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> CooldownRecord:
    """Poll until the GPU is below `threshold_c` or `cap_s` elapses; record which."""
    started = clock()
    samples: List[List[Optional[float]]] = []
    while True:
        temperature = read_temperature()
        elapsed = clock() - started
        samples.append([elapsed, temperature])
        verdict = cooldown_verdict(temperature, elapsed, threshold_c, cap_s)
        if verdict != WAITING:
            return CooldownRecord(
                enabled=True, outcome=verdict, threshold_c=threshold_c, cap_s=cap_s,
                waited_s=elapsed, final_temperature_c=temperature, samples=samples,
            )
        sleep(poll_interval_s)
