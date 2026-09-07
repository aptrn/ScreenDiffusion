"""The occupancy gate: measure a card nobody else is drawing on, or say so.

The sibling of `bench.cooldown`, and it exists for the same reason one sentence
further on. A GPU benchmarked while another application is using it measures the
contention instead of the change under test - and unlike a thermal throttle,
nothing else in the record shows it. The cooldown gate reads `reached`, the clock
regime reads whatever it always read, the fingerprint names the same card, and
the number is three times what the same code measured an hour earlier.

That is not hypothetical: issue #33's first `selective-people` re-run measured
54.17 ms/frame against a committed 16.89 on the same 4090, at 50 degC, with a
live real-time application holding ~45% of the SMs. It passed every door the
harness had and was appended to the README.

The sample is taken *before* the timed region, while this process is idle, so
what it reads is what else is on the card rather than what this run is about to
put there.

The decision is a pure function of one utilization reading (`occupancy_verdict`);
the sampling loop takes its meter and its sleep as arguments. Both are therefore
testable without a GPU.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

# 10%: a desktop compositor and a browser idle below this; a real-time application
# drawing frames sits far above it. The gate is for the second.
DEFAULT_BUSY_THRESHOLD_PCT = 10.0
# Four readings half a second apart. Long enough that a single frame of someone
# else's compositor does not decide it, short enough not to lengthen a run.
DEFAULT_SAMPLES = 4
DEFAULT_INTERVAL_S = 0.5

# Outcomes. All three are terminal and land in the result file.
CLEAR = "clear"
BUSY = "busy"
UNKNOWN = "unknown"


def occupancy_verdict(utilization_pct: Optional[float],
                      threshold_pct: float = DEFAULT_BUSY_THRESHOLD_PCT) -> str:
    """Whether the card is this run's alone, somebody else's, or unreadable.

    An unreadable meter (`None`) is `UNKNOWN` and never `CLEAR`, for the reason
    `cooldown_verdict` never reads an unreadable temperature as cool: a gate that
    cannot see the card has not seen an empty one.
    """
    if utilization_pct is None:
        return UNKNOWN
    return CLEAR if utilization_pct <= threshold_pct else BUSY


@dataclass(frozen=True)
class OccupancyRecord:
    """What the gate saw, as it lands in the result file.

    The outcome is never absent, so a comparison can always tell a run measured on
    a clear card from one measured beside something else from one that could not
    tell.
    """

    outcome: str
    threshold_pct: float = DEFAULT_BUSY_THRESHOLD_PCT
    mean_utilization_pct: Optional[float] = None
    # Every reading, not just the mean, so a run that only just passed is visible
    # rather than implied - the rule `CooldownRecord.samples` follows.
    samples: List[Optional[float]] = field(default_factory=list)

    @property
    def clear(self) -> bool:
        return self.outcome == CLEAR

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "clear": self.clear,
            "threshold_pct": self.threshold_pct,
            "mean_utilization_pct": self.mean_utilization_pct,
            "samples": list(self.samples),
        }


def measure_occupancy(
    read_utilization: Callable[[], Optional[float]],
    threshold_pct: float = DEFAULT_BUSY_THRESHOLD_PCT,
    samples: int = DEFAULT_SAMPLES,
    interval_s: float = DEFAULT_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
) -> OccupancyRecord:
    """Sample the card a few times and decide on the mean of what answered.

    The mean rather than the peak: a run lasts tens of seconds, so a single frame
    of someone else's window manager is not what confounds it and a sustained load
    is. Readings that did not answer are kept in the record and left out of the
    mean - some evidence is not none.
    """
    readings: List[Optional[float]] = []
    for index in range(max(1, samples)):
        if index:
            sleep(interval_s)
        readings.append(read_utilization())
    answered = [value for value in readings if value is not None]
    mean = round(statistics.fmean(answered), 2) if answered else None
    return OccupancyRecord(outcome=occupancy_verdict(mean, threshold_pct),
                           threshold_pct=threshold_pct,
                           mean_utilization_pct=mean, samples=readings)


def occupancy_summary(record: OccupancyRecord) -> str:
    """One line for the log and for the refusal. Quotes no figure it did not read."""
    if record.mean_utilization_pct is None:
        return f"GPU occupancy {record.outcome} (utilization unreadable)"
    return (f"GPU occupancy {record.outcome}: "
            f"{record.mean_utilization_pct:.0f}% of the SMs in use before the run, "
            f"against a {record.threshold_pct:.0f}% threshold")


def measure_occupancy_now() -> OccupancyRecord:
    """`measure_occupancy` pointed at this machine's own meter.

    The binding lives here rather than in the caller because both callers want the
    same one: the CLI's gate, which refuses before an engine build, and the runner,
    which takes the reading that lands in the result file. `read_utilization_pct`
    is imported inside the call so this module's own tests need no `nvidia-smi`.
    """
    from bench.fingerprint import read_utilization_pct

    return measure_occupancy(read_utilization_pct)
