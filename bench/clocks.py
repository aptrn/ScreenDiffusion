"""Which clock regime a run was measured under, and what to report when it is not locked.

Issue #13. Comparative cells are only comparable if they ran at the same clock. On
the RTX 3080 laptop they do not: under the 120 W limit the SM clock falls from boost
to the floor within about two seconds of a run starting, so a short call catches
boost and a long one spends its whole run throttled. The committed 512x512 sweep
shows it - batch 1 at a mean 1290 MHz against batch 4 at 1141 MHz - and the 8.6%
gain those two cells appear to show is 22.8% once the clocks are equalised.

The fix is to lock the clock. This harness cannot: `nvidia-smi --lock-gpu-clocks`
needs an elevated shell on Windows and the agent loop does not have one. So the
harness does the two things it *can* do, and neither of them is guessing:

- detect the regime, and record it on every result. A lock it cannot see is not a
  lock: `UNKNOWN` is a third state, never folded into either of the other two.
- when the clocks are not locked, report a clock-normalised estimate beside the raw
  figure and label it an estimate. `ms x f / f_ref` is first-order and assumes the
  work is compute-bound; it is a way to read a confounded sweep, not a substitute
  for locking.

`bench.marginal` normalises too, but for a different job and to a different basis:
it equalises the *committed sweep* against the fastest clock any of its cells
reached, so a table of cells is internally comparable. This module normalises one
run as it is measured, against a basis fixed by the hardware, so a result is
comparable before any other result exists.

Everything here is pure - arithmetic over a sample trace and one string out of
`nvidia-smi`. `bench.fingerprint` does the querying and hands the result in, which
is what keeps the whole of this covered in the GPU-free tier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Optional, Sequence

# The three regimes a result can be measured under.
LOCKED = "locked"
UNLOCKED = "unlocked"
# Not a synonym for UNLOCKED: it means the question was asked and not answered - an
# `nvidia-smi` that has no such field, or a driver that deprecated it. It fails
# `--require-locked-clocks` for the same reason UNLOCKED does.
UNKNOWN = "unknown"
REGIMES = (LOCKED, UNLOCKED, UNKNOWN)

# The `nvidia-smi --query-gpu=` fields that answer "is a lock in force, and at what
# clock". Named one by one because `bench.fingerprint` reads each back out of the row
# by name: the query and the reads then cannot drift apart.
#
# `clocks_event_reasons.applications_clocks_setting` is NVML's
# `nvmlClocksEventReasonApplicationsClocksSetting`: the clocks are being held to a
# software setting rather than to the driver's own boost decision. That is the only
# lock signal this driver (595.79) exposes - it reports no "locked clocks" field, and
# answers the applications-clock queries with a deprecation notice on consumer
# Ampere. Hence `UNKNOWN` rather than a second opinion when the field says nothing.
EVENT_REASON_FIELD = "clocks_event_reasons.applications_clocks_setting"
APPLIED_CLOCK_FIELD = "clocks.applications.graphics"
CURRENT_CLOCK_FIELD = "clocks.sm"
# What the basis clock is read from, carried in the result so the number is traceable
# to a field rather than to a convention.
BASIS_SOURCE = "clocks.max.sm"
LOCK_FIELDS = (EVENT_REASON_FIELD, APPLIED_CLOCK_FIELD, BASIS_SOURCE, CURRENT_CLOCK_FIELD)

NORMALIZATION_METHOD = "ms_per_frame * time-weighted mean SM clock / basis clock"


def interpret_event_reason(value: Optional[str]) -> str:
    """The regime one `nvidia-smi` event-reason cell implies.

    Only the two documented answers decide anything. `[N/A]`, `[Not Supported]`, a
    deprecation notice and an absent field all mean the same thing - nobody knows -
    and the harness says so rather than defaulting to the convenient answer.
    """
    text = (value or "").strip().strip("[]").strip().lower()
    if text == "active":
        return LOCKED
    if text == "not active":
        return UNLOCKED
    return UNKNOWN


@dataclass(frozen=True)
class ClockLock:
    """The clock state of the machine at the moment a run started.

    `evidence` is the raw cell the state was read from, kept for the same reason the
    fingerprint keeps the raw `nvidia-smi` dump: months later, "unlocked" is a claim,
    and the string it was derived from is the thing that can be re-read.
    """

    state: str
    applied_clock_mhz: Optional[float]
    max_sm_clock_mhz: Optional[float]
    current_sm_clock_mhz: Optional[float]
    evidence: str

    @property
    def locked(self) -> bool:
        """Only a detected lock counts. An undetectable one is not one."""
        return self.state == LOCKED

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ClockNormalization:
    """The clock-comparability block of a result: which regime, and the estimate.

    `ms_per_frame` is None whenever there is no honest number to put there - the
    clocks were locked and the raw figure already is the comparable one, or the
    trace or the basis clock is missing. `note` always says which of those it is.
    """

    regime: str
    ms_per_frame: Optional[float]
    basis_mhz: Optional[float]
    basis_source: Optional[str]
    method: Optional[str]
    sampled_mean_sm_clock_mhz: Optional[float]
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


Sample = Sequence[Optional[float]]


def time_weighted_mean_clock(samples: Sequence[Sample]) -> Optional[float]:
    """The mean SM clock over a run: the trace integrated, divided by its span.

    Time-weighted rather than a plain mean of the samples, because the samples are
    not evenly spaced. The sampler polls `nvidia-smi` on a thread and a call that
    takes a second leaves a second-long gap; averaging every reading equally would
    let one cheap boost-clock sample at the start of a run count as much as a long
    spell at the floor, which is the confound this module exists to correct.

    Trapezoidal, so the weights sum to the span of the trace exactly and evenly
    spaced samples come back as their plain mean. Samples with no clock reading are
    dropped, which stretches the neighbouring interval over them - the previous
    reading is what is actually known about that gap.
    """
    trace = [(float(elapsed), float(clock)) for elapsed, clock, *_ in samples
             if clock is not None]
    if not trace:
        return None

    span = trace[-1][0] - trace[0][0]
    if span <= 0:  # one sample, or a sample clock that did not advance
        return fmean(clock for _, clock in trace)
    area = sum((later_t - earlier_t) * (earlier_c + later_c) / 2.0
               for (earlier_t, earlier_c), (later_t, later_c) in zip(trace, trace[1:]))
    return area / span


def normalize_ms(ms: float, clock_mhz: float, basis_mhz: float) -> float:
    """What `ms` would have been at `basis_mhz`, to first order.

    Diffusion here is compute-bound, so time scales as 1/clock and the correction is
    a multiplication. First order: it ignores the memory clock, which does not move
    with the SM clock, and every other term. An estimate, and labelled one wherever
    it is reported.
    """
    return ms * clock_mhz / basis_mhz


def clock_normalization(
    lock: ClockLock,
    samples: Sequence[Sample],
    raw_ms_per_frame: float,
) -> ClockNormalization:
    """The comparability block for one run, given its clock state and its trace.

    Locked clocks get no normalised figure at all - the raw one is already
    comparable, and computing an estimate beside it would invite someone to quote
    the estimate. An unknown regime is normalised like an unlocked one: the harness
    cannot see a lock, so it must not assume the raw figure is comparable.
    """
    sampled = time_weighted_mean_clock(samples)
    basis = lock.max_sm_clock_mhz

    def block(note: str, *, ms_per_frame: Optional[float] = None,
              basis_mhz: Optional[float] = None,
              basis_source: Optional[str] = BASIS_SOURCE,
              method: Optional[str] = NORMALIZATION_METHOD) -> ClockNormalization:
        """One block, with the regime and the sampled clock already filled in.

        The defaults are the unlocked case - the regime this module exists for. Each
        branch then states only what is different about it.
        """
        return ClockNormalization(
            regime=lock.state, ms_per_frame=ms_per_frame, basis_mhz=basis_mhz,
            basis_source=basis_source, method=method,
            sampled_mean_sm_clock_mhz=None if sampled is None else round(sampled, 1),
            note=note,
        )

    if lock.locked:
        return block(
            basis_mhz=lock.applied_clock_mhz, basis_source=None, method=None,
            note="Clocks were locked for this run, so the raw ms/frame is the "
                 "comparable figure and nothing is normalised. The sampled clock is "
                 "carried as evidence the lock held.",
        )
    if sampled is None:
        return block(
            basis_mhz=basis,
            note="Clocks were not locked and no SM clock was sampled during the run, "
                 "so this result cannot be compared with another cell.",
        )
    if basis is None:
        return block(
            note=f"Clocks were not locked and {BASIS_SOURCE} could not be read, so "
                 f"there is no basis to normalise to.",
        )
    return block(
        ms_per_frame=round(normalize_ms(raw_ms_per_frame, sampled, basis), 4),
        basis_mhz=basis,
        note=f"Clocks were not locked ({lock.state}), so the raw figure carries this "
             f"run's clock. This is a first-order estimate of the same work at "
             f"{basis:.0f} MHz, not a measurement, and not a substitute for "
             f"`nvidia-smi --lock-gpu-clocks` from an elevated shell.",
    )


def regime_of(result: dict) -> str:
    """The regime a result file was measured under, absent field included.

    Issue #13 keeps the pre-existing results exactly as they are, "marked unlocked
    by absence": every one of them was measured on this laptop with no lock in force,
    so reading a missing field as `UNLOCKED` states a fact rather than papering over
    one. Results written since carry the field and answer for themselves.
    """
    hardware = result.get("hardware") or {}
    state = (hardware.get("clock_lock") or {}).get("state")
    return state if state in REGIMES else UNLOCKED


def regime_summary(lock: ClockLock) -> str:
    """One line for a log or an error message."""
    parts = [f"clocks {lock.state}"]
    if lock.applied_clock_mhz:
        parts.append(f"at {lock.applied_clock_mhz:.0f} MHz")
    if lock.max_sm_clock_mhz:
        parts.append(f"(max {lock.max_sm_clock_mhz:.0f} MHz)")
    return " ".join(parts)
