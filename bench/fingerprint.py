"""The hardware fingerprint, and the live GPU samples taken during a run.

Spec 7.4: absolute ms/frame and VRAM ceilings do not transfer between the RTX 3080
laptop this is developed on and the 3090 Ti / 4090 it deploys to, so a result
without its machine attached is not a result. The raw `nvidia-smi` dump with a
timestamp is carried verbatim for the same reason the parsed fields are: it is the
evidence the number came off a real device.

Parsing is separated from running `nvidia-smi` so the awkward cases - a laptop that
reports `[N/A]` for its power limit, an absent driver - are covered by GPU-free tests.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Sequence, Tuple

from bench.clocks import (
    EVENT_REASON_FIELD,
    LOCK_FIELDS,
    UNKNOWN,
    ClockLock,
    interpret_event_reason,
)

NVIDIA_SMI = "nvidia-smi"

# Constant for the machine, queried once per run.
# Both power limits: this laptop reports `[N/A]` for `power.limit` and 120 W for
# `enforced.power.limit`, and spec 7.4 wants the limit the silicon is actually held to.
FINGERPRINT_FIELDS: Tuple[str, ...] = ("name", "memory.total", "driver_version",
                                       "power.limit", "enforced.power.limit")
# Varying during the run, sampled per rep so a throttled run stays visible.
SAMPLE_FIELDS: Tuple[str, ...] = ("clocks.sm", "temperature.gpu")

Runner = Callable[[Sequence[str]], str]


class NvidiaSmiUnavailable(RuntimeError):
    """`nvidia-smi` is missing or failed. Without it there is no fingerprint, and
    without a fingerprint there is no result."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_number(value: Optional[str]) -> Optional[float]:
    """A numeric `nvidia-smi` field, or None when it says it has no value.

    `[N/A]` is what a laptop reports for `power.limit`. That is a fact about the
    machine, so it is recorded as None rather than dropped or faked.
    """
    if value is None:
        return None
    text = value.strip().strip("[]").strip()
    if not text or text.upper().startswith("N/A"):
        return None
    # `nounits` is requested, but tolerate a unit suffix if it appears anyway.
    head = text.split()[0]
    try:
        return float(head)
    except ValueError:
        return None


def parse_csv_row(fields: Sequence[str], text: str) -> Dict[str, str]:
    """The first GPU's row of `--format=csv,noheader` output, keyed by field name.

    First GPU only: this project targets a single NVIDIA GPU, and mixing two
    devices into one fingerprint would misattribute the number.
    """
    for line in text.splitlines():
        if not line.strip():
            continue
        values = [cell.strip() for cell in line.split(",")]
        return {field: value for field, value in zip(fields, values)}
    return {}


def _run(command: Sequence[str]) -> str:
    if shutil.which(command[0]) is None:
        raise NvidiaSmiUnavailable(f"{command[0]} is not on PATH")
    result = subprocess.run(list(command), capture_output=True, text=True)
    if result.returncode != 0:
        raise NvidiaSmiUnavailable(
            f"{' '.join(command)} exited {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout


def query(fields: Sequence[str], run: Runner = _run) -> Dict[str, str]:
    """`nvidia-smi --query-gpu=<fields>` for the first GPU."""
    return parse_csv_row(fields, run([
        NVIDIA_SMI,
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]))


@dataclass(frozen=True)
class GpuSample:
    """One live reading. Clocks and temperature sit next to every timing so a
    throttled run is visible rather than silently polluting a comparison."""

    sm_clock_mhz: Optional[float]
    temperature_c: Optional[float]


def read_gpu_sample(run: Runner = _run) -> GpuSample:
    row = query(SAMPLE_FIELDS, run=run)
    return GpuSample(
        sm_clock_mhz=parse_number(row.get("clocks.sm")),
        temperature_c=parse_number(row.get("temperature.gpu")),
    )


def read_clock_lock(run: Runner = _run) -> ClockLock:
    """Whether a clock lock is in force right now, and the clocks around it.

    Issue #13 step 1. The harness never *sets* a lock - `nvidia-smi
    --lock-gpu-clocks` needs an elevated shell it does not have - so this only ever
    reports what someone else has already done. A query that cannot be answered
    yields `UNKNOWN`, carrying the error as its evidence: a failed detection must
    not read as a lock, and must not read as the absence of one either.
    """
    try:
        row = query(LOCK_FIELDS, run=run)
    except NvidiaSmiUnavailable as error:
        return ClockLock(state=UNKNOWN, applied_clock_mhz=None, max_sm_clock_mhz=None,
                         current_sm_clock_mhz=None, evidence=str(error))
    reason = row.get(EVENT_REASON_FIELD)
    return ClockLock(
        state=interpret_event_reason(reason),
        applied_clock_mhz=parse_number(row.get("clocks.applications.graphics")),
        max_sm_clock_mhz=parse_number(row.get("clocks.max.sm")),
        current_sm_clock_mhz=parse_number(row.get("clocks.sm")),
        evidence=f"{EVENT_REASON_FIELD}={(reason or '').strip() or '<absent>'}",
    )


@dataclass(frozen=True)
class Fingerprint:
    """Which machine produced a number, and the proof it was a real one."""

    gpu_name: str
    total_vram_mib: Optional[float]
    driver_version: str
    power_limit_w: Optional[float]
    enforced_power_limit_w: Optional[float]
    # Issue #13: the clock regime this machine was in when the run started. A result
    # measured at an unlocked clock is not comparable with another at face value, so
    # the regime travels with the machine rather than with the reader's memory.
    clock_lock: ClockLock
    torch_version: Optional[str]
    cuda_version: Optional[str]
    hostname: str
    nvidia_smi_captured_utc: str
    nvidia_smi_raw: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_fingerprint(
    query_row: Dict[str, str],
    nvidia_smi_raw: str,
    captured_utc: str,
    torch_version: Optional[str],
    cuda_version: Optional[str],
    hostname: str,
    lock: ClockLock,
) -> Fingerprint:
    """Assemble a fingerprint from already-collected pieces. Pure, hence tested."""
    return Fingerprint(
        gpu_name=(query_row.get("name") or "").strip(),
        total_vram_mib=parse_number(query_row.get("memory.total")),
        driver_version=(query_row.get("driver_version") or "").strip(),
        power_limit_w=parse_number(query_row.get("power.limit")),
        enforced_power_limit_w=parse_number(query_row.get("enforced.power.limit")),
        clock_lock=lock,
        torch_version=torch_version,
        cuda_version=cuda_version,
        hostname=hostname,
        nvidia_smi_captured_utc=captured_utc,
        nvidia_smi_raw=nvidia_smi_raw,
    )


def capture_fingerprint(run: Runner = _run) -> Fingerprint:
    """Query the machine now. Raises `NvidiaSmiUnavailable` rather than guessing."""
    torch_version = cuda_version = None
    try:  # torch is imported here and nowhere at module scope - see bench/__init__.py
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except Exception:
        pass

    return build_fingerprint(
        query_row=query(FINGERPRINT_FIELDS, run=run),
        nvidia_smi_raw=run([NVIDIA_SMI]),
        captured_utc=utc_now(),
        torch_version=torch_version,
        cuda_version=cuda_version,
        hostname=platform.node(),
        lock=read_clock_lock(run=run),
    )
