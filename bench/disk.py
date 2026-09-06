"""Free disk space: checked before an engine build, recorded in every result.

A TensorRT engine is ~5.1 GB and the machine this is developed on runs with tens
of gigabytes free, so the spec 7.3 confirmation runs are a capacity decision before
they are a timing one. The check is made *before* the build, because a shortfall
discovered halfway through an ONNX export has already spent the minutes and left a
partial engine behind.

Reading and refusing are two functions on purpose. Every run records its headroom -
that is the evidence the gate the issue asks for was actually applied - while only
a run that is about to compile something refuses to continue without it.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence, Union

from bench.fingerprint import utc_now

BYTES_PER_GIB = 1024 ** 3
# 15 GB, the floor issue #3 names: room for two engines plus their ONNX scratch.
MIN_FREE_BYTES_FOR_ENGINE_BUILD = 15 * BYTES_PER_GIB

# `shutil.disk_usage`: (total, used, free).
Usage = Callable[[Union[str, Path]], Sequence[int]]


class NotEnoughDiskSpace(RuntimeError):
    """The volume cannot hold the engine that is about to be built. Stop, do not start."""


@dataclass(frozen=True)
class DiskRecord:
    """The headroom one run had, as it lands in the result file."""

    path: str
    total_bytes: int
    free_bytes: int
    required_bytes: int
    sufficient: bool
    checked_utc: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["free_gib"] = round(self.free_bytes / BYTES_PER_GIB, 2)
        return data


def read_disk(path: Union[str, Path],
              required_bytes: int = MIN_FREE_BYTES_FOR_ENGINE_BUILD,
              usage: Usage = shutil.disk_usage) -> DiskRecord:
    """How much room `path`'s volume has, and whether that clears `required_bytes`."""
    total, _used, free = usage(path)
    return DiskRecord(
        path=str(path),
        total_bytes=int(total),
        free_bytes=int(free),
        required_bytes=int(required_bytes),
        sufficient=int(free) >= int(required_bytes),
        checked_utc=utc_now(),
    )


def require_free_space(record: DiskRecord) -> None:
    """Raise `NotEnoughDiskSpace` unless the volume can hold what is about to be built."""
    if record.sufficient:
        return
    raise NotEnoughDiskSpace(
        f"{record.free_bytes / BYTES_PER_GIB:.1f} GiB free on {record.path}, "
        f"and a TensorRT engine build needs {record.required_bytes / BYTES_PER_GIB:.1f} GiB. "
        f"Stopping before the build rather than partway through it."
    )
