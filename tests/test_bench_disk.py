"""The free-disk gate on TensorRT engine builds (issue #3).

Each engine is ~5.1 GB. Spec 7.3's confirmation runs build two or three of them,
and the machine they build on has ~76 GB free - close enough that a run which
discovers the shortfall halfway through an ONNX export has already wasted minutes
and left a partial engine behind. The check happens before the build and lands in
the result, so a reviewer can see the headroom the number was measured with.
"""

import json
import sys
import types

import pytest

from bench import cli
from bench.disk import (
    MIN_FREE_BYTES_FOR_ENGINE_BUILD,
    DiskRecord,
    NotEnoughDiskSpace,
    read_disk,
    require_free_space,
)
from bench.results import write_result
from bench.scenarios import SCENARIOS
from test_bench_results import a_result

GIB = 1024 ** 3


def usage(free_gib: float, total_gib: float = 954.0):
    """A `shutil.disk_usage`-shaped stand-in, so the test does not need a real volume."""
    def _usage(_path):
        return (int(total_gib * GIB), int((total_gib - free_gib) * GIB), int(free_gib * GIB))
    return _usage


def test_the_threshold_is_the_one_floor_the_app_and_the_harness_share():
    """Issue #3 set it at 15 GB; issue #38 raised it to 20 and moved it into
    `engine_cache`, where `StreamGUI` reads the same number before Start - a
    build the window allows and the harness refuses is two floors, not one."""
    from engine_cache import MIN_FREE_BYTES_FOR_ENGINE_BUILD as shared

    assert MIN_FREE_BYTES_FOR_ENGINE_BUILD == 20 * GIB == shared


def test_a_reading_records_the_headroom_and_the_verdict(tmp_path):
    record = read_disk(tmp_path, usage=usage(free_gib=76.0))
    assert isinstance(record, DiskRecord)
    assert record.free_bytes == int(76.0 * GIB)
    assert record.required_bytes == MIN_FREE_BYTES_FOR_ENGINE_BUILD
    assert record.sufficient is True
    assert record.checked_utc.endswith("Z")
    assert str(tmp_path) in record.path


def test_a_reading_below_the_threshold_is_recorded_rather_than_raised(tmp_path):
    """Reading and refusing are separate: a `none` run records headroom without stopping."""
    record = read_disk(tmp_path, usage=usage(free_gib=9.0))
    assert record.sufficient is False


def test_enough_space_lets_the_build_proceed(tmp_path):
    require_free_space(read_disk(tmp_path, usage=usage(free_gib=76.0)))


def test_too_little_space_stops_cleanly_and_says_how_much_is_free(tmp_path):
    record = read_disk(tmp_path, usage=usage(free_gib=9.0))
    with pytest.raises(NotEnoughDiskSpace) as excinfo:
        require_free_space(record)
    message = str(excinfo.value)
    assert "9.0" in message and "20.0" in message


def test_the_record_serialises_to_plain_types(tmp_path):
    data = read_disk(tmp_path, usage=usage(free_gib=76.0)).to_dict()
    assert data["free_gib"] == pytest.approx(76.0, abs=0.05)
    assert data["sufficient"] is True
    assert set(data) >= {"path", "free_bytes", "total_bytes", "required_bytes",
                         "sufficient", "checked_utc", "free_gib"}


def test_a_real_volume_can_be_read(tmp_path):
    """The default `usage` is `shutil.disk_usage`; nothing here is a stub-only path."""
    record = read_disk(tmp_path)
    assert record.total_bytes > 0
    assert 0 <= record.free_bytes <= record.total_bytes


def test_the_engine_build_guard_refuses_a_build_on_a_full_volume(tmp_path):
    """Step 3 of the issue: check free disk before each build, stop cleanly under it."""
    trt = SCENARIOS["img2img-tensorrt-512x512-b4"]

    with pytest.raises(NotEnoughDiskSpace):
        cli.engine_build_guard(trt, engines_root=tmp_path, allow_build=True,
                               usage=usage(free_gib=9.0))

    record = cli.engine_build_guard(trt, engines_root=tmp_path, allow_build=True,
                                    usage=usage(free_gib=76.0))
    assert record.sufficient is True


def test_a_cached_engine_records_its_headroom_but_is_never_refused(tmp_path):
    """Reading and refusing are separate: loading an engine already on the volume
    needs no room, but issue #3's gate wants the reading in the file regardless."""
    trt = SCENARIOS["img2img-tensorrt-512x512-b1"]
    cached = tmp_path / cli.engine_dir_name(trt)
    cached.mkdir(parents=True)
    (cached / "unet.engine").write_bytes(b"")

    record = cli.engine_build_guard(trt, engines_root=tmp_path, usage=usage(free_gib=0.5))
    assert record is not None and record.sufficient is False


def test_a_none_run_has_no_engines_volume_to_report(tmp_path):
    """The `none` accelerator builds and loads no engine, so there is nothing to gate."""
    assert cli.engine_build_guard(SCENARIOS["img2img-none-512x512-b1"],
                                  engines_root=tmp_path, usage=usage(free_gib=0.5)) is None


def test_a_result_carries_no_disk_reading_when_nothing_was_built():
    """The `none` sweep compiles nothing, so there is no gate to evidence."""
    assert a_result().to_dict()["disk"] is None


def test_a_result_records_the_headroom_the_engine_was_built_with(tmp_path):
    """The issue's gate: each TensorRT confirmation run has its free-disk check recorded.

    Recorded *in the result*, not only printed - a reviewer reading the JSON months
    later is the one who has to see that the gate was applied.
    """
    record = read_disk(tmp_path, usage=usage(free_gib=76.0))
    path = write_result(a_result(disk=record), results_dir=tmp_path)
    written = json.loads(path.read_text(encoding="utf-8"))["disk"]
    assert written["sufficient"] is True
    assert written["required_bytes"] == MIN_FREE_BYTES_FOR_ENGINE_BUILD
    assert written["free_gib"] == pytest.approx(76.0, abs=0.05)


def test_the_cli_hands_the_guards_reading_to_the_run(tmp_path, monkeypatch):
    """The guard's reading has to reach `run_scenario`, or it is never written down."""
    captured = {}

    stub = types.ModuleType("bench.runner")
    stub.run_scenario = lambda scenario, **kwargs: captured.update(kwargs)
    monkeypatch.setitem(sys.modules, "bench.runner", stub)

    record = read_disk(tmp_path, usage=usage(free_gib=76.0))
    monkeypatch.setattr(cli, "engine_build_guard", lambda *a, **k: record)

    assert cli.main(["img2img-none-512x512-b1", "--results-dir", str(tmp_path)]) == 0
    assert captured["disk"] is record
