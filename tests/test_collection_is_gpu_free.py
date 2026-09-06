"""The tier split's own guard rail.

`pytest -m "not gpu"` still *collects* every test module before deselecting the
GPU ones, so an import at module scope runs on a machine with no CUDA device.
These assertions fail the moment something drags torch - or the GUI module and
its DLL priming - into collection.
"""

import sys


def test_torch_was_not_imported_during_collection():
    assert "torch" not in sys.modules


def test_the_application_modules_were_not_imported_during_collection():
    assert "main_gpu_addon" not in sys.modules
    assert "wrapper" not in sys.modules
