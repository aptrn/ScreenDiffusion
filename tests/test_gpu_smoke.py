"""The GPU tier. Everything here is `@pytest.mark.gpu` and skipped by the merge gate.

torch is imported inside the tests, never at collection time, so `pytest -m "not gpu"`
stays importable on a machine with no CUDA device.
"""

import pytest

pytestmark = pytest.mark.gpu


def test_cuda_is_available():
    import torch

    assert torch.cuda.is_available(), "no CUDA device visible to torch"
    assert torch.cuda.device_count() >= 1


def test_a_tensor_round_trips_through_the_device():
    import torch

    x = torch.ones(8, 8, dtype=torch.float16, device="cuda")
    assert (x + 1).cpu().sum().item() == pytest.approx(128.0)
