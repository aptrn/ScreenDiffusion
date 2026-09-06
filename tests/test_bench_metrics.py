"""The summary statistics `bench.runner` derives from the timed reps (issue #2).

`bench.runner` is the module that touches the GPU, but it imports torch inside its
functions - so importing it here is both how these pure helpers get tested and the
assertion that the deferred-import discipline still holds.
"""

import sys

from bench.runner import _percentile


def test_importing_the_runner_does_not_import_torch():
    """The claim bench/runner.py's docstring makes, checked rather than trusted."""
    assert "torch" not in sys.modules


def test_a_single_rep_has_a_percentile():
    """`statistics.quantiles` needs n >= 2; a 1-rep run is a legal scenario."""
    assert _percentile([42.0], 0.95) == 42.0


def test_the_percentile_is_the_nearest_rank_value():
    values = [float(n) for n in range(1, 11)]  # 1..10
    assert _percentile(values, 0.95) == 10.0  # ceil(9.5) -> 10th
    assert _percentile(values, 0.5) == 5.0  # ceil(5.0) -> 5th
    assert _percentile(values, 0.0) == 1.0  # clamped to the 1st


def test_the_input_order_does_not_matter():
    assert _percentile([9.0, 1.0, 5.0, 3.0, 7.0], 0.5) == 5.0


def test_p95_is_not_the_maximum_when_the_rank_lands_exactly():
    """Regression: `round(0.95 * n + 0.5)` ties to even, so n=20 reported its max.

    20 reps is an ordinary `--reps` value, and a p95 that silently equals the max
    hides exactly the tail this harness exists to see.
    """
    values = [float(n) for n in range(1, 21)]  # 1..20, so 0.95 * n == 19 exactly
    assert _percentile(values, 0.95) == 19.0
    assert _percentile(values, 0.95) < max(values)
