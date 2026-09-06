"""`_broadcast_list` (wrapper.py) - per-LoRA scale lists stretched to fit."""

from sourceloader import load_symbols

_broadcast_list = load_symbols("wrapper.py", ["_broadcast_list"])["_broadcast_list"]


def test_empty_when_no_slots():
    assert _broadcast_list([0.5, 0.7], 0) == []
    assert _broadcast_list(None, -1) == []


def test_missing_values_become_the_default():
    assert _broadcast_list(None, 3) == [1.0, 1.0, 1.0]
    assert _broadcast_list([], 2) == [1.0, 1.0]
    assert _broadcast_list(None, 2, default=0.8) == [0.8, 0.8]


def test_exact_length_passes_through():
    assert _broadcast_list([0.1, 0.2, 0.3], 3) == [0.1, 0.2, 0.3]


def test_too_many_values_are_truncated():
    assert _broadcast_list([0.1, 0.2, 0.3], 2) == [0.1, 0.2]


def test_too_few_values_repeat_the_first():
    # Not a zip-with-default: a short list broadcasts its head, so [0.4, 0.9]
    # over three slots is 0.4 everywhere and 0.9 is dropped.
    assert _broadcast_list([0.4, 0.9], 3) == [0.4, 0.4, 0.4]
