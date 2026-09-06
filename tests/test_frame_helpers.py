"""GPU-free helpers from main_gpu_addon.py: capture-frame and monitor geometry."""

import ctypes
import sys

import numpy as np
import pytest

from sourceloader import load_symbols

_frame_to_rgb = load_symbols(
    "main_gpu_addon.py",
    ["_frame_to_rgb"],
    {"np": np},
)["_frame_to_rgb"]


def test_alpha_channel_is_dropped():
    frame = np.zeros((2, 2, 4), dtype=np.uint8)
    frame[:, :, :] = (10, 20, 30, 255)
    out = _frame_to_rgb(frame)
    assert out.shape == (2, 2, 3)
    assert tuple(out[0, 0]) == (10, 20, 30)


def test_three_channel_frame_is_untouched():
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    frame[0, 0] = (1, 2, 3)
    assert tuple(_frame_to_rgb(frame)[0, 0]) == (1, 2, 3)


def test_force_swap_rb_reverses_the_channels():
    frame = np.zeros((1, 1, 4), dtype=np.uint8)
    frame[0, 0] = (10, 20, 30, 255)
    assert tuple(_frame_to_rgb(frame, force_swap_rb=True)[0, 0]) == (30, 20, 10)


def test_none_and_empty_frames_pass_straight_through():
    # The capture deque can hand us either; neither may raise on the frame path.
    assert _frame_to_rgb(None) is None
    empty = np.zeros((0, 0, 4), dtype=np.uint8)
    assert _frame_to_rgb(empty) is empty


@pytest.mark.skipif(sys.platform != "win32", reason="MonitorFromPoint is Win32-only")
class TestMonitorRectFromPoint:
    """No GPU, but a real user32 call - Windows-only, like the app itself."""

    @staticmethod
    def _load():
        user32 = ctypes.windll.user32
        symbols = load_symbols(
            "main_gpu_addon.py",
            ["RECT", "MONITORINFO", "POINT", "MONITOR_DEFAULTTONEAREST", "_monitor_rc_from_point"],
            {"ctypes": ctypes, "_user32": user32},
        )
        return symbols["_monitor_rc_from_point"]

    def test_origin_lands_in_a_non_empty_monitor_rect(self):
        left, top, right, bottom = self._load()(0, 0)
        assert right > left and bottom > top
        assert left <= 0 < right and top <= 0 < bottom

    def test_a_point_far_off_screen_still_returns_the_nearest_monitor(self):
        # MONITOR_DEFAULTTONEAREST, so this must not return a degenerate rect.
        left, top, right, bottom = self._load()(-999_999, -999_999)
        assert right > left and bottom > top
