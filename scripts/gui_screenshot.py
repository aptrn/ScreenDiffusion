"""Grab a picture of the app's window, for the human half of a UI review.

Issue #40's Gate asks for before/after screenshots, and a screenshot pasted into a
comment is gone by the next iteration. This builds a real `StreamGUI` - no worker,
no GPU, nothing started - raises it to the top and grabs exactly its own rectangle,
so what lands in `docs/gui/` is the window and not whatever else is on the desktop.

    uv run python scripts/gui_screenshot.py docs/gui/after.png
    uv run python scripts/gui_screenshot.py docs/gui/after-advanced.png --advanced
    uv run python scripts/gui_screenshot.py docs/gui/after-selective.png --selective
    uv run python scripts/gui_screenshot.py docs/gui/after-cfg.png --advanced --cfg=full

`--selective` fills the two fields and hands the window a **stand-in** fps payload -
the shape `detection.fps_payload` puts on the queue - so the detection readout can
be photographed without a GPU. It is a picture of the readout, not a measurement of
anything; every measured number in this repo comes from `bench/`.

Windows only, and it needs a desktop session: `ImageGrab` reads the screen. It is
not part of any test tier - a picture is not something a test can judge.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Let the window paint before it is grabbed. customtkinter draws its widgets over
# several idle passes, so one `update()` gets a half-built frame.
SETTLE_PASSES = 40
SETTLE_SECONDS = 0.05

# What a run with a live detector puts on the fps queue. Shape only - see above.
SAMPLE_PAYLOAD = {"fps": 30, "detections": 3, "detector_ms": 21.4,
                  "detect_every_n": 5, "amortised_ms": 4.3,
                  "concepts": ("person",), "ticks": 12}


def capture(out_path: Path, expand_advanced: bool = False,
            selective: bool = False, cfg_type: str = "") -> Path:
    from PIL import ImageGrab

    import main_gpu_addon

    app = main_gpu_addon.StreamGUI()
    app.geometry("+0+0")
    if cfg_type:
        # Through the window's own handler, so what is photographed is the state a
        # user reaches by picking the type - the scale its companion brings with
        # it, and the line under the row (issue #45).
        app._apply_cfg_type(cfg_type)
    if selective:
        app.target_var.set("person")
        app.style_var.set("wet denim, studio light")
        app._fps_payload = dict(SAMPLE_PAYLOAD)
        app.running = True
        app._refresh_plan_state()
        app.running = False
    if expand_advanced:
        app._toggle_advanced()
        # The left panel scrolls, and the section that just opened is below the
        # fold on a 840 px window. Scroll to it, or the picture shows nothing.
        app.update_idletasks()
        app.left_panel._parent_canvas.yview_moveto(1.0)
    app.attributes("-topmost", True)
    app.lift()
    app.focus_force()
    for _ in range(SETTLE_PASSES):
        app.update()
        app.update_idletasks()
        time.sleep(SETTLE_SECONDS)
    x, y = app.winfo_rootx(), app.winfo_rooty()
    box = (x, y, x + app.winfo_width(), y + app.winfo_height())
    image = ImageGrab.grab(bbox=box)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    app.destroy()
    return out_path


def main(argv: list) -> int:
    flags = {"--advanced", "--selective"}
    cfg_type = next((a.split("=", 1)[1] for a in argv if a.startswith("--cfg=")), "")
    args = [a for a in argv if a not in flags and not a.startswith("--cfg=")]
    out = Path(args[0]) if args else ROOT / "docs" / "gui" / "app.png"
    saved = capture(out, expand_advanced="--advanced" in argv,
                    selective="--selective" in argv, cfg_type=cfg_type)
    print(f"saved {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
