"""Grab a picture of the app's window, for the human half of a UI review.

Issue #40's Gate asks for before/after screenshots, and a screenshot pasted into a
comment is gone by the next iteration. This builds a real `StreamGUI` - no worker,
no GPU, nothing started - raises it to the top and grabs exactly its own rectangle,
so what lands in `docs/gui/` is the window and not whatever else is on the desktop.

    uv run python scripts/gui_screenshot.py docs/gui/after.png
    uv run python scripts/gui_screenshot.py docs/gui/after-advanced.png --advanced
    uv run python scripts/gui_screenshot.py docs/gui/after-selective.png --selective
    uv run python scripts/gui_screenshot.py docs/gui/after-mask-off.png --mask
    uv run python scripts/gui_screenshot.py docs/gui/after-mask-on.png --mask --overlay

`--selective` fills the two fields and hands the window a **stand-in** fps payload -
the shape `detection.fps_payload` puts on the queue - so the detection readout can
be photographed without a GPU. It is a picture of the readout, not a measurement of
anything; every measured number in this repo comes from `bench/`.

`--mask` (issue #47) goes one further and puts a real frame in the preview: a frame
of the committed clip `bench/clips/people.mp4`, its real detections from the
committed track beside it, put through the **shipped** `RegionScheduler` and
`Compositor` under `priority_case_plan()`. So the rectangles in the picture are the
regions that plan would actually restyle, at the geometry the transform actually
produces - not boxes drawn to look right. What it is *not* is a restyled frame:
photographing one needs the GPU and a running worker. `--overlay` turns the switch
on; without it the same window is photographed with the overlay off, which is the
pair the Gate asks for.

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

# Which frame of the committed clip the mask shots use. The track covers the same
# 48 frames every committed selective run renders, and this one has the people
# spread across the frame, so several distinct regions are visible at once.
MASK_CLIP = "people"
MASK_FRAME = 36


def mask_preview(app) -> None:
    """Put a real frame and its real selection in front of the window.

    Everything but the picture is the shipped path: the boxes come from the clip's
    committed track, the plan is `priority_case_plan()`, and the selection and the
    frame render come from `RegionScheduler` and `Compositor` themselves. The
    payload is then built by the same three functions the worker calls.
    """
    import json

    import cv2
    from PIL import Image

    from compositor import Compositor
    from detection import Box, Track, Tracks, fps_payload
    from mask_overlay import overlay_status
    from region_scheduler import RegionScheduler
    from render_plan import priority_case_plan

    clips = ROOT / "bench" / "clips"
    track = json.loads((clips / f"{MASK_CLIP}.track.json").read_text(encoding="utf-8"))
    boxes = next(entry["boxes"] for entry in track["frames"]
                 if entry["index"] == MASK_FRAME)
    capture = cv2.VideoCapture(str(clips / f"{MASK_CLIP}.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, MASK_FRAME)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"could not read frame {MASK_FRAME} of {MASK_CLIP}.mp4")
    height, width = frame.shape[:2]

    plan = priority_case_plan()
    tracks = Tracks(tracks=tuple(
        Track(track_id=index, box=Box(*box), concept=track["target"],
              confidence=0.9)
        for index, box in enumerate(boxes)), ticks=8, detector_ms=21.4)
    scheduler = RegionScheduler()
    selection = scheduler.select(tracks, plan, width, height)
    render = Compositor().frame(selection, width, height,
                                plan.settings.primitive)
    payload = fps_payload(30, tracks, plan.settings.detect_every_n)
    payload.update(scheduler.status(selection))
    payload.update(overlay_status(selection, render, width, height))

    app.target_var.set(track["target"])
    app.style_var.set(plan.effective_prompt)
    app._fps_payload = payload
    app.running = True
    app._refresh_plan_state()
    app.running = False
    app._update_preview(Image.fromarray(frame[:, :, ::-1]))


def capture(out_path: Path, expand_advanced: bool = False,
            selective: bool = False, mask: bool = False,
            overlay: bool = False) -> Path:
    from PIL import ImageGrab

    import main_gpu_addon

    app = main_gpu_addon.StreamGUI()
    app.geometry("+0+0")
    if selective:
        app.target_var.set("person")
        app.style_var.set("wet denim, studio light")
        app._fps_payload = dict(SAMPLE_PAYLOAD)
        app.running = True
        app._refresh_plan_state()
        app.running = False
    if mask:
        app.overlay_var.set(bool(overlay))
        mask_preview(app)
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
    flags = {"--advanced", "--selective", "--mask", "--overlay"}
    args = [a for a in argv if a not in flags]
    out = Path(args[0]) if args else ROOT / "docs" / "gui" / "app.png"
    saved = capture(out, expand_advanced="--advanced" in argv,
                    selective="--selective" in argv, mask="--mask" in argv,
                    overlay="--overlay" in argv)
    print(f"saved {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
