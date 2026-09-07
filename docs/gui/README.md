# The window, before and after issue #40

Regenerate any of these with `scripts/gui_screenshot.py`, which builds a real
`StreamGUI` - no worker, no GPU, nothing started - raises it and grabs its own
rectangle. "Ugly" is not something a test can assert, so this is how the visual
half of that issue's Gate gets judged.

| | what it shows |
| --- | --- |
| [`before.png`](before.png) | HEAD before the issue. Model path blank, so the app cannot start. Acceleration `xformers`, which no benchmark in this repo has ever measured. Target and Style at the bottom, below the preview, among the engine knobs, with nothing saying detection is idle. |
| [`after.png`](after.png) | Restyle leads. The model path resolved itself from the models root, the plan state reads `Plan: global — the whole frame.  Detection: off (no target).`, and the engine knobs are folded into **Advanced**. |
| [`after-advanced.png`](after-advanced.png) | The same window with Advanced open: the standing rebuild warning, and acceleration on `tensorrt`. |
| [`after-selective.png`](after-selective.png) | A target and a style typed: `Plan: selective — every person.  Detection: on, 3 objects held, every 5 frames.` |

`after.png` and the two beside it were taken with `SD_MODELS_DIR` pointed at the
main checkout's `models/`, because a fresh worktree has none and the point of the
shot is what a machine with a model does.

`after-selective.png` was taken with `--selective`, which hands the window a
**stand-in** fps payload of the shape `detection.fps_payload` produces, so the
detection readout can be photographed without running the GPU. It is a picture of
the readout and not a measurement of anything - every measured number in this repo
comes from `bench/` and carries a hardware fingerprint.
