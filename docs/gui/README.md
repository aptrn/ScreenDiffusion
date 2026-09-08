# The window, before and after issues #40, #38 and #45

Regenerate any of these with `scripts/gui_screenshot.py`, which builds a real
`StreamGUI` - no worker, no GPU, nothing started - raises it and grabs its own
rectangle. "Ugly" is not something a test can assert, so this is how the visual
half of that issue's Gate gets judged.

| | what it shows |
| --- | --- |
| [`before.png`](before.png) | HEAD before the issue. Model path blank, so the app cannot start. Acceleration `xformers`, which no benchmark in this repo has ever measured. Target and Style at the bottom, below the preview, among the engine knobs, with nothing saying detection is idle. |
| [`after.png`](after.png) | Restyle leads. The model path resolved itself from the models root, the plan state reads `Plan: global — the whole frame.  Detection: off (no target).`, and the engine knobs are folded into **Advanced**. |
| [`after-advanced.png`](after-advanced.png) | The same window with Advanced open: the standing rebuild warning, acceleration on `tensorrt`, and the three guidance controls issue #45 unhid - on the shipped setting, `none` at guidance 1.0, which draws no note at all. |
| [`after-selective.png`](after-selective.png) | A target and a style typed: `Plan: selective — every person.  Detection: on, 3 objects held, every 5 frames.` |
| [`after-model-choice.png`](after-model-choice.png) | Issue #38 step 6. **Base model** is a picker over the diffusers folders under the models root, not a path someone has to know, and the line under it says which engine that choice needs and whether it exists. |
| [`after-cfg.png`](after-cfg.png) | Issue #45. `CFG type` moved to `full`: the guidance scale came with it (1.4 - the rung spec 8.11's ladder measured best, and one the pipeline does not ignore), the line under the row says that `delta` is read by nothing under `full` and that this cfg type needs its own engine, and the line under the model has changed from `Engine: cached for sd-turbo-fp16 at 1 step.` to `No engine yet for sd-turbo-fp16 at 1 step, CFG full` - because `full` compiles a batch-2 UNet and the app has only a batch-1 one. |
| [`after-model-sd15.png`](after-model-sd15.png) | The same window with `sd-v1-5-fp16` picked. Choosing it turned LCM-LoRA on and moved the schedule to four steps - what SD 1.5 cannot render without - and the line reads `Engine: cached for sd-v1-5-fp16 at 4 steps.` |

`after.png` and the two beside it were taken with `SD_MODELS_DIR` pointed at the
main checkout's `models/`, because a fresh worktree has none and the point of the
shot is what a machine with a model does.

`after-model-sd15.png` and `after-cfg.png` were taken by driving
`_on_model_chosen` and `_apply_cfg_type` rather than by clicking, for the reason
the selective shot uses a stand-in payload: this loop has no hands. What they
photograph is the real handler's real effect on the real window.

`after-selective.png` was taken with `--selective`, which hands the window a
**stand-in** fps payload of the shape `detection.fps_payload` produces, so the
detection readout can be photographed without running the GPU. It is a picture of
the readout and not a measurement of anything - every measured number in this repo
comes from `bench/` and carries a hardware fingerprint.
