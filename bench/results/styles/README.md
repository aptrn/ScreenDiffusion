# Style LoRAs on SD 1.5

Written by `uv run python -m bench style-sd15`, never by hand. Issue #38 steps
3-5: whether a style LoRA loads on the SD 1.5 arm and whether it visibly changes
the output, with the format it is in recorded either way.

`net change` is the render against the source with the resize control
subtracted; `vs base` is the arm against the same arm with no LoRA fused, which
is the number that says the LoRA did anything.

| finished (UTC) | case | GPU | arm | format | loaded | ms/frame | net change | vs base | flicker | verdict | file |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-07T18:31:58Z | style-sd15 | NVIDIA GeForce RTX 4090 | base | no LoRA | yes | 53.33 | 14.58 | 0.00 | 2.51 | control | [style-sd15-20260907-183158Z.json](style-sd15-20260907-183158Z.json) |
| 2026-09-07T18:31:58Z | style-sd15 | NVIDIA GeForce RTX 4090 | loving-vincent | LoRA (kohya, linear only) | yes | 46.69 | 19.74 | 19.23 | 1.31 | pass | [style-sd15-20260907-183158Z.json](style-sd15-20260907-183158Z.json) |
| 2026-09-07T18:31:58Z | style-sd15 | NVIDIA GeForce RTX 4090 | illusion-pattern | LoRA (kohya, linear only) | yes | 46.72 | 15.17 | 5.32 | 2.55 | pass | [style-sd15-20260907-183158Z.json](style-sd15-20260907-183158Z.json) |
| 2026-09-07T18:31:58Z | style-sd15 | NVIDIA GeForce RTX 4090 | locon-probe | LoCon / LyCORIS (198 conv keys) | NO | 0.00 | 0.00 | 0.00 | 0.00 | FAIL | [style-sd15-20260907-183158Z.json](style-sd15-20260907-183158Z.json) |
