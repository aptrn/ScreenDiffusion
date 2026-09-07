# Selective render path results

Written by `uv run python -m bench <case>`, never by hand - issue #8, spec 8.8.
One run drives the *shipped* path over a committed clip: the detector on its own
thread, the tracker, the region scheduler, the 512x512 TensorRT engine and the
compositor, under the hardcoded priority-case plan the worker itself starts on.

`ms/frame` is the frame path alone; `+detect` adds what the detector amortises to
at the plan's cadence. `background` is the sharp criterion - every pixel outside
the rendered regions identical to the capture, on every frame. `flicker` is the
mean absolute difference between consecutive outputs over pixels static in the
source and painted in both, lower is steadier.

Absolute figures belong to the GPU in the row (spec 7.4), and rows from two GPUs
are two answers rather than one superseding the other. Whether 30 FPS is met is
a deploy-hardware question: `python -m bench --portability-report` answers it
from these rows, and spec 7.4 carries the answer.

| finished (UTC) | case | GPU | clip | plan | regions/frame | ms/frame | +detect | FPS | flicker | background | gate | cooldown | clock regime | ms/frame at basis clock | clip file | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-07T18:38:27Z | selective-people-sd15 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.19 | 34.49 | 40.99 | 24.4 | 2.31 | identical | pass | reached | unlocked | 29.37 @ 3150 MHz | [selective-people-sd15-20260907-183827Z-comparison.mp4](selective-people-sd15-20260907-183827Z-comparison.mp4) | [selective-people-sd15-20260907-183827Z.json](selective-people-sd15-20260907-183827Z.json) |
| 2026-09-07T18:39:23Z | selective-people-sd-turbo | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.19 | 17.81 | 21.87 | 45.7 | 1.49 | identical | pass | reached | unlocked | 14.53 @ 3150 MHz | [selective-people-sd-turbo-20260907-183923Z-comparison.mp4](selective-people-sd-turbo-20260907-183923Z-comparison.mp4) | [selective-people-sd-turbo-20260907-183923Z.json](selective-people-sd-turbo-20260907-183923Z.json) |
