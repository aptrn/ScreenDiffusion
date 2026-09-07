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
| 2026-09-06T20:25:15Z | selective-people | NVIDIA GeForce RTX 3080 Laptop GPU | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 55.06 | 74.73 | 13.4 | 1.49 | identical | pass | reached | unlocked | 43.97 @ 2100 MHz | [selective-people-20260906-202515Z-comparison.mp4](selective-people-20260906-202515Z-comparison.mp4) | [selective-people-20260906-202515Z.json](selective-people-20260906-202515Z.json) |
| 2026-09-07T10:09:43Z | selective-people | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 24.46 | 32.06 | 31.2 | 1.49 | identical | pass | reached | unlocked | 21.17 @ 3150 MHz | - | [selective-people-20260907-100943Z.json](selective-people-20260907-100943Z.json) |
| 2026-09-07T10:10:21Z | selective-people | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 24.65 | 32.04 | 31.2 | 1.49 | identical | pass | reached | unlocked | 21.27 @ 3150 MHz | - | [selective-people-20260907-101021Z.json](selective-people-20260907-101021Z.json) |
| 2026-09-07T10:10:36Z | selective-people | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 24.24 | 31.96 | 31.3 | 1.49 | identical | pass | reached | unlocked | 20.89 @ 3150 MHz | - | [selective-people-20260907-101036Z.json](selective-people-20260907-101036Z.json) |
| 2026-09-07T10:10:51Z | selective-people | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 24.39 | 31.88 | 31.4 | 1.49 | identical | pass | reached | unlocked | 21.03 @ 3150 MHz | - | [selective-people-20260907-101051Z.json](selective-people-20260907-101051Z.json) |
| 2026-09-07T10:11:06Z | selective-people | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 24.81 | 32.35 | 30.9 | 1.49 | identical | pass | reached | unlocked | 21.39 @ 3150 MHz | [selective-people-20260907-101106Z-comparison.mp4](selective-people-20260907-101106Z-comparison.mp4) | [selective-people-20260907-101106Z.json](selective-people-20260907-101106Z.json) |
