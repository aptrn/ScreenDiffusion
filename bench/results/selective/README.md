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

Absolute figures belong to the GPU in the row (spec 7.4). 30 FPS is not a gate
here and cannot be judged on this laptop at all.

| finished (UTC) | case | GPU | clip | plan | regions/frame | ms/frame | +detect | FPS | flicker | background | gate | cooldown | clock regime | ms/frame at basis clock | clip file | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-06T20:25:15Z | selective-people | NVIDIA GeForce RTX 3080 Laptop GPU | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 55.06 | 74.73 | 13.4 | 1.49 | identical | pass | reached | unlocked | 43.97 @ 2100 MHz | [selective-people-20260906-202515Z-comparison.mp4](selective-people-20260906-202515Z-comparison.mp4) | [selective-people-20260906-202515Z.json](selective-people-20260906-202515Z.json) |
