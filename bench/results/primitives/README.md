# Rendering-primitive comparison results

Written by `uv run python -m bench <case>`, never by hand - issue #5,
spec 8.2. One run measures *both* primitives on one case, interleaved frame by
frame, and writes one row per primitive plus the side-by-side clip a human
watches.

`flicker` is the mean absolute difference between consecutive outputs over the
pixels that were static in the source *and* painted by the primitive, in 0-255
units - lower is steadier. `t_index` is the denoise setting the case turned out
to need; higher is *less* denoise. `expresses` is whether the case's own
criterion was met at any rung of the ladder, and it is the half of the decision
no millisecond figure can answer.

Absolute ms/frame belongs to the GPU in the row (spec 7.4). Compare rows for the
A-against-B ratio, not for whether 30 FPS is met.

| finished (UTC) | case | primitive | option | GPU | clip | objects/frame | calls/frame | t_index | strength | ms/frame | flicker | expresses | cooldown | clock regime | ms/frame at basis clock | clip file | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-06T18:15:35Z | restyle-people | crop | A | NVIDIA GeForce RTX 3080 Laptop GPU | people.mp4 | 6.00 | 6.00 | 35 | 0.64 | 931.23 | 1.62 | yes | capped | unlocked | 461.26 @ 2100 MHz | restyle-people-20260906-181553Z-crop.mp4 | [restyle-people-20260906-181553Z.json](restyle-people-20260906-181553Z.json) |
| 2026-09-06T18:15:35Z | restyle-people | masked | B | NVIDIA GeForce RTX 3080 Laptop GPU | people.mp4 | 6.00 | 1.00 | 40 | 0.49 | 158.57 | 1.43 | yes | capped | unlocked | 78.55 @ 2100 MHz | restyle-people-20260906-181553Z-masked.mp4 | [restyle-people-20260906-181553Z.json](restyle-people-20260906-181553Z.json) |
| 2026-09-06T18:18:30Z | identity-dog | crop | A | NVIDIA GeForce RTX 3080 Laptop GPU | dog.mp4 | 1.00 | 1.00 | 20 | 0.92 | 81.75 | 17.38 | no | capped | unlocked | 46.40 @ 2100 MHz | identity-dog-20260906-181841Z-crop.mp4 | [identity-dog-20260906-181841Z.json](identity-dog-20260906-181841Z.json) |
| 2026-09-06T18:18:30Z | identity-dog | masked | B | NVIDIA GeForce RTX 3080 Laptop GPU | dog.mp4 | 1.00 | 1.00 | 25 | 0.85 | 85.73 | 19.50 | yes | capped | unlocked | 48.67 @ 2100 MHz | identity-dog-20260906-181841Z-masked.mp4 | [identity-dog-20260906-181841Z.json](identity-dog-20260906-181841Z.json) |
