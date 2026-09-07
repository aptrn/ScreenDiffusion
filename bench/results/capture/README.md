# Capture geometry results

Written by `uv run python -m bench <capture case>`, never by hand - issue
#39, spec 8.2. One run renders the same committed clip at every capture
geometry under both rendering primitives at K=1, through the shipped
compositor and the one cached 512x512 engine.

`frame path` is the worker's own cost - the resize onto the canvas, the
diffusion call, the composite including its one device-to-host copy, and
handing the frame to the GUI process. `object px` is how many canvas pixels
across the rendered region actually got, which is what the crop primitive
buys and the only reason to pay for a bigger capture.

Absolute figures belong to the GPU in the row (spec 7.4).

| finished (UTC) | case | GPU | arm | capture | frame path (ms) | FPS | object px | net change | flicker | background | file |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-07T17:29:29Z | capture-people | NVIDIA GeForce RTX 4090 | masked-512x512 | 512x512 | 15.97 | 62.6 | 123 | 9.1 | 1.07 | identical | capture-people-20260907-172929Z.json |
| 2026-09-07T17:29:29Z | capture-people | NVIDIA GeForce RTX 4090 | crop-512x512 | 512x512 | 16.10 | 62.1 | 512 | 8.8 | 1.30 | identical | capture-people-20260907-172929Z.json |
| 2026-09-07T17:29:29Z | capture-people | NVIDIA GeForce RTX 4090 | masked-1280x720 | 1280x720 | 18.54 | 53.9 | 69 | 10.0 | 1.18 | identical | capture-people-20260907-172929Z.json |
| 2026-09-07T17:29:29Z | capture-people | NVIDIA GeForce RTX 4090 | crop-1280x720 | 1280x720 | 17.34 | 57.7 | 512 | 9.8 | 1.62 | identical | capture-people-20260907-172929Z.json |
| 2026-09-07T17:29:29Z | capture-people | NVIDIA GeForce RTX 4090 | masked-1920x1080 | 1920x1080 | 20.60 | 48.5 | 69 | 10.3 | 1.18 | identical | capture-people-20260907-172929Z.json |
| 2026-09-07T17:29:29Z | capture-people | NVIDIA GeForce RTX 4090 | crop-1920x1080 | 1920x1080 | 20.63 | 48.5 | 512 | 9.0 | 1.31 | identical | capture-people-20260907-172929Z.json |
| 2026-09-07T17:30:54Z | capture-dog | NVIDIA GeForce RTX 4090 | masked-512x512 | 512x512 | 16.65 | 60.1 | 156 | 15.8 | 5.82 | identical | capture-dog-20260907-173054Z.json |
| 2026-09-07T17:30:54Z | capture-dog | NVIDIA GeForce RTX 4090 | crop-512x512 | 512x512 | 16.58 | 60.3 | 512 | 15.9 | 9.72 | identical | capture-dog-20260907-173054Z.json |
| 2026-09-07T17:30:54Z | capture-dog | NVIDIA GeForce RTX 4090 | masked-1280x720 | 1280x720 | 19.91 | 50.2 | 99 | 28.7 | 16.53 | identical | capture-dog-20260907-173054Z.json |
| 2026-09-07T17:30:54Z | capture-dog | NVIDIA GeForce RTX 4090 | crop-1280x720 | 1280x720 | 18.25 | 54.8 | 512 | 22.6 | 14.40 | identical | capture-dog-20260907-173054Z.json |
| 2026-09-07T17:30:54Z | capture-dog | NVIDIA GeForce RTX 4090 | masked-1920x1080 | 1920x1080 | 21.82 | 45.8 | 99 | 29.0 | 16.74 | identical | capture-dog-20260907-173054Z.json |
| 2026-09-07T17:30:54Z | capture-dog | NVIDIA GeForce RTX 4090 | crop-1920x1080 | 1920x1080 | 21.75 | 46.0 | 512 | 23.4 | 14.79 | identical | capture-dog-20260907-173054Z.json |
