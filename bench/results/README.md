# Benchmark results

Written by `uv run python -m bench <scenario>`, never by hand. Each row points at
the JSON file holding the full scenario config and hardware fingerprint.

Absolute ms/frame and VRAM figures belong to the GPU in the row - spec 7.4. Compare
rows across GPUs for curve shape and ranking only.

`clock regime` says whether the GPU clocks were locked while the row was measured.
Unlocked, the last column carries a first-order estimate of the same work at one
clock - an estimate, not a measurement. A row with neither cell was written before
the field existed, and was measured unlocked (issue #13).

| finished (UTC) | scenario | GPU | accel | res | batch | steps | ms/frame | FPS | peak VRAM (MiB) | SM clock (MHz) | max temp (C) | cooldown | file | clock regime | ms/frame at basis clock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-06T11:28:59Z | img2img-none-256x256-b1 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 256x256 | 1 | 1 | 47.96 | 20.8 | 2515 | 1675 | 68 | reached | [img2img-none-256x256-b1-20260906-112859Z.json](img2img-none-256x256-b1-20260906-112859Z.json) |
| 2026-09-06T11:29:18Z | img2img-none-384x384-b1 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 384x384 | 1 | 1 | 51.00 | 19.6 | 2549 | 1690 | 75 | reached | [img2img-none-384x384-b1-20260906-112918Z.json](img2img-none-384x384-b1-20260906-112918Z.json) |
| 2026-09-06T11:29:37Z | img2img-none-512x512-b1 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 1 | 1 | 79.09 | 12.6 | 2594 | 1290 | 79 | reached | [img2img-none-512x512-b1-20260906-112937Z.json](img2img-none-512x512-b1-20260906-112937Z.json) |
| 2026-09-06T11:30:02Z | img2img-none-512x512-b2 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 2 | 1 | 75.33 | 13.3 | 2695 | 1150 | 77 | reached | [img2img-none-512x512-b2-20260906-113002Z.json](img2img-none-512x512-b2-20260906-113002Z.json) |
| 2026-09-06T11:30:39Z | img2img-none-512x512-b4 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 4 | 1 | 72.80 | 13.7 | 2900 | 1141 | 80 | reached | [img2img-none-512x512-b4-20260906-113039Z.json](img2img-none-512x512-b4-20260906-113039Z.json) |
| 2026-09-06T11:31:44Z | img2img-tensorrt-512x512-b1 | NVIDIA GeForce RTX 3080 Laptop GPU | tensorrt | 512x512 | 1 | 1 | 54.78 | 18.3 | 2500 | 1342 | 79 | reached | [img2img-tensorrt-512x512-b1-20260906-113144Z.json](img2img-tensorrt-512x512-b1-20260906-113144Z.json) |
| 2026-09-06T11:45:24Z | img2img-none-256x256-b1 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 256x256 | 1 | 1 | 46.97 | 21.3 | 2516 | 1775 | 64 | reached | [img2img-none-256x256-b1-20260906-114524Z.json](img2img-none-256x256-b1-20260906-114524Z.json) |
| 2026-09-06T11:45:35Z | img2img-none-256x256-b2 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 256x256 | 2 | 1 | 25.51 | 39.2 | 2541 | 1785 | 73 | reached | [img2img-none-256x256-b2-20260906-114535Z.json](img2img-none-256x256-b2-20260906-114535Z.json) |
| 2026-09-06T11:45:48Z | img2img-none-256x256-b4 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 256x256 | 4 | 1 | 17.95 | 55.7 | 2594 | 1496 | 77 | reached | [img2img-none-256x256-b4-20260906-114548Z.json](img2img-none-256x256-b4-20260906-114548Z.json) |
| 2026-09-06T11:46:08Z | img2img-none-256x256-b8 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 256x256 | 8 | 1 | 17.44 | 57.4 | 2696 | 1211 | 76 | reached | [img2img-none-256x256-b8-20260906-114608Z.json](img2img-none-256x256-b8-20260906-114608Z.json) |
| 2026-09-06T11:46:27Z | img2img-none-384x384-b1 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 384x384 | 1 | 1 | 60.08 | 16.6 | 2548 | 1519 | 75 | reached | [img2img-none-384x384-b1-20260906-114627Z.json](img2img-none-384x384-b1-20260906-114627Z.json) |
| 2026-09-06T11:46:47Z | img2img-none-384x384-b2 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 384x384 | 2 | 1 | 42.89 | 23.3 | 2607 | 1317 | 78 | reached | [img2img-none-384x384-b2-20260906-114647Z.json](img2img-none-384x384-b2-20260906-114647Z.json) |
| 2026-09-06T11:47:10Z | img2img-none-384x384-b4 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 384x384 | 4 | 1 | 40.21 | 24.9 | 2720 | 1130 | 77 | reached | [img2img-none-384x384-b4-20260906-114710Z.json](img2img-none-384x384-b4-20260906-114710Z.json) |
| 2026-09-06T11:47:40Z | img2img-none-384x384-b8 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 384x384 | 8 | 1 | 38.07 | 26.3 | 2951 | 1137 | 80 | reached | [img2img-none-384x384-b8-20260906-114740Z.json](img2img-none-384x384-b8-20260906-114740Z.json) |
| 2026-09-06T11:48:04Z | img2img-none-512x512-b1 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 1 | 1 | 82.30 | 12.2 | 2593 | 1239 | 75 | reached | [img2img-none-512x512-b1-20260906-114804Z.json](img2img-none-512x512-b1-20260906-114804Z.json) |
| 2026-09-06T11:48:46Z | img2img-none-512x512-b2 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 2 | 1 | 72.89 | 13.7 | 2695 | 1182 | 76 | reached | [img2img-none-512x512-b2-20260906-114846Z.json](img2img-none-512x512-b2-20260906-114846Z.json) |
| 2026-09-06T11:49:13Z | img2img-none-512x512-b4 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 4 | 1 | 70.70 | 14.1 | 2900 | 1112 | 80 | reached | [img2img-none-512x512-b4-20260906-114913Z.json](img2img-none-512x512-b4-20260906-114913Z.json) |
| 2026-09-06T11:50:00Z | img2img-none-512x512-b8 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 8 | 1 | 95.88 | 10.4 | 3310 | 787 | 84 | reached | [img2img-none-512x512-b8-20260906-115000Z.json](img2img-none-512x512-b8-20260906-115000Z.json) |
| 2026-09-06T12:13:35Z | img2img-tensorrt-512x512-b1 | NVIDIA GeForce RTX 3080 Laptop GPU | tensorrt | 512x512 | 1 | 1 | 58.37 | 17.1 | 2500 | 960 | 78 | reached | [img2img-tensorrt-512x512-b1-20260906-121335Z.json](img2img-tensorrt-512x512-b1-20260906-121335Z.json) |
| 2026-09-06T12:40:21Z | img2img-tensorrt-512x512-b2 | NVIDIA GeForce RTX 3080 Laptop GPU | tensorrt | 512x512 | 2 | 1 | 54.96 | 18.2 | 2511 | 1172 | 80 | reached | [img2img-tensorrt-512x512-b2-20260906-124021Z.json](img2img-tensorrt-512x512-b2-20260906-124021Z.json) |
| 2026-09-06T12:59:00Z | img2img-tensorrt-512x512-b4 | NVIDIA GeForce RTX 3080 Laptop GPU | tensorrt | 512x512 | 4 | 1 | 67.44 | 14.8 | 2537 | 1348 | 85 | capped | [img2img-tensorrt-512x512-b4-20260906-125900Z.json](img2img-tensorrt-512x512-b4-20260906-125900Z.json) |
