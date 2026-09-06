# Benchmark results

Written by `uv run python -m bench <scenario>`, never by hand. Each row points at
the JSON file holding the full scenario config and hardware fingerprint.

Absolute ms/frame and VRAM figures belong to the GPU in the row - spec 7.4. Compare
rows across GPUs for curve shape and ranking only.

| finished (UTC) | scenario | GPU | accel | res | batch | steps | ms/frame | FPS | peak VRAM (MiB) | SM clock (MHz) | max temp (C) | cooldown | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-06T11:28:59Z | img2img-none-256x256-b1 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 256x256 | 1 | 1 | 47.96 | 20.8 | 2515 | 1675 | 68 | reached | [img2img-none-256x256-b1-20260906-112859Z.json](img2img-none-256x256-b1-20260906-112859Z.json) |
| 2026-09-06T11:29:18Z | img2img-none-384x384-b1 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 384x384 | 1 | 1 | 51.00 | 19.6 | 2549 | 1690 | 75 | reached | [img2img-none-384x384-b1-20260906-112918Z.json](img2img-none-384x384-b1-20260906-112918Z.json) |
| 2026-09-06T11:29:37Z | img2img-none-512x512-b1 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 1 | 1 | 79.09 | 12.6 | 2594 | 1290 | 79 | reached | [img2img-none-512x512-b1-20260906-112937Z.json](img2img-none-512x512-b1-20260906-112937Z.json) |
| 2026-09-06T11:30:02Z | img2img-none-512x512-b2 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 2 | 1 | 75.33 | 13.3 | 2695 | 1150 | 77 | reached | [img2img-none-512x512-b2-20260906-113002Z.json](img2img-none-512x512-b2-20260906-113002Z.json) |
| 2026-09-06T11:30:39Z | img2img-none-512x512-b4 | NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 4 | 1 | 72.80 | 13.7 | 2900 | 1141 | 80 | reached | [img2img-none-512x512-b4-20260906-113039Z.json](img2img-none-512x512-b4-20260906-113039Z.json) |
| 2026-09-06T11:31:44Z | img2img-tensorrt-512x512-b1 | NVIDIA GeForce RTX 3080 Laptop GPU | tensorrt | 512x512 | 1 | 1 | 54.78 | 18.3 | 2500 | 1342 | 79 | reached | [img2img-tensorrt-512x512-b1-20260906-113144Z.json](img2img-tensorrt-512x512-b1-20260906-113144Z.json) |
