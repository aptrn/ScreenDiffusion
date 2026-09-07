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
| 2026-09-07T18:14:14Z | img2img-none-512x512-b1-s1 | NVIDIA GeForce RTX 4090 | none | 512x512 | 1 | 1 | 26.02 | 38.4 | 2593 | 2715 | 57 | reached | [img2img-none-512x512-b1-s1-20260907-181414Z.json](img2img-none-512x512-b1-s1-20260907-181414Z.json) | unlocked | 22.43 @ 3150 MHz |
| 2026-09-07T18:14:24Z | img2img-none-512x512-b1-s2 | NVIDIA GeForce RTX 4090 | none | 512x512 | 1 | 2 | 32.73 | 30.6 | 2658 | 2708 | 59 | reached | [img2img-none-512x512-b1-s2-20260907-181424Z.json](img2img-none-512x512-b1-s2-20260907-181424Z.json) | unlocked | 28.13 @ 3150 MHz |
| 2026-09-07T18:14:37Z | img2img-none-512x512-b1-s4 | NVIDIA GeForce RTX 4090 | none | 512x512 | 1 | 4 | 48.63 | 20.6 | 2757 | 2705 | 62 | reached | [img2img-none-512x512-b1-s4-20260907-181437Z.json](img2img-none-512x512-b1-s4-20260907-181437Z.json) | unlocked | 41.74 @ 3150 MHz |
| 2026-09-07T18:14:48Z | img2img-tensorrt-512x512-b1-s1 | NVIDIA GeForce RTX 4090 | tensorrt | 512x512 | 1 | 1 | 20.67 | 48.4 | 2500 | 2662 | 55 | reached | [img2img-tensorrt-512x512-b1-s1-20260907-181448Z.json](img2img-tensorrt-512x512-b1-s1-20260907-181448Z.json) | unlocked | 17.47 @ 3150 MHz |
| 2026-09-07T18:18:08Z | img2img-none-512x512-b1-sd15-s4 | NVIDIA GeForce RTX 4090 | none | 512x512 | 1 | 4 | 49.97 | 20.0 | 2929 | 2715 | 56 | reached | [img2img-none-512x512-b1-sd15-s4-20260907-181808Z.json](img2img-none-512x512-b1-sd15-s4-20260907-181808Z.json) | unlocked | 43.07 @ 3150 MHz |
| 2026-09-07T18:23:41Z | img2img-tensorrt-512x512-b1-sd15-s4 | NVIDIA GeForce RTX 4090 | tensorrt | 512x512 | 1 | 4 | 33.88 | 29.5 | 2673 | 2722 | 46 | reached | [img2img-tensorrt-512x512-b1-sd15-s4-20260907-182341Z.json](img2img-tensorrt-512x512-b1-sd15-s4-20260907-182341Z.json) | unlocked | 29.28 @ 3150 MHz |
| 2026-09-07T18:28:48Z | img2img-tensorrt-512x512-b1-sd15-loving-vincent-s4 | NVIDIA GeForce RTX 4090 | tensorrt | 512x512 | 1 | 4 | 33.57 | 29.8 | 2672 | 2722 | 47 | reached | [img2img-tensorrt-512x512-b1-sd15-loving-vincent-s4-20260907-182848Z.json](img2img-tensorrt-512x512-b1-sd15-loving-vincent-s4-20260907-182848Z.json) | unlocked | 29.02 @ 3150 MHz |
| 2026-09-07T18:30:15Z | img2img-none-512x512-b1-sd15-loving-vincent-s4 | NVIDIA GeForce RTX 4090 | none | 512x512 | 1 | 4 | 52.02 | 19.2 | 2930 | 2720 | 48 | reached | [img2img-none-512x512-b1-sd15-loving-vincent-s4-20260907-183015Z.json](img2img-none-512x512-b1-sd15-loving-vincent-s4-20260907-183015Z.json) | unlocked | 44.90 @ 3150 MHz |
