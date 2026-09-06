# Detector benchmark results

Written by `uv run python -m bench <detector>`, never by hand - issue #4,
spec 8.1. Each row points at the JSON file holding the full detector config,
the hardware fingerprint, the vocabulary-change cost and the per-concept
detection evidence.

`amortised` is ms/detect divided by the detect cadence in the same row: spec 7.1
gives detection 4-8 ms amortised at one detect every 3rd frame, and that is
the figure `fits` judges. Absolute milliseconds and VRAM belong to the GPU in
the row (spec 7.4); compare rows for ranking, not for whether 30 FPS is met.

`with diffusion (MiB)` is `nvidia-smi` memory in use with the diffusion engine
*and* the detector resident - the only figure that answers whether they fit
together. `vocab change` is the cold-path text encode, paid when the user edits
the prompt and never on the frame path.

| finished (UTC) | detector | GPU | role | vocabulary | input | ms/detect | p95 ms | cadence | amortised ms/frame | fits 4-8 ms | torch peak (MiB) | with diffusion (MiB) | vocab change (ms) | concepts resolved | cooldown | clock regime | ms/detect at basis clock | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-06T14:12:41Z | yolo-world-s-640 | NVIDIA GeForce RTX 3080 Laptop GPU | candidate | open | 640x640 | 14.32 | 16.98 | 1 per 3 | 4.77 | yes | 3577 | 7182 | 16.2 | 3/3 | reached | unlocked | 11.00 @ 2100 MHz | [yolo-world-s-640-20260906-141241Z.json](yolo-world-s-640-20260906-141241Z.json) |
| 2026-09-06T14:15:03Z | yolov8n-640 | NVIDIA GeForce RTX 3080 Laptop GPU | speed floor | 80 COCO classes | 640x640 | 12.76 | 16.50 | 1 per 3 | 4.25 | yes | 2541 | 6157 | n/a | 1/3 | capped | unlocked | 9.56 @ 2100 MHz | [yolov8n-640-20260906-141503Z.json](yolov8n-640-20260906-141503Z.json) |
