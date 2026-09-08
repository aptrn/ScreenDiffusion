# Classifier-free guidance: does the prompt land?

Written by `uv run python -m bench cfg-dog`, never by hand. Issue #45: the app
ships with CFG switched off and has never measured anything else, so the weak
prompt adherence has never been shown to be the checkpoint rather than a
setting.

`adherence` is the fraction of rendered frames the open-vocabulary detector
reads back as the concept the prompt asked for - spec 8.2's identity probe -
and `net drift` is how far the region moved from the capture, net of its own
round trip. The two are the trade, and an arm is only worth something when
the first one moved.

| finished (UTC) | case | GPU | arm | cfg | guidance | delta | ms/frame | adherence | conf | net drift | UNet batch | background | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | none | none | 1 | n/a | 22.74 | 35% | 0.29 | 15.04 | 1 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | self-g10-d10 | self | 1.05 | 1 | 24.19 | 35% | 0.29 | 15.97 | 1 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | self-g11-d10 | self | 1.1 | 1 | 24.91 | 33% | 0.27 | 17.08 | 1 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | self-g12-d10 | self | 1.2 | 1 | 25.12 | 25% | 0.21 | 20.21 | 1 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | self-g14-d10 | self | 1.4 | 1 | 25.10 | 2% | 0.02 | 34.42 | 1 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | self-g20-d10 | self | 2 | 1 | 23.00 | 0% | 0.00 | 90.37 | 1 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | self-g30-d10 | self | 3 | 1 | 24.88 | 0% | 0.00 | 108.29 | 1 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | self-g14-d05 | self | 1.4 | 0.5 | 24.99 | 33% | 0.28 | 20.24 | 1 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | initialize-g10-d10 | initialize | 1.05 | 1 | 28.95 | 35% | 0.29 | 15.11 | 2 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | initialize-g11-d10 | initialize | 1.1 | 1 | 30.83 | 40% | 0.32 | 15.18 | 2 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | initialize-g12-d10 | initialize | 1.2 | 1 | 30.71 | 44% | 0.34 | 15.34 | 2 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | initialize-g14-d10 | initialize | 1.4 | 1 | 29.61 | 44% | 0.35 | 15.72 | 2 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | initialize-g20-d10 | initialize | 2 | 1 | 30.50 | 40% | 0.31 | 17.25 | 2 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | initialize-g30-d10 | initialize | 3 | 1 | 30.76 | 10% | 0.08 | 20.93 | 2 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | initialize-g14-d05 | initialize | 1.4 | 0.5 | 31.25 | 52% | 0.43 | 16.06 | 2 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | full-g10 | full | 1.05 | n/a | 30.45 | 35% | 0.29 | 15.11 | 2 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | full-g11 | full | 1.1 | n/a | 29.72 | 40% | 0.32 | 15.18 | 2 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | full-g12 | full | 1.2 | n/a | 30.71 | 44% | 0.34 | 15.34 | 2 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | full-g14 | full | 1.4 | n/a | 30.30 | 44% | 0.35 | 15.72 | 2 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | full-g20 | full | 2 | n/a | 30.06 | 40% | 0.31 | 17.25 | 2 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:19:16Z | cfg-dog | NVIDIA GeForce RTX 4090 | full-g30 | full | 3 | n/a | 29.63 | 10% | 0.08 | 20.93 | 2 | identical | [cfg-dog-20260908-121916Z.json](cfg-dog-20260908-121916Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | none | none | 1 | n/a | 47.56 | 19% | 0.16 | 25.86 | 4 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | self-g10-d10 | self | 1.05 | 1 | 48.16 | 21% | 0.17 | 27.65 | 4 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | self-g11-d10 | self | 1.1 | 1 | 48.30 | 19% | 0.16 | 30.29 | 4 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | self-g12-d10 | self | 1.2 | 1 | 48.06 | 10% | 0.08 | 38.84 | 4 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | self-g14-d10 | self | 1.4 | 1 | 48.23 | 0% | 0.00 | 62.21 | 4 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | self-g20-d10 | self | 2 | 1 | 47.71 | 0% | 0.00 | 87.03 | 4 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | self-g30-d10 | self | 3 | 1 | 47.64 | 0% | 0.00 | 93.09 | 4 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | self-g14-d05 | self | 1.4 | 0.5 | 48.19 | 12% | 0.11 | 35.52 | 4 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | initialize-g10-d10 | initialize | 1.05 | 1 | 58.19 | 23% | 0.18 | 26.59 | 5 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | initialize-g11-d10 | initialize | 1.1 | 1 | 58.39 | 25% | 0.19 | 27.57 | 5 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | initialize-g12-d10 | initialize | 1.2 | 1 | 57.77 | 15% | 0.12 | 30.69 | 5 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | initialize-g14-d10 | initialize | 1.4 | 1 | 57.83 | 4% | 0.03 | 42.27 | 5 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | initialize-g20-d10 | initialize | 2 | 1 | 57.67 | 0% | 0.00 | 74.56 | 5 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | initialize-g30-d10 | initialize | 3 | 1 | 57.98 | 0% | 0.00 | 90.64 | 5 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | initialize-g14-d05 | initialize | 1.4 | 0.5 | 57.64 | 35% | 0.27 | 29.05 | 5 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | full-g10 | full | 1.05 | n/a | 84.05 | 23% | 0.19 | 25.87 | 8 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | full-g11 | full | 1.1 | n/a | 84.12 | 23% | 0.19 | 25.89 | 8 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | full-g12 | full | 1.2 | n/a | 83.40 | 31% | 0.25 | 25.94 | 8 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | full-g14 | full | 1.4 | n/a | 83.56 | 40% | 0.31 | 26.04 | 8 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | full-g20 | full | 2 | n/a | 83.68 | 52% | 0.42 | 26.40 | 8 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
| 2026-09-08T12:22:05Z | cfg-dog-sd15 | NVIDIA GeForce RTX 4090 | full-g30 | full | 3 | n/a | 83.67 | 58% | 0.47 | 27.30 | 8 | identical | [cfg-dog-sd15-20260908-122205Z.json](cfg-dog-sd15-20260908-122205Z.json) |
