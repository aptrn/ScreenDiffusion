# Denoising steps: what one buys, and how it is paid for

Written by `uv run python -m bench steps-dog`, never by hand. Issue #46: the
same clip rendered at 1 / 2 / 4 / 8 denoising steps on both routes - the
pre-built ladder (`use_denoising_batch` on, one engine per rung) and the
unbatched one (off, one engine for every rung).

`adherence` is the fraction of rendered frames the detector reads back as what
the prompt asked for; `swap s` is how long that arm took to become the live
engine, which is a *load* wherever `cached` says yes.

| finished (UTC) | case | GPU | arm | route | steps | UNet batch | cached | swap s | ms/frame | adherence | net change | flicker | response | background | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-08T13:36:04Z | steps-dog | NVIDIA GeForce RTX 4090 | s1 | pre-built ladder | 1 | 1 | yes | 3.35 | 67.40 | 33% | 15.04 | 5.35 | 18.98 | 48/48 | [steps-dog-20260908-133604Z.json](steps-dog-20260908-133604Z.json) |
| 2026-09-08T13:36:04Z | steps-dog | NVIDIA GeForce RTX 4090 | s2 | pre-built ladder | 2 | 2 | yes | 3.59 | 69.90 | 44% | 19.04 | 6.48 | 18.08 | 48/48 | [steps-dog-20260908-133604Z.json](steps-dog-20260908-133604Z.json) |
| 2026-09-08T13:36:04Z | steps-dog | NVIDIA GeForce RTX 4090 | s4 | pre-built ladder | 4 | 4 | yes | 3.60 | 107.47 | 52% | 24.23 | 7.68 | 15.46 | 48/48 | [steps-dog-20260908-133604Z.json](steps-dog-20260908-133604Z.json) |
| 2026-09-08T13:36:04Z | steps-dog | NVIDIA GeForce RTX 4090 | s8 | pre-built ladder | 8 | 8 | yes | 3.63 | 188.80 | 77% | 28.77 | 8.29 | 14.99 | 48/48 | [steps-dog-20260908-133604Z.json](steps-dog-20260908-133604Z.json) |
| 2026-09-08T13:36:04Z | steps-dog | NVIDIA GeForce RTX 4090 | s1-nobatch | unbatched | 1 | 1 | yes | 3.54 | 69.09 | 33% | 15.04 | 5.35 | 18.98 | 48/48 | [steps-dog-20260908-133604Z.json](steps-dog-20260908-133604Z.json) |
| 2026-09-08T13:36:04Z | steps-dog | NVIDIA GeForce RTX 4090 | s2-nobatch | unbatched | 2 | 1 | yes | 3.59 | 107.29 | 31% | 17.24 | 6.39 | 20.33 | 48/48 | [steps-dog-20260908-133604Z.json](steps-dog-20260908-133604Z.json) |
| 2026-09-08T13:36:04Z | steps-dog | NVIDIA GeForce RTX 4090 | s4-nobatch | unbatched | 4 | 1 | yes | 3.46 | 210.84 | 12% | 35.04 | 15.99 | 31.26 | 48/48 | [steps-dog-20260908-133604Z.json](steps-dog-20260908-133604Z.json) |
| 2026-09-08T13:36:04Z | steps-dog | NVIDIA GeForce RTX 4090 | s8-nobatch | unbatched | 8 | 1 | yes | 3.49 | 384.60 | 0% | 69.84 | 26.74 | 41.62 | 48/48 | [steps-dog-20260908-133604Z.json](steps-dog-20260908-133604Z.json) |
