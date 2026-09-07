# Detector cadence sweep results

Written by `uv run python -m bench <case> --detect-every-n N`, never by hand -
issue #23, spec 8.8. One row is one arm of the cadence sweep: the same
shipped path over the same clip under the same plan, with
`global.detect_every_n` the only field that moved. So no pixel the diffusion
produces differs between arms - what a higher cadence spends is the freshness
of the boxes, which the record's `staleness` block measures and
`python -m bench --cadence-report` tabulates.

These rows are deliberately not in `../selective/`: that directory is reduced
to the newest run per (case, GPU) for spec 8.8 and 7.4, and an arm at another
cadence sitting there would quietly become the figure those sections quote.

| finished (UTC) | case | GPU | clip | plan | regions/frame | ms/frame | +detect | FPS | flicker | background | gate | cooldown | clock regime | ms/frame at basis clock | clip file | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-07T11:06:35Z | selective-people-n2 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 4.98 | 24.25 | 35.45 | 28.2 | 1.49 | identical | pass | reached | unlocked | 20.81 @ 3150 MHz | - | [selective-people-n2-20260907-110635Z.json](selective-people-n2-20260907-110635Z.json) |
| 2026-09-07T11:06:50Z | selective-people-n3 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 23.73 | 31.28 | 32.0 | 1.49 | identical | pass | reached | unlocked | 20.45 @ 3150 MHz | - | [selective-people-n3-20260907-110650Z.json](selective-people-n3-20260907-110650Z.json) |
| 2026-09-07T11:07:05Z | selective-people-n5 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.19 | 24.21 | 28.28 | 35.4 | 1.49 | identical | pass | reached | unlocked | 20.87 @ 3150 MHz | - | [selective-people-n5-20260907-110705Z.json](selective-people-n5-20260907-110705Z.json) |
| 2026-09-07T11:07:20Z | selective-people-n8 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 4.81 | 24.15 | 26.48 | 37.8 | 1.49 | identical | pass | reached | unlocked | 20.82 @ 3150 MHz | - | [selective-people-n8-20260907-110720Z.json](selective-people-n8-20260907-110720Z.json) |
| 2026-09-07T11:08:31Z | selective-people-n2 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 4.98 | 24.52 | 36.09 | 27.7 | 1.49 | identical | pass | reached | unlocked | 21.13 @ 3150 MHz | [selective-people-n2-20260907-110831Z-comparison.mp4](selective-people-n2-20260907-110831Z-comparison.mp4) | [selective-people-n2-20260907-110831Z.json](selective-people-n2-20260907-110831Z.json) |
| 2026-09-07T11:08:51Z | selective-people-n3 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 24.03 | 31.34 | 31.9 | 1.49 | identical | pass | reached | unlocked | 20.71 @ 3150 MHz | [selective-people-n3-20260907-110851Z-comparison.mp4](selective-people-n3-20260907-110851Z-comparison.mp4) | [selective-people-n3-20260907-110851Z.json](selective-people-n3-20260907-110851Z.json) |
| 2026-09-07T11:09:10Z | selective-people-n5 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.19 | 23.87 | 28.15 | 35.5 | 1.49 | identical | pass | reached | unlocked | 20.58 @ 3150 MHz | [selective-people-n5-20260907-110910Z-comparison.mp4](selective-people-n5-20260907-110910Z-comparison.mp4) | [selective-people-n5-20260907-110910Z.json](selective-people-n5-20260907-110910Z.json) |
| 2026-09-07T11:09:25Z | selective-people-n8 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 4.81 | 23.92 | 26.40 | 37.9 | 1.49 | identical | pass | reached | unlocked | 20.61 @ 3150 MHz | [selective-people-n8-20260907-110925Z-comparison.mp4](selective-people-n8-20260907-110925Z-comparison.mp4) | [selective-people-n8-20260907-110925Z.json](selective-people-n8-20260907-110925Z.json) |
