# Temporal stability sweep results

Written by `uv run python -m bench <case> --seed-policy P --output-ema E`,
never by hand - issue #32, spec 8.5. One row is one arm of the temporal-
stability sweep: the same shipped path over the same clip under the same
plan, with the seed policy and the output EMA the only fields that moved.

`flicker` is what an arm was run to move, and it is only half the reading -
`python -m bench --stability-report` puts it beside the responsiveness
figure and the visible-change figure net of a control, because an arm that
lowers flicker by rendering less is disqualified rather than recommended.

These rows are deliberately not in `../selective/`: that directory is
reduced to the newest run per (case, GPU) for spec 8.8 and 7.4, and an arm
at another setting sitting there would quietly become the figure those
sections quote.

| finished (UTC) | case | GPU | clip | plan | regions/frame | ms/frame | +detect | FPS | flicker | background | gate | cooldown | clock regime | ms/frame at basis clock | clip file | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-07T13:32:59Z | selective-people-fixed-ema00 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 16.76 | 23.67 | 42.3 | 1.49 | identical | pass | reached | unlocked | 14.48 @ 3150 MHz | [selective-people-fixed-ema00-20260907-133259Z-comparison.mp4](selective-people-fixed-ema00-20260907-133259Z-comparison.mp4) | [selective-people-fixed-ema00-20260907-133259Z.json](selective-people-fixed-ema00-20260907-133259Z.json) |
| 2026-09-07T13:33:14Z | selective-people-per_track-ema00 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 17.26 | 24.06 | 41.6 | 1.74 | identical | pass | reached | unlocked | 14.91 @ 3150 MHz | [selective-people-per_track-ema00-20260907-133314Z-comparison.mp4](selective-people-per_track-ema00-20260907-133314Z-comparison.mp4) | [selective-people-per_track-ema00-20260907-133314Z.json](selective-people-per_track-ema00-20260907-133314Z.json) |
| 2026-09-07T13:33:30Z | selective-people-random-ema00 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 17.43 | 24.12 | 41.5 | 8.72 | identical | pass | reached | unlocked | 15.05 @ 3150 MHz | [selective-people-random-ema00-20260907-133330Z-comparison.mp4](selective-people-random-ema00-20260907-133330Z-comparison.mp4) | [selective-people-random-ema00-20260907-133330Z.json](selective-people-random-ema00-20260907-133330Z.json) |
| 2026-09-07T13:33:46Z | selective-people-fixed-ema25 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.02 | 16.81 | 23.88 | 41.9 | 1.28 | identical | pass | reached | unlocked | 14.57 @ 3150 MHz | [selective-people-fixed-ema25-20260907-133346Z-comparison.mp4](selective-people-fixed-ema25-20260907-133346Z-comparison.mp4) | [selective-people-fixed-ema25-20260907-133346Z.json](selective-people-fixed-ema25-20260907-133346Z.json) |
| 2026-09-07T13:34:02Z | selective-people-fixed-ema50 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 17.03 | 24.00 | 41.7 | 0.88 | identical | pass | reached | unlocked | 14.76 @ 3150 MHz | [selective-people-fixed-ema50-20260907-133402Z-comparison.mp4](selective-people-fixed-ema50-20260907-133402Z-comparison.mp4) | [selective-people-fixed-ema50-20260907-133402Z.json](selective-people-fixed-ema50-20260907-133402Z.json) |
| 2026-09-07T13:34:17Z | selective-people-fixed-ema75 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.02 | 16.82 | 23.84 | 42.0 | 0.63 | identical | pass | reached | unlocked | 14.58 @ 3150 MHz | [selective-people-fixed-ema75-20260907-133417Z-comparison.mp4](selective-people-fixed-ema75-20260907-133417Z-comparison.mp4) | [selective-people-fixed-ema75-20260907-133417Z.json](selective-people-fixed-ema75-20260907-133417Z.json) |
| 2026-09-07T13:34:33Z | selective-people-fixed-ema00 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 16.90 | 24.25 | 41.2 | 1.49 | identical | pass | reached | unlocked | 14.61 @ 3150 MHz | - | [selective-people-fixed-ema00-20260907-133433Z.json](selective-people-fixed-ema00-20260907-133433Z.json) |
| 2026-09-07T13:34:49Z | selective-people-per_track-ema00 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 17.46 | 24.27 | 41.2 | 1.74 | identical | pass | reached | unlocked | 15.09 @ 3150 MHz | - | [selective-people-per_track-ema00-20260907-133449Z.json](selective-people-per_track-ema00-20260907-133449Z.json) |
| 2026-09-07T13:35:04Z | selective-people-random-ema00 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 16.97 | 23.89 | 41.9 | 8.72 | identical | pass | reached | unlocked | 14.67 @ 3150 MHz | - | [selective-people-random-ema00-20260907-133504Z.json](selective-people-random-ema00-20260907-133504Z.json) |
| 2026-09-07T13:35:20Z | selective-people-fixed-ema25 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 16.94 | 24.21 | 41.3 | 1.28 | identical | pass | reached | unlocked | 14.68 @ 3150 MHz | - | [selective-people-fixed-ema25-20260907-133520Z.json](selective-people-fixed-ema25-20260907-133520Z.json) |
| 2026-09-07T13:35:35Z | selective-people-fixed-ema50 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 16.67 | 23.59 | 42.4 | 0.88 | identical | pass | reached | unlocked | 14.45 @ 3150 MHz | - | [selective-people-fixed-ema50-20260907-133535Z.json](selective-people-fixed-ema50-20260907-133535Z.json) |
| 2026-09-07T13:35:50Z | selective-people-fixed-ema75 | NVIDIA GeForce RTX 4090 | people.mp4 1280x720 | person / lower_half / t40 | 5.04 | 16.92 | 23.90 | 41.8 | 0.63 | identical | pass | reached | unlocked | 14.63 @ 3150 MHz | - | [selective-people-fixed-ema75-20260907-133550Z.json](selective-people-fixed-ema75-20260907-133550Z.json) |
