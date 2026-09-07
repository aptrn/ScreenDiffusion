# Plan swap results

Written by `uv run python -m bench <swap case>`, never by hand - issue #30,
spec 8.9 and acceptance criteria 1 and 3. One run renders the committed clip
under one instruction, submits a second instruction mid-clip through the
shipped `ActivePlan`, and keeps rendering.

`keystroke -> pixel` is criterion 1's figure and includes the GUI's 400 ms
debounce; `worker` is the same measurement from where the worker accepts the
plan, which is the half an optimisation would move. `worst across swap` is
the longest inter-frame interval between the swap and the pixels, and it is
read against `control worst` - the dearest frame in the same run's steady
state - because 33.33 ms alone cannot say whether a swap cost anything.

Absolute figures belong to the GPU in the row (spec 7.4).

| finished (UTC) | swap | GPU | kind | instruction | keystroke -> pixel (s) | worker (ms) | criterion 1 | worst across swap (ms) | control worst (ms) | over budget | criterion 3 | rebuilds | background | clip file | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-07T12:24:23Z | swap-target | NVIDIA GeForce RTX 4090 | vocabulary | target person -> shoes | 0.50 | 102 | MET | 27.77 | 32.07 | 0/11 vs 0/44 | MET | 0 | identical | - | [swap-target-20260907-122423Z.json](swap-target-20260907-122423Z.json) |
| 2026-09-07T12:24:43Z | swap-style | NVIDIA GeForce RTX 4090 | runtime | style, denoise 0.49 -> 0.62 | 0.43 | 33 | MET | 37.71 | 32.18 | 1/1 vs 0/44 | MET | 0 | identical | - | [swap-style-20260907-122443Z.json](swap-style-20260907-122443Z.json) |
| 2026-09-07T12:24:58Z | swap-target | NVIDIA GeForce RTX 4090 | vocabulary | target person -> shoes | 0.50 | 104 | MET | 27.97 | 31.66 | 0/11 vs 0/44 | MET | 0 | identical | [swap-target-20260907-122458Z-comparison.mp4](swap-target-20260907-122458Z-comparison.mp4) | [swap-target-20260907-122458Z.json](swap-target-20260907-122458Z.json) |
| 2026-09-07T12:25:18Z | swap-style | NVIDIA GeForce RTX 4090 | runtime | style, denoise 0.49 -> 0.62 | 0.43 | 33 | MET | 37.85 | 32.03 | 1/1 vs 0/44 | MET | 0 | identical | [swap-style-20260907-122518Z-comparison.mp4](swap-style-20260907-122518Z-comparison.mp4) | [swap-style-20260907-122518Z.json](swap-style-20260907-122518Z.json) |
