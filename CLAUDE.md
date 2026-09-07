# ScreenDiffusion

Real-time img2img screen renderer on StreamDiffusion. Windows-only, CUDA + TensorRT,
single NVIDIA GPU. Fork of `rudyaa-sd/ScreenDiffusion`.

The active project is **object-aware selective restyling** — read
[`docs/prompt-orchestrator-spec.md`](docs/prompt-orchestrator-spec.md) before working on
detection, region selection, or anything touching the frame loop. It defines the
vocabulary the issues use: **hot path**, **cold path**, **Render Plan**, **slot**,
**track**, **concept**.

Read it for that vocabulary and for the measurement discipline in §7.4 — not for scope.
The spec's LLM prompt-compiler (§1's two planes, §5.1 components C1/C2, §8.4, §8.6, and
all of §11) is **cut from v1**, and those sections are historical. **The GitHub issues
are the source of truth for what is being built.** The Render Plan's producer is the
GUI: a target field whose text goes to the open-vocabulary detector, and a style field
whose text goes to StreamDiffusion.

## Environment

`uv` owns the environment; `pyproject.toml` + `uv.lock` are the source of truth.
`uv sync` builds it, `uv run python main_gpu_addon.py` launches the app.

Install packages with `uv add`, which updates the lockfile. A bare `pip install`
lands outside the locked environment and vanishes on the next `uv sync`.

Python is pinned to exactly 3.11, and `transformers` / `diffusers` / `huggingface-hub`
are pinned to exact versions because StreamDiffusion 0.1.1 breaks on any other majors.
Treat those five pins as fixed unless the task is specifically to move them.

`package.json` exists only for the Sandcastle agent loop. The application is Python.

`SD_MODELS_DIR` and `SD_ENGINES_DIR` point the model downloads (5.4 GB) and the
compiled TensorRT engines (~5.1 GB each) at one shared location. Both directories are
gitignored, so a fresh worktree has neither - set the two variables to the main
checkout's `models/` and `engines/` and every worktree reuses the caches instead of
rebuilding them. Unset, they resolve to `models/` and `engines/` beside
`main_gpu_addon.py`, which is the old behaviour. Resolution (`resolve_models_dir()` /
`resolve_engines_dir()` in `main_gpu_addon.py`, `_resolve_engine_dir()` in
`wrapper.py`) always returns an absolute path and anchors a relative value to the repo
root, never to the cwd; the worker logs both roots at startup.

Tests are two tiers. `uv run pytest -m "not gpu"` is the merge gate's tier: no CUDA
device, no torch import at collection time. `uv run pytest -m gpu` is everything that
needs the GPU. `python scripts/verify.py` runs the GPU-free tier through the venv's
interpreter, so it works from a bare shell.

`bench/` is the benchmark harness. `uv run python -m bench <scenario>` (`--list` for
the names) measures one configuration and writes `bench/results/<scenario>-<ts>.json`
plus a row in `bench/results/README.md`. Both are **tracked** - a committed result is
the deliverable - and both are written by a run, never by hand: if a run did not
happen there is no record. No result reaches disk without a hardware fingerprint,
the cooldown gate is on unless `--no-cooldown` and records `reached` / `capped`
either way, and `--per-module` splits UNet / VAE-encode / VAE-decode. Only
`bench/runner.py` imports torch, and only inside its functions.

Every result also records its **clock regime** - `locked`, `unlocked` or `unknown`,
under `hardware.clock_lock` - and a result that does not cannot be written.
Unlocked, it carries a clock-normalised ms/frame beside the raw one
(`clock_normalization`, basis `clocks.max.sm`, `ms x sampled clock / basis`); locked,
it carries the raw figure and says so. The normalised figure is a first-order
estimate and is labelled one everywhere it appears - it makes a confounded sweep
readable, it does not make it a locked one. `--require-locked-clocks` exits non-zero
unless a lock is detected, for the runs that decide something; `unknown` fails it
too, since an undetectable lock is not a lock.

The same positional slot takes a **detector** name (`yolo-world-s-640`,
`yolov8n-640`; `--list` shows both registries). A detector run writes to
`bench/results/detectors/` - its own directory, because `--marginal` reads every
JSON beside it as a diffusion cell - and goes through the same fingerprint and
clock-regime doors. It is measured with the diffusion engine resident by default
(`--with-diffusion`, `--no-diffusion` to opt out), since a detector benchmarked
alone says nothing about whether it fits. Weights and evidence photographs live
under `$SD_MODELS_DIR/detectors` and `$SD_MODELS_DIR/bench-images`, both
gitignored, and are only fetched behind `--allow-download`.
`python -m bench --detector-report` regenerates the measured block in spec 8.1,
which a test holds to a byte match.

The same slot also takes a **rendering-primitive case** (`restyle-people`,
`identity-dog`). One run renders *both* primitives - A crop-and-composite and B
full-frame-masked - over the same committed clip, interleaved frame by frame, and
writes to `bench/results/primitives/`: one JSON, one README row per primitive, a
source|A|B triptych mp4, a full-resolution triptych still, and one mp4 per arm.
Those clips are the Gate's manual-verification artefact and are **tracked**.
`--primitive-report` regenerates the decision block in spec 8.2, held to a byte
match by a test.

The same slot also takes an **end-to-end selective case** (`selective-people`).
That run drives the *shipped* modules — detector on its thread, tracker, region
scheduler, engine, device compositor — under `render_plan.priority_case_plan()` over a
committed clip resized to the app's capture canvas, and writes to
`bench/results/selective/`: one JSON, one README row, a source|render comparison
mp4 and a full-resolution still, which are the Gate's manual-verification
artefact and are **tracked**. Its record carries the four Gate checks as numbers
with thresholds — background bit-identity, visible change net of a control,
the `ceil(N/K)` round-robin bound, and whether the loop ever stalled — and
`--selective-report` regenerates the block in spec 8.8, held to a byte match by
a test. It also records how stale the boxes a frame renders got (`staleness`):
their age in frames, how far a track moved between two refreshes, and whether
identity survived.

`--detect-every-n N` runs that same case at one detector cadence instead of the
plan's — one arm of issue #23's sweep. The arm is named `<case>-nN` and writes to
`bench/results/cadence/`, **never beside the baselines**: the selective directory
is reduced to the newest run per (case, GPU) for specs 8.8 and 7.4, so an arm at
another cadence sitting there would quietly become the row those sections quote.
`--cadence-report` regenerates the sweep block in 8.8 and is the only report that
reads two directories — the arms, plus issue #24's baselines, from which it takes
the statement of whether there was a gap to close at all rather than re-deriving
one. It recommends the **freshest** cadence that fits the frame budget with 10%
headroom, disqualifies any arm that broke background bit-identity, and says
whether the recommendation is outside the run-to-run spread of the repeats.
Measured on a 4090: `detect_every_n: 5`, 28.15 ms/frame with detection against
33.33, at no cost in image quality. That **is** the shipped default since issue
#33, and a test holds the two together, so a re-sweep that moves the
recommendation fails rather than leaving the field behind. Every arm predates
issue #31's device composite and none has been re-run, so read the recommendation
as a bound: the ~7 ms it removes applies to every arm equally and can only move
the pick to a *fresher* cadence. **No committed `selective-people` baseline has
been re-measured at 5** - both cards are outstanding - so the figures spec 7.4 and
8.8 quote are a floor for the shipped configuration, not its margin.

The same slot also takes a **plan-swap case** (`swap-target`, `swap-style`) —
issue #30, acceptance criteria 1 and 3. That run renders the committed clip under
one instruction, submits a second one mid-clip through the shipped producer and
`ActivePlan`, and keeps rendering; it writes to `bench/results/swaps/`. It records
the whole per-frame **interval series**, how long the new instruction took to
reach the screen with and without the GUI's 400 ms debounce, and whether the
engine was rebuilt — a number read off the step count and the engine object's
identity either side, not an assumption. `--swap-report` regenerates the block in
spec 8.9. `swap-target` changes the target concept, so it pays the vocabulary
re-encode and the throwaway detect; `swap-style` changes only style and denoise.
Measured on a 4090: **0.50 s and 0.43 s keystroke to pixel** against a 3 s
criterion, worst interval across a swap 37.85 ms against 32.03 in the same run's
steady state, **0 rebuilds**. A stutter is judged against that steady state plus
one frame budget, never against 33.33 ms alone.

The same slot also takes a **temporal-stability arm** (issue #32, spec 8.5):
`--seed-policy {per_track,fixed,random}` and `--output-ema E` sweep the two
levers §8.5 named and nothing had built. The arm is named
`<case>-<policy>-ema<NN>` and writes to `bench/results/stability/` — its own
directory, for the reason the cadence arms have one. Every arm carries a
**responsiveness** figure beside its flicker: the same metric over the pixels
that *moved* in the source rather than the ones that stood still, because an EMA
that kills boiling also kills the restyle's response to motion and one number
alone would recommend the strongest smear measured. `--stability-report`
regenerates the block in spec 8.5. It disqualifies an arm whose region stopped
changing visibly (an EMA that lowers flicker by rendering less) or whose
background stopped being bit-identical, and it recommends the steadiest arm that
keeps 90% of the control arm's response. Measured on a 4090: **neither lever
ships**. `random` scores 8.72 against `fixed`'s 1.49, so the metric plainly sees
noise; `per_track` scores **1.74 — worse** than fixed; and the EMA trades
steadiness for response roughly one for one (0.25 → −14% flicker, −13%
response). 1.49/255 is the floor this path renders at, not a figure waiting to
be improved.

The same slot also takes a **capture-geometry case** (`capture-people`,
`capture-dog`) - issue #39. One run renders the same committed clip at 512x512,
1280x720 and 1920x1080 under *both* rendering primitives at K=1, through the
shipped scheduler and compositor and the one 512x512 engine, and writes to
`bench/results/capture/`. Each arm sweeps its own denoise ladder first, because
spec 8.2 measured that the two primitives need different strengths for the same
job. It records the cost **per stage** - the resize onto the canvas, the diffusion
call, the composite, the device-to-host copy, and the IPC and preview the GUI
process pays - and how many canvas pixels across the rendered region actually got,
which is the whole claim in numbers. The timed pass renders and does nothing else:
folding the metrics, the host copy and the queue into it charged their allocator
churn to the diffusion call, which then appeared to grow from 15.8 ms to 43 ms as
the capture grew. It does not - the canvas is fixed, and a bare probe measures
15.5 / 15.3 / 16.8 ms at the three geometries. `--capture-report` regenerates the
block in spec 8.2.

All seven case-table blocks (`--detector-report`, `--primitive-report`,
`--capture-report`, `--selective-report`, `--cadence-report`, `--swap-report`,
`--stability-report`)
keep the newest run **per (thing measured, GPU)**, not per name. A 4090 run
therefore adds a row beside the 3080's instead of erasing it, which is what keeps
spec 7.4's portability table checkable. When rows span GPUs the table grows a
`GPU` column, the preamble names every machine, and the per-machine verdicts —
the recommendation, the primitive decision, the selective Gate lines, the two
acceptance-criterion verdicts — are stated once per machine; with one machine the
block renders byte-identically to before, so the byte-match tests do not churn.
`gpu_of` in `bench/results.py` is the one reading of which machine a record came
from.

`--portability-report` is the second question asked of those same records — not
"does the path work" but "which of its numbers survived the move to the hardware
this ships on" — and regenerates the block in spec 7.4, byte-matched by a test
like the other six. It refuses to draw a table at all when the two runs rendered
different regions/frame, because region count drives cost and a table across two
of them is the wrong answer in a quotable shape. The 30 FPS verdict it prints is
judged on `ms/frame with detection`, carries the region count and the clock
regime, and says whether every committed run on the card landed on the same side
of the target — a verdict inside the run-to-run spread is labelled one.
`scripts/regen_spec_blocks.py` pastes all nine generated blocks back into the
spec, so a regeneration is never a hand-copy that drops a digit.

The clips in `bench/clips/` are committed and so are their box tracks
(`*.track.json`). A case reads its boxes rather than detecting them, so two runs
render identical regions; `python -m bench <case> --write-track` regenerates a
track and is the only thing that needs the detector. Regenerating one invalidates
the committed comparisons.

`tests/sourceloader.py` executes named top-level definitions straight out of
`main_gpu_addon.py` / `wrapper.py`. Importing either module in the GPU-free tier is
not an option - one primes the DLL search path and pulls in the GUI stack, the other
imports torch.

## Architecture

Two processes, and the split is load-bearing:

- **GUI process** — `StreamGUI` (Tk/customtkinter) in `main_gpu_addon.py`. Never
  imports torch or touches the GPU.
- **Worker process** — `image_generation_process()` (`main_gpu_addon.py:594`) owns the
  model, the GPU, and a DXcam capture thread feeding a bounded deque.

They talk only over `multiprocessing.Queue`s. Live changes reach the worker as
`control_queue` messages (`set_prompt`, `set_region`, `set_t_index_list`, …); adding a
runtime control means adding a message type there, not a shared object.

The capture thread always yields the newest frame and sheds the rest. Under load the
pipeline drops frames rather than falling behind — preserve that.

**The capture is not the diffusion canvas** (issue #39, spec 8.2). Every TensorRT
engine this app builds is 512x512 whatever its directory name claims (spec 7.2),
and until this issue the capture window was the same 512x512 - too small to get a
whole object into frame. `DIFFUSION_CANVAS` and `CAPTURE_PRESETS` in
`main_gpu_addon.py` split the two: the capture-size box sets what is *grabbed* and
the capture thread's resolution, the wrapper is always built at the canvas, and the
frame loop resizes between them with `device_compositor.to_canvas`. Raising the
capture *alone* is a regression - under `masked` a bigger frame gives each object
fewer diffusion pixels - which is why the plan's `global.primitive` lands with it.
`crop` spends the frame's one diffusion call on a single region instead of on the
whole squeezed frame (`crop_to_canvas`), so an object gets the whole 512x512 canvas;
it is honoured only on a frame that selected exactly one region, and falls back to
`masked` on any other, because two regions would be two calls. `validate_plan` says
so in its notes the moment a producer asks for `crop` at `max_instances` above 1.
The composite is unchanged: `composite(source, rendered, alpha, origin)` is one body
for both primitives, and outside the alpha the output is the captured byte at any
capture size.

`render_plan.py` is the **Render Plan** — spec §6, what to restyle and how. Stdlib
only, so both processes import it and neither pays for it. `validate_plan` is the
only way to get a `RenderPlan`: it defaults every field, clamps numbers into their
range and says so in `notes`, drops unknown fields, and *rejects* — with a reason in
words — a value outside a fixed vocabulary, a number that is not a number, a
duplicate target id, or a concept the active detector cannot serve. It assigns
`plan_version` as `previous + 1`, so versions are monotonic in the worker rather than
in whatever sent one. `plan_from_fields(target, style)` is the GUI's producer - the
**Target** and **Style** fields beside the prompt boxes, debounced by
`PLAN_DEBOUNCE_MS` because a target edit re-encodes the detector's vocabulary; a
blank target is `global`, a blank style falls back to the prompt box, and a plan the
GUI's own validator refuses is never sent, its reason going to the status area
instead. There is no LLM in this path and none is coming in v1. The plan crosses the
queue as a plain dict (`set_plan`); the worker validates it and holds it in an
`ActivePlan`, whose `begin_frame()` is the frame loop's single read — a plan
arriving mid-frame lands on the next frame. Only the *first* target's `prompt` and
`denoise` reach the engine, because issue #5 chose the full-frame masked primitive
and that is one embedding per frame; the rest are carried and the validator says
when they differ.

`detection.py` is the **tracker** — spec §5.1 C4 and §8.5. Stdlib only, like
`render_plan.py`. `Tracker.update` takes one detector tick and returns tracks with
monotonic, never-reused IDs: matched by overlap *within a concept*, box smoothed by
an EMA, and carried across two ticks the object was not seen in before it is
dropped. `Tracks` is the frozen snapshot the frame loop reads — one reference, no
per-frame allocation — and `EMPTY_TRACKS` is what it reads before the first detect,
so "no tracks yet" is not a second state.

`detector_worker.py` is the **detector in the worker** — spec §5.1 C3. The only
shipped module that imports ultralytics, and it does so inside its methods.
`BackgroundDetector` holds the state: it `follow`s the active plan (the concepts
are the plan's targets), loads the weights on the first plan that names one, and
runs `step()` on a daemon thread. The frame loop `offer`s the newest capture every
`global.detect_every_n` frames and never waits — a detect in flight keeps the frame
it has, and an offered frame replaces whatever was queued. A detector that cannot
load or throws disables detection, once, with the reason on the status queue; the
render loop carries on. Detection count and detector ms ride the existing fps queue
as a dict (`fps_payload`), which `_format_fps` renders in the GUI.

`region_scheduler.py` (stdlib) and `compositor.py` (numpy) are the **selective
render path** — spec §5.1 C5 and C7. The scheduler turns a `Tracks` snapshot and a
plan into a `Selection`: at most K regions, K being the honoured target's
`max_instances`, banded by each target's `region`, dilated by its `box_scale`,
clipped to the frame, and rotated so that with N eligible tracks over K slots every
track is rendered inside `ceil(N/K)` frames. A region under `MIN_REGION_PX` is
skipped and *counted*, before slots are handed out. There is no crop-to-tile snap:
issue #5 chose the full-frame masked primitive, so K counts masked regions and one
diffusion call covers all of them. The compositor turns that selection into a
`FrameRender` — pass the capture through, render the whole frame, or render and
composite through a feathered alpha — and the ramp climbs *inwards* from each
region's edge, so outside the regions the output is the captured pixel byte for
byte. That is asserted, not asserted-ish: `bench/results/selective/` holds a run
where 48/48 frames were bit-identical outside the mask.

`seeding.py` is the plan's **`seed_policy`**, applied to the engine's latent noise
(issue #32, spec 8.5). Stdlib in the GUI process's sense — torch is imported inside
the methods that write a tensor. Three things it settles. StreamDiffusion draws
`init_noise` once at `prepare()` and reuses it for every frame, so the shipped path's
noise is already **fixed and canvas-pinned**, and `fixed` is a name for what the app
has always done rather than a new setting — which is why `DEFAULT_SEED_POLICY` is
`fixed`. `per_track` cannot mean one call per track under the full-frame masked
primitive; what it means here is one field composed of per-track patches, each track's
realisation drawn from its id and *rolled to its current centre* so the noise follows
the object. And `random` — a fresh field per frame — does not ship: it is the control
that says the flicker metric can see the noise at all. Writing the field is a runtime
write of a plain tensor `add_noise` reads in Python, so a seed change cannot cost a
TensorRT rebuild; a GPU test holds that. `NoiseField` is held across frames beside the
scheduler's cursor and the compositor's alpha cache, and under `fixed` it costs the
frame path not one tensor operation.

The compositor also holds the **output EMA** (issue #32): `global.output_ema` pulls
each rendered canvas back towards the previous one, `current + (previous - current) *
ema`, *before* the mask. That ordering is the whole safety argument - an EMA on the
composited frame would make a background pixel a function of history, and this one
cannot, because the blend it feeds still copies the captured byte wherever alpha is 0.
The history ends where a frame rendered nothing and at an engine rebuild. It defaults
to 0.0 and spec 8.5 says why it stays there.

`device_compositor.py` is that blend **on the GPU** (issue #31). `DeviceCompositor`
*is* a `Compositor` - it inherits the actions, the feather, the alpha cache and the
numpy blend, so every rule the GPU-free tier holds the reference implementation to
is a rule about the shipped object - and adds `blend_device`: the engine is asked
for `output_type="pt"`, the capture and the render are blended where they already
are, and the frame pays one device-to-host copy, of uint8 after the mask rather
than of float before it. Measured on a 4090, 2.75 ms -> 0.63 ms of composite and
24.81 -> 16.89 ms/frame, because what went with it was the whole round trip. The
numpy path is not deleted and must not be: it is what the merge gate tests and what
the device path is held to, byte for byte, in `tests/test_gpu_device_compositor.py`
and `tests/test_gpu_selective_render.py`. torch is imported inside the two methods
that need it, so the module stays importable in the GUI process and in the GPU-free
tier.

The plan's `denoise` reaches the engine through `render_plan.t_index_for_denoise`,
which picks the schedule index whose noise amplitude is nearest — the amplitudes
are computed from SD-Turbo's own beta schedule in stdlib and pinned to issue #5's
measured ladder by a test. Only the schedule *values* move, so a plan change is a
runtime update and never an engine rebuild; a plan with no target leaves the
t_index slider alone. `seed_policy` and `global.output_ema` reach the frame loop
the same way and are runtime writes too - `seeding.NoiseField` and the
compositor - so no stability setting can cost an engine rebuild.
`SD_DEMO_PLAN=1` starts the worker on `priority_case_plan()` — restyle the lower
half of every person, gently — which is the headless way to drive the whole path.
It survives the GUI fields: a blank target sends no plan at all, at start or
ever, so an empty field cannot overwrite it.

## Gotchas

- **TensorRT engines compile per configuration.** Resolution, batch size
  (`frame_buffer_size`), step count, and fused LoRA weights each key a distinct engine.
  Changing step *count* at runtime tears down the wrapper and rebuilds — minutes, not
  milliseconds. Any design that changes these per-frame is a stall. Each built engine
  is ~5.0 GB on disk and 15–25 minutes to build (measured, 512², RTX 3080 laptop), so
  a sweep across configurations is a capacity decision before it is a timing one.
- **The resolution in an engine directory name is a lie.** `create_prefix()` puts
  `res-WxH` in the cache key, but `wrapper.py` never forwards a resolution to
  `EngineBuilder.build`, whose `opt_image_height` / `opt_image_width` default to 512
  with `build_dynamic_shape=False`. So every engine this app builds is 512×512, and a
  directory labelled `--res-256x256--` holds a 512² one — it loads without complaint
  and then collides at inference. Batch size *is* forwarded and is trustworthy.
  `tests/test_trt_engine_resolution.py` pins this; spec §7.2 has the detail.
- **Dev and deploy hardware differ.** Development is an RTX 3080 laptop; deployment
  targets RTX 3090 Ti / 4090. Curve shapes and relative rankings carry across; absolute
  ms/frame, VRAM ceilings and engine build times do not. Any 30 FPS claim is a
  deploy-hardware claim. Measured on both (issue #24, spec 7.4): the selective path
  runs 2.3x faster on a 4090 and met 30 FPS at 5.04 regions/frame **by ~1 ms**,
  while everything that is not a millisecond — the selection, the call count, the
  round-robin bound, the flicker, the bit-identity — came across unchanged. The
  composite was the exception in the other direction: host numpy, so it barely
  moved (0.68x where the frame path went to 0.45x) and became a larger share of the
  frame than it was on the laptop — which is why issue #31 moved it onto the device.
  The margin is now 9.58 ms, and the laptop row in spec 7.4 is still a
  host-composite measurement, so that row is two designs as much as two cards until
  the laptop is re-run. The generated block says so itself.
- **Engines are per GPU architecture, and must be rebuilt, never copied.** An Ampere
  build will not load on Ada. A second machine needs its own `models/` (2.5 GB once
  the fp32 duplicates are excluded; `huggingface-cli download stabilityai/sd-turbo
  --local-dir $SD_MODELS_DIR/sd-turbo-fp16 --exclude sd_turbo.safetensors
  unet/diffusion_pytorch_model.safetensors vae/diffusion_pytorch_model.safetensors
  text_encoder/model.safetensors`), its own detector weights under
  `$SD_MODELS_DIR/detectors`, and its own `engines/` (~15 min for the 512² b1 build
  on a 4090). **Do not benchmark in the process that built the engine** — the first
  4090 run measured 29.5 FPS against 30.9–31.4 for five cold-started runs against
  the cached engine, which is what the app actually does.
- **GPU benchmarks need a thermal cooldown**, or you measure the throttle instead of
  the change. Wait for the GPU to fall below ~62 °C before each rep, cap the wait, and
  record whether the threshold was actually reached — a laptop under sustained load may
  never get there. Every result carries a hardware fingerprint (GPU name, VRAM, driver,
  power limit, raw `nvidia-smi` with a timestamp): it identifies the machine and is the
  evidence the number was measured rather than invented.
- **A card someone else is using measures the neighbour, not the change.** The
  cooldown gate catches a hot GPU and nothing caught a shared one: issue #33's
  first re-run measured 54.17 ms/frame against a committed 16.89 on the same 4090,
  at 50 °C, while a live real-time app held ~45% of the SMs - and it passed the
  fingerprint, the cooldown (`reached`) and the clock-regime doors, and was
  appended to the tracked README. `bench/contention.py` is the door now. Every
  selective record carries an `occupancy` block (`clear` / `busy` / `unknown`,
  the mean of four readings taken *before* the timed region, while the bench
  process is idle, so what it reads is the other tenants), and
  `--require-idle-gpu` refuses a run that decides something, exactly as
  `--require-locked-clocks` does; `unknown` refuses too. The harness cannot name
  the offending process - `nvidia-smi` reports per-process memory but not
  per-process SM time on consumer cards - so this is the one door that needs a
  human to look at the machine.
- **Cooling is not enough: a comparative sweep wants locked clocks.** The cooldown
  gate only cools *before* a rep, and under the 120 W limit this laptop falls from
  boost to its floor within about two seconds of a run starting — so a short call
  measures boost and a long one measures the throttle, and the two are not
  comparable. Locking the clock is a **manual step this loop cannot perform**:
  `nvidia-smi --lock-gpu-clocks` needs an elevated shell, and the agent loop does not
  have one. From an Administrator PowerShell, before a sweep that decides something:

  ```powershell
  nvidia-smi --lock-gpu-clocks=1200,1200   # pick a clock the card holds under load
  nvidia-smi --query-gpu=clocks_event_reasons.applications_clocks_setting --format=csv
  # ... run the sweep, e.g. uv run python -m bench <scenario> --require-locked-clocks
  nvidia-smi --reset-gpu-clocks            # always, or the machine stays clamped
  ```

  The harness only ever *detects* a lock — it never sets one, and a failed attempt
  must never be read as a lock. Detection is the `clocks_event_reasons.
  applications_clocks_setting` event reason, the only lock signal driver 595.79
  exposes on consumer Ampere; when it cannot be read the regime is `unknown`, which
  is not a synonym for `unlocked` and does not satisfy `--require-locked-clocks`.
- **A higher `t_index` is *less* denoise, not more.** `t_index_list` indexes the
  50-step LCM schedule, which descends: index 20 is timestep 599 (noise amplitude
  0.92) and index 45 is timestep 99 (0.32). The app's default 30 is timestep 399.
  Changing the *values* is a runtime update (`set_t_index_list`, clamped to 1-49);
  only changing the step *count* rebuilds the engine. So sweeping strength is cheap
  and sweeping step counts is not.
- **Both rendering primitives resize onto the 512x512 canvas and back, and that
  round trip changes the pixels on its own.** For the full-frame primitive on a
  1280x720 clip it changes them a lot. Any "did the render do anything" criterion
  has to subtract a control pass with the diffusion call taken out
  (`render_resample_only` in `bench/primitive_runner.py`), or a strength that does
  nothing passes on its own blur.
- **`models/` and `engines/` are gitignored** and hold multi-GB downloads and compiled
  engines. Leave them out of commits and out of test fixtures. Point `SD_MODELS_DIR` /
  `SD_ENGINES_DIR` at a shared copy rather than re-downloading or rebuilding per
  worktree.
- **PRs from this fork default to the upstream repo.** `gh pr create` targets
  `rudyaa-sd/ScreenDiffusion` unless `remote.origin.gh-resolved` is set locally
  (`gh repo set-default aptrn/ScreenDiffusion`). It lives in `.git/config`, so it does
  not survive a fresh clone — see [`docs/second-machine.md`](docs/second-machine.md).
- **The worker enforces offline mode** (`enforce_offline_mode()`). Network calls from
  the frame path will fail there by design.
- **`controlnet_paths` / `controlnet_scales`** are accepted by
  `image_generation_process()` and never passed to the wrapper. Inherited dead stub —
  wiring it up is real work, not a one-liner.
- **SD-Turbo is SD 2.1-based.** SD 1.5 and SDXL LoRAs will not load, and LoCon/LyCORIS
  convolution layers are unsupported by this diffusers version.
- **`YOLOWorld.set_classes` drops the predictor.** Changing the vocabulary sets
  `self.predictor = None`, so the *next* `predict` rebuilds it and costs ~108 ms more
  than a steady detect (measured; a bare `predictor = None` costs the same). The text
  encode itself is ~16 ms and genuinely cold-path. Anything that changes the
  vocabulary must issue one throwaway detect before the new plan goes live, or the
  frame after a prompt edit drops three frames' worth of budget. Spec 8.1.
  `UltralyticsDetector.set_concepts` is where that throwaway detect lives; anything
  else that calls `set_classes` owes one.
- **Call `.to("cuda")` before the first `set_classes`.** The CLIP text encoder caches
  the device it was built on; built on the CPU and moved afterwards, every later
  vocabulary change raises a device-mismatch from `torch.embedding`.
- **ultralytics writes to the cwd.** `WEIGHTS_DIR` defaults to `<cwd>/weights`, and
  the first `set_classes` drops a 338 MB CLIP checkpoint there - in this repo, into
  the repo. `bench/detector_runner.py` repoints it at the shared models root before
  loading anything; `ultralytics.nn.text_model` binds the value at import time, so it
  has to happen first.
- **DXcam has no Desktop Duplication context in every session.** It raises
  `The specified device interface or feature level is not supported` on this machine
  under the agent loop, so the detector bench falls back to `mss` and records which
  backend produced a frame. The app's own capture path is unaffected; this is a
  property of the session, not of the code.
- **The detector costs ~3x more beside the diffusion than resident behind it.**
  Spec 8.1's 14-19 ms per detect was measured with the engine loaded but idle; in
  the shipped path the detect thread and the UNet share the SMs and a detect
  measures ~59 ms (fastest in the same run: 15.7 ms). Contention, not the clock.
  Anything that budgets detection from 8.1's figure is budgeting the wrong number
  - use `bench/results/selective/`, or raise `global.detect_every_n`. The
  contention is itself a function of the cadence: measured across issue #23's
  sweep, one detect costs ~23 ms at `detect_every_n: 2` and ~20 ms at 8, so
  raising the cadence saves more than 1/N.
- **A plan swap re-encodes the prompt on the frame thread.** `update_prompt` and
  the schedule-cache rebuild run inside the frame that binds a changed plan, and
  they cost that one frame ~5.8 ms on a 4090 (measured, spec 8.9) — the only place
  in this design where cold-path work runs on the hot path. It is inside the
  budget a dropped frame would cost, so it is not a stutter and needs no fix at
  30 FPS; anything that makes it dearer wants re-measuring. A *target* swap is the
  other way round: it costs the frame path nothing and costs the output ~10 frames
  of unstyled capture while the detector re-encodes and re-detects.
- **`per_track` seeding makes flicker *worse*, and that is measured, not a bug.**
  Flicker is scored where the *source* stood still, and the shipped noise field is
  pinned to the canvas, so a static pixel already gets the same noise on every
  frame. Pinning the field to a track instead makes it translate, so the static
  background *inside* a moving person's box starts boiling: 1.74 against `fixed`'s
  1.49 on a run-to-run spread of 0.0004 (spec 8.5). The lever does what its name
  says and the name was aimed at the wrong thing. Do not "fix" it by reverting the
  default to `per_track` - a plan field nothing reads describing behaviour nothing
  produces is what issue #32 found and removed.
- **The output EMA buys steadiness at about one-for-one in responsiveness.** 0.25
  takes 14% off flicker and 13% off the output's response to the source's own
  motion; 0.75 takes 58% and 46%. It is not cheating - the net change inside the
  regions is 11.7-11.9/255 at every setting - it is simply priced. At a starting
  flicker of 1.49/255 there is nothing worth buying, which is why the default is
  0.0. That verdict is about *this* denoise on *this* case: a stronger `denoise`
  boils more, and the answer could change.
- **`crop` wins the sub-region restyle and loses the identity change, measured
  twice.** At K=1 the two primitives cost within 8% of each other at every capture
  geometry, so §8.2's 5.87x - which was entirely A's call count at six objects -
  does not apply; crop diffuses the region at 512 px against masked's 69 at 1080p
  and expresses the priority case. It still cannot do `identity-dog`: 9-10 frames
  of 48 read as a cat against masked's 38-39, because a 424x280 box stretched onto
  a square canvas comes back at the wrong aspect, and a bigger capture does not fix
  a distortion that is proportional. Spec 8.2. Do not read "crop is the primitive"
  as a general statement - it is a per-case one.
- **A feather that bleeds outside its region breaks the whole criterion.** The
  selective path's promise is that non-target pixels are the captured bytes, so
  the alpha ramp climbs inwards from the region's own edge and is exactly 0 one
  pixel outside it. Anything that softens the mask symmetrically - a blur, a
  dilation, a Gaussian - fails the gate `tests/test_compositor.py` and
  `tests/test_gpu_selective_render.py` hold. On either device: the blend runs on
  the GPU now, and `tests/test_gpu_device_compositor.py` holds it to the numpy one
  byte for byte, which is the only reason the criterion can still be checked in a
  tier with no CUDA.

## Working agreement

Every change ships as a PR into `main`. Work is tracked as GitHub issues specced in
the Goal / Context / Steps / Gate / Traps / Verification format; the Gate is what the
merge gate checks, so it names criteria a test can assert.

Measurement milestones land their numbers in the issue and in
`docs/prompt-orchestrator-spec.md` — a benchmark whose result is not written down has
to be run again.
