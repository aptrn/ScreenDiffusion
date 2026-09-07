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
scheduler, engine, compositor — under `render_plan.priority_case_plan()` over a
committed clip resized to the app's capture canvas, and writes to
`bench/results/selective/`: one JSON, one README row, a source|render comparison
mp4 and a full-resolution still, which are the Gate's manual-verification
artefact and are **tracked**. Its record carries the four Gate checks as numbers
with thresholds — background bit-identity, visible change net of a control,
the `ceil(N/K)` round-robin bound, and whether the loop ever stalled — and
`--selective-report` regenerates the block in spec 8.8, held to a byte match by
a test.

A selective run is kept **per case and per GPU**, not per case: the same case on
two cards is two answers, and the deploy run must not delete the dev baseline.
`--portability-report` is the second question asked of those same records — not
"does the path work" but "which of its numbers survived the move to the hardware
this ships on" — and regenerates the block in spec 7.4, byte-matched by a test
like the other four. It refuses to draw a table at all when the two runs rendered
different regions/frame, because region count drives cost and a table across two
of them is the wrong answer in a quotable shape. The 30 FPS verdict it prints is
judged on `ms/frame with detection`, carries the region count and the clock
regime, and says whether every committed run on the card landed on the same side
of the target — a verdict inside the run-to-run spread is labelled one.
`scripts/regen_spec_blocks.py` pastes all five generated blocks back into the
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

`render_plan.py` is the **Render Plan** — spec §6, what to restyle and how. Stdlib
only, so both processes import it and neither pays for it. `validate_plan` is the
only way to get a `RenderPlan`: it defaults every field, clamps numbers into their
range and says so in `notes`, drops unknown fields, and *rejects* — with a reason in
words — a value outside a fixed vocabulary, a number that is not a number, a
duplicate target id, or a concept the active detector cannot serve. It assigns
`plan_version` as `previous + 1`, so versions are monotonic in the worker rather than
in whatever sent one. `plan_from_fields(target, style)` is the GUI's producer; there
is no LLM in this path and none is coming in v1. The plan crosses the queue as a
plain dict (`set_plan`); the worker validates it and holds it in an `ActivePlan`,
whose `begin_frame()` is the frame loop's single read — a plan arriving mid-frame
lands on the next frame. Only the *first* target's `prompt` and `denoise` reach the
engine, because issue #5 chose the full-frame masked primitive and that is one
embedding per frame; the rest are carried and the validator says when they differ.

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

The plan's `denoise` reaches the engine through `render_plan.t_index_for_denoise`,
which picks the schedule index whose noise amplitude is nearest — the amplitudes
are computed from SD-Turbo's own beta schedule in stdlib and pinned to issue #5's
measured ladder by a test. Only the schedule *values* move, so a plan change is a
runtime update and never an engine rebuild; a plan with no target leaves the
t_index slider alone. `SD_DEMO_PLAN=1` starts the worker on
`priority_case_plan()` — restyle the lower half of every person, gently — which is
how the whole path is driven until a GUI field exists to drive it.

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
  runs 2.3x faster on a 4090 and meets 30 FPS at 5.04 regions/frame **by ~1 ms**,
  while everything that is not a millisecond — the selection, the call count, the
  round-robin bound, the flicker, the bit-identity — came across unchanged. The
  composite is the exception in the other direction: it is host numpy, so it barely
  moved (0.68x where the frame path went to 0.45x) and is now a larger share of the
  frame than it was on the laptop.
- **Engines are per GPU architecture, and must be rebuilt, never copied.** An Ampere
  build will not load on Ada. A second machine needs its own `models/` (5.4 GB
  download; `huggingface-cli download stabilityai/sd-turbo --local-dir
  $SD_MODELS_DIR/sd-turbo-fp16 --exclude sd_turbo.safetensors
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
  - use `bench/results/selective/`, or raise `global.detect_every_n`.
- **A feather that bleeds outside its region breaks the whole criterion.** The
  selective path's promise is that non-target pixels are the captured bytes, so
  the alpha ramp climbs inwards from the region's own edge and is exactly 0 one
  pixel outside it. Anything that softens the mask symmetrically - a blur, a
  dilation, a Gaussian - fails the gate `tests/test_compositor.py` and
  `tests/test_gpu_selective_render.py` hold.

## Working agreement

Every change ships as a PR into `main`. Work is tracked as GitHub issues specced in
the Goal / Context / Steps / Gate / Traps / Verification format; the Gate is what the
merge gate checks, so it names criteria a test can assert.

Measurement milestones land their numbers in the issue and in
`docs/prompt-orchestrator-spec.md` — a benchmark whose result is not written down has
to be run again.
