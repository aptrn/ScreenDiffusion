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
  deploy-hardware claim.
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
- **`models/` and `engines/` are gitignored** and hold multi-GB downloads and compiled
  engines. Leave them out of commits and out of test fixtures. Point `SD_MODELS_DIR` /
  `SD_ENGINES_DIR` at a shared copy rather than re-downloading or rebuilding per
  worktree.
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

## Working agreement

Every change ships as a PR into `main`. Work is tracked as GitHub issues specced in
the Goal / Context / Steps / Gate / Traps / Verification format; the Gate is what the
merge gate checks, so it names criteria a test can assert.

Measurement milestones land their numbers in the issue and in
`docs/prompt-orchestrator-spec.md` — a benchmark whose result is not written down has
to be run again.
