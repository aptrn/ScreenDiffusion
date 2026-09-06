# ScreenDiffusion

Real-time img2img screen renderer on StreamDiffusion. Windows-only, CUDA + TensorRT,
single NVIDIA GPU. Fork of `rudyaa-sd/ScreenDiffusion`.

The active project is the **Prompt-Driven Orchestrator** — read
[`docs/prompt-orchestrator-spec.md`](docs/prompt-orchestrator-spec.md) before working
on detection, region selection, plan compilation, or anything touching the frame loop.
It defines the vocabulary the issues use: **hot path**, **cold path**, **Render Plan**,
**slot**, **track**, **concept**.

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

## Working agreement

Every change ships as a PR into `main`. Work is tracked as GitHub issues specced in
the Goal / Context / Steps / Gate / Traps / Verification format; the Gate is what the
merge gate checks, so it names criteria a test can assert.

Measurement milestones land their numbers in the issue and in
`docs/prompt-orchestrator-spec.md` — a benchmark whose result is not written down has
to be run again.
