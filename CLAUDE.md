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
  is ~5.1 GB on disk, so a sweep across configurations is a capacity decision before it
  is a timing one.
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
  engines. Leave them out of commits and out of test fixtures.
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
