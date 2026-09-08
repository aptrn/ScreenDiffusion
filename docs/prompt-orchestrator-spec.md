# Prompt-Driven Orchestrator — Specification (Draft v0.1)

**Status:** idea capture / pre-research. Nothing here is implemented.
**Base:** fork of ScreenDiffusion v0.2 (`main_gpu_addon.py`, `wrapper.py`).
**Hardware:** Windows 10/11, CUDA 12.8 + TensorRT 10.8, single NVIDIA GPU.
Development happens on an **RTX 3080 laptop**; deployment targets **RTX 3090 Ti /
4090** desktops. The two are not interchangeable, and §7.4 says which conclusions
survive the move and which do not.

---

## 1. The idea in one paragraph

Today ScreenDiffusion applies **one global style prompt to the entire capture
region**, every frame. The idea is to let the user instead type a single
*intent* in plain language — "find all people on screen and give them a red
hat" — and have the system figure out on its own what to detect, where to apply
diffusion, and with what conditioning. The user never touches a class list, a
mask tool, or a second prompt box.

Crucially this cannot be a single model. Frontier multimodal models can read a
screen and generate video from one instruction, but they run at seconds-to-
minutes per frame, not 33 ms. So the natural-language layer is moved **off the
hot path**: a small local LLM acts as a *compiler*, run once per prompt change,
which emits a machine-readable **Render Plan**. A fast, fixed, real-time loop
(capture → detect → crop → diffuse → composite) then executes that plan at
30 FPS with no LLM in the frame path.

Two planes:

|                 | Control plane ("cold path")  | Data plane ("hot path")                          |
| --------------- | ---------------------------- | ------------------------------------------------ |
| Trigger         | User edits the prompt        | Every frame                                      |
| Latency budget  | 0.2–3 s, allowed to stall    | ≤ 33 ms, must never stall                        |
| Contains        | LLM prompt compiler          | capture, detector, tracker, diffusion, composite |
| Output          | Render Plan (JSON)           | Rendered frame                                   |

Changing the prompt mid-stream swaps the plan atomically; the render loop picks
up the new plan on its next frame and keeps running.

---

## 2. Motivation

- **Removes the setup burden.** The interesting capability (selective,
  object-aware real-time restyling) currently requires the user to know about
  detector class IDs, mask dilation, per-region prompts, denoise strength. One
  text box hides all of it.
- **Keeps the real-time property.** The value of this project is that it runs
  *live* on the desktop. Any design that puts a large model in the frame loop
  throws that away.
- **Composable.** Once the Render Plan is an explicit data structure, it can
  also be authored by hand, saved as a preset, driven by OSC/MIDI, or produced
  by something other than an LLM.

---

## 3. What the fork already gives us

Grounding facts, from the current code:

- `main_gpu_addon.py:594` `image_generation_process()` runs in a **separate
  process**; the Tk GUI lives in the parent. They communicate over
  `multiprocessing.Queue`s: `out_queue`, `fps_queue`, `status_queue`,
  `control_queue`, `close_queue`, plus a `monitor_receiver` pipe carrying the
  capture rectangle.
- The worker already accepts **live control messages** on `control_queue`:
  `set_region`, `set_prompt`, `set_negative_prompt`, `set_t_at`,
  `set_t_index_list`. This is the seam the orchestrator plugs into — the Render
  Plan becomes a new message type on the same channel.
- Capture is a **separate daemon thread** inside the worker
  (`_screen_capture_loop_dx`, DXcam, with an `mss` fallback) writing into a
  bounded `deque`; the render loop always consumes the newest frame and sheds
  the rest. Frame shedding under load is therefore already established
  behaviour.
- Diffusion is `StreamDiffusionWrapper.img2img()` (`wrapper.py:48`): SD-Turbo
  (SD 2.1 base), fp16, TinyVAE, `cfg_type="none"`, fixed `width`/`height` (512
  default), `t_index_list` selecting denoise strength.
- **TensorRT engines are compiled per configuration.** Engine identity depends
  on resolution, batch size (`frame_buffer_size`), number of steps, and fused
  LoRA weights. Changing step *count* today forces a full wrapper teardown and
  rebuild (`main_gpu_addon.py:~650`). This is the single biggest constraint on
  the design in §7.3.
- `image_generation_process()` takes `controlnet_paths` / `controlnet_scales`
  parameters that are **accepted but never passed** to the wrapper — an
  inherited stub, not working ControlNet support.
- **The benchmark harness is `bench/`** (M0, issue #2): `uv run python -m bench
  <scenario>` measures one configuration and writes
  `bench/results/<scenario>-<timestamp>.json` plus a row in
  `bench/results/README.md`. It keeps the two conventions the deleted scratch
  scripts had — wait for the GPU to fall below ~62 °C before the timed reps
  (otherwise you measure thermal throttle), and time UNet / VAE-encode /
  VAE-decode separately, here behind `--per-module`. It adds two rules those
  scripts lacked: nothing is written without the §7.4 hardware fingerprint, and
  results are written by a run and never by hand.

---

## 4. Scope

**In scope (v1)**

- Single natural-language instruction, typed live, compiled to a Render Plan.
- Detection of a bounded set of target concepts per plan.
- Selective diffusion applied to detected regions only, composited back over
  the untouched screen capture.
- Live prompt swap without restarting generation or rebuilding an engine.

**Out of scope (v1)**

- Multimodal understanding of screen *content* by the LLM (it reads the user's
  text only, not the pixels). See §8.6 for the optional vision-assist path.
- Video/temporal diffusion models, audio, multi-GPU.
- Edits requiring geometric change (adding an object where none exists, moving
  objects). v1 restyles *within* a detected region.
- Cloud inference in the frame loop.

**Explicit non-goal:** conversational chat. The LLM emits a plan; it does not
hold a dialogue during rendering.

---

## 5. Architecture

```
                 ┌───────────────────────── CONTROL PLANE (cold) ─────────────┐
   user text ───►│  Prompt Compiler (local LLM, structured/JSON output)       │
                 │      ↓ validate + repair + clamp                           │
                 │  Render Plan  ──► versioned, atomically published          │
                 └───────────────────────────────┬────────────────────────────┘
                                                 │ control_queue: {"type":"set_plan", ...}
 ┌───────────────────────── DATA PLANE (hot, ≤33 ms) ─────────────────────────┐
 │ DXcam capture thread ─► frame deque (newest wins, rest shed)               │
 │        ↓                                                                   │
 │ Detector (open-vocab or COCO)  ─► boxes/masks   [every Nth frame]          │
 │        ↓                                                                   │
 │ Tracker (ByteTrack-class)      ─► stable IDs, box smoothing [every frame]  │
 │        ↓                                                                   │
 │ Region scheduler ─► select ≤K regions, snap to engine tile size            │
 │        ↓                                                                   │
 │ StreamDiffusion img2img (fixed batch, fixed resolution)                    │
 │        ↓                                                                   │
 │ Compositor ─► feathered alpha blend back into the full frame ─► out_queue  │
 └────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Components

**C1 — Prompt Compiler.** Takes the raw user string plus a static system prompt
describing the Render Plan schema; returns JSON. Must be constrained (grammar /
JSON-schema-guided decoding), not free-form, so malformed output is structurally
impossible rather than caught after the fact. Runs in its own process or thread;
never blocks the render loop. On failure or timeout the previous plan stays
active and the user sees an error.

**C2 — Plan Validator.** Sits between the LLM and the render loop. Clamps
numeric ranges, drops unknown fields, rejects target concepts the active
detector cannot serve (with a user-visible reason), assigns a monotonically
increasing `plan_version`. The render loop trusts only validated plans.

**C3 — Detector.** Given a frame and the plan's target list, returns boxes (and
optionally masks). Two candidate families, see §8.1: fixed-vocabulary (YOLOv8,
80 COCO classes, fastest) vs open-vocabulary (YOLO-World, OWLv2, Grounding DINO
— accepts arbitrary text, slower). Runs at a *lower* rate than the render loop
(e.g. every 3rd–5th frame). **Implemented**: `detector_worker.py`, YOLO-World on
a daemon thread inside the worker process. The weights load on the first plan
that names a concept — a `global` plan pays no VRAM for a detector it will not
use — and the frame loop *offers* the newest capture every
`global.detect_every_n` frames rather than detecting on it. An offer never waits:
a detect in flight keeps the frame it has and the offered one replaces whatever
was queued behind it, which is the capture thread's newest-frame-wins rule
applied one stage later.

**C4 — Tracker.** Bridges the gap between detector ticks, gives each object a
stable ID, and smooths box jitter. Stable IDs are what let us pin a **seed and
prompt embedding per object**, which is the main lever against flicker.
**Implemented**: `detection.py`, stdlib-only and GPU-free. Greedy nearest-overlap
matching within a concept, an EMA on the matched box (§8.5's box smoothing), and
a track carried across two ticks it was not seen in before it is dropped — the
capture deque sheds frames and detectors blink, and neither should cost an object
its identity. IDs are monotonic and never reused. The frame loop reads one
immutable `Tracks` snapshot, published by the detector thread with a single
reference assignment, so a frame sees one tick or the next and never half of one.

**C5 — Region Scheduler.** The rate limiter. Detections are unbounded; the
diffusion engine has a *fixed* batch size. This component decides which ≤K
regions get diffused this frame, snaps their crops to the engine's native tile
size, and applies a round-robin / priority policy when there are more objects
than slots (see §7.3). **Implemented**: `region_scheduler.py`, stdlib-only and
GPU-free. One difference from the paragraph above, and it follows from §8.2's
decision: primitive B issues **one** diffusion call per frame whatever K is, so
there are no crops to snap to a tile size and K limits how much of the frame is
*composited* rather than how much is diffused. K comes from the honoured target's
`max_instances`. Regions below a size floor are skipped and counted — under two
VAE cells there is no restyled content in a region, only resampled blur — and
they are filtered before the slots are handed out, so a permanently tiny object
cannot starve the rotation. The rotation cursor is a track id rather than a
position, ids being monotonic, so with N eligible tracks and K slots every track
is rendered inside `ceil(N/K)` frames.

**C6 — Diffusion Executor.** The existing `StreamDiffusionWrapper`, driven with
a batch of crops instead of one full frame. Needs per-item prompt embeddings if
different objects carry different styles (§8.3). **Implemented**: as one
`img2img` call per frame — §8.2 chose B — and the plan's `denoise`
reaches it as a `t_index` on the live schedule (`render_plan.t_index_for_denoise`,
whose amplitudes are held to §8.2's measured ladder by a test). Only the schedule
*values* move, never the step count, so a plan change is a runtime update and
never an engine rebuild. A frame whose plan selected no region costs no diffusion
call at all. What that one call is *given* is the plan's `global.primitive`
(issue #39): the whole capture squeezed onto the 512×512 canvas under `masked`, or
one region blown up to fill it under `crop`. The capture is no longer the canvas —
`device_compositor.to_canvas` / `crop_to_canvas` resize between them — and the
engine is 512×512 either way, because it is 512×512 whatever a directory name
claims (§7.2).

**C7 — Compositor.** Pastes results back with feathered alpha, optional
colour/exposure match to surrounding pixels, and optional temporal EMA to
suppress flicker. Everything outside the regions is raw captured pixels.
**Implemented**: `compositor.py`, numpy and no torch. The alpha ramp climbs
*inwards* from each region's own edge and is exactly zero one pixel outside it,
so the soft seam costs the region a few pixels of strength and costs the
background nothing — which is what makes "bit-identical outside the regions"
literally true and measurable (§8.8). Overlapping regions take the stronger
alpha; the blend is written as `source + (rendered - source) * alpha`, whose two
endpoints are exact in floating point; and only the bounding rectangle of the
non-zero alpha is written at all. The temporal EMA **is** built (issue #32): it
smooths the rendered canvas *before* the mask, so history reaches only the pixels
the mask lets through and the criterion above survives it unchanged. It is a plan
field, `global.output_ema`, and it defaults to 0.0 — §8.5 measured what it costs.
Colour matching is still unbuilt.

**And on the device** (issue #31): `device_compositor.DeviceCompositor` **is** a
`Compositor` — the actions, the feather, the alpha cache and the numpy blend are
inherited, so every rule the merge gate holds the reference implementation to is a
rule about the shipped object. What it adds is `blend_device`: the engine is asked
for `output_type="pt"`, the capture and the render are blended where they already
are, and the frame pays **one** device-to-host copy, of uint8 after the mask rather
than of float before it. The arithmetic is not equivalent to the numpy path, it is
the same operations in the same order — `tests/test_gpu_device_compositor.py` and
`tests/test_gpu_selective_render.py` hold the two to each other byte for byte, on
synthetic inputs and through the real engine. numpy stays the reference: it is what
the GPU-free tier tests and what anything without a device can still use.

---

## 6. The Render Plan (control-plane ⇄ data-plane contract)

The whole design hinges on this being small, validated, and stable. It is
**implemented**: `render_plan.py` is the schema and the validator, and
`validate_plan` is the only way to obtain a `RenderPlan`. The block below is the
contract as built — every value in it is the default the validator assigns, and a
test regenerates it through `validate_plan` and demands a match, so it cannot drift
away from the code.

```jsonc
{
  "plan_version": 1,                // assigned by the validator, never by a producer
  "source_prompt": "",              // the style text, when no target carries one
  "negative_prompt": "",            // not in the straw-man: the app has this box today
  "mode": "selective",              // "selective" | "global" | "inverse"
  "targets": [
    {
      "id": "t0",                   // assigned by position when omitted
      "concept": "person",          // free text, straight to the open-vocabulary detector
      "detector_class": null,       // resolved COCO id, or null for open-vocab
      "region": "full_box",         // full_box | upper_third | upper_half | center | lower_half | lower_third
      "box_scale": 1.15,            // dilation before cropping, 1.0-2.0
      "prompt": "",                 // the style for this target
      "negative_prompt": "",
      "denoise": 0.45,              // 0.0-1.0; carried, and applied by the render path
      "seed_policy": "fixed",       // "per_track" | "fixed" | "random" (spec 8.5)
      "max_instances": 6,           // 1-16
      "priority": 1                 // 0-99, an ordering key between targets
    }
  ],
  "background": { "action": "passthrough", "prompt": "" },   // or "stylize"
  "global": { "fps_target": 30, "detect_every_n": 5, "output_ema": 0.0,   // 0.0-0.9, spec 8.5
              "primitive": "masked" },                                    // "masked" | "crop", spec 8.2
  "confidence": 1.0,
  "notes": ""
}
```

Design rules:

- **The GUI writes it; the validator disposes.** The straw-man said "only C2 may
  write it" — C2 was the LLM prompt compiler, and it is cut from v1. The producer
  is `plan_from_fields(target, style)`: a target field whose text goes to the
  detector and a style field whose text goes to StreamDiffusion. Any other hand can
  write one too — a test, a preset file — because the door is the validator, not the
  producer.
- **Every field has a safe default.** A plan carrying just `targets[0].concept`
  and `.prompt` must render something sensible. `mode` defaults by reading the
  plan: `global` with no target, `selective` with one.
- **Out of range is clamped and said; malformed is refused and said.** A number
  outside its range is clamped into it and the clamp lands in `notes`; an unknown
  field is dropped and named. A value outside a fixed vocabulary (`mode`, `region`,
  `seed_policy`, `background.action`), a number that is not a number, a duplicate
  target id or a concept the active detector cannot serve rejects the plan with a
  reason in words. The plan in force keeps rendering — spec 8.7, never a black
  screen.
- **`notes` is surfaced in the UI.** Both kinds: the plan's own `notes` field and
  the validator's, which say what it changed on the way through.
- **Versioned and atomic.** `plan_version` is assigned by the validator as
  `previous + 1`, so it is monotonic in the worker rather than in whichever producer
  sent one. `ActivePlan.begin_frame()` is the frame loop's single read: a plan
  submitted while a frame renders lands on the next frame and never on half of this
  one.
- **Only the first target's `prompt` and `denoise` are honoured.** Issue #5 chose
  the full-frame masked primitive — one diffusion call, so one prompt embedding and
  one strength per frame. The other targets keep their own values in the schema, and
  the validator says out loud when they differ from the first.

**The producer, as built** (issue #22). `StreamGUI` carries two fields beside the
prompt boxes - **Target** and **Style** - and an edit to either debounces for
`PLAN_DEBOUNCE_MS` (400 ms, well past the prompt box's 150: a target edit changes
the detector's vocabulary and §8.1 measured that at a ~108 ms throwaway detect) and
then sends `plan_from_fields(target, style)` as `set_plan`. Three rules make the two
fields safe to leave blank:

- **A blank target is `global`**, the whole frame under one prompt, which is what
  this app has always done. Clearing the target is how a user goes back to it.
- **A blank style falls back to the prompt box**, so naming a target never hands the
  engine an empty embedding and calls it a style.
- **A refused plan is never sent.** The GUI validates first and puts the validator's
  stated reason in the status area, where the field was typed; the worker validates
  the same plan again, because it owns the version. Accepted, the validator's `notes`
  land in the same place (§8.7).

The fields stay editable while generation runs - changing what is restyled must not
mean stopping the run - and `SD_DEMO_PLAN=1` still starts the worker on the hardcoded
priority case, so a headless run is unaffected: a blank target sends nothing at all
rather than overwriting it.

**Where they sit, and what the window says about them** (issue #40). Target and
Style lead the left panel, above the model path and the strength slider; the engine
knobs - seed, buffer, acceleration, LCM-LoRA, denoising batch, and the step *count* -
are folded into a collapsed **Advanced** section, which is the `SHOW` dict the file
already had, used for what it was for. Underneath the two fields the window carries
one sentence of plan state: whether the plan is global or selective, whether
detection is running, on what concept, how many objects it is holding and at what
cadence. The concept it names is the *detector's*, taken from the fps payload, not
the field's - an edit is debounced and then costs a vocabulary re-encode, so for a
moment the two disagree and the one worth showing is the one being detected. Before
this the only sign the object-aware path was alive at all was the tail of the FPS
line. `docs/gui/` holds the before and after.

---

## 7. Performance model

### 7.1 Budget

30 FPS = **33.3 ms/frame**. The allocation is a target; the Measured column is
what M0 has actually put on the clock so far:

| Stage               | Rate            | Budget (ms/frame) | Measured, RTX 3080 laptop | Notes                     |
| ------------------- | --------------- | ----------------- | ------------------------- | ------------------------- |
| Capture (DXcam)     | every frame     | 1–2               | not yet measured          | already threaded          |
| Detection           | every 3rd frame | 4–8 amortised     | **4.8** @ 640² YOLO-World | fits — §8.1               |
| Tracking            | every frame     | < 1               | not yet measured          | CPU                       |
| Preprocess crops    | every frame     | 1–2               | not yet measured          | resize + normalise on GPU |
| **Diffusion**       | every frame     | **15–20**         | **54.8–58.4** @ 512² TRT b1 | the dominant term         |
| Composite + present | every frame     | 2–3               | **4.03** @ 512², host composite | §8.8 — 0.63 on a 4090 since the blend moved onto the device (#31) |
| Headroom            |                 | ~5                | —                         |                           |

Diffusion is over budget by roughly 3×, and on its own is ~1.7× the entire
33.3 ms frame — for **one** 512² crop, before any of the other rows are paid.
Detection, by contrast, lands inside its allocation: 14.3 ms per detect at 640²
amortises to 4.8 ms/frame at one detect every 3rd frame (§8.1), with the
diffusion engine resident while it was measured. That is a **dev-hardware** number and §7.4 says so: the two committed 512²
TensorRT batch-1 runs came back at 54.8 ms (mean 1342 MHz) and 58.4 ms (960 MHz),
and the spread between them is the 120 W limit and the clock sampling described in
§7.2, not variance in the model. **Whether 33.3 ms is reachable is a question only
the deploy GPU can answer**, and it is not answered here.

Batching does not rescue it at this crop size: the same engine at batch 2 costs
55.0 ms/frame and at batch 4 costs 67.4 ms/frame, so the per-frame figure gets
*worse*, not better, once the batch grows (§7.2). Small crops are what moves this
row — a 256² `none` call at batch 8 amortises to 11.8 ms/frame normalised — which
is why §7.3 recommends a small slot rather than a large one.

What is settled, and is portable, is where the budget has to be spent: diffusion
dominates so completely that every other row could be free and the frame would
still miss on this machine. Detection being comfortably in budget is the second
half of that finding — the detector is not the problem and does not need to be
made cheaper. The batch curve in §7.2 is therefore the whole design question, not
a tuning detail.

The LLM appears nowhere in this table. That is the point.

### 7.2 The numbers we actually need

Not assumed — measured by the harness M0 builds, with the hardware recorded
alongside every number (§7.4):

1. SD-Turbo 1-step img2img at 512², TRT vs `none`, batch 1 — ms/frame. **Done.**
2. Same at batch 2 / 4 / 8 — **is the per-crop marginal cost sublinear?** This
   determines whether N-object rendering is viable at all. **Done** on `none` at
   all three resolutions, and confirmed on TensorRT at 512² (batch 1 / 2 / 4).
   The answer is *resolution-dependent*, which is the finding: strongly sublinear
   at 256², weakly at 384², and **not** sublinear at 512².
3. Same at 256² and 384² crops — small crops are the whole economic case.
   **Done on `none`.** Not confirmable on TensorRT today — see "The resolution
   axis" below.
4. Detector latency for each candidate in §8.1 at 640² input. **Done**, in
   PyTorch rather than TensorRT: YOLO-World's text head is the reason to use it
   and does not survive an export, so exporting it would have measured a
   different model. YOLO-World 14.3 ms, YOLOv8n 12.8 ms — §8.1.
5. Peak VRAM with diffusion engine + detector + (optional) local LLM resident.
   **Done for the two that remain**: 7,182 MiB in use across the device with the
   512² TensorRT engine and YOLO-World both loaded, against 5,620 MiB for the
   engine alone — the detector adds ~1.5 GiB. The LLM is cut from v1, so there is
   no third tenant.
6. Cost of swapping prompt embeddings per batch item. *Outstanding.*

Results land in `bench/results/` as JSON, one file per run, with a readable table
in `bench/results/README.md`. A number that is not in there has to be measured
again.

Measure the batch and resolution sweep on the **`none` accelerator first**. It
answers the question that actually matters — the *shape* of the marginal-cost
curve — at zero engine-build cost, and only then is it worth spending an engine
on the two or three configurations the curve says are interesting. The two built
for this milestone cost ~5.0 GB and 15–25 minutes each, measured.

#### Measured: RTX 3080 Laptop GPU, SD-Turbo fp16, 1 step, img2img

Twelve `none` cells (256/384/512 square × batch 1/2/4/8) and three TensorRT cells
at 512² (batch 1 / 2 / 4 — the cap issue #3 sets on engine builds). Every TensorRT
cell records the free-disk reading its build was gated on. Fourteen of the fifteen
cells reached the 62 °C cooldown threshold before running; the TensorRT batch-4 cell
started at 63 °C after its own engine build and is recorded `capped`, not `reached`.
Reproduce the whole table from the committed JSON with
`uv run python -m bench --marginal` — it is computed from those files, not
transcribed into this document.

`ms/call` is the number that answers the design question: one call diffuses the
whole batch, so the slope between two batch sizes is what one more crop costs.
`ms/frame` is that slope already averaged over the batch, which hides it.

<!-- BEGIN MEASURED TABLE -->

As measured

| GPU | accel | res | batch | ms/call | ms/frame | marginal ms/item | x first item | peak VRAM (MiB) | SM clock (MHz) | cooldown |
|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 256x256 | 1 | 47.0 | 46.97 | - | - | 2516 | 1775 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 256x256 | 2 | 51.0 | 25.51 | 4.1 | 0.09 | 2541 | 1785 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 256x256 | 4 | 71.8 | 17.95 | 10.4 | 0.22 | 2594 | 1496 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 256x256 | 8 | 139.5 | 17.44 | 16.9 | 0.36 | 2696 | 1211 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 384x384 | 1 | 60.1 | 60.08 | - | - | 2548 | 1519 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 384x384 | 2 | 85.8 | 42.89 | 25.7 | 0.43 | 2607 | 1317 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 384x384 | 4 | 160.8 | 40.21 | 37.5 | 0.62 | 2720 | 1130 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 384x384 | 8 | 304.6 | 38.07 | 35.9 | 0.60 | 2951 | 1137 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 1 | 82.3 | 82.30 | - | - | 2593 | 1239 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 2 | 145.8 | 72.89 | 63.5 | 0.77 | 2695 | 1182 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 4 | 282.8 | 70.70 | 68.5 | 0.83 | 2900 | 1112 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 8 | 767.0 | 95.88 | 121.1 | 1.47 | 3310 | 787 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | tensorrt | 512x512 | 1 | 58.4 | 58.37 | - | - | 2500 | 960 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | tensorrt | 512x512 | 2 | 109.9 | 54.96 | 51.6 | 0.88 | 2511 | 1172 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | tensorrt | 512x512 | 4 | 269.8 | 67.44 | 79.9 | 1.37 | 2536 | 1348 | capped |

none 256x256: sublinear - first item 47.0 ms, extra items 9%-36% of it
none 384x384: sublinear - first item 60.1 ms, extra items 43%-62% of it
none 512x512: NOT sublinear - first item 82.3 ms, extra items 77%-147% of it
tensorrt 512x512: NOT sublinear - first item 58.4 ms, extra items 88%-137% of it

Normalised to 1785 MHz (estimate: ms x clock / reference)

| GPU | accel | res | batch | ms/call | ms/frame | marginal ms/item | x first item | peak VRAM (MiB) | SM clock (MHz) | cooldown |
|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 256x256 | 1 | 46.7 | 46.70 | - | - | 2516 | 1785 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 256x256 | 2 | 51.0 | 25.51 | 4.3 | 0.09 | 2541 | 1785 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 256x256 | 4 | 60.2 | 15.04 | 4.6 | 0.10 | 2594 | 1785 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 256x256 | 8 | 94.7 | 11.83 | 8.6 | 0.18 | 2696 | 1785 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 384x384 | 1 | 51.1 | 51.12 | - | - | 2548 | 1785 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 384x384 | 2 | 63.3 | 31.65 | 12.2 | 0.24 | 2607 | 1785 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 384x384 | 4 | 101.8 | 25.45 | 19.3 | 0.38 | 2720 | 1785 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 384x384 | 8 | 194.1 | 24.26 | 23.1 | 0.45 | 2951 | 1785 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 1 | 57.1 | 57.13 | - | - | 2593 | 1785 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 2 | 96.5 | 48.25 | 39.4 | 0.69 | 2695 | 1785 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 4 | 176.1 | 44.04 | 39.8 | 0.70 | 2900 | 1785 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | none | 512x512 | 8 | 338.0 | 42.25 | 40.5 | 0.71 | 3310 | 1785 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | tensorrt | 512x512 | 1 | 31.4 | 31.39 | - | - | 2500 | 1785 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | tensorrt | 512x512 | 2 | 72.2 | 36.09 | 40.8 | 1.30 | 2511 | 1785 | reached |
| NVIDIA GeForce RTX 3080 Laptop GPU | tensorrt | 512x512 | 4 | 203.7 | 50.93 | 65.8 | 2.10 | 2536 | 1785 | capped |

none 256x256: sublinear - first item 46.7 ms, extra items 9%-18% of it
none 384x384: sublinear - first item 51.1 ms, extra items 24%-45% of it
none 512x512: sublinear - first item 57.1 ms, extra items 69%-71% of it
tensorrt 512x512: NOT sublinear - first item 31.4 ms, extra items 130%-210% of it

<!-- END MEASURED TABLE -->

**On the `none` sweep, read the normalised table for curve shape.** The laptop
holds a 120 W limit and a longer call sinks deeper into it: mean SM clock across the
sweep runs from 1785 MHz down to 787 MHz, so a raw comparison across cells is partly
a comparison of clocks. Raw, 512² `none` reads "NOT sublinear" — but only because
the batch-8 cell ran at 787 MHz. Normalised, all three `none` resolutions are
sublinear, and the *ordering* between them survives either reading.

**On the three TensorRT cells, read the raw table.** Their verdict does not depend
on the clock model — raw and normalised both say 512² is not sublinear — and the
normalisation is least trustworthy exactly there. `mean_sm_clock_mhz` is the mean of
0.5 s samples taken across the whole run, and a TensorRT batch-1 run is only ~1.8 s
long, so it gets four samples; in the committed batch-1 cell one of them caught the
GPU at **210 MHz** between reps and pulled the mean from ~1210 MHz to 960. That is
what makes its normalised figure (31.4 ms) the fastest cell in the table, and why
the earlier batch-1 run of the same configuration normalises to ~41 ms instead. The
raw numbers are unaffected; only the correction is. Sampling that survives short
runs is a harness fix, not a re-measurement, and it is not in this milestone.

The normalisation (`ms × clock ÷ reference`) is first-order and labelled an estimate
throughout. Treat it as evidence about *shape*, which is what §7.4 says is portable,
and never as a prediction of an absolute figure.

#### The resolution axis is not measurable on TensorRT today

Item 3's TensorRT half could not be run, and the reason is a bug in this repo
rather than a limit of the hardware. `EngineBuilder.build` takes
`opt_image_height` / `opt_image_width`, both defaulting to 512, with
`build_dynamic_shape=False`; `wrapper.py` forwards `opt_batch_size` to
`compile_unet` / `compile_vae_encoder` / `compile_vae_decoder` and nothing else. So
**every engine this app builds is 512×512**, while `create_prefix` writes
`res-{width}x{height}` into the cache directory name. A directory labelled
`--res-256x256--` holds a 512² engine.

Building `img2img-tensorrt-256x256-b1` demonstrated it: TensorRT rejected the input
shapes (`Set dimensions are [1,3,256,256]. Expected dimensions are [1,3,512,512].`)
and the run returned 53.7 ms/frame — the 512² engine's number. It was not
committed. This is also why the 384² engine raises a CUDA illegal memory access
while the 512² one is fine: 512 is the only directory whose label happens to be
true. It is the same failure ef85e1a fixed by putting the resolution *in* the cache
key — the key varies, but nothing downstream of it does.

The batch axis is unaffected: `opt_batch_size` *is* forwarded, which is why the
TensorRT batch curve at 512² is trustworthy. `tests/test_trt_engine_resolution.py`
pins the finding so it cannot rot; fixing it is a `wrapper.py` change plus ~10 GB
of rebuilds, and wants its own issue.

#### Measured: the step-count axis — **issue #38**

Every figure above is SD-Turbo at **one** denoising step, and until issue #38
nothing in this repo swept the step count at all — §7.2's measured table sweeps
*batch size*, which is `frame_buffer_size`. That mattered the moment a second base
model was considered: SD 1.5 is not a turbo model and needs LCM-LoRA at about four
steps, and the only estimate available was arithmetic on the 256×256 laptop
per-module split — UNet ~80% of the call, so four passes ≈ 3.4× it.

**That estimate was wrong, and the reason is instructive.** `use_denoising_batch`
puts the steps through the UNet as one batch of N rather than as N calls, and a
batch of four is not four calls. Reproduce with
`uv run python -m bench --steps-report`; the arms live in `bench/results/steps/`,
which is deliberately *not* `bench/results/` — `--marginal` reads every JSON there
as a (resolution, batch) cell, so a two-step arm at batch 1 would join the
committed batch curve as a second batch-1 point.

<!-- BEGIN STEP COUNT -->
Step count swept over 1, 2, 4 on NVIDIA GeForce RTX 4090, at 512x512 batch 1 on `sd-turbo-fp16`, `sd-v1-5-fp16`. Only the count moves: every arm opens at schedule index 35 and spends its extra steps after it (`render_plan.t_index_ladder`), so an arm is not also a strength change. Batch 1 throughout, so ms/call is ms/frame. The UNet, VAE-encode and VAE-decode columns come from `--per-module`, whose extra synchronises perturb the total - they are read as a split of the call, not as the call.

| steps | model | style LoRA | accel | t_index list | ms/call | x 1 step | UNet ms | VAE encode ms | VAE decode ms | UNet share | peak VRAM (MiB) | SM clock (MHz) | cooldown | ms/call at basis clock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sd-turbo-fp16 | - | none | 35 | 26.02 | 1.00x | 16.70 | 2.09 | 2.48 | 64% | 2593 | 2715 | reached | 22.43 @ 3150 MHz |
| 1 | sd-turbo-fp16 | - | tensorrt | 35 | 20.67 | 1.00x | 12.55 | 1.41 | 1.38 | 61% | 2500 | 2662 | reached | 17.47 @ 3150 MHz |
| 2 | sd-turbo-fp16 | - | none | 35,49 | 32.73 | 1.26x | 21.84 | 2.09 | 2.90 | 67% | 2658 | 2708 | reached | 28.13 @ 3150 MHz |
| 4 | sd-turbo-fp16 | - | none | 35,40,44,49 | 48.63 | 1.87x | 37.29 | 2.14 | 2.94 | 77% | 2757 | 2705 | reached | 41.74 @ 3150 MHz |
| 4 | sd-v1-5-fp16 | - | none | 35,40,44,49 | 49.97 | - | 39.63 | 2.12 | 2.96 | 79% | 2929 | 2715 | reached | 43.07 @ 3150 MHz |
| 4 | sd-v1-5-fp16 | - | tensorrt | 35,40,44,49 | 33.88 | - | 25.69 | 1.37 | 1.41 | 76% | 2673 | 2722 | reached | 29.28 @ 3150 MHz |
| 4 | sd-v1-5-fp16 | loving-vincent | none | 35,40,44,49 | 52.02 | - | 41.09 | 2.15 | 3.02 | 79% | 2930 | 2720 | reached | 44.90 @ 3150 MHz |
| 4 | sd-v1-5-fp16 | loving-vincent | tensorrt | 35,40,44,49 | 33.57 | - | 25.27 | 1.33 | 1.38 | 75% | 2672 | 2722 | reached | 29.02 @ 3150 MHz |

**`sd-turbo-fp16`, `none`.** At one step the UNet is 64% of the 26.02 ms call (16.70 ms against VAE-encode 2.09 and VAE-decode 2.48). 2 steps cost **1.26x** the one-step call (32.73 ms against 26.02), so each step after the first cost 6.71 ms. That is 1.31x the UNet for 2x the passes (21.84 ms against 16.70): `use_denoising_batch` puts the steps through the UNet as one batch of 2, and a batch of 2 is not 2 calls. 4 steps cost **1.87x** the one-step call (48.63 ms against 26.02), so each step after the first cost 7.54 ms. That is 2.23x the UNet for 4x the passes (37.29 ms against 16.70): `use_denoising_batch` puts the steps through the UNet as one batch of 4, and a batch of 4 is not 4 calls. Issue #38 estimated ~3.4x for four steps, from the 256x256 laptop per-module split. Measured here it is 1.87x, below it - the estimate was pessimistic. That estimate is replaced by this table.

**`sd-turbo-fp16`, `tensorrt`.** At one step the UNet is 61% of the 20.67 ms call (12.55 ms against VAE-encode 1.41 and VAE-decode 1.38).

No one-step arm in 4 configurations (`sd-v1-5-fp16`, `none`; `sd-v1-5-fp16`, `none`, `loving-vincent` fused; `sd-v1-5-fp16`, `tensorrt`; `sd-v1-5-fp16`, `tensorrt`, `loving-vincent` fused), so their `x 1 step` cell is empty - which for a model that cannot render at one step is the honest answer. What those arms are read against is the base-model comparison below.

**Base model, `none`.** `sd-v1-5-fp16` (SD 1.5) at its working 4 steps costs 49.97 ms/call against `sd-turbo-fp16`'s 26.02 at 1 - **1.92x** the diffusion call. The step count is the whole of it: at 4 steps the two models are within 3% of each other.

**Base model, `tensorrt`.** `sd-v1-5-fp16` (SD 1.5) at its working 4 steps costs 33.88 ms/call against `sd-turbo-fp16`'s 20.67 at 1 - **1.64x** the diffusion call.
<!-- END STEP COUNT -->

Two consequences the rest of issue #38 rests on. Four steps is affordable on the
diffusion call — 1.87× rather than 3.4× on `none` — and four steps is a **batch-4
UNet engine**, so the step count keys its own ~5 GB build exactly as the batch size
does. `engine_cache.unet_batch_size` is that rule, shared by the window and the
harness.

The table also carries the answer to *which model*, on the same axis and in the
same session: SD 1.5 at its working four steps costs 1.64× SD-Turbo at its one on
TensorRT, and at four steps the two models are within 3% of each other. **The cost
of moving to SD 1.5 is the steps it needs, not the checkpoint.** §7.5 takes that
end to end.

### 7.3 The batch-size problem (the hard one)

A TensorRT engine is built for a **fixed batch size**. The scene contains a
*variable* number of objects. Options:

- **(a) Fixed slot count K.** Build for batch K (say 4). Fewer objects → pad
  with dummies (wasted compute). More objects → the Region Scheduler
  round-robins across frames, so each object updates at 30/⌈N/K⌉ FPS. Simple,
  predictable, no rebuilds.
- **(b) Multiple engines** (K = 1, 2, 4, 8) resident, chosen per frame. Costs
  VRAM and build time; avoids padding waste.
- **(c) Dynamic-shape TRT profiles.** An optimisation profile with a min/max
  batch range. Best answer if StreamDiffusion's engine builder supports it —
  it does; see the recommendation below.
- **(d) Single canvas.** Pack all crops into one 512² atlas and diffuse it as a
  single image. Constant cost, one engine, no batching problem — but objects
  bleed across tile seams and share one prompt. Cheap to prototype; may be good
  enough for uniform edits.

#### Recommendation

Decided on the §7.2 sweep, not on taste. That sweep says one thing loudly: **a
call is expensive and a *small* item inside it is cheap.** A batch of one costs
47–57 ms whatever the crop size — 512² costs only 1.2x what 256² does at batch 1,
despite four times the pixels, because a single small crop is overhead-bound rather
than pixel-bound. Adding a second 256² crop to that same call costs 4.3 ms.

The italics are the correction the TensorRT confirmation forced. Cheapness is not a
property of batching, it is a property of *small* crops being batched: the marginal
item costs 9–18% of the first at 256², 24–45% at 384², ~70% at 512² on `none`, and
**88–137% at 512² on TensorRT** — past parity, so a second full-size crop is no
cheaper than a second call. TensorRT's own advantage narrows the same way: 1.4x over
`none` at batch 1 (58.4 vs 82.3 ms/call) and 1.05x by batch 4 (269.8 vs 282.8). Two
independent accelerators agreeing that 512² does not batch is the durable part.

So the design should aim for exactly **one diffusion call per frame**, packed with
crops that are *small*, and each option is judged on how well it does that. "Render
more objects" and "render them larger" are the same budget spent twice; the plan
compiler has to trade them against each other rather than assume batching absorbs
both.

**Recommendation: (a) fixed slot count**, with (c) as the identified upgrade path
whenever someone spends the `wrapper.py` change it needs.

- **(a) Fixed slot count.** Take it. One call per frame, one engine, no rebuild —
  and the padding waste it is usually criticised for is small for exactly the
  reason above. A K=4 engine handed one real crop wastes three marginal items,
  which at 256² is ~13 ms, not three times the cost of a crop.
- **(b) Multiple engines.** Reject. It buys back that padding waste — the
  cheapest thing in the system — and pays in the most expensive. The two engines
  built for this milestone put numbers on it: ~5.0 GB per configuration on disk
  (a 1.77–1.87 GB UNet plus 3.47 GB of ONNX scratch that is never cleaned up), and
  15–25 minutes to build, of which 11–20 is the UNet alone. A K∈{1,2,4,8} set is
  ~20 GB and over an hour before anything renders, and the chosen ones are resident
  in VRAM together. Dominated by (c) on every axis.
- **(c) Dynamic-shape profiles.** The right long-term answer, and reachable. This
  milestone's step 4 settled the open question by reading the installed source:
  pinned StreamDiffusion 0.1.1 *does* support it — `accelerate_with_tensorrt`
  takes `min_batch_size` / `max_batch_size`, `build_static_batch` defaults to
  `False`, and `get_minmax_dims` widens the profile to `min_batch..max_batch`
  accordingly. **This app collapses it:** `wrapper.py` passes min == max at all
  three call sites, which is precisely what `--max_batch-1--min_batch-1--` in
  every engine directory name records. So (c) is a `wrapper.py` change plus a
  rebuild, *not* a dependency change, and `use_cuda_graph` is already `False` so
  nothing conflicts. `tests/test_trt_dynamic_shape.py` pins the finding. It is
  not adopted here only because it was outside this milestone's scope.
- **(d) Single canvas.** Keep it as a *quality* experiment. It is not a cost win.
  A 512² atlas holding four 256² crops is one 512² call at 57.1 ms normalised,
  against 60.2 ms for a batch-4 256² call — a wash, well inside the spread the
  power limit puts on these numbers. The "constant cost" argument for (d) does
  not survive measurement; it has to stand on seams, prompt sharing and
  simplicity instead. §8.2 is where it belongs.

**Consequence for K.** Because the marginal item is only cheap while the crop is
small, K and the crop size are one decision, not two. On the `none` curve the four
cheapest ways to spend ~60 ms of normalised call time are 1×512², 4×256², 2×384² and
8×256² (94.7 ms) — and only the 256² ones scale further. A slot of 256² is what
makes (a) worth having; a slot of 512² makes (a) indistinguishable from calling once
per object.

**Portable:** (§7.4 — conclusions about shape) that the per-call floor dominates
a single small crop; that extra *small* items in a call are cheap; that their
cheapness fades as the crop grows and is gone by 512², where the marginal item
reaches and passes the cost of the first (9–18% of it at 256² on `none`, 88–137% at
512² on TensorRT); that TensorRT's advantage over `none` shrinks with batch size at
512²; and therefore that (a) beats (b), that (d) is not a cost win, and that K is
only useful paired with a small slot. None of these depend on the absolute clock:
they are ratios within one machine, and the two accelerators agree.

**Not portable:** the value of **K**, and the crop size to pair it with. Both come
from where the 33.3 ms line falls, which is a deploy-GPU question — nothing measured
here is inside budget on the laptop. Likewise non-portable: the VRAM figures
(2.5–3.3 GiB on a 16 GB card) and every engine build time. And specifically
non-portable, because it is a laptop power-limit artefact rather than an
architectural one: **where** on the batch axis 512² crosses from sublinear to
superlinear. That a crossing exists is portable; a 3090 Ti or 4090 will put it
somewhere else, and finding it is a first task on deploy hardware.

### 7.4 Dev and deploy hardware are different

Development runs on an RTX 3080 laptop; deployment targets RTX 3090 Ti / 4090
desktops. Laptop silicon is power- and thermally-limited in a way desktop silicon
is not, and the VRAM ceiling differs sharply. A measurement is only meaningful
with its hardware attached.

**Every result record carries a hardware fingerprint**: GPU name, total VRAM,
driver version, power limit, and raw `nvidia-smi` output with a timestamp. That
fingerprint does two jobs — it says which machine a number came from, and it is
the evidence that the number was measured rather than invented.

What survives the move from laptop to desktop:

| Conclusion                                     | Portable?                                          |
| ---------------------------------------------- | -------------------------------------------------- |
| Shape of the marginal-cost curve vs batch size | Yes — the sublinearity question is architectural   |
| Relative ranking of detector candidates        | Yes                                                |
| Relative ranking of the §8.2 primitives        | Yes                                                |
| What the selective path selects, and in how many calls | Yes — **measured**, identical on both cards |
| Bit-identity of non-target pixels              | Yes — **measured**, 48/48 frames on both           |
| Absolute ms/frame, and whether 30 FPS is met   | **No** — **measured**, 2.3× apart on one design; re-measure on the deploy GPU |
| Where the frame path's cost sits (host vs device) | **No** — **measured**, the host composite was 7.3% of a laptop frame and 11.1% of a 4090 one (issue #31) |
| VRAM ceilings and OOM thresholds               | **No** — a laptop OOM says nothing about 24 GB     |
| Engine build times                             | **No**                                             |

The 30 FPS gate in §11 is therefore a **deploy-hardware** criterion. On the
laptop the same run is a regression check, not an acceptance test.

A laptop also may never reach the §7.2 cooldown threshold under sustained load.
The harness caps the wait and records that the threshold was not met, rather than
blocking forever or silently reporting a throttled number.

**The generated blocks keep one row per (thing measured, GPU)** — issue #25. The
§8.1, §8.2 and §8.8 tables reduce the committed JSON to the newest run of each
case or detector *on each machine*, so a deploy-card run adds a row beside the
laptop's rather than replacing it: the table above is a claim about two machines
and it is only checkable while both rows exist. A block whose rows span GPUs
grows a `GPU` column, names every machine in its preamble, and takes its
per-machine verdicts — the §8.1 recommendation, the §8.2 decision, the §8.8 Gate
lines — once per machine, because a ranking across two GPUs is not a ranking. A
single-machine repo renders exactly what it rendered before. A record with no
`hardware.gpu_name` (there are none, but the fingerprint post-dates the first
results) is labelled `unknown GPU` and never merged with another one.

#### Measured: the move to an RTX 4090 (issue #24)

Until 2026-09-07 the table above was reasoning, not measurement — every figure in
the repo came from one RTX 3080 laptop at a 120 W limit. The shipped selective path
has now been run on the deploy hardware, over the same committed clip under the same
plan, through an engine **rebuilt for Ada** (engines are architecture-specific; an
Ampere build will not load, and copying one across is the trap this measurement
exists to avoid). The block is generated from the committed records with
`uv run python -m bench --portability-report`; re-measure and the merge gate fails
until it is regenerated.

<!-- BEGIN DEPLOY HARDWARE -->
`selective-people` measured on NVIDIA GeForce RTX 4090 (unlocked clocks, 500 W) against the NVIDIA GeForce RTX 3080 Laptop GPU baseline of 2026-09-06 (unlocked clocks, 120 W): the same img2img-tensorrt-512x512-b1 engine rebuilt for this architecture, the same 48 frames of `people.mp4` at the app's 512x512 capture canvas, the same `person / lower_half / t_index 40` plan at `detect_every_n: 3`.

| measure | NVIDIA GeForce RTX 3080 Laptop GPU (dev) | NVIDIA GeForce RTX 4090 (deploy) | deploy / dev |
|---|---|---|---|
| regions/frame | 5.04 | 5.04 | 1.00x |
| diffusion calls/frame | 1.00 | 1.00 | 1.00x |
| ms/frame, frame path | 55.06 | 16.89 | 0.31x |
| ms/frame, with detection | 74.73 | 23.75 | 0.32x |
| FPS | 13.4 | 42.1 | 3.15x |
| ms/detect | 59.03 | 20.58 | 0.35x |
| composite ms/frame | 4.03 | 0.63 | 0.16x |
| flicker (static px) | 1.49 | 1.49 | 1.00x |
| peak VRAM (MiB) | 3742 | 3744 | 1.00x |
| mean SM clock (MHz) | 1668 | 2730 | 1.64x |

**The `composite ms/frame` row is not a hardware ratio**: the blend ran on the host (numpy) on NVIDIA GeForce RTX 3080 Laptop GPU and on the device (torch) on NVIDIA GeForce RTX 4090 (issue #31), so those two figures are two designs as much as two cards.

Both machines detected every 3 frames (`detect_every_n: 3`).

**What the cadence costs is box age, and only its frames travel.** 1.9 frames on NVIDIA GeForce RTX 3080 Laptop GPU (projected: that record predates the staleness block, so the frames are the other machine's) and a frame rendered boxes 1.9 frames old on NVIDIA GeForce RTX 4090 at `detect_every_n: 3` - 107 ms against 33 ms at their own frame paths, 3.3x apart on the same setting. The frames are the portable figure; the milliseconds are what a viewer sees the mask lag the subject by, and they are the reason a cadence measured on the deploy card is a decision about the deploy card.

**Acceptance criterion 2 (30 FPS): MET.** 42.1 FPS at 5.04 regions/frame on NVIDIA GeForce RTX 4090 - 23.75 ms per frame with detection amortised against the 33.33 ms a 30 FPS budget allows, 9.58 ms to spare; detect_every_n 3, clocks unlocked. 5 committed runs on NVIDIA GeForce RTX 4090 span 41.5-42.3 FPS, clear of the 30 FPS target (5 earlier runs on the card blended on the host (numpy) and are not in this spread).

What carried across the move, measured:

| Conclusion | Carried? | Evidence |
|---|---|---|
| Non-target pixels stay bit-identical to the capture | Yes | 48/48 frames dev -> 48/48 frames deploy |
| How much of the frame the scheduler picks: regions and calls per frame | Yes | 5.04 dev -> 5.04 deploy regions/frame, 1.00 dev -> 1.00 deploy calls/frame |
| The ceil(N/K) round-robin bound | Yes | worst gap 2 of 3 allowed dev -> 2 of 3 deploy |
| Flicker over pixels static in the source | Yes | 1.49 dev -> 1.49 deploy |
| Absolute ms/frame on the frame path | **No** | 55.06 ms dev -> 16.89 ms deploy |
| Whether the 30 FPS criterion is met | **No** | 13.4 FPS dev -> 42.1 FPS deploy |
| What one detect costs beside the diffusion | **No** | 59.03 ms dev -> 20.58 ms deploy |
| Peak VRAM the path allocates | Yes | 3742 MiB dev -> 3744 MiB deploy |
| The clock the card holds under load, against its own maximum | **No** | 79% of maximum dev (120 W limit) -> 87% deploy (500 W) |

Comparable because NVIDIA GeForce RTX 3080 Laptop GPU rendered 5.04 regions/frame and NVIDIA GeForce RTX 4090 rendered 5.04, the same selection to within 5%.
<!-- END DEPLOY HARDWARE -->

Four things the block does not say for itself.

**The criterion is met, and the margin is now nine milliseconds — it was one.**
The block above is the *second* measurement of this card. Issue #24 measured
32.35 ms against a 33.33 ms budget on the slowest of five runs, 31.88 ms on the
fastest: it cleared the gate and was nobody's idea of headroom. §8.8's two known
costs — detection contention and a host-side composite — were the levers, and both
have now been pulled. Issue #23 raised the cadence (5.18 ms, at one extra frame of
box age); issue #31 moved the composite onto the device, which costs nothing
anyone can see and is what the block above measures:
**23.75 ms with detection, 9.58 ms of headroom, at `detect_every_n: 3`.**
Issue #33 has since made 5 the shipped default and **neither row above has been
re-measured at it**, so the block states the cadence each machine ran at and this
margin is a floor rather than the margin — §8.8 says what closing that costs.
Because there was never a gap, **no lower-resolution engine was built.** The five
host-composite runs stay in `bench/results/selective/` as the before, and are named
out of the FPS spread above rather than averaged into it.

**The first run on this card measured 29.5 FPS and is not among issue #24's five.**
It was the run that compiled the engine, and it diffused in a process still holding
the TensorRT builder's state. Every committed run on this card - the five with the
host composite and the five with the device one - is cold-started against the cached
engine, which is what the app does; the excluded one is recorded here rather than in
`bench/results/` because it measured a condition the product never enters.

**Everything that is not a millisecond carried exactly.** Same regions, same calls,
the same round-robin bound, the same flicker to three figures, and the same 48/48
frames bit-identical outside the mask. That is the useful half of this measurement:
the design decisions taken on the laptop — issue #5's primitive, #8's scheduler and
feathered composite — did not need re-taking. Peak VRAM matched too, but read that
narrowly: what the path *allocates* is a property of the path, and the VRAM
*ceiling* remains untested, because nothing here came near either card's.

**The composite was the part that did not scale, and that is why it moved.** Issue
#24 measured the frame path down to 0.45× and one detect to 0.38×, while the numpy
blend only reached 0.68× — host code, which a faster GPU does not make faster. It
was 7.3% of the laptop's frame path and 11.1% of the 4090's, so on this card the
ranking of what to optimise had changed. Issue #31 acted on that: the blend now runs
on the device and the same measurement reads **0.63 ms against 2.75**, which is why
that row's ratio above carries a caveat rather than a conclusion — it is two designs
as much as two cards. The frame path fell further than the 2.75 ms it removed
(24.81 → 16.89 ms), because the round trip it removed was the whole of it: the
render used to come home as float32 and be rebuilt as a PIL image before a host
blend, and that cost was charged to the engine call rather than to the composite.
The laptop row is still a host-composite measurement; re-running it there is what
would make this row a hardware ratio again.

Clocks were unlocked on both machines: `nvidia-smi --lock-gpu-clocks` needs an
elevated shell the agent loop does not have (issue #13), and the 4090 refused it
here for the same reason the laptop did. The desktop is nonetheless the steadier of
the two — 86% of its maximum SM clock at 48 °C against the laptop's 79% at 75 °C —
which is the difference the move was expected to show, recorded rather than
normalised away.

---

### 7.5 The base model is a choice — **issue #38**

Every number in §7 above is SD-Turbo, which is **SD 2.1-based**. That came with
choosing SD-Turbo; it was never a separate decision and had never been revisited.
The reason to revisit it is not speed — SD-Turbo is the faster model and always will
be — it is **style control**. Live testing on 2026-09-07 found the prompt respected
only at low `t_index`, and low `t_index` is where the output stops correlating with
what is on screen. That trade is structural on the strength axis, so the levers left
are conditioning and the model, and the model's lever is the LoRA ecosystem: SD-Turbo
can load no SD 1.5 or SDXL LoRA at all.

Both models were run through the **shipped** selective path over the same clip under
the same plan, each at the step count it needs. `uv run python -m bench
selective-people --base-model {sd-turbo,sd15}` writes the arms to
`bench/results/base-models/` — their own directory, because an arm at another
checkpoint and another step count must not become the row §8.8 and §7.4 quote for
the shipped path. `base-models` and not `models`: `.gitignore` carries a bare
`models/` for the multi-GB downloads and it matches at any depth, so the obvious
name would have left every record here silently untracked. Regenerate with `uv run python -m bench --model-report`.

<!-- BEGIN BASE MODEL -->
The shipped selective path on each base model, over the same 48 frames of `people.mp4` under the same `person / lower_half / denoise 0.49` plan, on NVIDIA GeForce RTX 4090. Each model runs at the step count it needs rather than at a shared one: SD-Turbo is distilled to a single step and SD 1.5 is not, so an equal-step table would compare one model against a crippled one. The denoise is shared and *means* the same thing on both - the two checkpoints carry the same `scaled_linear` beta schedule, so `render_plan.t_index_for_denoise`'s ladder is the same ladder.

| base model | steps | regions/frame | ms/frame | +detect | FPS | headroom (ms) | flicker | background | 30 FPS |
|---|---|---|---|---|---|---|---|---|---|
| `sd-turbo-fp16` | 1 | 5.19 | 17.81 | 21.87 | 45.7 | +11.46 | 1.49 | identical | yes |
| `sd-v1-5-fp16` | 4 | 5.19 | 34.49 | 40.99 | 24.4 | -7.65 | 2.31 | identical | **no** |

**`sd-turbo-fp16` at 1 step: 30 FPS MET.** 45.7 FPS at 5.19 regions/frame on NVIDIA GeForce RTX 4090 - 21.87 ms per frame with detection amortised against the 33.33 ms a 30 FPS budget allows, 11.46 ms to spare; detect_every_n 5, clocks unlocked. Background: 48/48 frames left every pixel outside the rendered regions exactly as captured (164676 background pixels on the frame with the most painted).

**`sd-v1-5-fp16` at 4 steps: 30 FPS NOT MET.** 24.4 FPS at 5.19 regions/frame on NVIDIA GeForce RTX 4090 - 40.99 ms per frame with detection amortised against the 33.33 ms a 30 FPS budget allows, 1.23x the budget; detect_every_n 5, clocks unlocked. Background: 48/48 frames left every pixel outside the rendered regions exactly as captured (164676 background pixels on the frame with the most painted).

Every arm above left the background bit-identical to the capture.

Manual-verification artefacts, source | render: `selective-people-sd-turbo-20260907-183923Z-comparison.mp4`, `selective-people-sd15-20260907-183827Z-comparison.mp4`.
<!-- END BASE MODEL -->

**The denoise ladder survives the move, and that is measured rather than assumed.**
Issue #38's third trap warns that `render_plan.t_index_for_denoise` computes noise
amplitudes from SD-Turbo's own beta schedule. Both checkpoints' scheduler configs
carry `scaled_linear`, `beta_start 0.00085`, `beta_end 0.012`, 1000 train timesteps,
and `LCMScheduler.from_config` over either produces the same 50-step grid — index 30
is timestep 399 and index 45 is timestep 99 on both, with identical
`alphas_cumprod`. So the ladder means the same thing on SD 1.5, and the two arms
above are at one denoise honestly. What is *not* equal is the visible change it
buys: 14.5/255 on SD 1.5 against 11.8 on SD-Turbo, so 1.5 restyles slightly harder
at the same setting.

**30 FPS does not survive the move, and it is the steps rather than the model.**
§7.2's step-count block measures the two models within a few percent of each other
at four steps on `none`; what SD 1.5 costs is that it needs four. On TensorRT the
diffusion call alone is 33.9 ms against a 33.33 ms budget, before detection or
compositing — the frame lands at 40.99 ms with detection amortised, 24.4 FPS.

**Neither number disqualifies it.** The trade is visible and is a product decision
rather than a benchmark one: SD-Turbo at 45.7 FPS with no style LoRAs, or SD 1.5 at
24.4 FPS with the ecosystem's. §8.10 measures the second half of that sentence.
Background bit-identity held 48/48 on both arms, so the selective path's own
criterion is indifferent to the model.

## 8. Open questions (the research agenda)

### 8.1 Which detector?

**Settled: YOLO-World.** M0 measured it against YOLOv8n as a speed floor; the
numbers below are generated from `bench/results/detectors/`, not transcribed.

| Candidate      | Vocabulary            | Est. speed | Measured (RTX 3080 laptop, 640²) | Note                                                                       |
| -------------- | --------------------- | ---------- | -------------------------------- | -------------------------------------------------------------------------- |
| YOLOv8n/s      | 80 COCO classes       | fastest    | **12.8 ms/detect**, 1/3 concepts | speed floor only — "person", "car", "dog"                                  |
| YOLO-World (s, v2) | open, text-prompted | fast-ish  | **14.3 ms/detect**, 3/3 concepts | class embeddings precomputed per plan — fits the compile-once model exactly |
| OWLv2          | open                  | slower     | not measured — contingency        | better on unusual concepts                                                 |
| Grounding DINO | open, phrase grounding | slowest    | not measured — contingency        | handles "the red mug on the left"                                          |
| SAM2 / FastSAM | promptable segmentation | varies    | not measured                      | masks not boxes; may be what we actually want                              |

The open vocabulary costs **12% of one detect** against the closed-vocabulary
floor — 14.3 ms against 12.8 ms — and buys the two concepts the floor could not
reach. That is the whole decision. It is a *ranking*, and rankings transfer
between GPUs (§7.4); the milliseconds do not.

#### Measured

<!-- BEGIN DETECTOR TABLE -->

Measured: NVIDIA GeForce RTX 3080 Laptop GPU, 640x640 input, PyTorch. Diffusion resident during the timing: img2img-tensorrt-512x512-b1.

| detector | role | vocabulary | ms/detect | p95 ms | ms/detect at basis clock | amortised ms/frame | fits 4-8 ms | torch peak (MiB) | with diffusion resident (MiB) | vocabulary change (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo-world-s-640 | candidate | open | 14.32 | 16.98 | 11.00 @ 2100 MHz | 4.77 | yes | 3577 | 7182 | 16.2 |
| yolov8n-640 | speed floor | 80 COCO classes | 12.76 | 16.50 | 9.56 @ 2100 MHz | 4.25 | yes | 2541 | 6157 | n/a |

Vocabulary evidence - what each detector returned when asked for the concept, and what a closed vocabulary had to be asked for instead:

| concept | kind | detector | asked for | resolved | top confidence | strongest other label | frame |
|---|---|---|---|---|---|---|---|
| person | COCO class | yolo-world-s-640 | person | yes | 0.827 | - | desktop capture (mss) with the photo composited in |
| red mug | open vocabulary | yolo-world-s-640 | red mug | yes | 0.976 | - | desktop capture (mss) with the photo composited in |
| dog | non-COCO animal | yolo-world-s-640 | dog | yes | 0.916 | - | desktop capture (mss) with the photo composited in |
| person | COCO class | yolov8n-640 | person | yes | 0.801 | bus 0.73 | desktop capture (mss) with the photo composited in |
| red mug | open vocabulary | yolov8n-640 | - | no | - | - | desktop capture (mss) with the photo composited in |
| dog | non-COCO animal | yolov8n-640 | dog | no | - | cat 0.81 | desktop capture (mss) with the photo composited in |

**Recommendation: yolo-world-s-640.** yolo-world-s-640 resolved all 3 probed concepts and 11.00 ms per detect at one detect every 3rd frame is 3.67 ms/frame amortised, inside the 4-8 ms budget. It is not the fastest (yolov8n-640 is), and that is not the criterion: with the prompt compiler cut from v1, a closed vocabulary caps the product at the 80 COCO nouns whatever it costs. Ranked on the clock-normalised estimate - the `ms/detect at basis clock` column, not the raw one - because the clocks were not locked and these rows were measured minutes apart, so their raw figures carry two different clocks (issue #13). The estimate is an estimate; the ranking is what it is used for.

<!-- END DETECTOR TABLE -->

Regenerate with `uv run python -m bench --detector-report`. A new detector result
fails the merge gate until this block is regenerated.

#### What the numbers say

- **It fits, at one detect every 3rd frame.** 14.3 ms amortises to 4.8 ms/frame,
  inside §7.1's 4–8 ms detection budget, with the 512² TensorRT diffusion engine
  resident throughout. At every frame it would be 14.3 ms and would not fit; the
  cadence is part of the claim, not a footnote.
- **A vocabulary change costs 16 ms and does not touch the frame path — almost.**
  Re-encoding three terms with CLIP ViT-B/32 takes ~16 ms on the cold path, and
  detecting against the changed vocabulary is no dearer than before it (−7.9%,
  inside the noise of an unlocked laptop clock). But ultralytics'
  `YOLOWorld.set_classes` sets `self.predictor = None`, so **the first detect
  after a change costs ~108 ms more than a steady one** while the predictor is
  rebuilt. Confirmed as the cause: a bare `predictor = None` costs the same.
  Mitigation: issue one throwaway detect on the cold path, after the change and
  before the new plan goes live. The frame path then never sees it.
- **Combined VRAM: 7,182 MiB** with the diffusion engine and the detector both
  resident (`nvidia-smi`, whole device), against 5,620 MiB for the engine on its
  own — YOLO-World adds ~1.5 GiB. The floor adds far less (6,157 MiB total), which
  is the other thing the open vocabulary is paid for in. On a 16 GB laptop that
  leaves headroom; **this is the §7.4 non-portable kind of number and has to be
  re-measured on the deploy GPU.**
- **The floor is an accuracy floor too.** Asked for `dog` on a desktop showing
  one, YOLOv8n returned no dog and a **cat at 0.81**. "Fastest" was never the
  criterion, but it is worth recording that the cheap option was also the wrong
  answer here.
- Evidence frames are real desktop captures with the source photograph
  composited in, since restyling the screen is what the product does. The raw
  screen is captured as a control. DXcam — what the app uses — cannot open a
  Desktop Duplication context in this session, so the captures came from `mss`;
  that is a property of the capture session, not of the detector.

#### In the worker (issue #7)

The detector now runs where the product needs it rather than only in the harness,
and the mitigation above is built rather than recommended: `BackgroundDetector`
sets the vocabulary and issues the throwaway detect on its own thread, before the
first frame is offered against the new plan, so a prompt edit never lands the
~108 ms predictor rebuild on a frame.

Re-measured through the shipped path, on the committed 1280×720 `people.mp4`
clip with `person` as the target, RTX 3080 laptop, **unlocked clocks at a sampled
1575 MHz**: **18.8 ms per detect**, six to seven people per frame, which amortises
to **4.7 ms/frame at one detect every 3rd frame** and 2.4 ms/frame at every 6th —
inside §7.1's 4–8 ms row on the same reading as the 640² figure above. Handing a
frame to the detector costs the frame loop **0.03 ms**, and one track ID survived
all 30 consecutive frames of the clip.

Absolutes here are worth exactly what §7.4 and issue #13 say they are. The same
detector through the same `bench.detector_runner` code measured **191 ms per
detect** on this machine an hour earlier, with the card sitting at 435–900 MHz and
19–24 W instead of boost. Nothing about the code changed; the clock did. The two
claims that survive it are the ratio (half as many detects cost half as much) and
the identity (an ID is the same ID), and those are what the tests assert.

The instruction "give **them** a red hat" needs a *head region*, not a person
box. Boxes may be too coarse; a segmentation stage or a body-part heuristic
(`region: "upper_third"`) may be required. **Still open** — YOLO-World settles
*which* detector, not whether a box is the right primitive (§8.2).

### 8.2 Is crop-and-diffuse the right primitive at all?

Honest risk: img2img on a 64×64 person crop upscaled to 512² will hallucinate
detail, drift in identity frame to frame, and flicker. The four options:

- **A. Crop → diffuse → composite** (this spec's original default).
- **B. Full-frame diffuse once, masked composite.** Constant cost, temporally
  smoother, but every object gets the same prompt.
- **C. Latent-space masking / inpainting.** Diffuse the full latent, blending
  masked and unmasked latents per step. Cheaper than A, more localised than B —
  but SD-Turbo's 1-step schedule leaves little room to blend.
- **D. ControlNet-conditioned.** Would need the dead `controlnet_paths` stub
  wired up plus a TRT engine that supports it. Best structure preservation,
  highest cost.

A and B were implemented offline and measured against each other on the committed
reference clips; C and D were assessed from the source rather than built, which is
what issue #5 asked for.

#### Decision (2026-09-06)

Generated from `bench/results/primitives/`, not transcribed. Regenerate with
`uv run python -m bench --primitive-report`; a new comparison fails the merge gate
until this block is regenerated.

<!-- BEGIN PRIMITIVE DECISION -->

Measured: NVIDIA GeForce RTX 3080 Laptop GPU, engine img2img-tensorrt-512x512-b1. Committed clips: dog.mp4 (48 consecutive frames), people.mp4 (48 consecutive frames). Both primitives are timed on the same frames, interleaved frame by frame, so the laptop's clock drift lands on both arms.

| case | primitive | option | denoise t_index | strength | objects/frame | calls/frame | ms/frame | ms/frame at basis clock | flicker (static px) | expresses |
|---|---|---|---|---|---|---|---|---|---|---|
| restyle-people | crop | A | 35 | 0.64 | 6.00 | 6.00 | 931.23 | 461.26 @ 2100 MHz | 1.62 (225146) | yes |
| restyle-people | masked | B | 40 | 0.49 | 6.00 | 1.00 | 158.57 | 78.55 @ 2100 MHz | 1.43 (225146) | yes |
| identity-dog | crop | A | 20 | 0.92 | 1.00 | 1.00 | 81.75 | 46.40 @ 2100 MHz | 17.38 (16341) | no |
| identity-dog | masked | B | 25 | 0.85 | 1.00 | 1.00 | 85.73 | 48.67 @ 2100 MHz | 19.50 (16341) | yes |

Denoise strength each case turned out to need:

- **restyle-people / crop:** t_index 35 - timestep 299, denoise strength 0.64. Selected as the least denoise whose mean absolute change inside the region, net of the resize control, reached 8/255; it changed the region by 9.4/255 - 8.6 of it net of the 0.8/255 the resize alone costs - against 0.0/255 outside it.
- **restyle-people / masked:** t_index 40 - timestep 199, denoise strength 0.49. Selected as the least denoise whose mean absolute change inside the region, net of the resize control, reached 8/255; it changed the region by 13.1/255 - 10.5 of it net of the 2.6/255 the resize alone costs - against 0.0/255 outside it.
- **identity-dog / crop:** No rung of the ladder met the criterion (the least denoise at which the detector read at least 50% of the probed frames as the new identity). The strongest denoise tried was t_index 20 (timestep 599, strength 0.92), which changed the region by 24.1/255 net of resizing and was read as the new identity in 1/4 frames. The comparison was timed at that setting and the case is recorded as not achieved.
- **identity-dog / masked:** t_index 25 - timestep 499, denoise strength 0.85. Selected as the least denoise at which the detector read at least 50% of the probed frames as the new identity; it changed the region by 36.5/255 - 33.5 of it net of the 3.0/255 the resize alone costs - against 0.0/255 outside it, and was read as the new identity in 3/4 probed frames.

- **A, crop -> diffuse -> composite (`crop`) cannot express:** A small crop, cheaply. Every region gets the engine's full 512x512 canvas whatever its source size, so a 45 px-wide region is upscaled 11x before it is diffused and the model invents the detail it finds there - the small-crop quality floor. Cost scales with the object count, so the number of objects becomes a frame-budget decision rather than a detection one; and each object is diffused with no sight of the rest of the frame, so nothing ties two objects' output together.
- **B, full-frame diffuse, masked composite (`masked`) cannot express:** One prompt and one denoise per frame. Every object in the frame is rendered from the same text embedding at the same strength, so "turn the dog into a cat and the man into a statue" is two frames' work, not one. Nor can it spend detail where it matters: the whole frame is squeezed onto one 512x512 canvas, so a region occupying 45 px of a 1280 px-wide frame is diffused at ~18 px and comes back with roughly that much detail.

Findings:

- **Small objects, people.mp4:** 288 rendered regions, smallest side 45 px and largest smallest-side 293 px. 48 have a side under 96 px; 0 are under 96 px in *both* dimensions. The Gate's small-object case is therefore met on the smallest-side reading and not on the small-in-area one - the gap issue #17 recorded when it committed these clips, and it is still open.
- **identity-dog separates the two primitives:** crop did not express it at any rung of the ladder and masked did. That is the half of the comparison no millisecond figure carries.
- **Small objects, dog.mp4:** 48 rendered regions, smallest side 138 px and largest smallest-side 192 px. 0 have a side under 96 px; 0 are under 96 px in *both* dimensions. The Gate's small-object case is therefore met on the smallest-side reading and not on the small-in-area one - the gap issue #17 recorded when it committed these clips, and it is still open.

**Decision: masked.** **B, full-frame diffuse, masked composite** (`masked`). On the priority case (restyle-people) it costs 158.6 ms/frame at 6.0 objects per frame, against 931.2 ms/frame for crop (5.87x), which expressed it too. Flicker over the pixels static in the source, 0-255 units, lower steadier: crop 1.62, masked 1.43. On the eventual case (identity-dog) it held up at 85.7 ms/frame. What it cannot express: One prompt and one denoise per frame. Every object in the frame is rendered from the same text embedding at the same strength, so "turn the dog into a cat and the man into a statue" is two frames' work, not one. Nor can it spend detail where it matters: the whole frame is squeezed onto one 512x512 canvas, so a region occupying 45 px of a 1280 px-wide frame is diffused at ~18 px and comes back with roughly that much detail. That limitation is accepted for v1 - one concept at a time - and is recorded here rather than discovered later.

Side-by-side clips for human judgement (source | A | B), under `bench/results/primitives/`: `restyle-people-20260906-181553Z-triptych.mp4`, `identity-dog-20260906-181841Z-triptych.mp4`. **The metric ranks cost and temporal stability, not beauty** - a human still has to watch these and confirm the priority case is acceptable.

<!-- END PRIMITIVE DECISION -->

#### How the comparison was run

- **One clip, one committed box track, per case.** `bench/clips/people.mp4` for the
  priority case and `bench/clips/dog.mp4` for the identity change (issue #17). The
  boxes come from `bench/clips/*.track.json`, generated once with YOLO-World and
  committed, so two runs render exactly the same regions. Box smoothing — one of
  §8.5's own levers — is applied when the track is written, so neither primitive is
  charged for detector jitter and both see identical regions.
- **Both primitives are timed on the same frames, interleaved frame by frame.** One
  arm after the other would have compared a boost clock against a throttled one
  (§7.4, issue #13).
- **What is timed is the whole primitive** — resizes and composite included —
  because the frame loop pays those too. Both render through the one cached
  512² batch-1 TensorRT engine, so the A-against-B ratio is a property of the
  primitives and not of two engines.
- **The denoise strength is measured, not assumed.** Each case is swept over
  `t_index` 20–45 before the timed pass; higher `t_index` is *less* denoise (index
  20 is timestep 599, noise amplitude 0.92; index 45 is timestep 99, 0.32). The
  restyle case selects the least denoise whose change inside the region reaches
  8/255. The identity case cannot be selected that way — a frame can change
  enormously and still be a dog — so it is judged by asking YOLO-World whether the
  rendered subject now reads as a cat.
- **Flicker** is the mean absolute difference between consecutive outputs over the
  pixels that were static in the source *and* painted by the primitive, in 0–255
  units. Both restrictions matter: a moving subject is the render working, and the
  untouched pass-through pixels are identical by construction, so leaving them in
  would rank the primitive that restyles least as the steadiest. `bench/flicker.py`,
  pure, unit-tested.
- **B's `region_change` includes its own resampling loss.** B squeezes the whole
  frame onto the 512² canvas and stretches it back, so part of what the change
  figure measures inside the region is blur rather than style. That is not an
  artefact of the measurement — it is what B costs — but it does mean B's change
  figure is not a like-for-like measure of style against A's.

#### What the comparison changed about this section's assumptions

Two of them, and both went against what §8.2 assumed when it was written.

- **The identity change is where A was supposed to be necessary, and A is the one
  that failed it.** Asked for `dog` and `cat` on the rendered frames, YOLO-World
  read B's output as a cat in 37 of 48 frames and A's in 12. A upscales a 424×280
  dog to 512² and diffuses it with no sight of the rest of the frame; B diffuses the
  whole scene, and the scene is what carries a cat. So "A's dedicated 512² canvas
  per object may be necessary for an identity change" is not what the measurement
  says.
- **A is not steadier, and B is not dramatically smoother either.** On the priority
  case the two are within 12% of each other on flicker (1.62 against 1.43, in 0–255
  units over 225k static painted pixels per pair), which is far less separation than
  the cost difference. Temporal stability was expected to be B's argument; it is not
  the argument that decides this.

Two defects the stills show that no metric here scores, both visible in
`*-triptych.jpg`:

- **A squashes a non-square region onto a square canvas.** Every engine this app
  builds is 512×512 (§7.2 — the resolution in an engine directory name is a lie), so
  a 424×280 dog box is stretched to a square, diffused, and stretched back. In the
  identity-case still that is most of why A returns a dog's face at the wrong scale.
  Letterboxing the crop would fix the aspect and throw away canvas; a non-square
  engine is a rebuild. Neither is free, and it is a cost that belongs to A alone.
- **Both primitives leave a hard rectangular seam at the region boundary** once the
  denoise is high enough to change anything. `outside_change` is 0.0/255 by
  construction — the composite is exact — so the discontinuity lands entirely on the
  box edge. Feathering the composite is the obvious mitigation and is unbuilt; it
  belongs to whoever implements the chosen primitive, and it is cheap for B (one
  alpha blend per frame) and K times dearer for A.

One caveat on the absolute figures, and it is §7.4's: the priority-case run was
timed with the cooldown gate **capped rather than reached** — the laptop never got
under 62 °C — so its milliseconds are throttled ones. An earlier run of the same
comparison on a cooler die measured 490 ms/frame for A and 87 for B, against 931 and
159 here. The **ratio** moved from 5.62× to 5.87×; the absolutes moved by ~90%. The
ratio is the portable conclusion, the absolutes are not, and neither is a claim
about hitting 30 FPS.

#### C and D, assessed rather than built

**C — latent-space masking — is not a third primitive on a one-step schedule.**
Its idea is to blend masked and unmasked latents *between* denoising steps, and
`DEFAULT_T_INDEX_LIST` is a single rung: every engine this app has is built for one
step. With one step there is no intermediate latent, so the blend happens once,
after the only UNet pass — which is B, with the mask moved from pixel space into a
64×64 latent grid. That costs exactly what B costs and buys an 8×-coarser region
edge. It becomes interesting only once the pipeline has more than one step, and
more steps is the engine rebuild §7.2 prices, not a setting.
`tests/test_primitive_options_c_and_d.py` pins the step count so this assessment
fails rather than rots.

**D — ControlNet-conditioned — is a fork, not a flag.** `controlnet_paths` and
`controlnet_scales` are accepted by `image_generation_process()` and reach nothing:
the worker never passes them on, `wrapper.py` does not mention ControlNet, and
neither does the pinned StreamDiffusion 0.1.1. So D means a ControlNet-aware
pipeline that does not exist here, plus a TensorRT engine taking the extra
conditioning inputs — another ~5.0 GB and 15–25 minutes per configuration
(§7.2) — plus the per-frame cost of the conditioning pass itself. The issue's
instruction was to implement D only if A and B both failed the priority case. They
did not, so it stays unbuilt. The same test pins the three places ControlNet is
absent, so if a dependency ever ships one this assessment fails rather than rots.

**This comparison happened before anything else was built**, which is what this
section asked for.

#### Re-taken at K=1, with the capture decoupled from the canvas (issue #39)

The decision above was taken under a premise that has since been reversed. It ranked
the primitives at **six objects a frame**, where A's 5.87× penalty is entirely its
call count — and it recorded, as B's cost, exactly the thing live testing later ran
into:

> "Nor can it spend detail where it matters: the whole frame is squeezed onto one
> 512×512 canvas, so a region occupying 45 px of a 1280 px-wide frame is diffused at
> ~18 px and comes back with roughly that much detail."

Two things changed on 2026-09-07. The product decision: **one object at high detail
is worth more than all objects at low detail**, so K is 1. And the app's capture
geometry stopped being the engine's canvas — a 512×512 capture window cannot get a
whole object into frame, the engine is 512×512 whatever is captured (§7.2), and the
frame loop resizes between them. Under `crop` that resize is of one region rather
than of the whole frame, which is where the detail comes from.

At K=1 the cost argument that decided the section above does not apply: A and B are
**one diffusion call each**. §8.2's own table already showed it — on `identity-dog`,
at one object, crop measured 81.75 ms/frame against masked's 85.73.

The block below is generated from `bench/results/capture/`, not transcribed.
Regenerate it with `uv run python -m bench --capture-report`; a new comparison fails
the merge gate until it is regenerated. Every arm renders the same region of the same
frame of the same committed clip through the same 512×512 engine, at the strength its
own denoise sweep selected, at three capture geometries.

Two things the run does *not* claim. The clip's pixels at 1920×1080 are an upscale of
its own 1280×720, because the committed tracks are what make two arms comparable and
regenerating one invalidates every committed comparison — so the *cost* figures are
exact and the detail figures measure what each primitive does with one source rather
than what a true 1080p source would give. And detection is not in the frame path
here, for the same reason: the boxes come from the committed track.

<!-- BEGIN CAPTURE GEOMETRY -->
Measured: NVIDIA GeForce RTX 4090, engine img2img-tensorrt-512x512-b1 - one 512x512 canvas, whatever the capture is (spec 7.2). Committed clips: dog.mp4 (48 frames), people.mp4 (48 frames), resized to each capture geometry the way the worker's capture thread resizes a screen region. Every arm renders the same regions of the same frames at K=1 through the same engine at the same strength, so what differs between two rows is the primitive and the geometry.

| case | arm | capture | denoise | object px | detail gain | resize in | diffuse | composite | host copy | IPC put | frame path | FPS | 30 FPS | net change | flicker | background |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| capture-dog | masked | 512x512 | 0.49 | 156 | 1.00x | 0.01 | 15.64 | 0.95 | 0.15 | 0.05 | 16.65 | 60.1 | yes | 15.8 | 5.82 | identical |
| capture-dog | crop | 512x512 | 0.85 | 512 | 3.29x | 0.08 | 15.54 | 0.95 | 0.16 | 0.01 | 16.58 | 60.3 | yes | 15.9 | 9.72 | identical |
| capture-dog | masked | 1280x720 | 0.85 | 99 | 0.40x | 0.11 | 17.58 | 2.21 | 0.34 | 0.01 | 19.91 | 50.2 | yes | 28.7 | 16.53 | identical |
| capture-dog | crop | 1280x720 | 0.92 | 512 | 2.08x | 0.09 | 16.06 | 2.09 | 0.34 | 0.01 | 18.25 | 54.8 | yes | 22.6 | 14.40 | identical |
| capture-dog | masked | 1920x1080 | 0.85 | 99 | 0.27x | 0.16 | 16.56 | 5.09 | 1.35 | 0.01 | 21.82 | 45.8 | yes | 29.0 | 16.74 | identical |
| capture-dog | crop | 1920x1080 | 0.92 | 512 | 1.39x | 0.12 | 16.76 | 4.85 | 1.35 | 0.01 | 21.75 | 46.0 | yes | 23.4 | 14.79 | identical |
| capture-people | masked | 512x512 | 0.32 | 123 | 1.00x | 0.01 | 15.38 | 0.46 | 0.22 | 0.12 | 15.97 | 62.6 | yes | 9.1 | 1.07 | identical |
| capture-people | crop | 512x512 | 0.76 | 512 | 4.16x | 0.06 | 15.50 | 0.53 | 0.16 | 0.01 | 16.10 | 62.1 | yes | 8.8 | 1.30 | identical |
| capture-people | masked | 1280x720 | 0.49 | 69 | 0.40x | 0.11 | 17.08 | 1.34 | 0.52 | 0.01 | 18.54 | 53.9 | yes | 10.0 | 1.18 | identical |
| capture-people | crop | 1280x720 | 0.76 | 512 | 2.96x | 0.08 | 15.96 | 1.29 | 0.37 | 0.01 | 17.34 | 57.7 | yes | 9.8 | 1.62 | identical |
| capture-people | masked | 1920x1080 | 0.49 | 69 | 0.27x | 0.14 | 17.12 | 3.32 | 0.67 | 0.01 | 20.60 | 48.5 | yes | 10.3 | 1.18 | identical |
| capture-people | crop | 1920x1080 | 0.64 | 512 | 1.98x | 0.10 | 17.25 | 3.27 | 1.38 | 0.01 | 20.63 | 48.5 | yes | 9.0 | 1.31 | identical |

**Non-target pixels stayed bit-identical to the capture** on all 576 rendered frames, at every geometry measured (1280x720, 1920x1080, 512x512) and under both primitives. The criterion does not get easier because the frame got bigger, and it did not have to.

At 512x512, `crop` diffused the region at 512 px against `masked`'s 156 - 3.3x the detail - for 16.58 ms against 16.65 ms (1.00x). Visible change inside the region net of the resize control: crop 15.9/255 at strength 0.85, masked 15.8/255 at 0.49, against a 8/255 threshold. Read as the new identity in: crop 9/48, masked 23/48 frames.

At 1280x720, `crop` diffused the region at 512 px against `masked`'s 99 - 5.2x the detail - for 18.25 ms against 19.91 ms (0.92x). Visible change inside the region net of the resize control: crop 22.6/255 at strength 0.92, masked 28.7/255 at 0.85, against a 8/255 threshold. Read as the new identity in: crop 10/48, masked 38/48 frames.

At 1920x1080, `crop` diffused the region at 512 px against `masked`'s 99 - 5.2x the detail - for 21.75 ms against 21.82 ms (1.00x). Visible change inside the region net of the resize control: crop 23.4/255 at strength 0.92, masked 29.0/255 at 0.85, against a 8/255 threshold. Read as the new identity in: crop 9/48, masked 39/48 frames.

**What 512x512 -> 1920x1080 (7.9x the pixels) cost, per stage, under `masked`:** resize in 0.01 -> 0.16 ms (19.1x); diffuse 15.64 -> 16.56 ms (1.1x); composite 0.95 -> 5.09 ms (5.3x); host copy 0.15 -> 1.35 ms (9.3x); ipc put 0.05 -> 0.01 ms (0.2x); ipc roundtrip 0.65 -> 9.03 ms (13.9x); preview 0.36 -> 10.00 ms (28.1x). The diffusion call is the one stage that *cannot* move - the canvas is fixed at 512x512 (spec 7.2), and a bare probe on this card measures it at 15.5 / 15.3 / 16.8 ms at the three geometries - so what the table shows in that column is the allocation pressure a bigger frame puts on the same call, not a bigger call. Everything else in the list is the price of the capture, and issue #31's 9.58 ms of headroom is a 512x512 figure that does not transfer.

Denoise strength each arm turned out to need:
- **masked-512x512:** t_index 40 - timestep 199, denoise strength 0.49. Selected as the least denoise at which the detector read at least 50% of the probed frames as the new identity; it changed the region by 17.7/255 - 17.7 of it net of the 0.0/255 the resize alone costs - against 0.0/255 outside it, and was read as the new identity in 2/4 probed frames.
- **crop-512x512:** t_index 25 - timestep 499, denoise strength 0.85. Selected as the least denoise at which the detector read at least 50% of the probed frames as the new identity; it changed the region by 18.7/255 - 16.8 of it net of the 1.9/255 the resize alone costs - against 0.0/255 outside it, and was read as the new identity in 3/4 probed frames.
- **masked-1280x720:** t_index 25 - timestep 499, denoise strength 0.85. Selected as the least denoise at which the detector read at least 50% of the probed frames as the new identity; it changed the region by 31.5/255 - 28.9 of it net of the 2.5/255 the resize alone costs - against 0.0/255 outside it, and was read as the new identity in 4/4 probed frames.
- **crop-1280x720:** t_index 20 - timestep 599, denoise strength 0.92. Selected as the least denoise at which the detector read at least 50% of the probed frames as the new identity; it changed the region by 24.5/255 - 23.4 of it net of the 1.0/255 the resize alone costs - against 0.0/255 outside it, and was read as the new identity in 2/4 probed frames.
- **masked-1920x1080:** t_index 25 - timestep 499, denoise strength 0.85. Selected as the least denoise at which the detector read at least 50% of the probed frames as the new identity; it changed the region by 31.9/255 - 29.4 of it net of the 2.5/255 the resize alone costs - against 0.0/255 outside it, and was read as the new identity in 4/4 probed frames.
- **crop-1920x1080:** t_index 20 - timestep 599, denoise strength 0.92. Selected as the least denoise at which the detector read at least 50% of the probed frames as the new identity; it changed the region by 24.9/255 - 24.4 of it net of the 0.5/255 the resize alone costs - against 0.0/255 outside it, and was read as the new identity in 2/4 probed frames.

30 FPS on the frame path, per arm:
- **masked-512x512:** MET - 16.65 ms on the frame path against 33.33 ms, 60.1 FPS at 1.00 regions/frame on NVIDIA GeForce RTX 4090.
- **crop-512x512:** MET - 16.58 ms on the frame path against 33.33 ms, 60.3 FPS at 1.00 regions/frame on NVIDIA GeForce RTX 4090.
- **masked-1280x720:** MET - 19.91 ms on the frame path against 33.33 ms, 50.2 FPS at 1.00 regions/frame on NVIDIA GeForce RTX 4090.
- **crop-1280x720:** MET - 18.25 ms on the frame path against 33.33 ms, 54.8 FPS at 1.00 regions/frame on NVIDIA GeForce RTX 4090.
- **masked-1920x1080:** MET - 21.82 ms on the frame path against 33.33 ms, 45.8 FPS at 1.00 regions/frame on NVIDIA GeForce RTX 4090.
- **crop-1920x1080:** MET - 21.75 ms on the frame path against 33.33 ms, 46.0 FPS at 1.00 regions/frame on NVIDIA GeForce RTX 4090.

Detection is **not** in these figures: every arm reads its boxes from the committed track, which is what makes two arms comparable (issue #5's second trap). Spec 8.8's cadence sweep measures detection at ~4.3 ms per frame amortised at `detect_every_n` 5 on this card at 512x512, so an arm with less than that in hand is not a 30 FPS claim about the shipped app.

**Recommended for capture-dog on NVIDIA GeForce RTX 4090: `masked-1280x720`** - the region diffused at 99 px (0.40x its size in the capture), 19.91 ms on the frame path (50.2 FPS), background identical. It is the most canvas an object got among the arms that fit the budget and expressed the case. It is **not** a recommendation against a larger capture: under `masked` the object keeps the same fraction of a frame that is squeezed onto one canvas, so it lands on the same canvas pixels either way. Detail therefore cannot separate the geometries and the tie falls to cost. What 1920x1080 buys instead is field of view - a screen region big enough to hold the object at all, which is what issue #39 was opened about and the one thing no metric here scores - and what it costs is 1.91 ms/frame (21.82 ms, 45.8 FPS).

At 512x512, `crop` diffused the region at 512 px against `masked`'s 123 - 4.2x the detail - for 16.10 ms against 15.97 ms (1.01x). Visible change inside the region net of the resize control: crop 8.8/255 at strength 0.76, masked 9.1/255 at 0.32, against a 8/255 threshold. Both expressed the case.

At 1280x720, `crop` diffused the region at 512 px against `masked`'s 69 - 7.4x the detail - for 17.34 ms against 18.54 ms (0.94x). Visible change inside the region net of the resize control: crop 9.8/255 at strength 0.76, masked 10.0/255 at 0.49, against a 8/255 threshold. Both expressed the case.

At 1920x1080, `crop` diffused the region at 512 px against `masked`'s 69 - 7.4x the detail - for 20.63 ms against 20.60 ms (1.00x). Visible change inside the region net of the resize control: crop 9.0/255 at strength 0.64, masked 10.3/255 at 0.49, against a 8/255 threshold. Both expressed the case.

**What 512x512 -> 1920x1080 (7.9x the pixels) cost, per stage, under `masked`:** resize in 0.01 -> 0.14 ms (20.9x); diffuse 15.38 -> 17.12 ms (1.1x); composite 0.46 -> 3.32 ms (7.1x); host copy 0.22 -> 0.67 ms (3.1x); ipc put 0.12 -> 0.01 ms (0.1x); ipc roundtrip 1.28 -> 9.95 ms (7.8x); preview 0.37 -> 9.80 ms (26.2x). The diffusion call is the one stage that *cannot* move - the canvas is fixed at 512x512 (spec 7.2), and a bare probe on this card measures it at 15.5 / 15.3 / 16.8 ms at the three geometries - so what the table shows in that column is the allocation pressure a bigger frame puts on the same call, not a bigger call. Everything else in the list is the price of the capture, and issue #31's 9.58 ms of headroom is a 512x512 figure that does not transfer.

Denoise strength each arm turned out to need:
- **masked-512x512:** t_index 45 - timestep 99, denoise strength 0.32. Selected as the least denoise whose mean absolute change inside the region, net of the resize control, reached 8/255; it changed the region by 9.1/255 - 9.1 of it net of the 0.0/255 the resize alone costs - against 0.0/255 outside it.
- **crop-512x512:** t_index 30 - timestep 399, denoise strength 0.76. Selected as the least denoise whose mean absolute change inside the region, net of the resize control, reached 8/255; it changed the region by 11.2/255 - 9.5 of it net of the 1.6/255 the resize alone costs - against 0.0/255 outside it.
- **masked-1280x720:** t_index 40 - timestep 199, denoise strength 0.49. Selected as the least denoise whose mean absolute change inside the region, net of the resize control, reached 8/255; it changed the region by 12.2/255 - 9.7 of it net of the 2.6/255 the resize alone costs - against 0.0/255 outside it.
- **crop-1280x720:** t_index 30 - timestep 399, denoise strength 0.76. Selected as the least denoise whose mean absolute change inside the region, net of the resize control, reached 8/255; it changed the region by 10.8/255 - 9.9 of it net of the 0.9/255 the resize alone costs - against 0.0/255 outside it.
- **masked-1920x1080:** t_index 40 - timestep 199, denoise strength 0.49. Selected as the least denoise whose mean absolute change inside the region, net of the resize control, reached 8/255; it changed the region by 12.2/255 - 9.9 of it net of the 2.3/255 the resize alone costs - against 0.0/255 outside it.
- **crop-1920x1080:** t_index 35 - timestep 299, denoise strength 0.64. Selected as the least denoise whose mean absolute change inside the region, net of the resize control, reached 8/255; it changed the region by 9.4/255 - 9.0 of it net of the 0.4/255 the resize alone costs - against 0.0/255 outside it.

30 FPS on the frame path, per arm:
- **masked-512x512:** MET - 15.97 ms on the frame path against 33.33 ms, 62.6 FPS at 1.00 regions/frame on NVIDIA GeForce RTX 4090.
- **crop-512x512:** MET - 16.10 ms on the frame path against 33.33 ms, 62.1 FPS at 1.00 regions/frame on NVIDIA GeForce RTX 4090.
- **masked-1280x720:** MET - 18.54 ms on the frame path against 33.33 ms, 53.9 FPS at 1.00 regions/frame on NVIDIA GeForce RTX 4090.
- **crop-1280x720:** MET - 17.34 ms on the frame path against 33.33 ms, 57.7 FPS at 1.00 regions/frame on NVIDIA GeForce RTX 4090.
- **masked-1920x1080:** MET - 20.60 ms on the frame path against 33.33 ms, 48.5 FPS at 1.00 regions/frame on NVIDIA GeForce RTX 4090.
- **crop-1920x1080:** MET - 20.63 ms on the frame path against 33.33 ms, 48.5 FPS at 1.00 regions/frame on NVIDIA GeForce RTX 4090.

Detection is **not** in these figures: every arm reads its boxes from the committed track, which is what makes two arms comparable (issue #5's second trap). Spec 8.8's cadence sweep measures detection at ~4.3 ms per frame amortised at `detect_every_n` 5 on this card at 512x512, so an arm with less than that in hand is not a 30 FPS claim about the shipped app.

**Recommended for capture-people on NVIDIA GeForce RTX 4090: `crop-512x512`** - the region diffused at 512 px (4.16x its size in the capture), 16.10 ms on the frame path (62.1 FPS), background identical. It is the most canvas an object got among the arms that fit the budget and expressed the case. It is **not** a recommendation against a larger capture: under `crop` the object gets the whole canvas whatever is captured. Detail therefore cannot separate the geometries and the tie falls to cost. What 1920x1080 buys instead is field of view - a screen region big enough to hold the object at all, which is what issue #39 was opened about and the one thing no metric here scores - and what it costs is 4.54 ms/frame (20.63 ms, 48.5 FPS).

Side-by-side clips for human judgement, under `bench/results/capture/`: `capture-dog-20260907-173054Z-triptych.mp4`, `capture-people-20260907-172929Z-triptych.mp4`. **The metric ranks cost, detail and steadiness, not beauty** - a human still has to watch these and confirm the crop's upscaled invention is acceptable.
<!-- END CAPTURE GEOMETRY -->

#### What this changed about the decision above

**The decision splits by case, and the 5.87× that made it does not survive K=1.**
At one object A and B are one diffusion call each and cost within 8% of one another
at every geometry measured (0.92–1.01×). What separates them is what they spend that
call on.

- **For the priority case — a sub-region restyle — `crop` is the primitive.** It
  diffuses the region at the full 512 px against masked's 69 at 1080p (7.4×), it
  expresses the case (9.0/255 net against an 8/255 threshold), and it costs 20.63 ms
  against masked's 20.60. That is the limitation §8.2 recorded and accepted,
  removed, at no measurable cost in milliseconds.
- **For the identity change `masked` still wins, and by the margin §8.2 already
  found.** Asked whether the rendered subject is a cat, YOLO-World read masked's
  output as one in 38–39 of 48 frames at 720p and 1080p, and crop's in 9–10 — at
  crop's *own* selected strength, on the same frames. The reason is the one this
  section already names: a 424×280 box stretched onto a square canvas comes back at
  the wrong aspect and the wrong scale, and a larger capture does not fix a
  distortion that is proportional. `capture-dog-*-triptych.jpg` is what it looks
  like. **Crop lost `identity-dog` at the new geometry too, and that is the
  finding.**

**Raising the capture on its own would have been a regression, and the table shows
it.** Under `masked` the same object drops from 123 canvas px at 512×512 to 69 at
1080p — 0.27× its own size in the frame — because the whole frame is squeezed onto
one canvas whatever the frame is. Under `crop` it is 512 px at every geometry. That
is why the capture size and `global.primitive` landed in one change.

**What the larger capture costs is the frame's periphery, not its centre.** Per
stage, 512×512 → 1920×1080 under `masked` on the priority case: the resize onto the
canvas 0.01 → 0.14 ms, the composite 0.46 → 3.32 ms, the device-to-host copy 0.22 →
0.67 ms, and the diffusion call flat. In the GUI process, which the frame budget
does not pay but a viewer does: the IPC round trip 1.28 → 9.95 ms and the preview
0.37 → 9.80 ms. Issue #31's 9.58 ms of headroom was a 512×512 figure and does not
transfer, which the trap predicted; what does transfer is that **30 FPS survives**
— 20.60 ms on the frame path at 1080p at 1.00 region/frame on an RTX 4090, leaving
12.7 ms for the ~4.3 ms/frame §8.8 measures detection at.

**What ships, and what does not.** The lever is a plan field (`global.primitive`)
and a GUI box — *Detail*, beside Target and Style, which sets the primitive and
`max_instances` together because `crop` above one slot falls back to `masked` on
every frame. The **defaults do not move**: `masked`, `max_instances: 6`, a 512×512
capture. Every committed measurement in §7.4, §8.5, §8.8 and §8.9 was taken there,
and adopting crop as the default means moving `max_instances` to 1 with it — one
product decision and a re-measurement of those four sections, which is its own
change. The recommendation this run makes is on the record; the field behind it is
not moved by it.

### 8.3 Per-object prompts

Different objects with different prompts means different text embeddings in one
batch. StreamDiffusion caches a single prompt embedding. We need to confirm the
embedding can be supplied per batch item at runtime under TRT, or accept the
constraint "one prompt per frame, cycle prompts across frames".

### 8.4 Where does the LLM run?

| Option                              | VRAM       | Latency          | Note                                            |
| ----------------------------------- | ---------- | ---------------- | ----------------------------------------------- |
| Local 7–8B, GGUF Q4, llama.cpp, GPU | ~5 GB      | 0.3–1 s          | contends with diffusion VRAM                    |
| Local 3–4B on GPU                   | ~2.5 GB    | ~0.3 s           | plan generation is an easy task; small may do   |
| Local on **CPU**                    | 0 GB VRAM  | 1–4 s            | cold path — this may be perfectly fine          |
| Cloud API                           | 0          | 0.5–2 s + network | breaks the offline guarantee                    |

Note the worker currently *enforces offline mode* (`enforce_offline_mode()`). A
cloud compiler would be a deliberate policy change, opt-in, and must never sit
on the frame path.
**Leaning:** CPU-side small model first (simplest, zero VRAM contention),
revisit if the latency annoys.

### 8.5 Temporal stability — **measured**

The known failure mode of per-frame diffusion is boiling/flicker. Five levers were
named here: per-track fixed seed, prompt-embedding caching per track, latent reuse
across frames, output EMA in the compositor, box smoothing in the tracker. Two were
built by earlier issues — box smoothing in `detection.Tracker`, and the feathered
composite that keeps a moving seam from becoming one — and this section asked for a
**quantitative flicker metric** so that a change could be judged rather than argued
about. The metric has existed since issue #5 (`bench.flicker`). Issue #32 built the
two remaining levers, wired both to plan fields, and used the metric on them.

**What `per_track` can mean, and what it cannot.** Stated first, because it is a
finding rather than an implementation note. Issue #5 chose the full-frame masked
primitive: one diffusion call per frame, and therefore one latent noise field
covering all K regions. So a per-track *strength*, *prompt* or *sampler* is not
expressible without a second call, and no amount of seeding makes it so. What *is*
expressible is one field composed of per-track patches — each track owns a
canvas-sized noise realisation drawn from its own id, and the frame pastes that
realisation, rolled to the track's current centre, into the region the compositor
is about to paint. That is `seeding.NoiseField`, and it is the whole of what the
name can mean here.

**And the shipped path was already `fixed`.** `StreamDiffusion.prepare()` draws
`init_noise` once from a seeded generator and `encode_image` adds it to every
frame's latent thereafter; with one denoising step nothing overwrites it. So the
noise this app renders on is already fixed and canvas-pinned, and `fixed` names
what it has always done rather than a new setting — which is why it is
`DEFAULT_SEED_POLICY`, and why the previous default of `per_track` was a field
nothing read describing behaviour nothing produced. `random` — a fresh field every
frame — does not ship and is here as the control: if the metric cannot tell it from
`fixed`, the metric is not measuring noise.

The sweep is generated from the committed records; re-run an arm and the merge gate
fails until 8.5 is regenerated with `uv run python -m bench --stability-report`.

<!-- BEGIN STABILITY SWEEP -->
The shipped path's committed runs measure flicker at 1.49 (spec 8.8); this sweep's control arm - `seed_policy: fixed`, `output_ema: 0.00` - measures 1.49, which is the same figure, so the control is measuring the shipped path.

`seed_policy` swept over fixed, per_track, random and `global.output_ema` over 0.00, 0.25, 0.50, 0.75 on NVIDIA GeForce RTX 4090, through the shipped selective path: the same img2img-tensorrt-512x512-b1 engine, the same 48 frames of `people.mp4` at the app's 512x512 capture canvas, the same `person / lower_half / t_index 40` plan and the same detector cadence. `flicker` is the mean absolute difference between consecutive outputs where the source stood still and the mask painted both frames - lower is steadier; `response` is the same figure where the source *moved* - higher is more responsive; `net change` is how much the regions changed against the capture, net of the capture's own round trip, and an arm under 8 is disqualified.

| seed policy | output EMA | flicker | vs shipped | response | net change | regions/frame | ms/frame | +detect | background |
|---|---|---|---|---|---|---|---|---|---|
| fixed | 0.00 | 1.49 | control | 4.76 | 11.8 | 5.04 | 16.90 | 24.25 | identical |
| fixed | 0.25 | 1.28 | -0.21 | 4.16 | 11.7 | 5.04 | 16.94 | 24.21 | identical |
| fixed | 0.50 | 0.88 | -0.60 | 3.46 | 11.7 | 5.04 | 16.67 | 23.59 | identical |
| fixed | 0.75 | 0.63 | -0.86 | 2.58 | 11.9 | 5.04 | 16.92 | 23.90 | identical |
| per_track | 0.00 | 1.74 | +0.26 | 5.38 | 11.7 | 5.04 | 17.46 | 24.27 | identical |
| random | 0.00 | 8.72 | +7.24 | 19.38 | 11.9 | 5.04 | 16.97 | 23.89 | identical |

**Recommended: `seed_policy: fixed`, `output_ema: 0.00` on NVIDIA GeForce RTX 4090 - the shipped default. Every arm that lowered flicker did it by low-passing the output: the closest, `seed_policy: fixed` / `output_ema: 0.25`, took flicker 1.49 -> 1.28 and the response to motion 4.76 -> 4.16, keeping 87% of it against the 90% a recommendation has to keep. So neither lever pays for itself here.**

Nothing was traded, because nothing was adopted. That is the answer this section asked for rather than a gap in it: the levers are built, they are wired to plan fields, and the measurement says what they cost.

Each arm was measured at least twice and two runs of one never differed in flicker by more than 0.00035 - unlike a millisecond, a flicker figure here carries essentially no run-to-run noise, because the render is deterministic given the clip, the plan and the seed. So the differences between arms are the arms.

Manual verification artefact at the recommended setting: `selective-people-fixed-ema00-20260907-133259Z-comparison.mp4` (source | selective render) and `selective-people-fixed-ema00-20260907-133259Z-comparison.jpg`.

What the metric's upper bound looks like: `selective-people-random-ema00-20260907-133330Z-comparison.mp4` is the `random` arm, a fresh noise field every frame.

Every arm above left the background bit-identical to the capture, and every one changed the region by 11.7-11.9/255 net of the control against a 8/255 threshold - so no arm bought its steadiness by rendering less, and the trade the table shows is the whole trade.
<!-- END STABILITY SWEEP -->

Four things the block does not say for itself.

**The control works, and the metric is measuring noise.** `random` scores 8.72
against `fixed`'s 1.49 on the same clip, the same plan and the same engine — nearly
6×. A metric that could not separate a redrawn noise field from a reused one would
have had nothing to say about either lever, and every number above would have been
a number about something else.

**`per_track` makes flicker *worse*, and the geometry says why.** 1.74 against
1.49, +0.26 — outside a run-to-run spread of 0.0004. Flicker is measured where the
*source* stood still, and a canvas-pinned field already gives a static pixel the
same noise on every frame; pinning the field to a track instead makes the noise
translate, so the static background *inside* a moving person's box now boils where
it did not before. The lever does exactly what its name says and the name was
aimed at the wrong thing: under one noise field per frame, "per-track" and "steady
where nothing moved" are opposed rather than aligned. It is built, it is a plan
field, and it is measured — and it is not the default.

**The EMA works and is priced fairly.** Flicker falls monotonically with the
coefficient (1.49 → 1.28 → 0.88 → 0.63) and so does the output's response to the
source's own motion (4.76 → 4.16 → 3.46 → 2.58). Nothing is bought free: at 0.25
it takes 14% off the boiling and 13% off the response, and at 0.75 it takes 58%
and 46%. And it is not cheating — the net change inside the regions is 11.7–11.9/255
at *every* setting against an 8/255 threshold, so no arm lowered flicker by
rendering less, which is the failure the visible-change control exists to catch.
Background bit-identity is 48/48 at every setting too: the EMA smooths the
*rendered canvas* before the mask, so history cannot reach a pixel the composite
copies from the capture.

**The reason to ship neither is the starting point, not the levers.** 1.49/255 is
0.6% of range on a clip the whole pipeline was tuned on; the boiling this section
was written to worry about is not what this path does at `t_index 40` on the
priority case. Spending a seventh of the restyle's response to motion to remove
something already near the floor is a bad trade, and that is the whole of the
verdict. Both defaults therefore stay where the measurement found them —
`seed_policy: fixed`, `output_ema: 0.0` — and the levers stay built, because the
answer changes if the denoise, the clip or the primitive does: a stronger `denoise`
boils more, and an EMA that is worth nothing at 1.49 may be worth something at 5.

Two of the five levers are still unbuilt and were not measured: prompt-embedding
caching per track and latent reuse across frames. Neither is expressible under one
diffusion call per frame either — one embedding per frame is the same constraint
that limits `per_track`, and latent reuse would make a frame a function of the
previous *latent* rather than the previous output, which is the EMA one stage
earlier and with bit-identity much harder to keep. Both want the primitive to
change before they want an issue.

### 8.6 Does the LLM ever see the screen?

Optional later mode: on prompt change only, send **one** captured frame to a
small VLM so the compiler can ground vague references ("the thing in the
corner"). Still cold path, once per prompt. Deferred — but the Render Plan
schema should not preclude it.

### 8.7 Failure UX

What does the user see when the plan asks for something impossible ("turn the
music into a bird")? Proposal: the plan carries `confidence` and `notes`; low
confidence surfaces as a non-blocking banner, and the previous plan keeps
rendering. Never a black screen, never a crash.

**Built** (issue #22), for the half a two-field producer can reach. A plan the
validator refuses is never sent, and its reason - in words, from the validator -
goes to the GUI status area beside the field that caused it; a plan it accepts
carries its `notes` to the same place. What the worker refuses, it says on the
status queue and keeps rendering the plan in force. `confidence` is carried and
nothing reads it yet: with no LLM in the path there is nothing to be unconfident.

Issue #40 moved that half a step closer to the user: the refusal and the notes are
also drawn on their own row *under the two fields*, the refusal in the error colour,
the row not there at all when there is nothing to read. The status bar still gets
both, but it is shared with the worker's own messages and the next one replaces
whatever was there - which is the failure mode #22's own notes named.

### 8.8 Does the selective path work end to end? — **yes, measured**

Issue #8, the M1 finish line. The question §5's diagram poses and no component test
answers: with the detector, the tracker, the region scheduler, the engine and the
compositor all wired together under one plan, does the priority case come out of
the pipeline — and does everything else come out untouched?

It is measured through the *shipped* modules rather than a harness copy of them.
`python -m bench selective-people` drives `detector_worker.BackgroundDetector` on
its own thread, `detection.Tracker`, `region_scheduler.RegionScheduler`, the cached
512×512 TensorRT engine and `device_compositor.DeviceCompositor` over a committed clip resized
to the app's own capture canvas, under `render_plan.priority_case_plan()` — the
same hardcoded plan the worker starts on behind `SD_DEMO_PLAN`. The block below is
generated from the committed record; re-run the case and the merge gate fails until
it is regenerated with `uv run python -m bench --selective-report`.

<!-- BEGIN SELECTIVE PATH -->
Measured on NVIDIA GeForce RTX 3080 Laptop GPU, NVIDIA GeForce RTX 4090, img2img-tensorrt-512x512-b1, 48 consecutive frames of `people.mp4` resized to the app's 512x512 capture canvas. Clocks: NVIDIA GeForce RTX 3080 Laptop GPU unlocked, NVIDIA GeForce RTX 4090 unlocked; absolute figures belong to the GPU in the row (spec 7.4), and 30 FPS is M2's gate, not this one's. Rows are one per case per GPU.

| case | GPU | plan | regions/frame | diffusion calls/frame | ms/frame | +detect | FPS | flicker (static px) | gate |
|---|---|---|---|---|---|---|---|---|---|
| selective-people | NVIDIA GeForce RTX 3080 Laptop GPU | person / lower_half / t_index 40 | 5.04 | 1.00 | 55.1 | 74.7 | 13.4 | 1.49 | pass |
| selective-people | NVIDIA GeForce RTX 4090 | person / lower_half / t_index 40 | 5.04 | 1.00 | 16.9 | 23.7 | 42.1 | 1.49 | pass |

The Gate, measured on NVIDIA GeForce RTX 3080 Laptop GPU:

- **Non-target pixels bit-identical** - yes. 48/48 frames left every pixel outside the rendered regions exactly as captured (164676 background pixels on the frame with the most painted).
- **The region is visibly restyled** - yes. The rendered regions changed by 11.8/255 against the capture - 11.8 net of the 0.00/255 the capture's own round trip costs - against a 8/255 threshold.
- **No track starved by the round robin** - yes. With 6 tracks over 2 slots no track waited more than 2 frames to be rendered, against a ceil(N/K) bound of 3.
- **The loop never stalls** - yes. 48/48 frames produced an output; the worst frame spent 0.168 ms offering the capture to the detector (budget 5 ms), and 0 frames had nothing to restyle and passed the capture through.

What the cadence cost: **not recorded** - this run predates the staleness block (issue #23).

Manual verification artefact: `selective-people-20260906-202515Z-comparison.mp4` (source | selective render) and `selective-people-20260906-202515Z-comparison.jpg`.

The Gate, measured on NVIDIA GeForce RTX 4090:

- **Non-target pixels bit-identical** - yes. 48/48 frames left every pixel outside the rendered regions exactly as captured (164676 background pixels on the frame with the most painted).
- **The region is visibly restyled** - yes. The rendered regions changed by 11.8/255 against the capture - 11.8 net of the 0.00/255 the capture's own round trip costs - against a 8/255 threshold.
- **No track starved by the round robin** - yes. With 6 tracks over 2 slots no track waited more than 2 frames to be rendered, against a ceil(N/K) bound of 3.
- **The loop never stalls** - yes. 48/48 frames produced an output; the worst frame spent 0.014 ms offering the capture to the detector (budget 5 ms), and 0 frames had nothing to restyle and passed the capture through.

What the cadence cost: At detect_every_n 3 a frame rendered boxes 1.9 frames old on average (worst 3, 33 ms); between refreshes a track's box kept 0.98 IoU (worst 0.77) and its centre moved 0.8 px (worst 18.0); 6 concurrent objects held 6 identities over 17 detects.

Manual verification artefact: `selective-people-20260907-115310Z-comparison.mp4` (source | selective render) and `selective-people-20260907-115310Z-comparison.jpg`.
<!-- END SELECTIVE PATH -->

The table carries one row per machine. Issue #24 re-ran the case on the deploy
hardware, and both rows are kept rather than the newer replacing the older: they
are two answers to one question, and §7.4 is the section that says so.

Three things the numbers say that the Gate does not.

**Detection costs three times more beside diffusion than alone.** §8.1 measured
YOLO-World at 14–19 ms per detect with the diffusion engine merely *resident*; on
the laptop it runs concurrently with a UNet on the same SMs and each detect costs
~59 ms, so at `detect_every_n: 3` it amortises to ~20 ms/frame rather than ~6.
Contention rather than the clock — the fastest detect in the same run was 15.7 ms.
The contention is not a laptop artefact: the 4090 pays it too, at ~23 ms against
§8.1's resident-but-idle figure. Raising the cadence is the cheap lever, and M2
spent it: §8.8's sweep sized it and issue #33 moved the default to 5.

**The composite was ~4 ms/frame of numpy, and is now 0.63 ms on the device.**
The bounding rectangle of the regions used to be blended on the host, on a frame
that was otherwise entirely on the GPU — and it was the only cost here that barely
moved on the faster card (§7.4): 2.75 ms against the laptop's 4.03, where the frame
path around it came down to 0.45×, so it was a *larger* share of a 4090 frame than
of a laptop one, 11.1% against 7.3%. Issue #31 moved it, and the interface really
was "an alpha map and a blend" on either device: `DeviceCompositor` inherits the
whole of the numpy one and adds the blend. The 4090 row above is the device path,
the laptop row is still the host one, and the frame path fell by more than the
composite cost (24.81 → 16.89 ms) because what went with it was the round trip —
the render used to come home as float32 and be rebuilt as a PIL image first. The
five host-composite runs stay in `bench/results/selective/` as the before.

**Flicker is 1.49 where §8.2 measured 1.43 on the same clip.** The two are not the
same measurement — this one renders detected boxes at the app's capture geometry,
that one rendered a committed track at clip resolution — but they are the same
order, so neither the tracker's box smoothing nor the feathered composite made the
boiling worse. Per-track seed pinning and the output EMA are now built and measured
(§8.5), and neither is on: the EMA trades response to motion roughly one for one
for steadiness, and per-track seeding makes flicker *worse* under a primitive with
one noise field per frame. 1.49 is the floor this path renders at, not a figure
waiting to be improved.

**30 FPS is not this section's claim and is not a gate here.** This section asks
whether the path works, and both rows say it does. What it costs is a fact about
the card in the row — 13.4 FPS on an RTX 3080 laptop under a 120 W limit, 30.9 on
an RTX 4090 — and the frame-rate *verdict* is §7.4's, where issue #24 measured the
two against each other and answered acceptance criterion 2.

#### Measured: closing the frame budget — what `detect_every_n` buys (issue #23)

M2 was written from laptop numbers, where the budget was missed by more than 2×
and trading resolution for speed was obviously worth measuring. §7.4 then measured
the deploy card and the premise went away: the **unmodified 512×512 path already
meets the budget**, so a 384² or 256² engine would have spent ~5.1 GB and 15–25
minutes each to recommend a quality downgrade nobody needs. **No lower-resolution
engine was built, and that is the finding rather than an omission** — the first
paragraph of the block below is that decision, computed from #24's committed
records rather than retyped from them.

The sweep is the other half, and it was worth running either way: cadence is the
one lever on the frame budget that costs no image quality. Every arm renders the
same regions through the same engine at the same strength; what a higher cadence
spends is the *freshness* of the boxes, which is why the block puts the
milliseconds bought and the staleness spent in one table. Each arm was measured
twice, cold-started, and the block is generated with
`uv run python -m bench --cadence-report` from `bench/results/cadence/`.

<!-- BEGIN CADENCE SWEEP -->
Step 1, taken from issue #24's committed baseline rather than re-derived: 42.1 FPS at 5.04 regions/frame on NVIDIA GeForce RTX 4090 - 23.75 ms per frame with detection amortised against the 33.33 ms a 30 FPS budget allows, 9.58 ms to spare; detect_every_n 3, clocks unlocked. 5 committed runs on NVIDIA GeForce RTX 4090 span 41.5-42.3 FPS, clear of the 30 FPS target (5 earlier runs on the card blended on the host (numpy) and are not in this spread) - so **there is no gap to close by lowering the resolution, and no lower-resolution engine was built** - a 384x384 or 256x256 engine would trade what the model can see for speed nobody needs.

`detect_every_n` swept over 2, 3, 5, 8 on NVIDIA GeForce RTX 4090, through the shipped selective path: the same img2img-tensorrt-512x512-b1 engine, the same 48 frames of `people.mp4` at the app's 512x512 capture canvas, the same `person / lower_half / t_index 40` plan. Only the cadence moves, so no pixel the diffusion produces changes - what a higher cadence spends is the freshness of the boxes, which is the right half of the table.

| detect_every_n | regions/frame | ms/detect | amortised ms/frame | ms/frame | +detect | FPS | headroom (ms) | box age (frames) | refresh IoU | ids/objects | flicker | background |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4.98 | 23.14 | 11.57 | 24.52 | 36.09 | 27.7 | -2.76 | 1.46 | 0.99 | 6/6 | 1.49 | identical |
| 3 | 5.04 | 21.92 | 7.31 | 24.03 | 31.34 | 31.9 | +1.99 | 1.94 | 0.98 | 6/6 | 1.49 | identical |
| 5 | 5.19 | 21.38 | 4.28 | 23.87 | 28.15 | 35.5 | +5.18 | 2.88 | 0.98 | 6/6 | 1.49 | identical |
| 8 | 4.81 | 19.85 | 2.48 | 23.92 | 26.40 | 37.9 | +6.94 | 4.33 | 0.97 | 5/5 | 1.49 | identical |

**Recommended: `detect_every_n: 5` on NVIDIA GeForce RTX 4090 - 28.15 ms per frame with detection amortised, +5.18 ms against the 33.33 ms a 30 FPS budget allows: the freshest cadence measured that leaves at least 3.33 ms of the 33.33 ms budget free.**

What that costs: At detect_every_n 5 a frame rendered boxes 2.9 frames old on average (worst 5, 69 ms); between refreshes a track's box kept 0.98 IoU (worst 0.77) and its centre moved 1.2 px (worst 18.0); 6 concurrent objects held 6 identities over 11 detects.

Each cadence was measured twice and the two runs of one never differed by more than 0.64 ms, against 1.28 ms from the nearest arm to the headroom line - so the recommendation is outside the run-to-run spread.

Manual verification artefact at the recommended cadence: `selective-people-n5-20260907-110910Z-comparison.mp4` (source | selective render) and `selective-people-n5-20260907-110910Z-comparison.jpg`.

Every arm above left the background bit-identical to the capture; every arm rendered 4.81-5.19 regions/frame, the same selection to within 15%.
<!-- END CADENCE SWEEP -->

Four things the block does not say for itself.

**Detection contention falls with the cadence, so the saving is better than
linear.** One detect costs ~23 ms at `detect_every_n: 2` and ~20 ms at 8 — the
same detector on the same frames, differing only in how much UNet work it is
sharing the SMs with. The amortised figure therefore falls faster than 1/N, and
the ~59 ms the laptop measured (§7.4) is the same effect at the other end.

**The frame path itself is flat across the sweep**, at 23.9–24.5 ms. That is the
control this sweep needed: the cadence is not quietly changing what is rendered,
so the whole of the difference between arms is detection.

**Above `detect_every_n: 5` the tracker starts losing objects, not just freshness.**
At 8 the run held five concurrent tracks where every faster arm held six: an
object that is present for under eight frames can now be missed entirely, and no
millisecond figure shows that. It is the `ids/objects` column, and it is the
reason the recommendation is the *freshest* setting that fits rather than the
cheapest one.

**The remaining lever was the composite, and it has since been pulled.** §7.4
measured it at 2.75 ms of host numpy that a faster GPU did not shrink — 11.1% of a
4090 frame and, at the recommended cadence, ~10% of a 28 ms one. Issue #31 moved
the blend onto the device: 0.63 ms, and 23.75 ms/frame with detection at
`detect_every_n: 3`, which is 4.4 ms cheaper than this sweep's best arm was at a
cadence it had to spend box freshness to reach.

**So the sweep above is measured on a frame path that has since got faster**, and
its arms are not re-run here — every arm predates issue #31 and the table says what
it said when it was measured. What that changes is only which arm the rule picks,
and it can only pick a *fresher* one: every arm loses the same ~7 ms, so the n=2
arm's −2.76 ms of headroom becomes a surplus and the recommendation moves towards
the fresher end rather than away from it. Re-running the sweep is issue #23's
ground, not #31's; until it is, read the recommendation as a bound and the
staleness half of the table — which no composite touches — as measured.

**The recommendation is now the shipped default (issue #33).**
`DEFAULT_DETECT_EVERY_N` is 5. Issue #23 measured the recommendation and
deliberately left the field behind it, so that both cards' committed baselines
stayed at one cadence; #33 moved it, and a test now holds the two together —
`test_the_shipped_default_is_the_cadence_the_deploy_card_recommends` fails if a
re-sweep moves the recommendation, so the field is re-decided rather than quietly
left where the last measurement put it.

**Neither committed baseline has been re-measured at it, and both re-runs are
outstanding.** Every `selective-people` run in §8.8 and §7.4 was measured at 3:
the 4090's because the card was busy with another application when the re-run was
attempted — the occupancy gate below refused it, which is what that gate is for —
and the laptop's because it is a different machine. So read the shipped figures as
a **floor**. Cadence 5 removes ~3 ms of amortised detection from a frame path this
sweep measured flat across the cadence, so the 23.75 ms and the 9.58 ms of
headroom §7.4 quotes can only improve at the default the app now ships; the
criterion-2 verdict cannot flip in the direction that would matter. What is *not*
knowable until both cards are re-run is the freshness half, which is the half a
viewer sees.

**What the cadence costs is box age, and only its frames travel.** At
`detect_every_n: 5` a frame renders boxes 2.9 frames old. That is ~49 ms at the
4090's current frame path and ~158 ms at the laptop's — the same setting, **3.3x
apart** in what a viewer sees the mask lag the subject by, and the reason a
cadence chosen on the deploy card's milliseconds is a decision about the deploy
card. §7.4's generated block states both currencies for whatever the committed
runs measured, and labels the projected one. If ~160 ms proves unacceptable on the
dev machine, the honest answer is a per-machine default rather than a different
global one.

**A run measured beside another application is not a measurement**, and until
issue #33 nothing in the harness said so. The first attempt at the re-run above
measured 54.17 ms/frame against a committed 16.89 on the same 4090, at 50 °C,
with a live real-time application holding ~45% of the SMs — and it passed every
door there was: fingerprint present, cooldown `reached`, clock regime recorded. It
was written to `bench/results/selective/` and appended to the README, where it
would have become the row §8.8 and §7.4 quote and flipped criterion 2 to NOT MET
at 15.3 FPS. `bench/contention.py` is the door that refuses it: every
`selective-people` record now carries an `occupancy` block — `clear`, `busy` or
`unknown`, sampled before the timed region while this process is idle — and
`--require-idle-gpu` exits non-zero for a run that decides something, the way
`--require-locked-clocks` does. `unknown` refuses too, for the same reason it
refuses there.

---

### 8.9 Does a plan swap take effect in time, and without a stutter? — **yes, measured**

Acceptance criteria 1 and 3 (§11) were the two nobody had run. Everything the
repo knew about them was indirect and additive: `PLAN_DEBOUNCE_MS = 400` in the
GUI, a ~16 ms text encode, a `set_classes` that drops the predictor and costs the
next `predict` ~108 ms, a schedule update that is a runtime call rather than a
rebuild. On paper that sums to well under three seconds — but criterion 3 is not
the same question as criterion 1. "No stutter" is about the *inter-frame
interval* while that work happens, and nothing in the repo had ever looked at
inter-frame intervals.

Two swaps, because they take different paths. The **expensive** one changes the
target concept, which re-encodes the detector's vocabulary and owes the throwaway
detect §8.1 measured; the **cheap** one changes only the style and the strength,
which is a prompt re-encode on the frame thread and a `set_t_index_list`. Both
start from the priority-case plan the worker itself boots on, so the frames
before the swap are the steady state §7.4 and §8.8 already have baselines for.

Three rules the block below obeys.

**Criterion 1's clock starts at the keystroke.** The 400 ms debounce is time the
user waits, so it is in the figure; the worker-side half is reported beside it,
because that is the one an optimisation would move. What the harness does *not*
measure is the `multiprocessing.Queue` hop between the two processes — it runs in
one process, and the omission is named rather than absorbed.

**The pixels are the event, not the plan version.** A frame that bound the new
plan but has no boxes for it yet renders the capture untouched, which is the old
instruction still on screen. The clock stops at the first frame whose rendered
regions came from the new plan's own tracks.

**Criterion 3 is judged against a steady-state control from the same run.** A
40 ms frame on a card rendering 32 ms frames has not stuttered, and 33.33 ms
alone cannot tell you that. The bar is the dearest frame in the same run's steady
state, plus one frame budget — past that the stream has lost a frame to the swap.
The control is the worse of the two steady states either side, so it is not
whichever half flatters the verdict.

Each swap was measured twice, cold-started. The block is generated with
`uv run python -m bench --swap-report` from `bench/results/swaps/`.

<!-- BEGIN PLAN SWAP -->
Measured on NVIDIA GeForce RTX 4090, img2img-tensorrt-512x512-b1, 96 consecutive frames of `people.mp4` resized to the app's 512x512 capture canvas, the new instruction submitted on frame 48. Clocks unlocked. The swap travels the shipped cold path - `plan_from_fields`, `validate_plan`, `ActivePlan.submit`, and the frame loop's own prompt re-encode, `BackgroundDetector.follow` and `set_t_index_list` - so what is timed is the work the worker does. It omits one hop the app has and this harness does not: the `multiprocessing.Queue` between the GUI and the worker process.

| swap | what moved | keystroke -> pixel | worker | frames to pixel | criterion 1 | worst across swap | control worst | over budget (swap / steady) | criterion 3 | rebuilds | background |
|---|---|---|---|---|---|---|---|---|---|---|---|
| swap-style | style, denoise 0.49 -> 0.62 | 0.43 s | 33 ms | 1 | MET | 37.85 ms | 32.03 ms | 1/1 vs 0/44 | MET | 0 | identical |
| swap-target | target person -> shoes | 0.50 s | 104 ms | 11 | MET | 27.97 ms | 31.66 ms | 0/11 vs 0/44 | MET | 0 | identical |

**Acceptance criterion 1 - a typed instruction takes effect within 3 s: MET.** The slower of the two swaps is `swap-target`, where a typed instruction reached the screen 0.50 s after the keystroke - 400 ms of GUI debounce, 0.1 ms to validate the plan and 104 ms in the worker (11 frames) - against the 3.00 s the criterion allows.

**Acceptance criterion 3 - swapping the instruction stutters nothing and rebuilds nothing: MET.** The least steady of the two swaps is `swap-style`, where the worst inter-frame interval across the swap was 37.85 ms over 1 frames, against 32.03 ms in the same run's steady state before (1.18x): +5.82 ms, inside the 33.33 ms a dropped frame would cost; 1/1 frames across the swap were over the 33.33 ms budget against 0/44 in steady state. Across both swaps there were 0 TensorRT rebuilds: the schedule went from t_index [40] to [36] on the same engine object at 1 step(s) either side, so 0 TensorRT rebuilds.

What a swap costs when it does not cost milliseconds: `swap-target` showed the capture untouched for 10 of the 11 frames it took to arrive (2 detects). A plan change drops the tracks that were about the old concept, and a frame with no boxes costs no diffusion call - which is why the frame path across a vocabulary swap is *cheaper* than steady state rather than dearer.

Criterion 4 held across the swap too: 96/96 frames left every pixel outside the rendered regions exactly as captured (164676 background pixels on the frame with the most painted).

Each swap was measured twice and the two runs of one never differed by more than 2 ms in keystroke-to-pixel or 0.20 ms in the worst interval across the swap, against margins of 2.50 s and 27.51 ms to the two thresholds - so both verdicts are outside the run-to-run spread.

Manual verification artefacts, source | render: `swap-style-20260907-122518Z-comparison.mp4`, `swap-target-20260907-122458Z-comparison.mp4`. The still beside each is the frame the new instruction first reached, which is the frame there is anything to look at on.
<!-- END PLAN SWAP -->

Three things the block does not say for itself.

**The expensive swap is the cheap one on the frame path.** A vocabulary change
costs the frame loop *less* than steady state, not more, because `follow` drops
the tracks that were about the old concept and a frame with no boxes costs no
diffusion call at all. What it costs instead is ten frames of unstyled capture —
about 0.1 s of output showing the screen as it is. That is the trade a
millisecond figure alone would report as free, and it is the one a viewer sees.

**The cheap swap is the dearer one on the frame path**, at +5.8 ms on the single
frame it lands on: `update_prompt` re-encodes the prompt on the frame thread, and
the schedule caches are rebuilt beside it. It is one frame, it is inside the
budget a dropped frame would cost, and it is the only place in this design where
cold-path work runs on the hot path. Moving the text encode off the frame thread
is the obvious optimisation and is not needed at 30 FPS.

**No rebuild is arithmetic here, not an assumption.** The step *count* is what
keys a TensorRT engine (§7.2); a plan's `denoise` reaches the engine as a
schedule *value* through `render_plan.t_index_for_denoise`, so t_index 40 → 36 is
a runtime update. The record carries both the count and the engine object's
identity either side of the swap, so a reviewer recomputes the zero rather than
trusting it.

---

### 8.10 Do style LoRAs work on SD 1.5, and how does a style ship? — **measured**

Issue #38 steps 3-5. §7.5 prices the move to SD 1.5; this is the half that says
whether the move buys anything, and the Gate's sentence is sharp: **a LoRA that
loads and changes nothing is a failure, not a pass.** So an arm carries two numbers.
`net change` is the render against the source with the resize control subtracted —
§8.2's own criterion — and `vs base` is the arm against the *same arm with no LoRA
fused*, which is the number that says the LoRA did anything. A plain SD 1.5 render
already clears the first.

A third thing is recorded rather than inferred: **which formats load**. One arm is
deliberately a LoCon file, because reporting "LoRAs work" from one plain LoRA that
happened to would be reporting a coincidence.

`uv run python -m bench style-sd15`; regenerate with
`uv run python -m bench --style-report`.

<!-- BEGIN STYLE LORAS -->
3 style LoRAs on `img2img-none-512x512-b1-sd15` at 4 steps, over 24 frames of `people.mp4` at 512x512, on NVIDIA GeForce RTX 4090. Every arm renders the same frames at the same denoise (0.62) under the same prompt ("a painting"); only the fused LoRA moves. `net change` is the render against the source with the resize control subtracted - spec 8.2's criterion - and `vs base` is the arm against the same arm with no LoRA fused, which is the number that says the LoRA did anything.

| arm | format | loaded | ms/frame | net change | vs base | flicker | verdict |
|---|---|---|---|---|---|---|---|
| `base` | no LoRA | yes | 53.33 | 14.58 | 0.00 | 2.51 | control |
| `loving-vincent` | LoRA (kohya, linear only) | yes | 46.69 | 19.74 | 19.23 | 1.31 | pass |
| `illusion-pattern` | LoRA (kohya, linear only) | yes | 46.72 | 15.17 | 5.32 | 2.55 | pass |
| `locon-probe` | LoCon / LyCORIS (198 conv keys) | **no** | 0.00 | 0.00 | 0.00 | 0.00 | **FAIL** |

**2 of 3 style LoRAs loaded and visibly changed the output.**

- `loving-vincent` loaded and moved the output 19.2/255 against the same arm without it (threshold 4), on a render that is itself 19.7/255 from the source net of the control.
- `illusion-pattern` loaded and moved the output 5.3/255 against the same arm without it (threshold 4), on a render that is itself 15.2/255 from the source net of the control.
- `locon-probe` (LoCon / LyCORIS (198 conv keys)) did not load: Failed to load LoRA 'style-locon-probe.safetensors': 'UNet2DConditionModel' object has no attribute 'conv'

The LoRA must match the base model's architecture (sd-turbo is SD 2.1-based), and LoRAs containing convolution layers (LoCon/LyCORIS) are not supported by diffusers 0.24.0..
The artefact a human judges this by, source | base | one panel per style: `style-sd15-20260907-183158Z-styles.jpg`, `style-sd15-20260907-183158Z-styles.mp4`.

**Neither path ships a real-time style at this step count.** The same LoRA (`loving-vincent`) fused into a TensorRT engine costs 33.57 ms/call against 52.02 ms without one - 1.55x - and against a 33.33 ms budget, neither path fits the frame budget on the diffusion call alone. What the engine costs is that it *is* an engine: each distinct style and scale keys its own build - ~5 GB on disk and 5-25 minutes, depending on the GPU, because `create_prefix` puts the fused-LoRA fingerprint in the cache key - so styles are a release-time set rather than something a user types. The `none` path swaps a style for a model reload and no compile, which is what LoRA *experimentation* needs.
<!-- END STYLE LORAS -->

**The delivery decision, in one sentence: pre-built engines, one per style.** A
fused LoRA keys its own TensorRT engine — `create_prefix` puts a sha1 of the fused
`lora_dict` in the cache directory name, so `loving-vincent` at 0.9 and the same
LoRA at 0.5 are two ~5 GB builds. That is not a detail to route around; it is the
shape of the feature. Styles are a **release-time set** the app ships engines for,
not something a user types, and the non-TensorRT path is where LoRA
*experimentation* happens: it swaps a style for a model reload and no compile, at
52.02 ms/call against 33.57.

Two things the block does not settle. The engine cost is per (LoRA, scale) pair, so
a "style strength" slider is a slider over engines and cannot exist on the TensorRT
path — a fixed scale per shipped style is the only expressible form. And neither
path clears the frame budget on the diffusion call alone at four steps, so shipping
a style on SD 1.5 is shipping §7.5's 24.4 FPS with it.

#### A release-time set only works if its engines are findable — **issue #44**

Live testing on 2026-09-08 found that **no style LoRA in the window ever found a
pre-compiled engine**, with one sitting on disk. The cache key was the raw path
*string*: the same file at the same scale hashed to `c85d6678` spelled with
backslashes — the engine the harness built — and to `7a007080` spelled with forward
slashes, which is what Tk's file dialog returns on Windows. So the two producers of
a `lora_dict` could never meet, and adding one LoRA by two routes cost two ~5 GB
builds.

`engine_cache.normalize_lora_key` is the fix, and it is in `engine_cache` rather
than in the window on purpose: that module exists so the harness, the window and
`wrapper.py` cannot answer this question three ways. `wrapper.py` had the third
copy — an inline `hashlib.sha1` — and it was the copy that keyed every engine on
disk; it now calls `engine_cache.lora_fingerprint` like the other two. The
normalisation is `Path.resolve()`: it absolutises, folds the separators, follows
links and on Windows returns the file's own case, so the key is the *file* rather
than the route taken to it. Both halves of an entry are read for what they are —
the scale as a float, so `1` and `1.0` are one engine and not two `repr`s.

**No committed engine was orphaned**, which was the risk worth checking before the
change rather than after it: an absolute backslash spelling is already its own
resolution, so `c85d6678` is still `c85d6678` and the two `sd-v1-5-fp16 … lora-c85d6678`
directories under `engines/` are found from either spelling now.
`tests/test_gpu_engine_reuse.py` asks that of this machine's actual cache rather
than of prose.

The affordance moved with it. The style LoRAs are three files in one known
directory (`$SD_MODELS_DIR/loras`), so the window picks them from a list —
`local_lora_paths` / `lora_label` / `_lora_choices`, the same shape as §7.5's base-model
picker — and a file dialog for three files in a known place is both the wrong
control and how the wrong spelling got in. Browse stays, because a LoRA outside the
models root is still a legal answer. The LCM-LoRA sits in the same directory and is
*not* offered: it is applied by the `use_lcm_lora` switch, and listing it invites
fusing it twice. And because the scale keys the engine as much as the file does,
the standing sentence under the model names both — `Engine: cached for sd-v1-5-fp16
at 4 steps with style-loving-vincent.safetensors @ 1.00.` — and follows the slider
(`docs/gui/after-lora-choice.png`).


## 9. Risks

| Risk                                              | Impact                  | Mitigation                                             |
| ------------------------------------------------- | ----------------------- | ------------------------------------------------------ |
| Per-object diffusion can't hit 30 FPS at any useful K | kills the design        | Measure §7.2 **first**; fall back to option B / 7.3(d) |
| TRT fixed batch forces rebuilds mid-stream        | multi-second stalls     | Fixed slot count 7.3(a) or dynamic profiles 7.3(c)     |
| Small crops produce mush                          | unusable output quality | Minimum crop-size floor; skip objects below it         |
| Flicker makes it unwatchable                      | unusable                | §8.5 — measured: 1.49/255 on the priority case, against 8.72 for a redrawn noise field. Two levers built and neither needed |
| LLM emits plausible-but-wrong plans               | user confusion          | Constrained decoding, validator, visible `notes`, manual override |
| VRAM exhaustion (diffusion + detector + LLM)      | crash                   | CPU LLM; measure peak; hard budget                     |
| Detector vocabulary too narrow                    | "it doesn't understand me" | Open-vocab detector, or an honest UI error listing what *is* supported |
| Laptop numbers used as deploy numbers             | ship a design that misses 30 FPS on the target | §7.4 — fingerprint every result; treat absolutes and VRAM ceilings as non-portable |
| Disk exhaustion from TensorRT engines             | benchmark run dies partway  | 5.1 GB per engine; sweep on `none` first, build TRT only for the configurations that matter |

---

## 10. Phased plan

**M0 — Measure (no features).** Build a tracked benchmark harness, then use it:
diffusion cost vs batch size and vs resolution; detector candidates' latency;
VRAM ceilings. Deliverable is a numbers table that either validates or kills
§7.3.

**M1 — Hand-authored plans, no LLM.** Add `set_plan` to `control_queue`.
Hardcode a plan (person → red hat). Build detector + crop + composite. No
tracker, no scheduler, K=1. Goal: see the pipeline work end to end at *any*
frame rate, and compare approaches A/B/C/D from §8.2 on real footage.

**M2 — Real-time hardening.** Tracker, region scheduler, K slots, seed pinning,
flicker metric and mitigations. Goal: hold 30 FPS with 3–4 objects.

**M3 — The compiler.** Add the LLM with constrained JSON output, the validator,
and a prompt box in the GUI that hot-swaps plans. Goal: the demo in §1 works
from typed text alone.

**M4 — Polish.** Plan presets, save/load, `notes` surfacing, graceful
degradation, docs.

Each milestone is independently useful, and M0/M1 can invalidate the whole idea
cheaply — which is the point of ordering them this way.

---

## 11. Acceptance criteria for v1

1. User types "find all people and give them a red hat" and, within 3 s and
   without restarting generation, people on screen render with red hats.
   **Met, measured (issue #30, 2026-09-07):** 0.50 s from keystroke to pixel for a
   new *target* on an RTX 4090 — 0.40 s of that is the GUI's own debounce, 0.10 s
   is the worker — and 0.43 s for a new style. Generation never restarts: the
   engine object and its step count are unchanged across both swaps. §8.9 has the
   block and what the two halves of the figure mean.
2. Sustained ≥ 30 FPS output with up to 4 tracked objects at 512² diffusion,
   measured **on deploy hardware** (RTX 3090 Ti / 4090) — see §7.4.
   **Met, measured (issue #24, 2026-09-07):** 30.9–31.4 FPS over five runs on an
   RTX 4090 at **5.04 regions/frame** — more than the four objects the criterion
   asks for — with the whole shipped selective path running. The margin is ~1 ms
   of a 33.33 ms frame, on a card with nothing else on it; §7.4 has the comparison
   against the dev laptop and what it does and does not license.
   **Widened, measured (issue #23, 2026-09-07):** at `detect_every_n: 5` the same
   path measures 35.5 FPS with 5.18 ms of headroom and no cost in image quality —
   §8.8's cadence block.
   **Widened again, measured (issue #31, 2026-09-07):** with the composite moved
   off the host the same path measures **41.5–42.3 FPS over five runs, 23.75 ms
   with detection against the 33.33 ms budget — 9.58 ms of headroom — at
   `detect_every_n: 3`**, and still 48/48 frames bit-identical outside the
   rendered regions. Unlike the cadence lever this one costs no box freshness and
   nothing else anyone can see.
   **Restated at the shipped cadence (issue #33, 2026-09-07):**
   `DEFAULT_DETECT_EVERY_N` is now **5**, and the figures above were measured at
   3, so **the 9.58 ms of headroom is a floor rather than the margin**: the sweep
   measured the frame path flat across the cadence and cadence 5 removes ~3 ms of
   amortised detection, so the shipped configuration is at least as fast as the
   verdict quotes. The criterion is MET and cannot flip in that direction — but
   the number it is met by has **not been re-measured on either card**, and until
   it is, §7.4's table and this verdict describe a cadence the app no longer
   ships. What the cadence costs is box age, not milliseconds: 2.9 frames, which
   is ~49 ms on the 4090 and ~158 ms on the laptop. Neither that nor the margin is
   the thing to watch here; what the criterion has never been measured with is a
   real screen, a capture thread and a GUI process beside it.
3. Typing a new instruction swaps behaviour with **no stutter** in the output
   stream and **no TensorRT rebuild**.
   **Met, measured (issue #30, 2026-09-07):** worst inter-frame interval across a
   swap 37.85 ms against 32.03 ms in the same run's steady state — +5.8 ms on the
   one frame the prompt re-encode lands on, well inside the 33.33 ms a dropped
   frame would cost — and **0 TensorRT rebuilds**, checked from the step count and
   the engine object's identity either side rather than assumed. A *target* swap
   is cheaper than steady state on the frame path and costs ten frames of unstyled
   capture instead; §8.9 says why that is the half worth watching.
4. Non-target pixels are bit-identical to the capture (verifiable).
   **Held across the §8.5 levers (issue #32, 2026-09-07):** 48/48 frames at every
   seed policy and every output EMA measured. The EMA smooths the *rendered canvas*
   before the mask rather than the composited frame, so blending across frames
   cannot reach a pixel the composite copies from the capture.
5. An unsatisfiable instruction leaves the previous render running and shows a
   readable explanation.

---

## 12. Glossary

- **Render Plan** — the validated JSON contract between control and data plane.
- **Control plane / cold path** — everything that runs on prompt change only.
- **Data plane / hot path** — the per-frame loop under the 33 ms budget.
- **Slot (K)** — one item in the diffusion engine's fixed batch.
- **Track** — a detected object with an identity stable across frames.
- **Concept** — the user's word for a target ("person"), resolved by the
  validator to whatever the active detector actually supports.
