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
full-frame `img2img` call per frame — §8.2 chose B — and the plan's `denoise`
reaches it as a `t_index` on the live schedule (`render_plan.t_index_for_denoise`,
whose amplitudes are held to §8.2's measured ladder by a test). Only the schedule
*values* move, never the step count, so a plan change is a runtime update and
never an engine rebuild. A frame whose plan selected no region costs no diffusion
call at all.

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
non-zero alpha is written at all. Colour matching and the temporal EMA are not
built — they are M2 levers (§8.5).

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
      "seed_policy": "per_track",   // "per_track" | "fixed" | "random"
      "max_instances": 6,           // 1-16
      "priority": 1                 // 0-99, an ordering key between targets
    }
  ],
  "background": { "action": "passthrough", "prompt": "" },   // or "stylize"
  "global": { "fps_target": 30, "detect_every_n": 3 },
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

What the worker does with a plan **today** is the `global` case of it: the plan's
effective prompt over the whole frame, which is what the app already did and what
`set_prompt` still does. `mode`, `region`, `box_scale`, `max_instances`,
`seed_policy` and `background` are the schema the selective render path consumes,
and that path is issues #7 (detector and tracks) and #8 (regions and compositing).

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
| Composite + present | every frame     | 2–3               | not yet measured          |                           |
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
| Absolute ms/frame, and whether 30 FPS is met   | **No** — **measured**, 2.3× apart; re-measure on the deploy GPU |
| VRAM ceilings and OOM thresholds               | **No** — a laptop OOM says nothing about 24 GB     |
| Engine build times                             | **No**                                             |

The 30 FPS gate in §11 is therefore a **deploy-hardware** criterion. On the
laptop the same run is a regression check, not an acceptance test.

A laptop also may never reach the §7.2 cooldown threshold under sustained load.
The harness caps the wait and records that the threshold was not met, rather than
blocking forever or silently reporting a throttled number.

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
`selective-people` measured on NVIDIA GeForce RTX 4090 (unlocked clocks, 500 W) against the NVIDIA GeForce RTX 3080 Laptop GPU baseline of 2026-09-06 (unlocked clocks, 120 W): the same img2img-tensorrt-512x512-b1 engine rebuilt for this architecture, the same 48 frames of `people.mp4` at the app's 512x512 capture canvas, the same `person / lower_half / t_index 40` plan.

| measure | NVIDIA GeForce RTX 3080 Laptop GPU (dev) | NVIDIA GeForce RTX 4090 (deploy) | deploy / dev |
|---|---|---|---|
| regions/frame | 5.04 | 5.04 | 1.00x |
| diffusion calls/frame | 1.00 | 1.00 | 1.00x |
| ms/frame, frame path | 55.06 | 24.81 | 0.45x |
| ms/frame, with detection | 74.73 | 32.35 | 0.43x |
| FPS | 13.4 | 30.9 | 2.31x |
| ms/detect | 59.03 | 22.62 | 0.38x |
| composite ms/frame | 4.03 | 2.75 | 0.68x |
| flicker (static px) | 1.49 | 1.49 | 1.00x |
| peak VRAM (MiB) | 3742 | 3738 | 1.00x |
| mean SM clock (MHz) | 1668 | 2715 | 1.63x |

**Acceptance criterion 2 (30 FPS): MET.** 30.9 FPS at 5.04 regions/frame on NVIDIA GeForce RTX 4090 - 32.35 ms per frame with detection amortised against the 33.33 ms a 30 FPS budget allows, 0.98 ms to spare; clocks unlocked. 5 committed runs on NVIDIA GeForce RTX 4090 span 30.9-31.4 FPS, clear of the 30 FPS target.

What carried across the move, measured:

| Conclusion | Carried? | Evidence |
|---|---|---|
| Non-target pixels stay bit-identical to the capture | Yes | 48/48 frames dev -> 48/48 frames deploy |
| How much of the frame the scheduler picks: regions and calls per frame | Yes | 5.04 dev -> 5.04 deploy regions/frame, 1.00 dev -> 1.00 deploy calls/frame |
| The ceil(N/K) round-robin bound | Yes | worst gap 2 of 3 allowed dev -> 2 of 3 deploy |
| Flicker over pixels static in the source | Yes | 1.49 dev -> 1.49 deploy |
| Absolute ms/frame on the frame path | **No** | 55.06 ms dev -> 24.81 ms deploy |
| Whether the 30 FPS criterion is met | **No** | 13.4 FPS dev -> 30.9 FPS deploy |
| What one detect costs beside the diffusion | **No** | 59.03 ms dev -> 22.62 ms deploy |
| Peak VRAM the path allocates | Yes | 3742 MiB dev -> 3738 MiB deploy |
| The clock the card holds under load, against its own maximum | **No** | 79% of maximum dev (120 W limit) -> 86% deploy (500 W) |

Comparable because NVIDIA GeForce RTX 3080 Laptop GPU rendered 5.04 regions/frame and NVIDIA GeForce RTX 4090 rendered 5.04, the same selection to within 5%.
<!-- END DEPLOY HARDWARE -->

Four things the block does not say for itself.

**The criterion is met, and the margin is one millisecond.** 32.35 ms against a
33.33 ms budget on the slowest of five runs, 31.88 ms on the fastest. That clears
the gate and is nobody's idea of headroom: it is 5.04 regions on one 512² frame at
`detect_every_n: 3` with nothing else on the card, while the app also has a capture
thread, a GUI process and a real screen to feed. §8.8's two known costs — detection
contention and a host-side composite — are the levers, and #23 is where they get
pulled. This verdict is a floor to optimise from, not a finish line.

**The first run on this card measured 29.5 FPS and is not among the five.** It was
the run that compiled the engine, and it diffused in a process still holding the
TensorRT builder's state. The five committed runs are cold-started against the
cached engine, which is what the app does; the excluded one is recorded here rather
than in `bench/results/` because it measured a condition the product never enters.

**Everything that is not a millisecond carried exactly.** Same regions, same calls,
the same round-robin bound, the same flicker to three figures, and the same 48/48
frames bit-identical outside the mask. That is the useful half of this measurement:
the design decisions taken on the laptop — issue #5's primitive, #8's scheduler and
feathered composite — did not need re-taking. Peak VRAM matched too, but read that
narrowly: what the path *allocates* is a property of the path, and the VRAM
*ceiling* remains untested, because nothing here came near either card's.

**The composite is the part that did not scale.** The frame path came down to 0.45×
and one detect to 0.38×, but the numpy blend only reached 0.68× — it is host code,
and a faster GPU does not make it faster. It was 7.3% of the laptop's frame path and
it is 11.1% of the 4090's. On this card the ranking of what to optimise has changed.

Clocks were unlocked on both machines: `nvidia-smi --lock-gpu-clocks` needs an
elevated shell the agent loop does not have (issue #13), and the 4090 refused it
here for the same reason the laptop did. The desktop is nonetheless the steadier of
the two — 86% of its maximum SM clock at 48 °C against the laptop's 79% at 75 °C —
which is the difference the move was expected to show, recorded rather than
normalised away.

---

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

### 8.5 Temporal stability

The known failure mode of per-frame diffusion is boiling/flicker. Levers:
per-track fixed seed, prompt-embedding caching per track, latent reuse across
frames, output EMA in the compositor, box smoothing in the tracker. Needs a
**quantitative flicker metric** (e.g. mean absolute difference between
consecutive outputs in static regions) so we can tell whether a change helped
rather than arguing about it.

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

### 8.8 Does the selective path work end to end? — **yes, measured**

Issue #8, the M1 finish line. The question §5's diagram poses and no component test
answers: with the detector, the tracker, the region scheduler, the engine and the
compositor all wired together under one plan, does the priority case come out of
the pipeline — and does everything else come out untouched?

It is measured through the *shipped* modules rather than a harness copy of them.
`python -m bench selective-people` drives `detector_worker.BackgroundDetector` on
its own thread, `detection.Tracker`, `region_scheduler.RegionScheduler`, the cached
512×512 TensorRT engine and `compositor.Compositor` over a committed clip resized
to the app's own capture canvas, under `render_plan.priority_case_plan()` — the
same hardcoded plan the worker starts on behind `SD_DEMO_PLAN`. The block below is
generated from the committed record; re-run the case and the merge gate fails until
it is regenerated with `uv run python -m bench --selective-report`.

<!-- BEGIN SELECTIVE PATH -->
Measured through img2img-tensorrt-512x512-b1 over 48 consecutive frames of `people.mp4` resized to the app's 512x512 capture canvas, one row per machine. Clocks unlocked; absolute figures belong to the GPU in the row, and whether 30 FPS is met is a deploy-hardware question answered in spec 7.4, not here.

| case | GPU | plan | regions/frame | diffusion calls/frame | ms/frame | +detect | FPS | flicker (static px) | gate |
|---|---|---|---|---|---|---|---|---|---|
| selective-people | NVIDIA GeForce RTX 3080 Laptop GPU | person / lower_half / t_index 40 | 5.04 | 1.00 | 55.1 | 74.7 | 13.4 | 1.49 | pass |
| selective-people | NVIDIA GeForce RTX 4090 | person / lower_half / t_index 40 | 5.04 | 1.00 | 24.8 | 32.4 | 30.9 | 1.49 | pass |

The Gate, measured on NVIDIA GeForce RTX 4090:

- **Non-target pixels bit-identical** - yes. 48/48 frames left every pixel outside the rendered regions exactly as captured (164676 background pixels on the frame with the most painted).
- **The region is visibly restyled** - yes. The rendered regions changed by 11.8/255 against the capture - 11.8 net of the 0.00/255 the capture's own round trip costs - against a 8/255 threshold.
- **No track starved by the round robin** - yes. With 6 tracks over 2 slots no track waited more than 2 frames to be rendered, against a ceil(N/K) bound of 3.
- **The loop never stalls** - yes. 48/48 frames produced an output; the worst frame spent 0.015 ms offering the capture to the detector (budget 5 ms), and 0 frames had nothing to restyle and passed the capture through.

Manual verification artefact: `selective-people-20260907-101106Z-comparison.mp4` (source | selective render) and `selective-people-20260907-101106Z-comparison.jpg`.
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
§8.1's resident-but-idle figure. Raising the cadence is the cheap lever and is a
plan field already; spending it is M2's decision, not this one's.

**The composite is ~4 ms/frame of numpy** — the bounding rectangle of the regions,
blended on the host, on a frame that is otherwise entirely on the GPU. The
interface (an alpha map and a blend) is the same on either device, so moving it is
an M2 optimisation and not a redesign. It is also the only cost here that barely
moved on the faster card (§7.4): 2.75 ms against 4.03, where the frame path around
it came down to 0.45× — so it is a *larger* share of a 4090 frame than of a laptop
one, 11.1% against 7.3%.

**Flicker is 1.49 where §8.2 measured 1.43 on the same clip.** The two are not the
same measurement — this one renders detected boxes at the app's capture geometry,
that one rendered a committed track at clip resolution — but they are the same
order, so neither the tracker's box smoothing nor the feathered composite made the
boiling worse. Per-track seed pinning and the output EMA (§8.5) are still unbuilt.

**30 FPS is not this section's claim and is not a gate here.** This section asks
whether the path works, and both rows say it does. What it costs is a fact about
the card in the row — 13.4 FPS on an RTX 3080 laptop under a 120 W limit, 30.9 on
an RTX 4090 — and the frame-rate *verdict* is §7.4's, where issue #24 measured the
two against each other and answered acceptance criterion 2.

---

## 9. Risks

| Risk                                              | Impact                  | Mitigation                                             |
| ------------------------------------------------- | ----------------------- | ------------------------------------------------------ |
| Per-object diffusion can't hit 30 FPS at any useful K | kills the design        | Measure §7.2 **first**; fall back to option B / 7.3(d) |
| TRT fixed batch forces rebuilds mid-stream        | multi-second stalls     | Fixed slot count 7.3(a) or dynamic profiles 7.3(c)     |
| Small crops produce mush                          | unusable output quality | Minimum crop-size floor; skip objects below it         |
| Flicker makes it unwatchable                      | unusable                | §8.5 — measure, don't eyeball                          |
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
2. Sustained ≥ 30 FPS output with up to 4 tracked objects at 512² diffusion,
   measured **on deploy hardware** (RTX 3090 Ti / 4090) — see §7.4.
   **Met, measured (issue #24, 2026-09-07):** 30.9–31.4 FPS over five runs on an
   RTX 4090 at **5.04 regions/frame** — more than the four objects the criterion
   asks for — with the whole shipped selective path running. The margin is ~1 ms
   of a 33.33 ms frame, on a card with nothing else on it; §7.4 has the comparison
   against the dev laptop and what it does and does not license.
3. Typing a new instruction swaps behaviour with **no stutter** in the output
   stream and **no TensorRT rebuild**.
4. Non-target pixels are bit-identical to the capture (verifiable).
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
