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
- **No benchmark harness is versioned.** Scratch scripts have existed and been
  deleted; the measurements below currently have nowhere to live, which is why
  M0 builds a tracked one. Two conventions from those scripts are worth keeping:
  wait for the GPU to fall below ~62 °C before each rep (otherwise you measure
  thermal throttle), and time UNet / VAE-encode / VAE-decode separately by
  wrapping their `forward`.

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
(e.g. every 3rd–5th frame).

**C4 — Tracker.** Bridges the gap between detector ticks, gives each object a
stable ID, and smooths box jitter. Stable IDs are what let us pin a **seed and
prompt embedding per object**, which is the main lever against flicker.

**C5 — Region Scheduler.** The rate limiter. Detections are unbounded; the
diffusion engine has a *fixed* batch size. This component decides which ≤K
regions get diffused this frame, snaps their crops to the engine's native tile
size, and applies a round-robin / priority policy when there are more objects
than slots (see §7.3).

**C6 — Diffusion Executor.** The existing `StreamDiffusionWrapper`, driven with
a batch of crops instead of one full frame. Needs per-item prompt embeddings if
different objects carry different styles (§8.3).

**C7 — Compositor.** Pastes results back with feathered alpha, optional
colour/exposure match to surrounding pixels, and optional temporal EMA to
suppress flicker. Everything outside the regions is raw captured pixels.

---

## 6. The Render Plan (control-plane ⇄ data-plane contract)

The whole design hinges on this being small, validated, and stable. Straw-man:

```jsonc
{
  "plan_version": 7,
  "source_prompt": "find all people on screen and give them a red hat",
  "mode": "selective",              // "selective" | "global" | "inverse"
  "targets": [
    {
      "id": "t0",
      "concept": "person",          // free text; resolved by C2 to detector vocab
      "detector_class": 0,          // resolved COCO id, or null for open-vocab
      "region": "upper_third",      // "full_box" | "upper_third" | "lower_half" ...
      "box_scale": 1.15,            // dilation before cropping
      "prompt": "wearing a vibrant red hat",
      "negative_prompt": "blurry, deformed",
      "denoise": 0.45,              // maps to t_index_list selection
      "seed_policy": "per_track",   // "per_track" | "fixed" | "random"
      "max_instances": 6,
      "priority": 1
    }
  ],
  "background": { "action": "passthrough" },   // or {"action":"stylize","prompt":"..."}
  "global": { "fps_target": 30, "detect_every_n": 3 },
  "confidence": 0.82,
  "notes": "assumed 'red hat' means head region only"
}
```

Design rules:

- **Only C2 may write it.** The LLM proposes; the validator disposes.
- **Every field has a safe default.** A plan carrying just `targets[0].concept`
  and `.prompt` must render something sensible.
- **`notes` is surfaced in the UI.** The user should see what interpretation was
  chosen — "assumed head region only" — and be able to correct it in natural
  language rather than by hunting for a slider.
- **Versioned and atomic.** The render loop swaps plans between frames, never
  mid-frame.

---

## 7. Performance model

### 7.1 Budget

30 FPS = **33.3 ms/frame**. A first-cut allocation, to be replaced with
measurements:

| Stage               | Rate            | Budget (ms/frame) | Notes                     |
| ------------------- | --------------- | ----------------- | ------------------------- |
| Capture (DXcam)     | every frame     | 1–2               | already threaded          |
| Detection           | every 3rd frame | 4–8 amortised     | strongly model-dependent  |
| Tracking            | every frame     | < 1               | CPU                       |
| Preprocess crops    | every frame     | 1–2               | resize + normalise on GPU |
| **Diffusion**       | every frame     | **15–20**         | the dominant term         |
| Composite + present | every frame     | 2–3               |                           |
| Headroom            |                 | ~5                |                           |

The LLM appears nowhere in this table. That is the point.

### 7.2 The numbers we actually need

Not assumed — measured by the harness M0 builds, with the hardware recorded
alongside every number (§7.4):

1. SD-Turbo 1-step img2img at 512², TRT vs `none`, batch 1 — ms/frame.
2. Same at batch 2 / 4 / 8 — **is the per-crop marginal cost sublinear?** This
   determines whether N-object rendering is viable at all.
3. Same at 256² and 384² crops — small crops are the whole economic case.
4. Detector latency for each candidate in §8.1 at 640² input, TRT.
5. Peak VRAM with diffusion engine + detector + (optional) local LLM resident.
6. Cost of swapping prompt embeddings per batch item.

Measure the batch and resolution sweep on the **`none` accelerator first**. It
answers the question that actually matters — the *shape* of the marginal-cost
curve — at zero engine-build cost, and only then is it worth spending 5.1 GB and
several minutes per TensorRT engine to confirm the two or three configurations
the curve says are interesting.

### 7.3 The batch-size problem (the hard one)

A TensorRT engine is built for a **fixed batch size**. The scene contains a
*variable* number of objects. Options:

- **(a) Fixed slot count K.** Build for batch K (say 4). Fewer objects → pad
  with dummies (wasted compute). More objects → the Region Scheduler
  round-robins across frames, so each object updates at 30/⌈N/K⌉ FPS. Simple,
  predictable, no rebuilds. **Current preference.**
- **(b) Multiple engines** (K = 1, 2, 4, 8) resident, chosen per frame. Costs
  VRAM and build time; avoids padding waste.
- **(c) Dynamic-shape TRT profiles.** An optimisation profile with a min/max
  batch range. Best answer if StreamDiffusion's engine builder supports it —
  **needs investigation**.
- **(d) Single canvas.** Pack all crops into one 512² atlas and diffuse it as a
  single image. Constant cost, one engine, no batching problem — but objects
  bleed across tile seams and share one prompt. Cheap to prototype; may be good
  enough for uniform edits.

Option (d) is the fastest thing to test, and the answer might simply be (d) + (a).

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
| Absolute ms/frame, and whether 30 FPS is met   | **No** — re-measure on the deploy GPU              |
| VRAM ceilings and OOM thresholds               | **No** — a laptop OOM says nothing about 24 GB     |
| Engine build times                             | **No**                                             |

The 30 FPS gate in §11 is therefore a **deploy-hardware** criterion. On the
laptop the same run is a regression check, not an acceptance test.

A laptop also may never reach the §7.2 cooldown threshold under sustained load.
The harness caps the wait and records that the threshold was not met, rather than
blocking forever or silently reporting a throttled number.

---

## 8. Open questions (the research agenda)

### 8.1 Which detector?

| Candidate      | Vocabulary            | Est. speed | Note                                                                       |
| -------------- | --------------------- | ---------- | -------------------------------------------------------------------------- |
| YOLOv8n/s      | 80 COCO classes       | fastest    | "person", "car", "dog" only                                                |
| YOLO-World     | open, text-prompted   | fast-ish   | class embeddings precomputed per plan — fits the compile-once model exactly |
| OWLv2          | open                  | slower     | better on unusual concepts                                                 |
| Grounding DINO | open, phrase grounding | slowest    | handles "the red mug on the left"                                          |
| SAM2 / FastSAM | promptable segmentation | varies    | masks not boxes; may be what we actually want                              |

The instruction "give **them** a red hat" needs a *head region*, not a person
box. Boxes may be too coarse; a segmentation stage or a body-part heuristic
(`region: "upper_third"`) may be required. **Open.**

### 8.2 Is crop-and-diffuse the right primitive at all?

Honest risk: img2img on a 64×64 person crop upscaled to 512² will hallucinate
detail, drift in identity frame to frame, and flicker. Alternatives worth
benchmarking side by side before committing:

- **A. Crop → diffuse → composite** (this spec's default).
- **B. Full-frame diffuse once, masked composite.** Constant cost, temporally
  smoother, but every object gets the same prompt.
- **C. Latent-space masking / inpainting.** Diffuse the full latent, blending
  masked and unmasked latents per step. Cheaper than A, more localised than B —
  but SD-Turbo's 1-step schedule leaves little room to blend.
- **D. ControlNet-conditioned.** Would need the dead `controlnet_paths` stub
  wired up plus a TRT engine that supports it. Best structure preservation,
  highest cost.

**This comparison should probably happen before anything else is built.**

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
