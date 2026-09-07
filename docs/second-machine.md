# Running the loop on a second machine

Issue state lives on GitHub, so the Sandcastle loop resumes wherever you start it.
What follows is what the repo does *not* carry, and the one step that fails silently.

## One loop at a time

Two loops on the same repo race: both planners plan, both label issues `Sandcastle`,
both open PRs, both auto-merge. **Stop the loop on the other machine first.** The loop
resumes somewhere else; it does not run in addition.

## Setup

```bash
git clone https://github.com/aptrn/ScreenDiffusion.git
cd ScreenDiffusion
gh repo set-default aptrn/ScreenDiffusion   # do not skip - see below
```

Then `setup.bat`, which installs **uv** if it is missing, fetches Python 3.11 and runs
`uv sync`. Do not reach for a bare `uv sync` — a machine without uv gets no useful error
from it, and the loop needs uv on PATH for far more than the initial install (every
agent worktree runs `uv sync` as its sandbox-ready hook, so a missing uv fails every
sandbox, not just setup).

**Open a new terminal before going further.** The uv installer only adds
`%USERPROFILE%\.local\bin` to PATH for *new* shells. `setup.bat` patches its own session
so its `uv sync` succeeds, which means setup can finish cleanly and the loop still fail
afterwards if you reuse the old terminal.

Then `npm install` for the Node-side tooling, and create `.sandcastle/.env` with
`CLAUDE_CODE_OAUTH_TOKEN` and `GH_TOKEN` (a fine-grained PAT scoped to this repo:
Contents R/W, Issues R/W, Pull requests R/W, Metadata R).

`npm run sandcastle` starts the loop; the dashboard binds `127.0.0.1:4747`.

## The step that fails silently

This repo is a fork of `rudyaa-sd/ScreenDiffusion`. `gh pr create` inside a fork targets
the **parent** repository unless `remote.origin.gh-resolved` is set — and that setting
lives in `.git/config`, which a fresh clone does not have.

Skip `gh repo set-default` and the loop's pr-opener will try to open its pull requests
against the upstream project instead of yours. Nothing warns you until it happens.

## What does not travel

| | why | what to do on the new machine |
|---|---|---|
| `models/` | gitignored | copy it across, or fetch it - see below. The bench does **not** fetch it for you; point `SD_MODELS_DIR` at wherever it lands |
| `engines/` | gitignored **and** GPU-specific | **rebuild.** Ampere engines do not load on Ada. ~5.1 GB and 15-25 min each (measured ~5 min on a 4090); pass `--allow-engine-build` |
| `.sandcastle/.env` | secrets | recreate |
| `.venv/`, `node_modules/` | build artefacts | `setup.bat` (installs uv + Python 3.11 + syncs), then `npm install` |
| uv itself | not part of the repo | `setup.bat` installs it, then **use a new terminal** |
| `.sandcastle/logs/`, `worktrees/`, `monitor-history.json` | local run state | nothing - they regenerate |

Committed results, reference clips and box tracks **do** travel, so comparisons stay
reproducible across machines. Every result carries a hardware fingerprint, so 3080 and
4090 rows coexist in `bench/results/` without ambiguity (spec §7.4).

### Filling `models/` from scratch

`resolve_model_path` falls back to the *name* `sd-turbo-fp16` when there is no local
directory, and that is not a Hugging Face repo id - so nothing downloads it on your
behalf and the failure surfaces as a confusing load error. Fetch it explicitly. The
`--exclude` list drops the fp32 duplicates, exactly as the GUI's downloader does:

```sh
huggingface-cli download stabilityai/sd-turbo --local-dir "$SD_MODELS_DIR/sd-turbo-fp16" \
  --exclude sd_turbo.safetensors unet/diffusion_pytorch_model.safetensors \
  vae/diffusion_pytorch_model.safetensors text_encoder/model.safetensors
curl -L -o "$SD_MODELS_DIR/detectors/yolov8s-worldv2.pt" \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt
```

That is 2.5 GB and 26 MB. The 338 MB CLIP text checkpoint YOLO-World needs is fetched
by ultralytics on the first `set_concepts`, into `$SD_MODELS_DIR/detectors`. The
detector bench's evidence photographs are separate again and only arrive behind
`python -m bench yolo-world-s-640 --allow-download`; without them four GPU tests skip
rather than fail.

## On the 4090 specifically

- ~~Remove the `hold` label from the deploy-validation issue.~~ Done: issue #24 ran
  on 2026-09-07 and acceptance criterion 2 is met at 30.9-31.4 FPS, by about 1 ms of
  a 33.33 ms frame. Spec §7.4 carries the comparison.
- The frame-budget issue produces far more meaningful numbers here than on the laptop,
  where nothing came within 2x of the budget - and it now has a measured baseline to
  optimise against rather than an assumed one.
- **Do not benchmark in the process that built the engine.** The first 4090 run
  measured 29.5 FPS and every cold-started run since has measured 30.9-31.4. Build
  the engine, let the process exit, then measure.
- Clock locking still needs an elevated shell. A desktop card throttles less than a
  120 W laptop, but `--require-locked-clocks` is still the door for a run that decides
  something.
- `maxConcurrentIssues: 1` in `.sandcastle/main.mts` is set because the GPU is the
  instrument under measurement, not because the laptop is slow. The reasoning holds on
  any single-GPU machine.
- Spec §7.4 calls the 3080 laptop the dev machine and the 3090 Ti / 4090 the deploy
  target. If the 4090 becomes a dev machine too, update that table rather than letting
  the two roles blur - the portability claims depend on knowing which is which.
