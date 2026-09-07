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
npm install
uv sync
```

Then create `.sandcastle/.env` with `CLAUDE_CODE_OAUTH_TOKEN` and `GH_TOKEN` (a
fine-grained PAT scoped to this repo: Contents R/W, Issues R/W, Pull requests R/W,
Metadata R). `npm run sandcastle` starts it; the dashboard binds `127.0.0.1:4747`.

## The step that fails silently

This repo is a fork of `rudyaa-sd/ScreenDiffusion`. `gh pr create` inside a fork targets
the **parent** repository unless `remote.origin.gh-resolved` is set — and that setting
lives in `.git/config`, which a fresh clone does not have.

Skip `gh repo set-default` and the loop's pr-opener will try to open its pull requests
against the upstream project instead of yours. Nothing warns you until it happens.

## What does not travel

| | why | what to do on the new machine |
|---|---|---|
| `models/` (5.4 GB) | gitignored | copy it across, or let the app and bench re-download; point `SD_MODELS_DIR` at wherever it lands |
| `engines/` | gitignored **and** GPU-specific | **rebuild.** Ampere engines do not load on Ada. ~5.1 GB and 15-25 min each |
| `.sandcastle/.env` | secrets | recreate |
| `.venv/`, `node_modules/` | build artefacts | `uv sync`, `npm install` |
| `.sandcastle/logs/`, `worktrees/`, `monitor-history.json` | local run state | nothing - they regenerate |

Committed results, reference clips and box tracks **do** travel, so comparisons stay
reproducible across machines. Every result carries a hardware fingerprint, so 3080 and
4090 rows coexist in `bench/results/` without ambiguity (spec §7.4).

## On the 4090 specifically

- Remove the `hold` label from the deploy-validation issue. It is the only thing that
  tests the 30 FPS acceptance criterion, and it has never run.
- The frame-budget issue produces far more meaningful numbers here than on the laptop,
  where nothing came within 2x of the budget.
- Clock locking still needs an elevated shell. A desktop card throttles less than a
  120 W laptop, but `--require-locked-clocks` is still the door for a run that decides
  something.
- `maxConcurrentIssues: 1` in `.sandcastle/main.mts` is set because the GPU is the
  instrument under measurement, not because the laptop is slow. The reasoning holds on
  any single-GPU machine.
- Spec §7.4 calls the 3080 laptop the dev machine and the 3090 Ti / 4090 the deploy
  target. If the 4090 becomes a dev machine too, update that table rather than letting
  the two roles blur - the portability claims depend on knowing which is which.
