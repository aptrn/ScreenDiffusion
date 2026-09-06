import { runSandcastle } from 'sandcastle-kit';
import { noSandbox } from '@ai-hero/sandcastle/sandboxes/no-sandbox';

await runSandcastle({
  // No container: this repo's work is CUDA/TensorRT benchmarking, which needs the
  // host GPU. Docker is not installed here, and GPU passthrough on Windows would
  // buy isolation we cannot use.
  //
  // Isolation comes from git worktrees instead. Sandcastle checks each issue's
  // branch out under `.sandcastle/worktrees/<branch>/` (gitignored) and runs the
  // agent there, so this working directory — and whatever is uncommitted in it —
  // is never touched. Each worktree gets its own `.venv` from the `uv sync` hook
  // below; uv hardlinks from its global cache, so that is cheap after the first.
  sandbox: noSandbox(),

  // One issue at a time. There is a single GPU, and it is the thing under
  // measurement — two agents benchmarking concurrently contend for VRAM and clocks
  // and produce numbers that mean nothing.
  maxConcurrentIssues: 1,

  // GPU-free, dependency-free, seconds. Grows on its own once tests/ exists.
  verify: 'python scripts/verify.py',

  // uv owns the environment; the default `npm install` would set up the wrong one.
  // Warm uv cache makes this a hardlink pass, not a re-download.
  hooks: { sandbox: { onSandboxReady: [{ command: 'uv sync' }] } },

  // The dashboard is unauthenticated. Loopback-only; switch to '0.0.0.0' to reach
  // it from another device on a trusted network.
  monitor: { host: '127.0.0.1' },
});
