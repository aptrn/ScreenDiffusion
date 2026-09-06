"""The measuring half. This is the only module that touches the GPU.

torch is imported inside the functions, never at module scope, so importing the CLI
on a machine with no CUDA device still works and the merge gate's GPU-free tier
stays GPU-free.

Two things about the numbers this produces:

- Every timed region is bracketed by `torch.cuda.synchronize()`. Without it the
  clock measures queue submission, not compute, and every figure is fiction.
- `peak_vram_bytes` is `torch.cuda.max_memory_allocated`, so it counts what the
  torch allocator handed out. A TensorRT engine allocates its own device memory
  outside that allocator, so on the `tensorrt` accelerator this figure is a floor,
  not the resident total. The raw `nvidia-smi` dump in the fingerprint is the
  cross-check.
"""

from __future__ import annotations

import importlib.util
import math
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from bench.clocks import clock_normalization, regime_summary
from bench.cooldown import (
    DEFAULT_CAP_S,
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_THRESHOLD_C,
    CooldownRecord,
    skipped_cooldown,
    wait_for_cooldown,
)
from bench.disk import DiskRecord
from bench.fingerprint import capture_fingerprint, read_gpu_sample, utc_now
from bench.paths import (
    REPO_ROOT,
    RESULTS_DIR,
    resolve_engines_dir,
    resolve_model_path,
)
from bench.results import (
    BYTES_PER_MIB,
    README_NAME,
    BenchResult,
    RunMetrics,
    append_readme_row,
    write_result,
)
from bench.scenarios import ScenarioConfig

# Half a second: often enough to catch a clock drop inside a 30-rep run, rare enough
# that the `nvidia-smi` subprocesses do not crowd out the process being measured.
DEFAULT_SAMPLE_INTERVAL_S = 0.5


def load_wrapper_module():
    """Load `wrapper.py` by path, the way the worker does - it is not an importable module.

    Importing it by name would work here, but going through the same door the worker
    uses keeps the harness measuring the code path the app actually runs.
    """
    if "wrapper" in sys.modules:
        return sys.modules["wrapper"]
    spec = importlib.util.spec_from_file_location("wrapper", REPO_ROOT / "wrapper.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["wrapper"] = module
    spec.loader.exec_module(module)
    return module


def build_stream(scenario: ScenarioConfig, engines_root: Optional[Path] = None):
    """A prepared `StreamDiffusionWrapper` for `scenario`, with the app's own settings."""
    wrapper = load_wrapper_module()
    engines_root = resolve_engines_dir() if engines_root is None else Path(engines_root)
    stream = wrapper.StreamDiffusionWrapper(
        model_id_or_path=resolve_model_path(scenario.model),
        t_index_list=list(scenario.t_index_list),
        frame_buffer_size=scenario.batch_size,
        width=scenario.width,
        height=scenario.height,
        warmup=2,
        acceleration=scenario.acceleration,
        do_add_noise=scenario.do_add_noise,
        mode=scenario.mode,
        use_denoising_batch=scenario.use_denoising_batch,
        cfg_type=scenario.cfg_type,
        seed=scenario.seed,
        use_lcm_lora=scenario.use_lcm_lora,
        use_tiny_vae=scenario.use_tiny_vae,
        engine_dir=str(engines_root),
    )
    stream.prepare(prompt=scenario.prompt, num_inference_steps=50)
    return stream


def _input_batch(stream, scenario: ScenarioConfig):
    """One preprocessed input tensor of `batch_size` frames.

    Preprocessing is done once and left outside the timed region: in the app the
    capture thread already hands the frame loop a tensor, so folding a PIL resize
    into every rep would measure something the frame path does not do.
    """
    import torch
    from PIL import Image

    frame = Image.new("RGB", (scenario.width, scenario.height), (32, 96, 160))
    tensor = stream.preprocess_image(frame)
    if scenario.batch_size == 1:
        return tensor
    return torch.cat([tensor] * scenario.batch_size)


# Attribute on the inner `StreamDiffusion` -> the label it is timed under. One
# mapping, so the three submodules are named once instead of at every step of
# patch / run / restore / report.
TIMED_SUBMODULES: Dict[str, str] = {
    "unet": "unet",
    "encode_image": "vae_encode",
    "decode_image": "vae_decode",
}


class _TimedCall:
    """A callable that times what it wraps and forwards everything else to it.

    Used only under `--per-module`: the synchronise on each submodule perturbs the
    total, which is why the per-module figures never feed the headline number.
    """

    def __init__(self, name: str, inner: Callable, totals: Dict[str, float]):
        self._name = name
        self._inner = inner
        self._totals = totals

    def __call__(self, *args, **kwargs):
        import torch

        torch.cuda.synchronize()
        started = time.perf_counter()
        result = self._inner(*args, **kwargs)
        torch.cuda.synchronize()
        self._totals[self._name] = self._totals.get(self._name, 0.0) + (
            time.perf_counter() - started) * 1000.0
        return result

    def __getattr__(self, item):
        # `stream.unet.config` and friends still have to resolve.
        return getattr(self._inner, item)


def _time_modules(stream, scenario: ScenarioConfig, batch, reps: int) -> Dict[str, float]:
    """Mean ms/frame spent in the UNet, the VAE encoder and the VAE decoder.

    A separate pass after the headline timing, so the extra synchronises cannot
    contaminate it. Restores the originals before returning.
    """
    inner = stream.stream
    totals: Dict[str, float] = {}
    originals = {attribute: getattr(inner, attribute) for attribute in TIMED_SUBMODULES}
    for attribute, label in TIMED_SUBMODULES.items():
        setattr(inner, attribute, _TimedCall(label, originals[attribute], totals))
    try:
        for _ in range(reps):
            stream(image=batch)
    finally:
        for attribute, original in originals.items():
            setattr(inner, attribute, original)
    frames = reps * scenario.batch_size
    return {label: round(totals.get(label, 0.0) / frames, 4)
            for label in TIMED_SUBMODULES.values()}


def cooldown_gate(enabled: bool, threshold_c: float, cap_s: float,
                  poll_interval_s: float, log: Callable[[str], None]) -> CooldownRecord:
    """Wait for a cool GPU, or record that the wait was skipped or capped.

    Public because `bench.detector_runner` opens the same gate before its own timed
    region (issue #4): a detector measured on a throttled die is as misleading as a
    diffusion cell measured on one.
    """
    if not enabled:
        log("cooldown: skipped (--no-cooldown)")
        return skipped_cooldown(threshold_c=threshold_c, cap_s=cap_s)
    log(f"cooldown: waiting for <= {threshold_c:.0f} C (cap {cap_s:.0f} s)")
    record = wait_for_cooldown(
        lambda: read_gpu_sample().temperature_c,
        threshold_c=threshold_c, cap_s=cap_s, poll_interval_s=poll_interval_s,
    )
    log(f"cooldown: {record.outcome} after {record.waited_s:.1f} s "
        f"at {record.final_temperature_c} C")
    return record


class GpuSampler:
    """Polls clocks and temperature on a background thread while the reps run.

    Sampling from inside the timed loop was the first version and it was wrong:
    each `nvidia-smi` call costs tens of milliseconds of wall clock, and the gap it
    leaves between reps lets the GPU drop out of its boost state, so the harness
    measured a clock it had itself lowered. A thread never interrupts the loop; it
    only costs CPU, which is not what is being measured.
    """

    def __init__(self, interval_s: float = DEFAULT_SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        # [[elapsed_s, sm_clock_mhz, temperature_c], ...]
        self.samples: List[List[Optional[float]]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = read_gpu_sample()
            self.samples.append([time.perf_counter() - self._started,
                                 sample.sm_clock_mhz, sample.temperature_c])
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuSampler":
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @property
    def mean_sm_clock_mhz(self) -> Optional[float]:
        clocks = [clock for _, clock, _ in self.samples if clock is not None]
        return round(statistics.fmean(clocks), 1) if clocks else None

    @property
    def max_temperature_c(self) -> Optional[float]:
        temperatures = [temp for _, _, temp in self.samples if temp is not None]
        return max(temperatures) if temperatures else None


def _percentile(values: List[float], fraction: float) -> float:
    """Nearest-rank percentile. `statistics.quantiles` needs n >= 2; a 1-rep run is legal.

    Nearest rank is `ceil(fraction * n)`, 1-based. Spelling that as `round(x + 0.5)`
    silently disagrees with it whenever `fraction * n` is an odd integer, because
    `round` breaks a .5 tie towards even - a 20-rep run reported its max as its p95.
    """
    ordered = sorted(values)
    rank = min(len(ordered), max(1, math.ceil(fraction * len(ordered))))
    return ordered[rank - 1]


def run_scenario(
    scenario: ScenarioConfig,
    cooldown: bool = True,
    per_module: bool = False,
    results_dir: Path = RESULTS_DIR,
    threshold_c: float = DEFAULT_THRESHOLD_C,
    cap_s: float = DEFAULT_CAP_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    engines_root: Optional[Path] = None,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    disk: Optional[DiskRecord] = None,
    log: Callable[[str], None] = print,
) -> BenchResult:
    """Measure `scenario` once and write its result. Returns the record that was written."""
    import torch

    # First, so a machine that cannot be fingerprinted fails before it spends
    # minutes building an engine for a result that could never be written.
    fingerprint = capture_fingerprint()
    log(f"gpu: {fingerprint.gpu_name} | driver {fingerprint.driver_version} | "
        f"{regime_summary(fingerprint.clock_lock)}")

    log(f"building {scenario.name} ({scenario.acceleration}, "
        f"{scenario.width}x{scenario.height}, batch {scenario.batch_size})")
    stream = build_stream(scenario, engines_root=engines_root)
    batch = _input_batch(stream, scenario)

    cooldown_record = cooldown_gate(cooldown, threshold_c, cap_s, poll_interval_s, log)

    for _ in range(scenario.warmup_reps):
        stream(image=batch)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started_utc = utc_now()
    per_rep_ms: List[float] = []
    with GpuSampler(sample_interval_s) as sampler:
        for _ in range(scenario.reps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            stream(image=batch)
            torch.cuda.synchronize()
            per_rep_ms.append((time.perf_counter() - start) * 1000.0)
    peak_vram_bytes = int(torch.cuda.max_memory_allocated())
    finished_utc = utc_now()

    per_module_ms = _time_modules(stream, scenario, batch, scenario.reps) if per_module else None

    # Per *call*, which renders `batch_size` frames. The per-frame figures divide by
    # it - that ratio is the sublinearity question spec 7.2 item 2 asks.
    per_frame_ms = [ms / scenario.batch_size for ms in per_rep_ms]
    mean_ms = statistics.fmean(per_frame_ms)

    run = RunMetrics(
        started_utc=started_utc,
        finished_utc=finished_utc,
        warmup_reps=scenario.warmup_reps,
        reps=scenario.reps,
        per_rep_ms=[round(ms, 4) for ms in per_rep_ms],
        mean_ms_per_frame=round(mean_ms, 4),
        median_ms_per_frame=round(statistics.median(per_frame_ms), 4),
        p95_ms_per_frame=round(_percentile(per_frame_ms, 0.95), 4),
        min_ms_per_frame=round(min(per_frame_ms), 4),
        max_ms_per_frame=round(max(per_frame_ms), 4),
        stdev_ms_per_frame=round(statistics.stdev(per_frame_ms), 4) if len(per_frame_ms) > 1 else 0.0,
        fps=round(1000.0 / mean_ms, 3),
        mean_sm_clock_mhz=sampler.mean_sm_clock_mhz,
        max_temperature_c=sampler.max_temperature_c,
        peak_vram_bytes=peak_vram_bytes,
        per_module_ms=per_module_ms,
        gpu_samples=sampler.samples,
    )
    # From the trace the sampler already took, and from the lock state read at the
    # top of this function - not from a second reading, which could disagree with
    # the one the result records (issue #13).
    normalisation = clock_normalization(fingerprint.clock_lock, sampler.samples,
                                        raw_ms_per_frame=run.mean_ms_per_frame)
    result = BenchResult(scenario=scenario, run=run, cooldown=cooldown_record,
                         hardware=fingerprint, disk=disk,
                         clock_normalization=normalisation)

    results_dir = Path(results_dir)
    path = write_result(result, results_dir=results_dir)
    append_readme_row(result, results_dir / README_NAME, filename=path.name)
    log(f"{scenario.name}: {run.mean_ms_per_frame:.2f} ms/frame ({run.fps:.1f} FPS), "
        f"peak {peak_vram_bytes / BYTES_PER_MIB:.0f} MiB -> {path.name}")
    if normalisation.ms_per_frame is not None:
        log(f"clocks {normalisation.regime}: {normalisation.ms_per_frame:.2f} ms/frame "
            f"estimated at {normalisation.basis_mhz:.0f} MHz "
            f"(sampled {normalisation.sampled_mean_sm_clock_mhz:.0f} MHz) - an "
            f"estimate, not a measurement")
    return result
