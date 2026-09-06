"""What a detector run records, and the table spec 8.1 carries.

Issue #4. A detector result answers different questions from a diffusion one - what
a vocabulary change costs, whether the frame path noticed it, what the detector
found when it was asked for a concept COCO has never heard of - so it has its own
record and its own table, in `bench/results/detectors/`.

Its own *directory*, because `bench --marginal` reads every JSON beside it as a
diffusion cell: a detector record dropped in the parent would not be ignored, it
would crash the marginal report. What it shares is the door. `write_detector_result`
and `append_detector_readme_row` go through `bench.results.require_recordable` like
everything else, so a detector number cannot reach disk without a machine and a
clock regime attached either.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from bench import RESULT_SCHEMA_VERSION
from bench.clocks import ClockNormalization, regime_of
from bench.cooldown import CooldownRecord
from bench.detectors import (
    DEFAULT_CADENCE,
    BudgetVerdict,
    Candidate,
    DetectorConfig,
    FramePathVerdict,
    recommend,
)
from bench.fingerprint import Fingerprint
from bench.results import (
    BYTES_PER_MIB,
    append_row,
    normalised_cell,
    require_recordable,
    timestamp_from,
    write_record,
)

RECORD_KIND = "detector"

# spec 7.4 again, on the one figure most likely to be quoted out of its machine: a
# 16 GB laptop's combined ceiling says nothing about a 24 GB desktop's.
VRAM_NOTE = ("Measured on the development laptop. VRAM ceilings do not transfer "
             "(spec 7.4) - re-measure the combined figure on the deploy GPU.")


@dataclass(frozen=True)
class LatencySummary:
    """The distribution of one detector's per-detect times.

    Built from the samples rather than passed in beside them, so the record cannot
    disagree with the trace it was computed from.
    """

    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> "LatencySummary":
        ordered = sorted(samples)
        # Nearest rank, the same spelling `bench.runner._percentile` uses: `ceil`,
        # not `round(x + 0.5)`, which overshoots when `fraction * n` is an odd integer.
        rank = min(len(ordered), max(1, math.ceil(0.95 * len(ordered))))
        return cls(
            mean_ms=round(statistics.fmean(samples), 4),
            median_ms=round(statistics.median(samples), 4),
            p95_ms=round(ordered[rank - 1], 4),
            min_ms=round(ordered[0], 4),
            max_ms=round(ordered[-1], 4),
            stdev_ms=round(statistics.stdev(samples), 4) if len(samples) > 1 else 0.0,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VramRecord:
    """What was resident, and what the detector added on top of it.

    The issue's fourth trap: a detector benchmarked alone says nothing about whether
    it fits. So the run loads the diffusion engine first and reads `nvidia-smi`
    three times - empty, diffusion resident, both resident - and the difference
    between the last two is what the detector actually costs in place.

    `torch_peak_bytes` is `torch.cuda.max_memory_allocated`, which counts only what
    the torch allocator handed out; a TensorRT engine allocates outside it. That is
    why the `nvidia-smi` figures are the ones the gate reads.
    """

    diffusion_scenario: Optional[str]
    baseline_used_mib: Optional[float]
    diffusion_used_mib: Optional[float]
    combined_used_mib: Optional[float]
    torch_peak_bytes: int
    note: str = VRAM_NOTE

    @property
    def detector_delta_mib(self) -> Optional[float]:
        """What loading and running the detector added to an already-loaded GPU."""
        if self.combined_used_mib is None:
            return None
        base = self.diffusion_used_mib if self.diffusion_used_mib is not None \
            else self.baseline_used_mib
        return None if base is None else round(self.combined_used_mib - base, 1)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["torch_peak_mib"] = round(self.torch_peak_bytes / BYTES_PER_MIB, 1)
        data["detector_delta_mib"] = self.detector_delta_mib
        return data


@dataclass(frozen=True)
class VocabularyChange:
    """The cold-path cost of changing what the detector is looking for.

    `first_change_ms` is separate from the steady-state ones because it is a
    different event: it includes fetching and loading the text encoder, which
    happens once per process and not once per prompt edit. Reporting one mean over
    both would overstate the cost of editing a prompt by two orders of magnitude.
    """

    supported: bool
    terms: List[str]
    first_change_ms: Optional[float]
    change_ms: List[float]
    median_change_ms: Optional[float]
    text_encoder: Optional[str]
    note: str

    @classmethod
    def unsupported(cls, note: str) -> "VocabularyChange":
        """A closed-vocabulary detector: not a change costing zero, a change that
        cannot be made at all. The distinction is the whole of spec 8.1."""
        return cls(supported=False, terms=[], first_change_ms=None, change_ms=[],
                   median_change_ms=None, text_encoder=None, note=note)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Detection:
    """One box the detector returned, as it lands in the evidence."""

    label: str
    confidence: float
    box_xyxy: List[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ConceptEvidence:
    """Whether one concept was resolved, in one frame, with the boxes to show for it.

    `queried` is what was actually put in front of the detector and may differ from
    `concept`: a closed-vocabulary detector cannot be asked for `red mug`, so it is
    asked for the nearest COCO class instead and the substitution is recorded. That
    is not the same as resolving the concept, and `resolved` says so.

    `strongest_other` is what the detector returned under some *other* label, and it
    is the difference between "found nothing" and "found the thing and called it
    something else". Asked for `dog` on a desktop showing a dog, YOLOv8n returned no
    dog and a cat at 0.79 - which is a finding, and is invisible in a record that
    keeps only the boxes matching the label it was asked for.

    The image is identified by URL and SHA-256 rather than committed: the pictures
    are third-party, and a hash is what makes the claim re-checkable.
    """

    concept: str
    kind: str
    queried: Optional[str]
    frame: str
    image_name: str
    image_source: str
    image_sha256: str
    resolved: bool
    top_confidence: Optional[float]
    detections: List[Detection]
    note: str
    strongest_other: List[Detection] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["detections"] = [detection.to_dict() for detection in self.detections]
        data["strongest_other"] = [d.to_dict() for d in self.strongest_other]
        return data


@dataclass(frozen=True)
class DetectorMetrics:
    """Everything measured about one detector's hot path."""

    started_utc: str
    finished_utc: str
    warmup_reps: int
    reps: int
    imgsz: int
    per_rep_ms: List[float]
    latency: LatencySummary
    detects_per_second: float
    # ultralytics' own preprocess / inference / postprocess split, carried because
    # the frame loop pays all three and only the middle one is "the model".
    ultralytics_speed_ms: Dict[str, float]
    mean_sm_clock_mhz: Optional[float]
    max_temperature_c: Optional[float]
    gpu_samples: List[List[Optional[float]]] = field(default_factory=list)
    # What was in front of the detector while it was timed. A detector timed on an
    # empty frame gives NMS nothing to do and reports a latency the frame loop will
    # never see, so the frame is part of the measurement, not a detail of it.
    timing_frame: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["latency"] = self.latency.to_dict()
        return data


@dataclass(frozen=True)
class DetectorResult:
    """One detector, measured. The unit `bench/results/detectors/` stores."""

    detector: DetectorConfig
    run: DetectorMetrics
    cooldown: CooldownRecord
    hardware: Fingerprint
    vram: VramRecord
    budget: BudgetVerdict
    vocabulary_change: VocabularyChange
    # None for a detector whose vocabulary cannot change: there is no before and
    # after to compare, and a verdict invented from one measurement would be a claim
    # about a thing that did not happen.
    frame_path: Optional[FramePathVerdict]
    evidence: List[ConceptEvidence]
    clock_normalization: Optional[ClockNormalization] = None
    # The same detector on the *raw* screen, with nothing composited in. Kept out of
    # `evidence` deliberately: it is not a concept anyone asked to be resolved, and
    # counting it would make a detector that correctly found nothing look worse than
    # one that hallucinated a dog on an empty desktop.
    desktop_control: Optional[ConceptEvidence] = None

    def to_dict(self) -> dict:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "kind": RECORD_KIND,
            "detector": self.detector.to_dict(),
            "run": self.run.to_dict(),
            "cooldown": self.cooldown.to_dict(),
            "hardware": self.hardware.to_dict(),
            "vram": self.vram.to_dict(),
            "budget": self.budget.to_dict(),
            "vocabulary_change": self.vocabulary_change.to_dict(),
            "frame_path": None if self.frame_path is None else self.frame_path.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "desktop_control": (None if self.desktop_control is None
                                else self.desktop_control.to_dict()),
            "clock_normalization": (None if self.clock_normalization is None
                                    else self.clock_normalization.to_dict()),
        }


ResultLike = Union[DetectorResult, dict]


def _as_dict(result: ResultLike) -> dict:
    return result.to_dict() if isinstance(result, DetectorResult) else result


def detector_result_filename(detector_name: str, timestamp: str) -> str:
    return f"{detector_name}-{timestamp}.json"


def write_detector_result(result: ResultLike, results_dir: Path,
                          timestamp: Optional[str] = None) -> Path:
    """Write `<detector>-<timestamp>.json` under `results_dir`; return its path."""
    data = _as_dict(result)
    if timestamp is None:
        timestamp = timestamp_from(data)
    return write_record(data, results_dir,
                        detector_result_filename(data["detector"]["name"], timestamp))


DETECTOR_README_TITLE = "# Detector benchmark results"
DETECTOR_README_INTRO = (
    "Written by `uv run python -m bench <detector>`, never by hand - issue #4,\n"
    "spec 8.1. Each row points at the JSON file holding the full detector config,\n"
    "the hardware fingerprint, the vocabulary-change cost and the per-concept\n"
    "detection evidence.\n\n"
    "`amortised` is ms/detect divided by the detect cadence in the same row: spec 7.1\n"
    "gives detection 4-8 ms amortised at one detect every 3rd frame, and that is\n"
    "the figure `fits` judges. Absolute milliseconds and VRAM belong to the GPU in\n"
    "the row (spec 7.4); compare rows for ranking, not for whether 30 FPS is met.\n\n"
    "`with diffusion (MiB)` is `nvidia-smi` memory in use with the diffusion engine\n"
    "*and* the detector resident - the only figure that answers whether they fit\n"
    "together. `vocab change` is the cold-path text encode, paid when the user edits\n"
    "the prompt and never on the frame path.\n"
)
DETECTOR_README_HEADER = (
    "| finished (UTC) | detector | GPU | role | vocabulary | input | ms/detect |"
    " p95 ms | cadence | amortised ms/frame | fits 4-8 ms | torch peak (MiB) |"
    " with diffusion (MiB) | vocab change (ms) | concepts resolved | cooldown |"
    " clock regime | ms/detect at basis clock | file |"
)
DETECTOR_README_SEPARATOR = "|" + "---|" * (DETECTOR_README_HEADER.count("|") - 1)
DETECTOR_README_NAME = "README.md"

DETECTOR_README_PREAMBLE = (
    f"{DETECTOR_README_TITLE}\n\n{DETECTOR_README_INTRO}\n"
    f"{DETECTOR_README_HEADER}\n{DETECTOR_README_SEPARATOR}\n"
)


def _number(value: Optional[float], digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _resolved_cell(result: dict) -> str:
    """`2/3` - how many of the probed concepts this detector actually found."""
    evidence = result.get("evidence") or []
    return f"{sum(1 for item in evidence if item['resolved'])}/{len(evidence)}"


def detector_readme_row(result: dict, filename: str) -> str:
    detector, run, budget = result["detector"], result["run"], result["budget"]
    change = result["vocabulary_change"]
    return "| " + " | ".join([
        run["finished_utc"],
        detector["name"],
        result["hardware"]["gpu_name"],
        detector["role"],
        "open" if detector["open_vocabulary"] else "80 COCO classes",
        f"{run['imgsz']}x{run['imgsz']}",
        _number(run["latency"]["mean_ms"], 2),
        _number(run["latency"]["p95_ms"], 2),
        f"1 per {budget['cadence']}",
        _number(budget["amortised_ms"], 2),
        "yes" if budget["fits"] else "no",
        _number(result["vram"]["torch_peak_mib"], 0),
        _number(result["vram"]["combined_used_mib"], 0),
        _number(change["median_change_ms"], 1) if change["supported"] else "n/a",
        _resolved_cell(result),
        result["cooldown"]["outcome"],
        regime_of(result),
        normalised_cell(result),
        f"[{filename}]({filename})",
    ]) + " |"


def append_detector_readme_row(result: ResultLike, readme_path: Path, filename: str) -> None:
    """Append one readable row, creating the table if this is the first detector run."""
    data = _as_dict(result)
    require_recordable(data)
    append_row(detector_readme_row(data, filename), readme_path, DETECTOR_README_PREAMBLE)


# --- the report spec 8.1 carries -------------------------------------------------

def load_detector_results(results_dir: Path) -> Dict[str, dict]:
    """Every detector result under `results_dir`, keyed by filename."""
    return {path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(Path(results_dir).glob("*.json"))}


def latest_per_detector(results: Mapping[str, dict]) -> Dict[str, dict]:
    """One result per detector: the most recently finished run of each.

    A detector measured twice is two honest records and both stay on disk; a table
    built from a mixture would compare a cold run against a hot one.
    """
    newest: Dict[str, Tuple[str, str]] = {}
    for filename, result in results.items():
        name = result["detector"]["name"]
        finished = str(result["run"]["finished_utc"])
        if name not in newest or finished > newest[name][1]:
            newest[name] = (filename, finished)
    return {filename: results[filename] for filename, _ in newest.values()}


REPORT_HEADER = ("| detector | role | vocabulary | ms/detect | p95 ms |"
                 " ms/detect at basis clock | amortised ms/frame | fits 4-8 ms |"
                 " torch peak (MiB) | with diffusion resident (MiB) |"
                 " vocabulary change (ms) |")
REPORT_SEPARATOR = "|" + "---|" * (REPORT_HEADER.count("|") - 1)

EVIDENCE_HEADER = ("| concept | kind | detector | asked for | resolved |"
                   " top confidence | strongest other label | frame |")
EVIDENCE_SEPARATOR = "|" + "---|" * (EVIDENCE_HEADER.count("|") - 1)


def _report_preamble(results: Sequence[dict]) -> str:
    """One line saying which machine and which input these numbers are for.

    spec 7.4's rule applied to spec 8.1: a detector table with no GPU attached invites the
    reader to treat a laptop measurement as a deploy-hardware one.
    """
    gpus = sorted({result["hardware"]["gpu_name"] for result in results})
    sizes = sorted({f"{result['run']['imgsz']}x{result['run']['imgsz']}"
                    for result in results})
    resident = sorted({result["vram"]["diffusion_scenario"] or "nothing else resident"
                       for result in results})
    return (f"Measured: {', '.join(gpus)}, {', '.join(sizes)} input, PyTorch. "
            f"Diffusion resident during the timing: {', '.join(resident)}.")


def _report_row(result: dict) -> str:
    detector, run, budget = result["detector"], result["run"], result["budget"]
    change = result["vocabulary_change"]
    return "| " + " | ".join([
        detector["name"],
        detector["role"],
        "open" if detector["open_vocabulary"] else "80 COCO classes",
        _number(run["latency"]["mean_ms"], 2),
        _number(run["latency"]["p95_ms"], 2),
        normalised_cell(result),
        _number(budget["amortised_ms"], 2),
        "yes" if budget["fits"] else "no",
        _number(result["vram"]["torch_peak_mib"], 0),
        _number(result["vram"]["combined_used_mib"], 0),
        _number(change["median_change_ms"], 1) if change["supported"] else "n/a",
    ]) + " |"


def _other_cell(item: dict) -> str:
    """The strongest box the detector returned under a different label, or `-`.

    In the table because "did not resolve it" and "resolved it as something else"
    are different failures and the second one is the more informative.
    """
    others = item.get("strongest_other") or []
    return "-" if not others else f"{others[0]['label']} {others[0]['confidence']:.2f}"


def _evidence_rows(results: Sequence[dict]) -> List[str]:
    rows = []
    for result in results:
        for item in result.get("evidence") or []:
            rows.append("| " + " | ".join([
                item["concept"],
                item["kind"],
                result["detector"]["name"],
                item["queried"] or "-",
                "yes" if item["resolved"] else "no",
                _number(item["top_confidence"], 3),
                _other_cell(item),
                item["frame"],
            ]) + " |")
    return rows


def ranking_ms(result: dict) -> float:
    """The figure two detectors are ranked by: clock-normalised when it exists.

    On an unlocked laptop clock two detectors measured minutes apart are not
    comparable raw - issue #13 - and ranking is precisely what issue #4 asks for. A
    locked run has no estimate and needs none; its raw figure is already comparable.
    """
    normalisation = result.get("clock_normalization") or {}
    normalised = normalisation.get("ms_per_frame")
    return float(normalised if normalised is not None
                 else result["run"]["latency"]["mean_ms"])


def candidates_from(results: Sequence[dict]) -> List[Candidate]:
    """The recommendation's inputs, read back out of the committed records."""
    return [
        Candidate(
            name=result["detector"]["name"],
            ms_per_detect=ranking_ms(result),
            open_vocabulary=bool(result["detector"]["open_vocabulary"]),
            concepts_resolved=sum(1 for item in result.get("evidence") or []
                                  if item["resolved"]),
            concepts_probed=len(result.get("evidence") or []),
        )
        for result in results
    ]


def _ranking_basis(results: Sequence[dict]) -> str:
    """Which column the recommendation ranked on, when that is not the obvious one.

    It is the clock-normalised estimate whenever one exists, and the milliseconds in
    the recommendation are therefore not the milliseconds in the `ms/detect` column.
    Saying so is cheaper than a reader noticing the discrepancy and distrusting both.
    """
    if not any(ranking_ms(result) != result["run"]["latency"]["mean_ms"]
               for result in results):
        return ""
    return (" Ranked on the clock-normalised estimate - the `ms/detect at basis clock` "
            "column, not the raw one - because the clocks were not locked and these "
            "rows were measured minutes apart, so their raw figures carry two "
            "different clocks (issue #13). The estimate is an estimate; the ranking "
            "is what it is used for.")


def format_detector_report(results: Mapping[str, dict],
                           cadence: int = DEFAULT_CADENCE) -> str:
    """The spec 8.1 measured block: the table, the evidence, and the recommendation.

    Generated from the committed JSON rather than transcribed, for the reason
    spec 7.2's table is: a table pasted into Markdown drifts the moment a cell is
    re-measured, and nothing notices.
    """
    ordered = [results[name] for name in sorted(latest_per_detector(results),
                                                key=lambda n: results[n]["detector"]["name"])]
    if not ordered:
        return "no detector result committed yet"

    recommendation = recommend(candidates_from(ordered), cadence=cadence)
    sections = [
        _report_preamble(ordered),
        "\n".join([REPORT_HEADER, REPORT_SEPARATOR] + [_report_row(r) for r in ordered]),
        "Vocabulary evidence - what each detector returned when asked for the "
        "concept, and what a closed vocabulary had to be asked for instead:",
        "\n".join([EVIDENCE_HEADER, EVIDENCE_SEPARATOR] + _evidence_rows(ordered)),
        f"**Recommendation: {recommendation.name}.** {recommendation.reason}"
        f"{_ranking_basis(ordered)}",
    ]
    return "\n\n".join(sections)
