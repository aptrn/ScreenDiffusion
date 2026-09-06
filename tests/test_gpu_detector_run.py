"""One real detector run, end to end (issue #4). GPU tier.

Short - a handful of reps, and with `--no-diffusion`, so it needs the cached
weights but never loads a TensorRT engine. It asserts what the committed detector
results must satisfy: a fingerprint, a clock regime, a budget verdict with its
cadence named, and per-concept evidence with boxes to show for it.

Skips rather than downloads. Fetching ~370 MB of weights is a decision someone
makes with `--allow-download`, not something a test does behind their back.
"""

import pytest

from bench.detectors import DETECTORS, PRIMARY_DETECTOR, SPEED_FLOOR_DETECTOR, weights_path
from bench.paths import resolve_models_dir
from bench.results import require_recordable

pytestmark = pytest.mark.gpu


@pytest.fixture
def cached_weights():
    """The candidate's weights and the three evidence photographs, or a skip."""
    from bench.detector_runner import IMAGE_CACHE_SUBDIR
    from bench.detectors import CONCEPT_PROBES

    models_root = resolve_models_dir()
    needed = [weights_path(DETECTORS[PRIMARY_DETECTOR], models_root)]
    needed += [models_root / IMAGE_CACHE_SUBDIR / probe.image_name
               for probe in CONCEPT_PROBES]
    missing = [path for path in needed if not path.is_file()]
    if missing:
        pytest.skip(f"not cached under {models_root}: {[p.name for p in missing]}. "
                    f"Run `python -m bench {PRIMARY_DETECTOR} --allow-download` once.")
    return models_root


def test_a_short_detector_run_emits_a_complete_result(cached_weights, tmp_path):
    from bench.detector_runner import run_detector

    config = DETECTORS[PRIMARY_DETECTOR].replace(reps=6, warmup_reps=2)
    result = run_detector(config, diffusion_scenario=None, cooldown=False,
                          results_dir=tmp_path, models_dir=cached_weights)
    data = result.to_dict()

    require_recordable(data)
    assert data["run"]["latency"]["mean_ms"] > 0
    assert len(data["run"]["per_rep_ms"]) == 6
    assert data["run"]["imgsz"] == 640
    assert set(data["run"]["ultralytics_speed_ms"]) >= {"preprocess", "inference",
                                                       "postprocess"}
    assert data["budget"]["cadence"] == 3, "a verdict without its cadence is not one"
    assert data["budget"]["statement"]
    assert data["vram"]["torch_peak_bytes"] > 0

    assert len(written := list(tmp_path.glob("*.json"))) == 1
    assert (tmp_path / "README.md").exists()
    assert written[0].name.startswith(PRIMARY_DETECTOR)


def test_a_vocabulary_change_is_measured_rather_than_assumed(cached_weights, tmp_path):
    """The gate: the cold-path cost, and what the frame path did or did not pay."""
    from bench.detector_runner import run_detector

    config = DETECTORS[PRIMARY_DETECTOR].replace(reps=6, warmup_reps=2)
    data = run_detector(config, diffusion_scenario=None, cooldown=False,
                        results_dir=tmp_path, models_dir=cached_weights).to_dict()

    change = data["vocabulary_change"]
    assert change["supported"] is True
    assert change["median_change_ms"] > 0
    assert change["first_change_ms"] >= change["median_change_ms"], (
        "the first change loads the text encoder and is kept apart from the rest"
    )
    assert change["text_encoder"] == "clip:ViT-B/32"

    frame_path = data["frame_path"]
    assert frame_path["before_ms"] > 0 and frame_path["after_ms"] > 0
    assert frame_path["first_detect_ms"] is not None
    assert len(frame_path["passes"]) == 6, "alternating blocks, so drift hits both arms"


def test_the_detector_finds_the_concepts_on_a_real_screen(cached_weights, tmp_path):
    """Step 4, against this machine's actual desktop rather than a fixture."""
    from bench.detector_runner import run_detector

    config = DETECTORS[PRIMARY_DETECTOR].replace(reps=4, warmup_reps=1)
    data = run_detector(config, diffusion_scenario=None, cooldown=False,
                        results_dir=tmp_path, models_dir=cached_weights).to_dict()

    evidence = {item["concept"]: item for item in data["evidence"]}
    assert set(evidence) == {"person", "red mug", "dog"}
    for concept, item in evidence.items():
        assert item["resolved"] is True, f"{concept} went unresolved: {item['note']}"
        assert item["top_confidence"] > 0.3, concept
        assert "desktop capture" in item["frame"]

    control = data["desktop_control"]
    assert control is not None and "nothing composited" in control["frame"], (
        "the control is what separates finding the photo from returning boxes anyway"
    )


def test_a_closed_vocabulary_detector_says_so_rather_than_reporting_a_zero(tmp_path):
    """YOLOv8n has no text encoder: its vocabulary cannot change, which is not the
    same as changing for free."""
    from bench.detector_runner import run_detector

    models_root = resolve_models_dir()
    if not weights_path(DETECTORS[SPEED_FLOOR_DETECTOR], models_root).is_file():
        pytest.skip(f"no {SPEED_FLOOR_DETECTOR} weights under {models_root}")

    config = DETECTORS[SPEED_FLOOR_DETECTOR].replace(reps=4, warmup_reps=1)
    data = run_detector(config, diffusion_scenario=None, cooldown=False,
                        results_dir=tmp_path, models_dir=models_root).to_dict()

    assert data["vocabulary_change"]["supported"] is False
    assert data["vocabulary_change"]["median_change_ms"] is None
    assert data["frame_path"] is None, "there is no before and after to compare"

    mug = next(item for item in data["evidence"] if item["concept"] == "red mug")
    assert mug["queried"] is None and mug["resolved"] is False
    assert "Not expressible" in mug["note"]


def test_the_weights_are_refused_rather_than_fetched_behind_your_back(tmp_path):
    """Hundreds of MB is a decision, not a side effect - the issue's second trap."""
    from bench.detector_runner import DownloadRefused, run_detector

    with pytest.raises(DownloadRefused) as refusal:
        run_detector(DETECTORS[PRIMARY_DETECTOR], diffusion_scenario=None,
                     cooldown=False, results_dir=tmp_path, models_dir=tmp_path,
                     allow_download=False)
    assert "--allow-download" in str(refusal.value)
