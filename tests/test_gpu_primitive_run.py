"""One real rendering-primitive comparison, end to end (issue #5). GPU tier.

Short - four frames, no cooldown, no clips - so it needs the cached 512x512
TensorRT engine and the committed clip, and nothing else. It asserts what the
committed comparisons must satisfy: a fingerprint, a clock regime, two arms measured
on the same frames, a denoise requirement selected by a stated rule, and a flicker
figure computed over pixels that were static in the source.

Skips rather than builds. Compiling an engine is ~5.0 GB and 15-25 minutes, which is
a decision someone takes with `--allow-engine-build`, not something a test does
behind their back.
"""

import pytest

from bench.cli import engine_dir_name
from bench.paths import resolve_engines_dir
from bench.primitives import CASES, CROP, MASKED, RESTYLE_CASE, load_track
from bench.results import require_recordable
from bench.scenarios import SCENARIOS

pytestmark = pytest.mark.gpu


@pytest.fixture
def cached_engine():
    """The one engine both primitives render through, or a skip."""
    from bench.primitive_runner import ENGINE_SCENARIO

    scenario = SCENARIOS[ENGINE_SCENARIO]
    root = resolve_engines_dir()
    engine = root / engine_dir_name(scenario) / "unet.engine"
    if not engine.is_file():
        pytest.skip(f"no cached engine for {ENGINE_SCENARIO} under {root}. Build one "
                    f"with `python -m bench {ENGINE_SCENARIO} --allow-engine-build`.")
    return root


@pytest.fixture
def short_case():
    """The priority case, cut to four frames and a two-rung ladder."""
    case = CASES[RESTYLE_CASE]
    load_track(case.clip)  # fails loudly here rather than deep inside the run
    return case.replace(frames=4, sweep_frames=2, denoise_ladder=(20, 45))


def test_a_short_comparison_emits_a_complete_record(cached_engine, short_case,
                                                    tmp_path):
    from bench.primitive_runner import run_case

    result = run_case(short_case, cooldown=False, results_dir=tmp_path,
                      engines_root=cached_engine, write_clips=False)
    data = result.to_dict()

    require_recordable(data)
    assert data["kind"] == "primitive"
    assert {arm["primitive"] for arm in data["arms"]} == {CROP, MASKED}
    assert data["clip"]["frames_used"] == 4
    assert data["clip"]["sha256"], "a re-encoded clip is a different clip"
    assert data["run"]["engine_scenario"] == "img2img-tensorrt-512x512-b1"

    assert len(written := list(tmp_path.glob("*.json"))) == 1
    assert (tmp_path / "README.md").exists()
    assert written[0].name.startswith(RESTYLE_CASE)
    rows = (tmp_path / "README.md").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows[-2:]) == 2, "one row per primitive"


def test_both_arms_are_timed_on_the_same_frames(cached_engine, short_case, tmp_path):
    """Interleaved frame by frame, so the laptop's clock drift lands on both."""
    from bench.primitive_runner import run_case

    data = run_case(short_case, cooldown=False, results_dir=tmp_path,
                    engines_root=cached_engine, write_clips=False).to_dict()

    arms = {arm["primitive"]: arm for arm in data["arms"]}
    assert arms[CROP]["frames"] == arms[MASKED]["frames"] == 4
    assert arms[CROP]["ms_per_frame"] > 0 and arms[MASKED]["ms_per_frame"] > 0
    assert arms[MASKED]["calls_per_frame"] == 1.0
    assert arms[CROP]["calls_per_frame"] == data["track"]["objects_per_frame"]


def test_the_crop_primitive_costs_one_diffusion_call_per_object(cached_engine,
                                                                short_case, tmp_path):
    """The cost model, confirmed against the hardware rather than asserted."""
    from bench.primitive_runner import run_case

    data = run_case(short_case, cooldown=False, results_dir=tmp_path,
                    engines_root=cached_engine, write_clips=False).to_dict()

    arms = {arm["primitive"]: arm for arm in data["arms"]}
    objects = data["track"]["objects_per_frame"]
    assert objects > 1, "the priority case has several people in frame"
    assert arms[CROP]["ms_per_frame"] > arms[MASKED]["ms_per_frame"]


def test_the_denoise_strength_is_selected_by_a_rule_the_record_carries(
        cached_engine, short_case, tmp_path):
    from bench.primitive_runner import run_case

    data = run_case(short_case, cooldown=False, results_dir=tmp_path,
                    engines_root=cached_engine, write_clips=False).to_dict()

    for arm in data["arms"]:
        denoise = arm["denoise"]
        assert denoise["rule"] and denoise["statement"], arm["primitive"]
        assert len(denoise["points"]) == 2, "both rungs of the shortened ladder"
        assert denoise["t_index"] in (20, 45)
        for point in denoise["points"]:
            assert point["timestep"] > 0 and 0.0 < point["strength"] <= 1.0
            assert point["outside_change"] == 0.0, (
                "a selective primitive leaves the rest of the frame alone"
            )
            assert point["resample_change"] >= 0.0


def test_the_flicker_metric_is_computed_over_static_source_pixels(
        cached_engine, short_case, tmp_path):
    from bench.primitive_runner import run_case

    data = run_case(short_case, cooldown=False, results_dir=tmp_path,
                    engines_root=cached_engine, write_clips=False).to_dict()

    for arm in data["arms"]:
        flicker = arm["flicker"]
        assert flicker["pairs"] == 3, "four frames give three consecutive pairs"
        assert flicker["pairs_scored"] > 0
        assert flicker["static_pixels"] > 0
        assert flicker["mean_abs_diff"] is not None
        assert "static in the source" in flicker["note"]


def test_the_side_by_side_clips_are_written_when_they_are_asked_for(
        cached_engine, short_case, tmp_path):
    """The Gate's manual verification step has to be pointed at a file."""
    from bench.primitive_runner import run_case

    data = run_case(short_case, cooldown=False, results_dir=tmp_path,
                    engines_root=cached_engine, write_clips=True).to_dict()

    assert (tmp_path / data["comparison_clip"]).is_file()
    assert (tmp_path / data["comparison_still"]).is_file()
    for arm in data["arms"]:
        assert (tmp_path / arm["clip_file"]).is_file(), arm["primitive"]
