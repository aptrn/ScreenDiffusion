"""The measured table in spec §7.2 is the committed results, not a transcription.

Issue #3's verification asks that "a reviewer can recompute marginal cost per batch
item directly from the JSON". That only holds if the table in the spec still *is*
what the JSON says - and a table pasted into Markdown drifts the moment a cell is
re-measured, silently, because nothing checks it.

So the spec carries the table between two anchors and this test regenerates it from
`bench/results/` and demands a byte match. Add a result, and the merge gate fails
until §7.2 is updated with `python -m bench --marginal`. That is the intended
workflow, not an obstacle to it.

GPU-free: `bench.marginal` reads JSON and divides. It never imports torch.
"""

from pathlib import Path

import pytest

from bench import marginal
from bench.cli import report_marginal
from sourceloader import ROOT

SPEC = Path(ROOT) / "docs/prompt-orchestrator-spec.md"
RESULTS = Path(ROOT) / "bench/results"
BEGIN = "<!-- BEGIN MEASURED TABLE -->"
END = "<!-- END MEASURED TABLE -->"


def spec_table() -> str:
    """Whatever currently sits between the anchors in §7.2."""
    text = SPEC.read_text(encoding="utf-8")
    assert BEGIN in text and END in text, "§7.2 lost its measured-table anchors"
    return text.split(BEGIN, 1)[1].split(END, 1)[0].strip()


def computed_table() -> str:
    """The same report `python -m bench --marginal` prints."""
    import io

    buffer = io.StringIO()
    report_marginal(RESULTS, out=buffer)
    return buffer.getvalue().strip()


def test_the_spec_table_is_what_the_committed_results_say():
    assert spec_table() == computed_table(), (
        "spec §7.2 no longer matches bench/results/. Regenerate it with\n"
        "    uv run python -m bench --marginal\n"
        "and paste the output between the MEASURED TABLE anchors."
    )


def test_every_none_cell_of_the_sweep_is_present():
    """Issue #3's gate: a result for every (resolution, batch) cell on `none`."""
    latest = marginal.latest_result_per_scenario(marginal.load_results(RESULTS))
    measured = {result["scenario"]["name"] for result in latest.values()}

    missing = {
        f"img2img-none-{size}x{size}-b{batch}"
        for size in (256, 384, 512)
        for batch in (1, 2, 4, 8)
    } - measured
    assert not missing, f"unmeasured `none` cells: {sorted(missing)}"


def test_every_tensorrt_result_records_the_free_disk_check():
    """Issue #3's gate: each TensorRT confirmation run carries its disk reading.

    The `none` sweep builds nothing, so it has no build to gate and no reading to
    show; the two pre-issue-#3 TensorRT runs predate the recording and are exempt
    by name rather than by silence.
    """
    grandfathered = {"img2img-tensorrt-512x512-b1-20260906-113144Z.json"}

    for filename, result in marginal.load_results(RESULTS).items():
        if result["scenario"]["acceleration"] != "tensorrt" or filename in grandfathered:
            continue
        disk = result.get("disk")
        assert disk is not None, f"{filename} has no free-disk reading"
        assert disk["free_bytes"] > 0 and "required_bytes" in disk, filename


def test_at_most_three_tensorrt_confirmation_runs():
    """Issue #3 caps the confirmation at three runs; each is ~5.1 GB and minutes."""
    confirmations = [
        filename for filename, result in marginal.load_results(RESULTS).items()
        if result["scenario"]["acceleration"] == "tensorrt" and "disk" in result
    ]
    assert len(confirmations) <= 3, sorted(confirmations)


@pytest.mark.parametrize("size", (256, 384, 512))
def test_the_normalised_none_curve_is_sublinear_at_every_resolution(size):
    """The go/no-go answer, asserted rather than only narrated.

    Normalised to one SM clock, because raw 512² reads "NOT sublinear" purely from
    the batch-8 cell running at 787 MHz under the 120 W limit (§7.4).
    """
    curves = marginal.curves_from_results(marginal.load_results(RESULTS))
    reference = marginal.reference_clock_mhz(curves)
    matching = [c.normalised_to(reference) for c in curves
                if c.acceleration == "none" and c.width == size]

    assert matching, f"no `none` curve at {size}x{size}"
    for curve in matching:
        assert curve.sublinear is True, (
            f"{size}x{size} marginal cost is no longer sublinear once normalised - "
            f"that is the go/no-go answer changing, so re-read §7.3"
        )


def test_the_tensorrt_confirmation_covers_three_batch_sizes_at_512():
    """Issue #3 step 3: confirm the curve on TensorRT, at most three configurations.

    Three cells at one resolution rather than three resolutions, because the
    resolution axis does not survive `wrapper.py` today - see
    `tests/test_trt_engine_resolution.py`. The batch axis does.
    """
    curves = marginal.curves_from_results(marginal.load_results(RESULTS))
    trt = [c for c in curves if c.acceleration == "tensorrt"]

    assert len(trt) == 1, f"the confirmation is one curve, got {[c.width for c in trt]}"
    assert (trt[0].width, trt[0].height) == (512, 512)
    assert [p.batch_size for p in trt[0].points] == [1, 2, 4]


def test_batching_512_crops_does_not_pay_on_tensorrt():
    """The confirmation's actual answer, pinned so it cannot soften into prose.

    The `none` sweep says the marginal item gets dearer as the crop grows; at 512²
    TensorRT it reaches parity with the first item and passes it. Raw, so it needs
    no clock model to be true - and it agrees with raw `none` at the same size.
    That is why §7.3 recommends packing *small* crops, not more of them.
    """
    curves = marginal.curves_from_results(marginal.load_results(RESULTS))
    trt = next(c for c in curves if c.acceleration == "tensorrt")

    assert trt.sublinear is False, (
        "512² TensorRT batching turned sublinear - that changes §7.3's reasoning "
        "about crop size, so re-read the recommendation rather than editing this test"
    )
    assert max(step.fraction_of_first_item for step in trt.steps) > 1.0, (
        "the largest marginal item should still cost more than the first"
    )


def test_a_cell_that_missed_the_cooldown_says_so():
    """Issue #3's trap: a capped cooldown is recorded, never silently reported as clean.

    The batch-4 TensorRT cell started at 63 °C after the previous run's build, and
    the table has to carry that rather than imply every cell was equally cold.
    """
    outcomes = {result["scenario"]["name"]: result["cooldown"]["outcome"]
                for result in marginal.latest_result_per_scenario(
                    marginal.load_results(RESULTS)).values()}

    assert outcomes["img2img-tensorrt-512x512-b4"] == "capped"
    assert set(outcomes.values()) <= {"reached", "capped", "skipped"}
