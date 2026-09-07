"""How much a rendered sequence boils where the source stood still.

Issue #5's Gate, and the quantitative metric spec 8.5 asks for. The known failure
mode of per-frame diffusion is flicker: a wall that does not move in the capture
comes back a slightly different colour in every output frame. Ranking two rendering
primitives on ms/frame alone would be blind to it, and "it looks steadier to me" is
not a number anyone can re-check.

The definition, stated once:

    mean over consecutive output pairs of the mean absolute difference between
    them, taken only over pixels that were static in the *source* across the same
    pair - and, when a painted mask is supplied, only over pixels the primitive
    actually rendered in both frames of the pair.

Both restrictions are load-bearing.

- **Static in the source.** A subject walking across the frame changes every output
  pixel it crosses, and that is the render working, not flickering. The source is
  the only place to ask what moved, because it is the one thing the two primitives
  share.
- **Painted in both frames.** Outside its regions a primitive passes the source
  through unchanged, so those pixels are identical between consecutive outputs by
  construction. Left in, they drag every score towards zero in proportion to how
  little of the frame was restyled - which would rank a primitive that restyles
  nothing as the steadiest of all. Requiring *both* frames of a pair excludes the
  compositing edge a moving box leaves behind, which is also not flicker.

Pure: numpy in, a dataclass out. No torch, no cv2, no filesystem. Values are
expected in 0-255 image units, and the score is in those units too - 0 is a
sequence that never moves where the source did not, and the metric reads low-is
-steadier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

# A pixel that moved by no more than this in every channel counts as static.
# Capture and video compression put a couple of units of noise on a genuinely
# static wall; a threshold of zero would find almost no static pixels on a real
# clip and score nothing. Inclusive, so the threshold names the largest movement
# still called still.
STATIC_THRESHOLD = 2.0


@dataclass(frozen=True)
class FlickerScore:
    """One sequence's flicker, and what the figure was computed over.

    `mean_abs_diff` is None whenever there is no honest number - fewer than two
    frames, or no pixel that was both static and painted in any pair. `note` always
    says which, because a metric that returned 0.0 for "nothing to measure" would
    report the least measurable primitive as the steadiest.
    """

    mean_abs_diff: Optional[float]
    pairs: int
    pairs_scored: int
    static_pixels: int
    frame_pixels: int
    static_fraction: float
    threshold: float
    per_pair_abs_diff: List[float]
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


def _as_float_array(frame) -> np.ndarray:
    return np.asarray(frame, dtype=np.float64)


def static_pair_mask(before, after, threshold: float = STATIC_THRESHOLD) -> np.ndarray:
    """The HxW pixels that moved by no more than `threshold` in *every* channel.

    Max over the channels rather than mean: a pixel whose red channel jumped 255
    while green and blue held still moved, and averaging the three would hide it
    behind a comfortable 85.
    """
    difference = np.abs(_as_float_array(after) - _as_float_array(before))
    return difference.max(axis=-1) <= threshold


def _pair_abs_diff(before, after, mask: np.ndarray) -> float:
    """Mean absolute difference between two frames over `mask`, averaged over channels."""
    difference = np.abs(_as_float_array(after) - _as_float_array(before)).mean(axis=-1)
    return float(difference[mask].mean())


def _validate(sources: Sequence, outputs: Sequence) -> None:
    if len(sources) != len(outputs):
        raise ValueError(
            f"the source and the output must have the same number of frames: "
            f"{len(sources)} against {len(outputs)}"
        )
    shapes = {np.asarray(frame).shape for frame in list(sources) + list(outputs)}
    if len(shapes) > 1:
        raise ValueError(f"every frame must have the same shape; got {sorted(shapes)}")


@dataclass(frozen=True)
class _Scored:
    """What both metrics compute, before either says what it means.

    Flicker and responsiveness differ in exactly one place - which side of the
    static threshold a pixel has to fall on - so the walk over pairs, the painted
    intersection, the skipped-pair rule and the rounding are shared, and the two
    public functions are the two sentences said about the result.
    """

    mean_abs_diff: Optional[float]
    pairs: int
    pairs_scored: int
    selected_pixels: int
    frame_pixels: int
    selected_fraction: float
    per_pair_abs_diff: List[float]


PairMask = Callable[[object, object], np.ndarray]


def _score(sources: Sequence, outputs: Sequence, painted: Optional[Sequence],
           select: PairMask) -> _Scored:
    """`select(before, after)` picks the source pixels this metric is about."""
    _validate(sources, outputs)
    frame_pixels = 0
    if sources:
        height, width = np.asarray(sources[0]).shape[:2]
        frame_pixels = int(height * width)
    if len(sources) < 2:
        return _Scored(mean_abs_diff=None, pairs=0, pairs_scored=0,
                       selected_pixels=0, frame_pixels=frame_pixels,
                       selected_fraction=0.0, per_pair_abs_diff=[])

    per_pair: List[float] = []
    selected_total = 0
    for index in range(1, len(sources)):
        mask = select(sources[index - 1], sources[index])
        if painted is not None:
            mask = mask & np.asarray(painted[index - 1], dtype=bool) \
                        & np.asarray(painted[index], dtype=bool)
        selected_total += int(mask.sum())
        if not mask.any():
            continue
        per_pair.append(_pair_abs_diff(outputs[index - 1], outputs[index], mask))

    pairs = len(sources) - 1
    selected_mean = selected_total / pairs
    if not per_pair:
        return _Scored(mean_abs_diff=None, pairs=pairs, pairs_scored=0,
                       selected_pixels=0, frame_pixels=frame_pixels,
                       selected_fraction=0.0, per_pair_abs_diff=[])
    return _Scored(
        mean_abs_diff=round(float(np.mean(per_pair)), 6),
        pairs=pairs,
        pairs_scored=len(per_pair),
        selected_pixels=int(round(selected_mean)),
        frame_pixels=frame_pixels,
        selected_fraction=(round(selected_mean / frame_pixels, 6)
                           if frame_pixels else 0.0),
        per_pair_abs_diff=[round(float(value), 6) for value in per_pair],
    )


def flicker_score(
    sources: Sequence,
    outputs: Sequence,
    painted: Optional[Sequence] = None,
    threshold: float = STATIC_THRESHOLD,
) -> FlickerScore:
    """Score `outputs` for flicker against the `sources` they were rendered from.

    `painted[i]` is an HxW boolean mask of what the primitive rendered into frame
    `i`; None means the whole frame, which is what a full-frame primitive with no
    region restriction paints. A pair contributes only where the pixel was static
    in the source *and* painted in both of its frames.

    Pairs with no such pixel are counted and skipped rather than scored zero.
    """
    scored = _score(sources, outputs, painted,
                    lambda before, after: static_pair_mask(before, after, threshold))
    if scored.mean_abs_diff is None:
        note = ("a flicker metric needs at least two frames to compare"
                if scored.pairs == 0 else
                "no pixel was both static in the source and painted in both frames "
                "of any pair, so there is nothing to score")
    else:
        note = (f"mean absolute difference between consecutive outputs over the "
                f"{scored.selected_pixels} pixels per pair that were static in the "
                f"source (within {threshold:g}/255) and painted in both frames; "
                f"0-255 units, lower is steadier")
    return FlickerScore(
        mean_abs_diff=scored.mean_abs_diff, pairs=scored.pairs,
        pairs_scored=scored.pairs_scored, static_pixels=scored.selected_pixels,
        frame_pixels=scored.frame_pixels, static_fraction=scored.selected_fraction,
        threshold=threshold, per_pair_abs_diff=scored.per_pair_abs_diff, note=note,
    )


# --- the other half of the trade (issue #32, spec 8.5) -----------------------


@dataclass(frozen=True)
class ResponseScore:
    """How much the output moves where the *source* moved. Flicker's mirror image.

    Issue #32's second trap: a lower flicker number is not automatically better. An
    output EMA suppresses boiling by averaging consecutive renders, and an EMA
    strong enough to kill boiling also averages away the restyle's response to
    motion - a subject walking across the frame drags a smear behind them. Both
    numbers come from the same pairs and the same painted mask; the only difference
    is which side of the static threshold the pixel fell on.

    So flicker reads low-is-steadier and this reads **high-is-more-responsive**, and
    a recommendation that moves one has to say what it did to the other.
    """

    mean_abs_diff: Optional[float]
    pairs: int
    pairs_scored: int
    moving_pixels: int
    frame_pixels: int
    moving_fraction: float
    threshold: float
    per_pair_abs_diff: List[float]
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


def response_score(
    sources: Sequence,
    outputs: Sequence,
    painted: Optional[Sequence] = None,
    threshold: float = STATIC_THRESHOLD,
) -> ResponseScore:
    """Score `outputs` for responsiveness against the `sources` they came from.

    The same definition as `flicker_score` with the static test inverted: the mean
    absolute difference between consecutive outputs over the pixels that *moved* in
    the source and were painted in both frames of the pair. Same threshold, so the
    two partition the painted pixels of a pair between them and neither can be
    improved by moving the line.
    """
    scored = _score(sources, outputs, painted,
                    lambda before, after: ~static_pair_mask(before, after, threshold))
    return ResponseScore(
        mean_abs_diff=scored.mean_abs_diff, pairs=scored.pairs,
        pairs_scored=scored.pairs_scored, moving_pixels=scored.selected_pixels,
        frame_pixels=scored.frame_pixels, moving_fraction=scored.selected_fraction,
        threshold=threshold, per_pair_abs_diff=scored.per_pair_abs_diff,
        note=(f"mean absolute difference between consecutive outputs over the "
              f"{scored.selected_pixels} pixels per pair that moved in the source "
              f"(by more than {threshold:g}/255) and were painted in both frames; "
              f"0-255 units, higher is more responsive"
              if scored.mean_abs_diff is not None else
              "no pixel both moved in the source and was painted in both frames of "
              "any pair, so there is nothing to score"),
    )

