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
from typing import List, Optional, Sequence

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
    _validate(sources, outputs)
    if len(sources) < 2:
        return FlickerScore(
            mean_abs_diff=None, pairs=0, pairs_scored=0, static_pixels=0,
            frame_pixels=int(np.asarray(sources[0]).shape[0]
                             * np.asarray(sources[0]).shape[1]) if sources else 0,
            static_fraction=0.0, threshold=threshold, per_pair_abs_diff=[],
            note="a flicker metric needs at least two frames to compare",
        )

    height, width = np.asarray(sources[0]).shape[:2]
    frame_pixels = int(height * width)
    per_pair: List[float] = []
    static_total = 0
    for index in range(1, len(sources)):
        mask = static_pair_mask(sources[index - 1], sources[index], threshold)
        if painted is not None:
            mask = mask & np.asarray(painted[index - 1], dtype=bool) \
                        & np.asarray(painted[index], dtype=bool)
        static_total += int(mask.sum())
        if not mask.any():
            continue
        per_pair.append(_pair_abs_diff(outputs[index - 1], outputs[index], mask))

    pairs = len(sources) - 1
    static_mean = static_total / pairs
    if not per_pair:
        return FlickerScore(
            mean_abs_diff=None, pairs=pairs, pairs_scored=0, static_pixels=0,
            frame_pixels=frame_pixels, static_fraction=0.0, threshold=threshold,
            per_pair_abs_diff=[],
            note="no pixel was both static in the source and painted in both frames "
                 "of any pair, so there is nothing to score",
        )
    return FlickerScore(
        mean_abs_diff=round(float(np.mean(per_pair)), 6),
        pairs=pairs,
        pairs_scored=len(per_pair),
        static_pixels=int(round(static_mean)),
        frame_pixels=frame_pixels,
        static_fraction=round(static_mean / frame_pixels, 6) if frame_pixels else 0.0,
        threshold=threshold,
        per_pair_abs_diff=[round(float(value), 6) for value in per_pair],
        note=f"mean absolute difference between consecutive outputs over the "
             f"{static_mean:.0f} pixels per pair that were static in the source "
             f"(within {threshold:g}/255) and painted in both frames; 0-255 units, "
             f"lower is steadier",
    )
