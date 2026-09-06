# Reference clips and their box tracks

Committed fixtures for the rendering-primitive comparison (issue #5, spec 8.2).
Comparing on live capture makes runs non-reproducible, so the clip *and* the boxes
are both in the repo.

| file | resolution | length | intended case |
|---|---|---|---|
| `people.mp4` | 1280x720, 25 fps | 377 frames | the priority sub-region restyle (`lower_half` on people) |
| `dog.mp4` | 960x540, 24 fps | 228 frames | the identity change (dog → cat) |

Both unwatermarked and royalty-free CC. Added by #17.

## The tracks

`people.track.json` and `dog.track.json` hold the boxes each case renders, one entry
per frame, strongest detection first. They are **generated, then committed**:

```
uv run python -m bench restyle-people --write-track
uv run python -m bench identity-dog   --write-track
```

That runs YOLO-World (`yolo-world-s-640`, the detector issue #4 settled on) over the
frames the case uses, at confidence 0.25, and matches the detections across frames by
overlap with light smoothing. Reading the boxes back instead of detecting them is
what makes two comparisons comparable: the same regions are rendered every time, and
neither primitive is charged for detector jitter that a real tracker would absorb —
box smoothing is one of spec 8.5's own levers against flicker.

A track only covers the frames its case renders (`start_frame` to `start_frame +
frames`). Change either and the track has to be regenerated, which is why the case
config and the track are committed together.

Regenerating a track changes the measured numbers, so it also means re-running both
comparisons and regenerating the spec 8.2 block.
