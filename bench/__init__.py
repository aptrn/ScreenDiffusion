"""Benchmark harness (issue #2, spec 7.2 / 7.4).

`uv run python -m bench <scenario>` measures one configuration and writes a
machine-readable result under `bench/results/`, which is tracked - a committed
result is the deliverable, and a number that was never written down has to be
measured again.

Every result carries a hardware fingerprint. Development runs on an RTX 3080
laptop and deployment targets RTX 3090 Ti / 4090, so absolute timings do not
transfer; the fingerprint says which machine a number came from, and is the
evidence it was measured rather than invented.

Nothing here imports torch at module scope. `bench.runner` is the only module
that touches the GPU, and it imports torch inside its functions, so the CLI,
the cooldown gate, the fingerprint and the result records all stay testable in
the merge gate's GPU-free tier.
"""

RESULT_SCHEMA_VERSION = 1
