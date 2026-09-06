"""Spec 7.3 has to end in a decision, not a menu (issue #3 step 6).

The issue's gate asks for "an explicit recommendation among 7.3 (a)-(d), with
portable and non-portable conclusions labelled separately". Prose satisfies that
once and then rots: an option gets ruled out in a later commit, the recommendation
line stays, and nothing notices. So the shape of the answer is pinned here - one
named winner, a verdict on every option, and the two conclusion lists kept apart -
while what it *says* stays the spec's business.

GPU-free: this reads a Markdown file.
"""

import re
from pathlib import Path

from sourceloader import ROOT

SPEC = Path(ROOT) / "docs/prompt-orchestrator-spec.md"
OPTIONS = ("a", "b", "c", "d")
RECOMMENDATION = re.compile(r"^\*\*Recommendation: \(([a-d])\)", re.MULTILINE)


def section(heading: str) -> str:
    """The text under `heading`, up to the next heading of the same or higher level.

    Same *or higher*: "#### Recommendation" is followed by "### 7.4", and a helper
    that stopped only at a sibling heading would swallow the rest of the document -
    which is how a test like this quietly stops testing anything.
    """
    text = SPEC.read_text(encoding="utf-8")
    assert heading in text, f"the spec lost its {heading!r} heading"
    depth = len(heading.split(" ", 1)[0])

    lines = []
    for line in text.split(heading, 1)[1].splitlines():
        hashes = len(line) - len(line.lstrip("#"))
        if 0 < hashes <= depth:
            break
        lines.append(line)
    return "\n".join(lines)


def recommendation() -> str:
    return section("#### Recommendation")


def test_exactly_one_option_is_recommended():
    picked = RECOMMENDATION.findall(section("### 7.3"))
    assert len(picked) == 1, f"7.3 names {len(picked)} recommendations, want exactly 1"
    assert picked[0] in OPTIONS


def test_every_option_gets_a_verdict():
    """A recommendation that ignores three of the four options has not chosen."""
    body = recommendation()
    for option in OPTIONS:
        assert f"**({option})" in body, f"option ({option}) has no verdict in 7.3"


def test_portable_and_non_portable_conclusions_are_kept_apart():
    """Section 7.4's rule, applied to this recommendation rather than restated."""
    body = recommendation()
    assert "Portable:" in body and "Not portable:" in body, (
        "7.3's recommendation must label which conclusions survive the move to the "
        "deploy GPU and which have to be re-measured"
    )
    portable, non_portable = body.split("Portable:", 1)[1].split("Not portable:", 1)
    assert portable.strip() and non_portable.strip(), "both lists must say something"


def test_the_recommendation_cites_the_measured_curve():
    """It has to rest on the sweep, not on taste."""
    body = recommendation()
    assert "7.2" in body, "the recommendation should point at the measured table"
