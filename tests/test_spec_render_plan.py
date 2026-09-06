"""Spec section 6's block is the schema, not a description of it (issue #6).

The block is a plan carrying every default. Feeding it back through `validate_plan`
must return exactly itself, so a renamed field, a moved default or a region added to
the vocabulary fails the merge gate until section 6 is brought along with it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from render_plan import REGIONS, validate_plan

SPEC = Path(__file__).resolve().parent.parent / "docs" / "prompt-orchestrator-spec.md"
TEXT = SPEC.read_text(encoding="utf-8")


def section(number: str) -> str:
    """The body of a numbered top-level spec section, up to the next one."""
    start = TEXT.index(f"## {number}. ")
    rest = TEXT.index("\n## ", start + 1)
    return TEXT[start:rest]


SECTION_6 = section("6")


def plan_block() -> dict:
    """The jsonc block in section 6, with its trailing line comments stripped."""
    fenced = re.search(r"```jsonc\n(.*?)```", SECTION_6, re.DOTALL)
    assert fenced, "section 6 has no jsonc block"
    body = "\n".join(re.sub(r"\s*//.*$", "", line) for line in fenced.group(1).splitlines())
    return json.loads(body)


def test_the_documented_plan_is_a_fixed_point_of_the_validator():
    block = plan_block()
    result = validate_plan(block, previous_version=block["plan_version"] - 1)
    assert result.ok, result.reason
    assert result.plan.to_dict() == block


def test_the_documented_plan_needed_no_correcting():
    """A note means the validator changed something on the way through - which would
    mean the block documents a value the code does not accept as written."""
    block = plan_block()
    assert validate_plan(block, previous_version=block["plan_version"] - 1).notes == ()


@pytest.mark.parametrize("region", REGIONS)
def test_section_6_names_every_region_in_the_vocabulary(region):
    assert region in SECTION_6


def test_section_6_names_no_region_that_does_not_exist():
    documented = re.search(r'"region": "full_box",\s*//\s*(.*)', SECTION_6)
    assert documented, "the region line no longer carries its vocabulary"
    assert [name.strip() for name in documented.group(1).split("|")] == list(REGIONS)


def test_section_6_no_longer_says_the_llm_writes_the_plan():
    """The compiler is cut from v1 (see CLAUDE.md); section 6 is not historical, so
    it has to say who the producer actually is."""
    assert "plan_from_fields" in SECTION_6
    assert "Only C2 may write it" not in SECTION_6
