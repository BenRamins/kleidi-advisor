"""Tests for .claude/skills/kleidi-advisor/SKILL.md (F6.S1.T1, Spec F6).

pyyaml is banned (D-01/guardrails), so frontmatter is parsed by hand here —
just enough to confirm it's well-formed, not a general YAML parser.
"""

from __future__ import annotations

import re
from pathlib import Path

SKILL_PATH = (
    Path(__file__).resolve().parent.parent / ".claude" / "skills" / "kleidi-advisor" / "SKILL.md"
)
TEXT = SKILL_PATH.read_text(encoding="utf-8")

REAL_SUBCOMMANDS = {"scan", "fix", "bench", "report", "audit"}


def _split_frontmatter(text: str):
    assert text.startswith("---\n"), "SKILL.md must start with a '---' frontmatter delimiter"
    _, _, rest = text.partition("---\n")
    frontmatter, sep, body = rest.partition("\n---\n")
    assert sep, "SKILL.md frontmatter must be closed by a second '---' line"
    return frontmatter, body


def _parse_simple_keys(frontmatter: str) -> dict:
    """'key: value' top-level lines only — everything this frontmatter uses."""
    keys = {}
    for line in frontmatter.splitlines():
        if not line.strip() or line.startswith((" ", "\t")):
            continue
        key, sep, value = line.partition(":")
        if sep:
            keys[key.strip()] = value.strip()
    return keys


def test_frontmatter_delimited_and_has_name_and_description():
    frontmatter, _body = _split_frontmatter(TEXT)
    keys = _parse_simple_keys(frontmatter)
    assert keys.get("name"), "frontmatter 'name' must be present and non-empty"
    assert keys.get("description"), "frontmatter 'description' must be present and non-empty"


def test_trigger_phrases_present_in_description():
    frontmatter, _body = _split_frontmatter(TEXT)
    description = _parse_simple_keys(frontmatter)["description"]
    assert "GGUF" in description
    assert "KleidiAI" in description
    assert "why is my model slow on Arm/Graviton" in description


def test_every_kleidi_advisor_word_in_body_is_a_real_subcommand():
    _frontmatter, body = _split_frontmatter(TEXT)
    words = re.findall(r"kleidi-advisor\s+([A-Za-z][\w-]*)", body)
    assert words, "expected at least one kleidi-advisor invocation in the skill body"
    for word in words:
        assert word in REAL_SUBCOMMANDS, f"'kleidi-advisor {word}' is not a real subcommand"


def test_reveal_keeps_attribution_and_not_our_measurement_clause():
    assert "not our measurement" in TEXT
    assert "llama.cpp PR #9921" in TEXT


def test_pr_anchor_is_demoted_not_quoted_as_the_expected_uplift():
    # D-03 as amended 2026-08-14: the 2.5-2.9x figure predates the K-quant
    # CPU_REPACK path, so the skill must not hand it to a user as what they
    # should expect from `fix`.
    assert "superseded" in TEXT
    assert "Do **not** quote the older ~2.5–2.9× figure" in TEXT


def test_both_miss_verdicts_have_a_branch_in_the_skill():
    for verdict in ("NOT_KLEIDIAI_PATH", "FALLBACK_GENERIC", "OK_KLEIDIAI", "NOT_APPLICABLE"):
        assert verdict in TEXT, f"skill body never mentions {verdict}"


def test_measured_speedup_is_never_quoted_without_its_ppl_caveat():
    # The no-bare-throughput rule holds in the skill too: the 1.61x figure and
    # its quality cost live in one sentence, so the reveal can't be quoted
    # half-way, and the caveat travels with the number.
    assert "1.61×" in TEXT
    assert "ppl cost of the switch: +0.049" in TEXT
    assert "inside the error bars" in TEXT


def test_no_banned_d11_phrasing():
    banned = re.compile(r"our (optimization|speedup)|we made it faster|we optimized", re.IGNORECASE)
    assert not banned.search(TEXT)
