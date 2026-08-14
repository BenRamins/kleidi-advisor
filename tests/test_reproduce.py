"""Tests for REPRODUCE.md — the public reproduction procedure.

REPRODUCE.md is a cleaned derivative of a private working runbook, so the
tests that matter most here are the negative ones: nothing infrastructure-
specific, account-specific, or internal-to-the-build may leak into a file that
ships. A reader has this repository and nothing else.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPRODUCE = ROOT / "REPRODUCE.md"
REFERENCE = ROOT / "REFERENCE.md"
README = ROOT / "README.md"

TEXT = REPRODUCE.read_text(encoding="utf-8")

REAL_SUBCOMMANDS = {"scan", "fix", "bench", "report", "audit"}

STEP_HEADINGS = [f"## {n}." for n in range(1, 10)]

# Host and account identifiers from the machine the numbers were measured on,
# plus internal decision tags. Matched case-insensitively against the shipped docs.
LEAK_PATTERN = re.compile(
    r"kleidi-rg|kleidi-box|kleidi-user|D-[0-9]",
    re.IGNORECASE,
)


def _code_spans(text: str):
    fenced = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    inline = re.findall(r"(?<!`)`([^`\n]+)`(?!`)", text)
    return fenced + inline


def test_nine_steps_present_and_in_order():
    positions = [TEXT.find(h) for h in STEP_HEADINGS]
    for heading, pos in zip(STEP_HEADINGS, positions):
        assert pos != -1, f"missing heading {heading!r}"
    assert positions == sorted(positions), "steps are not in order"


def test_no_empty_code_blocks():
    for block in re.findall(r"```[^\n]*\n(.*?)```", TEXT, re.DOTALL):
        assert block.strip(), "found an empty fenced code block"


def test_every_kleidi_advisor_word_is_a_real_subcommand():
    words = []
    for span in _code_spans(TEXT):
        words.extend(re.findall(r"kleidi-advisor\s+([A-Za-z][\w-]*)", span))
    assert words, "expected at least one kleidi-advisor invocation"
    for word in words:
        assert word in REAL_SUBCOMMANDS, f"'kleidi-advisor {word}' is not a real subcommand"


def test_no_private_infrastructure_or_internal_tags_leak():
    for path in (REPRODUCE, REFERENCE):
        hits = LEAK_PATTERN.findall(path.read_text(encoding="utf-8"))
        assert not hits, f"{path.name} leaks {sorted(set(h.lower() for h in hits))}"


def test_hardware_requirement_is_generic_not_one_specific_vm():
    assert ">=8 cores" in TEXT
    assert ">=32 GB RAM" in TEXT
    assert ">=100 GB disk" in TEXT
    # The specific SKU may be named as provenance for the numbers, never as a
    # prerequisite for following the procedure.
    assert "any Arm64 Linux instance" in TEXT


def test_log_reading_step_precedes_and_justifies_verify():
    # The methodological core: read the raw -v logs before trusting --verify.
    raw_read = TEXT.find("Read the raw load logs before running `--verify` at all")
    verify_run = TEXT.find("Now run the on-device verify")
    assert raw_read != -1, "the raw-log-reading instruction is missing"
    assert verify_run != -1, "the --verify step is missing"
    assert raw_read < verify_run, "--verify must come after reading the raw logs"

    for trap in ("CPU_KLEIDIAI model buffer size", "CPU_REPACK model buffer size",
                 "cannot be used with preferred buffer type CPU_KLEIDIAI"):
        assert trap in TEXT, f"missing the {trap!r} evidence line"
    assert "-v" in TEXT and "llama-cli" in TEXT, "the -v / llama-cli caveats must survive"


def test_wall_clock_estimates_survive_on_every_step():
    # Useful to a reproducer; deliberately kept when deadlines were stripped.
    headings = re.findall(r"^## \d+\. .*$", TEXT, re.MULTILINE)
    assert len(headings) == 9
    for heading in headings:
        assert re.search(r"\([^)]*\bmin\b[^)]*\)", heading), f"no time estimate in {heading!r}"


def test_readme_points_at_reproduce_and_never_at_the_private_runbook():
    readme = README.read_text(encoding="utf-8")
    assert "[`REPRODUCE.md`](REPRODUCE.md)" in readme
    assert "RUNBOOK" not in readme
