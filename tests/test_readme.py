"""Tests for README.md (F5.S2.T2, Spec F5 acceptance criterion 2)."""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"
TEXT = README.read_text(encoding="utf-8")

ATTRIBUTION_LINE = (
    "Speedup comes from Arm's KleidiAI kernels; this tool detects the miss and measures the delta."
)

SECTION_HEADINGS = [
    "## 1. The Finding",
    "## 2. Why Nobody Notices",
    "## 3. What This Is / Is Not",
    "## 4. Quickstart",
    "## 5. Results",
    "## 6. How It Works",
    "## 7. Verify It Yourself",
    "## 8. What We Got Wrong",
    "## 9. Limitations",
    "## 10. Future Work",
    "## 11. License",
]


def test_ten_sections_present_in_order():
    positions = [TEXT.find(heading) for heading in SECTION_HEADINGS]
    for heading, pos in zip(SECTION_HEADINGS, positions):
        assert pos != -1, f"missing heading {heading!r}"
    assert positions == sorted(positions), "sections are not in the required order"


def test_no_placeholders_remain():
    # Every results surface is measured now, so the placeholder budget is zero.
    # Built by concatenation so this assertion never matches itself.
    token = "TODO" + "(box)"
    assert token not in TEXT, f"{TEXT.count(token)} unfilled placeholder(s) left in README.md"


def test_headline_pairs_throughput_with_its_quality_cost():
    # The no-bare-throughput rule, checked on the shipped prose rather than
    # only on `report`'s output.
    results = TEXT[TEXT.find("## 5. Results"):TEXT.find("## 6. How It Works")]
    assert "1.61×" in results
    assert "+0.049" in results
    assert "100 chunks" in results


def test_quality_cost_is_not_overclaimed_as_equivalence():
    # +0.049 is smaller than the ±0.14 error bars, which limits what this run
    # can resolve — it is not evidence that the two models are the same.
    for banned in ("identical quality", "no quality cost", "quality is unchanged", "lossless"):
        assert banned.lower() not in TEXT.lower(), f"README overclaims: {banned!r}"
    assert "inside the error bars" in TEXT


def test_limitations_names_every_axis_the_run_did_not_cover():
    limitations = TEXT[TEXT.find("## 9. Limitations"):TEXT.find("## 10. Future Work")]
    for required in ("8-vCPU", "Neoverse-N2", "Qwen2.5-7B-Instruct", "b10431",
                     "--chunks 100", "hand-assembled", "not a"):
        assert required in limitations, f"Limitations never states {required!r}"


def test_attribution_line_present_verbatim():
    assert ATTRIBUTION_LINE in TEXT


def test_quickstart_has_at_most_five_commands():
    start = TEXT.find("## 4. Quickstart")
    end = TEXT.find("## 5. Results")
    quickstart = TEXT[start:end]
    fenced = re.search(r"```bash\n(.*?)```", quickstart, re.DOTALL)
    assert fenced, "expected a fenced bash block in the Quickstart section"
    commands = [line for line in fenced.group(1).splitlines() if line.strip()]
    assert 1 <= len(commands) <= 5, f"expected 1-5 Quickstart commands, found {len(commands)}"
