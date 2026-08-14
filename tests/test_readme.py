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
    "## 8. Limitations",
    "## 9. Future Work",
    "## 10. License",
]


def test_ten_sections_present_in_order():
    positions = [TEXT.find(heading) for heading in SECTION_HEADINGS]
    for heading, pos in zip(SECTION_HEADINGS, positions):
        assert pos != -1, f"missing heading {heading!r}"
    assert positions == sorted(positions), "sections are not in the required order"


def test_todo_box_count_between_one_and_four():
    count = TEXT.count("TODO(box)")
    assert 1 <= count <= 4, f"expected 1-4 TODO(box) placeholders, found {count}"


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
