"""Tests for report headline + attribution lines (F4.S1.T2, Spec F4 rules 3-4, D-14)."""

from __future__ import annotations

import json
from pathlib import Path

from kleidi_advisor.report import ATTRIBUTION_LINE, ResultEntry, render_markdown

DATA_DIR = Path(__file__).resolve().parent / "data"


def _entry(name: str) -> ResultEntry:
    return ResultEntry(json.loads((DATA_DIR / name).read_text()))


def test_paired_fixtures_render_speedup_and_ppl_delta_in_one_line():
    rendered = render_markdown(
        [_entry("results-baseline.json"), _entry("results-fixed.json")], instance="c8g.8xlarge"
    )
    assert "× pp512 at +" in rendered


def test_ppl_less_fixture_renders_not_measured_with_no_other_times_token():
    baseline_data = json.loads((DATA_DIR / "results-baseline.json").read_text())
    fixed_data = json.loads((DATA_DIR / "results-fixed.json").read_text())
    fixed_data["ppl"] = None  # speedup is still computable; ppl is not

    rendered = render_markdown(
        [ResultEntry(baseline_data), ResultEntry(fixed_data)], instance="c8g.8xlarge"
    )

    assert "ppl: not measured" in rendered
    assert "×" not in rendered


def test_attribution_line_present_in_every_rendered_report():
    rendered = render_markdown(
        [_entry("results-baseline.json"), _entry("results-fixed.json")], instance="c8g.8xlarge"
    )
    assert ATTRIBUTION_LINE in rendered
    assert rendered.count(ATTRIBUTION_LINE) == 1
