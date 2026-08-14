"""Tests for report table building (F4.S1.T1, Spec F4 rule 1)."""

from __future__ import annotations

import shutil
from pathlib import Path

from kleidi_advisor.report import build_table, load_results

DATA_DIR = Path(__file__).resolve().parent / "data"


def test_both_committed_fixtures_render_with_exact_medians(tmp_path):
    shutil.copy(DATA_DIR / "results-baseline.json", tmp_path / "results-baseline.json")
    shutil.copy(DATA_DIR / "results-fixed.json", tmp_path / "results-fixed.json")

    entries = load_results(tmp_path)
    table = build_table(entries)

    assert len(entries) == 2
    assert "415.1" in table  # baseline pp512 median
    assert "1163.8" in table  # fixed pp512 median


def test_junk_json_in_results_dir_is_skipped_without_error(tmp_path):
    shutil.copy(DATA_DIR / "results-baseline.json", tmp_path / "results-baseline.json")
    (tmp_path / "junk.json").write_text("{not valid json")
    (tmp_path / "wrong-schema.json").write_text('{"schema": 2, "not": "ours"}')

    entries = load_results(tmp_path)  # must not raise

    assert len(entries) == 1
    assert entries[0].tag == "baseline"
