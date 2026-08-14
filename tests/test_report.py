"""Tests for report table building (F4.S1.T1, Spec F4 rule 1)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from kleidi_advisor.report import build_table, find_baseline_and_candidate, load_results

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


# --- Headline selection with more than one candidate -------------------------
#
# "First non-baseline" is unambiguous with two result files and arbitrary with
# three: it silently picked our own imatrix-fix row over the published-build
# comparison the README documents as the headline.


def _three_results(tmp_path):
    shutil.copy(DATA_DIR / "results-baseline.json", tmp_path / "a-baseline.json")
    for tag in ("imatrix-fix", "published-q4_0"):
        data = json.loads((DATA_DIR / "results-fixed.json").read_text(encoding="utf-8"))
        data["tag"] = tag
        data["metrics"]["pp512"]["median"] = 900.0 if tag == "imatrix-fix" else 1163.8
        (tmp_path / f"z-{tag}.json").write_text(json.dumps(data), encoding="utf-8")
    return load_results(tmp_path)


def test_headline_tag_selects_the_comparison_deliberately(tmp_path):
    entries = _three_results(tmp_path)

    baseline, candidate = find_baseline_and_candidate(entries, headline_tag="published-q4_0")

    assert baseline.tag == "baseline"
    assert candidate.tag == "published-q4_0"


def test_unknown_headline_tag_names_the_tags_that_do_exist(tmp_path):
    entries = _three_results(tmp_path)

    with pytest.raises(ValueError) as exc_info:
        find_baseline_and_candidate(entries, headline_tag="no-such-tag")

    message = str(exc_info.value)
    assert "imatrix-fix" in message and "published-q4_0" in message


def test_ambiguous_headline_warns_rather_than_choosing_silently(tmp_path, capsys):
    entries = _three_results(tmp_path)

    find_baseline_and_candidate(entries)

    assert "Pass --headline TAG" in capsys.readouterr().err


def test_single_candidate_needs_no_flag_and_warns_about_nothing(tmp_path, capsys):
    shutil.copy(DATA_DIR / "results-baseline.json", tmp_path / "results-baseline.json")
    shutil.copy(DATA_DIR / "results-fixed.json", tmp_path / "results-fixed.json")

    baseline, candidate = find_baseline_and_candidate(load_results(tmp_path))

    assert baseline.tag == "baseline" and candidate.tag == "fixed"
    assert capsys.readouterr().err == ""


def test_medians_render_at_two_decimals(tmp_path):
    # One decimal erased the 66.65 / 71.60 difference README §5 discusses.
    shutil.copy(DATA_DIR / "results-baseline.json", tmp_path / "results-baseline.json")
    table = build_table(load_results(tmp_path))
    assert "415.10" in table, table
