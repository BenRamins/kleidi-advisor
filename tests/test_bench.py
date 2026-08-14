"""Tests for kleidi_advisor.bench parsing and stats (F3.S1.T1, REFERENCE.md §6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kleidi_advisor.bench import BenchParseError, parse_bench_json

DATA_DIR = Path(__file__).resolve().parent / "data"


def test_baseline_fixture_pp512_median_is_415_1():
    raw = (DATA_DIR / "llama-bench-baseline.json").read_text()
    metrics = parse_bench_json(raw)
    assert metrics["pp512"].median == 415.1


def test_falls_back_to_avg_ts_when_samples_ts_absent():
    rows = json.loads((DATA_DIR / "llama-bench-baseline.json").read_text())
    del rows[0]["samples_ts"]  # pp512 row now has only avg_ts
    metrics = parse_bench_json(json.dumps(rows))
    assert metrics["pp512"].median == rows[0]["avg_ts"]
    assert metrics["pp512"].runs == [rows[0]["avg_ts"]]


def test_missing_both_keys_raises_listing_present_keys():
    rows = json.loads((DATA_DIR / "llama-bench-baseline.json").read_text())
    del rows[0]["samples_ts"]
    del rows[0]["avg_ts"]
    with pytest.raises(BenchParseError) as exc_info:
        parse_bench_json(json.dumps(rows))
    message = str(exc_info.value)
    for key in rows[0].keys():
        assert key in message
