"""Tests for kleidi_advisor.bench parsing and stats (F3.S1.T1, REFERENCE.md §6)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from stubs.make_stubs import make_stubs

from kleidi_advisor.bench import (
    BenchParseError,
    detect_llama_cpp_commit,
    parse_bench_json,
    run_bench,
)

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


# --- Provenance fields are recorded, never placeheld --------------------------
#
# These two fields once shipped in real result files carrying the repo's
# unfilled-slot token as their value. A placeholder that looks like data is
# worse than a null: it survives into everything that reads the file, and it
# trips every unfilled-slot sweep in the repo.


def test_instance_and_commit_are_recorded_not_placeheld(tmp_path, monkeypatch):
    bin_dir = make_stubs(tmp_path / "bin")
    monkeypatch.setenv("KA_STUB_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setenv("KA_STUB_STDOUT", str(DATA_DIR / "llama-bench-baseline.json"))
    monkeypatch.delenv("KA_STUB_EXIT", raising=False)

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    label = "Azure Standard_E8ps_v6 (Cobalt 100, Neoverse N2), 8 threads"

    result = run_bench(
        gguf,
        threads=8,
        tag="baseline",
        results_dir=tmp_path / "results",
        llama_bin_dir=str(bin_dir),
        instance=label,
    )

    payload = json.loads(result.results_path.read_text(encoding="utf-8"))
    assert payload["instance"] == label
    for field in ("instance", "llama_cpp_commit"):
        assert payload[field] != "TODO" + "(box)", f"{field} written as a placeholder"
    # The stub bin dir is not a git checkout, so the commit is honestly null
    # rather than invented.
    assert payload["llama_cpp_commit"] is None


def test_commit_detection_never_raises_and_returns_none_off_a_checkout(tmp_path):
    assert detect_llama_cpp_commit(tmp_path / "nope" / "llama-bench") is None


def test_commit_detection_reads_the_checkout_a_binary_was_built_in(tmp_path):
    repo = tmp_path / "llama.cpp"
    (repo / "build" / "bin").mkdir(parents=True)
    binary = repo / "build" / "bin" / "llama-bench"
    binary.write_text("#!/bin/sh\n")
    if subprocess.run(["git", "init", "-q", str(repo)], capture_output=True).returncode != 0:
        pytest.skip("git unavailable")
    for cmd in (["config", "user.email", "t@localhost"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(repo), *cmd], capture_output=True, check=True)
    (repo / "f.txt").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "f.txt"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "c"], capture_output=True, check=True)

    sha = detect_llama_cpp_commit(binary)

    assert sha and len(sha) >= 7 and all(c in "0123456789abcdef" for c in sha)
