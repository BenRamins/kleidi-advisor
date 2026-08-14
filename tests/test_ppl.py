"""Tests for kleidi_advisor.ppl parsing and the bench --gate quality gate
(F3.S2.T1, Spec F3 rules 2-3, REFERENCE.md §7).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kleidi_advisor import bench as bench_module
from kleidi_advisor.cli import main
from kleidi_advisor.ppl import PPLParseError, parse_ppl

DATA_DIR = Path(__file__).resolve().parent / "data"


def test_ppl_with_err_suffix_parses_to_6_7841():
    assert parse_ppl((DATA_DIR / "ppl-with-err.txt").read_text()) == 6.7841


def test_ppl_no_err_suffix_parses_to_6_7841():
    assert parse_ppl((DATA_DIR / "ppl-no-err.txt").read_text()) == 6.7841


def test_unparseable_output_raises_with_last_lines():
    raw = "line one\nline two\nline three\nno ppl estimate here\n"
    with pytest.raises(PPLParseError) as exc_info:
        parse_ppl(raw)
    assert "no ppl estimate here" in str(exc_info.value)


class _FakeCompleted:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _patch_bench_and_ppl_stubs(monkeypatch, *, candidate_ppl: float):
    """The REFERENCE.md §9 stub template is driven by one shared
    KA_STUB_STDOUT env var, but llama-bench and llama-perplexity need
    different canned output within the same `bench --perplexity` run — so
    for this gate test only, fake run_binary's return per binary name
    instead of fighting the shared-env-var stub mechanism.
    """
    bench_json = (DATA_DIR / "llama-bench-baseline.json").read_text()

    def fake_run_binary(binary_path, args, **kwargs):
        name = Path(binary_path).name
        if "llama-perplexity" in name:
            return _FakeCompleted(stdout=f"Final estimate: PPL = {candidate_ppl}\n")
        if "llama-bench" in name:
            return _FakeCompleted(stdout=bench_json)
        raise AssertionError(f"unexpected binary invoked: {binary_path}")

    monkeypatch.setattr(bench_module, "run_binary", fake_run_binary)
    monkeypatch.setattr(
        bench_module, "resolve_binaries", lambda names, llama_bin_dir=None: {n: Path(n) for n in names}
    )


def _bench_gate_args(tmp_path, gguf, calib, baseline, max_delta):
    return [
        "bench", str(gguf), "--threads", "4", "--tag", "candidate",
        "--results-dir", str(tmp_path / "results"),
        "--perplexity", "--calib", str(calib),
        "--gate", str(baseline), "--max-delta", str(max_delta),
    ]


def test_gate_fail_delta_0_4_exceeds_max_0_3_exits_5(tmp_path, monkeypatch, capsys):
    _patch_bench_and_ppl_stubs(monkeypatch, candidate_ppl=6.7)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"ppl": {"value": 6.3, "corpus": "x", "chunks": None}}))
    calib = tmp_path / "calib.txt"
    calib.write_text("hello\n")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")

    exit_code = main(_bench_gate_args(tmp_path, gguf, calib, baseline, 0.3))

    assert exit_code == 5
    stderr = capsys.readouterr().err
    assert "6.7" in stderr and "6.3" in stderr


def test_gate_pass_delta_0_1_within_max_0_3_exits_0(tmp_path, monkeypatch, capsys):
    _patch_bench_and_ppl_stubs(monkeypatch, candidate_ppl=6.4)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"ppl": {"value": 6.3, "corpus": "x", "chunks": None}}))
    calib = tmp_path / "calib.txt"
    calib.write_text("hello\n")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")

    exit_code = main(_bench_gate_args(tmp_path, gguf, calib, baseline, 0.3))

    assert exit_code == 0
    assert "gate: PASS" in capsys.readouterr().out
