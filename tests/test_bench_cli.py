"""Automated regression coverage for `bench` CLI wiring (F3.S1.T2).

The story's literal Verify: line is a one-off shell invocation against the
real results/ directory (run manually, not as a pytest file — see
RUN-REPORT.md). This file adds fast, isolated coverage using --results-dir
so routine `pytest -q` runs never touch the tracked results/ directory.
"""

from __future__ import annotations

from stubs.make_stubs import make_stubs

from kleidi_advisor.cli import main


def test_bench_cli_writes_tagged_results_file(gguf_q4_0_only, tmp_path, monkeypatch, capsys):
    bin_dir = make_stubs(tmp_path / "bin")
    monkeypatch.setenv("KA_STUB_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setenv("KA_STUB_STDOUT", str(__import__("pathlib").Path(__file__).resolve().parent / "data" / "llama-bench-baseline.json"))
    monkeypatch.delenv("KA_STUB_EXIT", raising=False)
    results_dir = tmp_path / "results"

    exit_code = main(
        [
            "bench", str(gguf_q4_0_only), "--threads", "4", "--tag", "stubtest",
            "--results-dir", str(results_dir), "--llama-bin-dir", str(bin_dir),
        ]
    )

    assert exit_code == 0
    matches = list(results_dir.glob("*-stubtest.json"))
    assert len(matches) == 1
    assert f"wrote {matches[0]}" in capsys.readouterr().out

    call_argv = (tmp_path / "calls.log").read_text().split()
    assert "-p" in call_argv and "512" in call_argv
    assert "-n" in call_argv and "128" in call_argv
    assert "-t" in call_argv and "4" in call_argv
    assert "-o" in call_argv and "json" in call_argv
