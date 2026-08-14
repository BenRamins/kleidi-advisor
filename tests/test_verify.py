"""Tests for kleidi_advisor.verify (F1.S4.T1, REFERENCE.md §8 outcome rules).

Rewritten 2026-08-14 against the measured b10431 log shapes. The log excerpts
below are the real ones from the box, trimmed — in particular the "cannot be
used with preferred buffer type CPU_KLEIDIAI" line, which both formats emit
and which the old substring scheme mistook for a KleidiAI hit.
"""

from __future__ import annotations

from kleidi_advisor.compat import (
    FALLBACK_GENERIC,
    NOT_APPLICABLE,
    NOT_KLEIDIAI_PATH,
    OK_KLEIDIAI,
)
from kleidi_advisor.verify import AGREE, DISAGREE, INCONCLUSIVE, VERIFY_ARGS, run_verify
from stubs.make_stubs import make_stubs

# Measured on Azure Standard_E8ps_v6 / llama.cpp 1692f9e50 (b10431).
LOG_Q4_0 = (
    "kleidiai: primary q4 kernel feature I8MM\n"
    "load_tensors: tensor 'token_embd.weight' cannot be used with preferred"
    " buffer type CPU_KLEIDIAI, using CPU instead\n"
    "load_tensors: CPU_KLEIDIAI model buffer size =  3500.45 MiB\n"
    "load_tensors:   CPU model buffer size =   308.23 MiB\n"
)
LOG_Q4_K_M = (
    "load_tensors: tensor 'token_embd.weight' cannot be used with preferred"
    " buffer type CPU_KLEIDIAI, using CPU instead\n"
    "repack: repack tensor blk.0.attn_q.weight with q4_K_8x8\n"
    "load_tensors: CPU_REPACK model buffer size =  4166.82 MiB\n"
)
LOG_NEITHER_BUFFER = (
    "load_tensors: tensor 'token_embd.weight' cannot be used with preferred"
    " buffer type CPU_KLEIDIAI, using CPU instead\n"
    "load_tensors:   CPU model buffer size =  4166.82 MiB\n"
)
LOG_NO_BUFFER_LINES = "llama_model_loader: loaded meta data with 26 key-value pairs\n"


def _run(tmp_path, monkeypatch, static_verdict, stderr_text):
    bin_dir = make_stubs(tmp_path / "bin")
    monkeypatch.setenv("KA_STUB_LOG", str(tmp_path / "calls.log"))
    monkeypatch.delenv("KA_STUB_EXIT", raising=False)
    monkeypatch.delenv("KA_STUB_STDOUT", raising=False)

    stderr_file = tmp_path / "stderr.txt"
    stderr_file.write_text(stderr_text)
    monkeypatch.setenv("KA_STUB_STDERR", str(stderr_file))

    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"GGUF")
    return run_verify(fake_gguf, static_verdict, bin_dir / "llama-bench")


def test_rule1_no_buffer_line_at_all_is_inconclusive(tmp_path, monkeypatch):
    # -v not passed, or a build that doesn't print buffer sizes: nothing was
    # observed, so a confident AGREE/DISAGREE would be fabricated.
    result = _run(tmp_path, monkeypatch, OK_KLEIDIAI, LOG_NO_BUFFER_LINES)
    assert result.outcome == INCONCLUSIVE
    assert result.signals.any_buffer_line is False


def test_rule2_kleidiai_buffer_and_ok_kleidiai_agrees(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, OK_KLEIDIAI, LOG_Q4_0)
    assert result.outcome == AGREE
    assert result.signals.kleidiai_buffer is True


def test_rule3_kleidiai_buffer_and_a_miss_verdict_disagrees(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, NOT_KLEIDIAI_PATH, LOG_Q4_0).outcome == DISAGREE
    assert _run(tmp_path, monkeypatch, FALLBACK_GENERIC, LOG_Q4_0).outcome == DISAGREE


def test_rule4_no_kleidiai_buffer_and_ok_kleidiai_disagrees(tmp_path, monkeypatch):
    # Under the old pattern scheme this was INCONCLUSIVE. With -v and buffer
    # lines present, the absence of a CPU_KLEIDIAI buffer is real evidence.
    result = _run(tmp_path, monkeypatch, OK_KLEIDIAI, LOG_Q4_K_M)
    assert result.outcome == DISAGREE


def test_rule5_not_kleidiai_path_agrees_only_with_a_repack_buffer(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, NOT_KLEIDIAI_PATH, LOG_Q4_K_M).outcome == AGREE
    # The class asserts a CPU_REPACK buffer exists; if neither buffer shows up
    # the model is really FALLBACK_GENERIC and the table is wrong.
    assert _run(tmp_path, monkeypatch, NOT_KLEIDIAI_PATH, LOG_NEITHER_BUFFER).outcome == DISAGREE


def test_rule6_fallback_generic_agrees_only_when_neither_buffer_appears(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, FALLBACK_GENERIC, LOG_NEITHER_BUFFER).outcome == AGREE
    assert _run(tmp_path, monkeypatch, FALLBACK_GENERIC, LOG_Q4_K_M).outcome == DISAGREE


def test_rule7_other_static_verdict_is_always_inconclusive(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, NOT_APPLICABLE, LOG_Q4_0)
    assert result.outcome == INCONCLUSIVE


def test_the_repack_word_alone_never_decides_the_outcome(tmp_path, monkeypatch):
    # The regression this whole rewrite exists for: "repack" and the token
    # CPU_KLEIDIAI both appear in the Q4_K_M log, and neither may produce AGREE
    # for an OK_KLEIDIAI prediction.
    assert "repack" in LOG_Q4_K_M
    assert "CPU_KLEIDIAI" in LOG_Q4_K_M
    assert _run(tmp_path, monkeypatch, OK_KLEIDIAI, LOG_Q4_K_M).outcome != AGREE


def test_verify_drives_llama_bench_with_verbose_and_no_generation(tmp_path, monkeypatch):
    # llama-cli goes interactive and hangs; the buffer lines are -v only.
    _run(tmp_path, monkeypatch, OK_KLEIDIAI, LOG_Q4_0)
    logged = (tmp_path / "calls.log").read_text()
    assert "llama-bench" in logged
    assert "llama-cli" not in logged
    for token in VERIFY_ARGS:
        assert token in logged.split(), f"missing {token!r} in {logged!r}"
