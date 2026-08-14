"""Tests for kleidi_advisor.verify (F1.S4.T1, REFERENCE.md §8 outcome rules)."""

from __future__ import annotations

from kleidi_advisor.compat import FALLBACK_GENERIC, NOT_APPLICABLE, OK_KLEIDIAI
from kleidi_advisor.verify import AGREE, DISAGREE, INCONCLUSIVE, run_verify
from stubs.make_stubs import make_stubs


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
    return run_verify(fake_gguf, static_verdict, bin_dir / "llama-cli")


def test_rule1_hit_and_ok_kleidiai_agrees(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, OK_KLEIDIAI, "loaded via repack path\n")
    assert result.outcome == AGREE


def test_rule2_hit_and_fallback_generic_disagrees(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, FALLBACK_GENERIC, "using aarch64 kleidi kernels\n")
    assert result.outcome == DISAGREE


def test_rule3_no_hit_and_fallback_generic_agrees(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, FALLBACK_GENERIC, "loading model...\n")
    assert result.outcome == AGREE


def test_rule4_no_hit_and_ok_kleidiai_is_inconclusive(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, OK_KLEIDIAI, "loading model...\n")
    assert result.outcome == INCONCLUSIVE


def test_rule5_other_static_verdict_is_always_inconclusive(tmp_path, monkeypatch):
    # Even a pattern hit doesn't matter once the static verdict is neither
    # OK_KLEIDIAI nor FALLBACK_GENERIC — this is a genuinely distinct 5th rule.
    result = _run(tmp_path, monkeypatch, NOT_APPLICABLE, "loaded via repack path\n")
    assert result.outcome == INCONCLUSIVE
