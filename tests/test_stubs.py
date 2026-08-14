"""Tests for tests/stubs/make_stubs.py invoked through binaries.run_binary (F2.S1.T2)."""

from __future__ import annotations

from kleidi_advisor.binaries import run_binary
from stubs.make_stubs import make_stubs


def test_stub_invocation_logs_full_argv(tmp_path, monkeypatch):
    bin_dir = make_stubs(tmp_path / "bin")
    monkeypatch.setenv("KA_STUB_LOG", str(tmp_path / "calls.log"))
    monkeypatch.delenv("KA_STUB_STDOUT", raising=False)
    monkeypatch.delenv("KA_STUB_STDERR", raising=False)
    monkeypatch.delenv("KA_STUB_EXIT", raising=False)

    result = run_binary(bin_dir / "llama-cli", ["-m", "model.gguf", "-n", "8"], capture_output=True, text=True)

    assert result.returncode == 0
    logged = (tmp_path / "calls.log").read_text()
    assert "llama-cli" in logged
    assert "-m" in logged and "model.gguf" in logged
    assert "-n" in logged and "8" in logged


def test_stub_exit_code_surfaces(tmp_path, monkeypatch):
    bin_dir = make_stubs(tmp_path / "bin")
    monkeypatch.setenv("KA_STUB_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setenv("KA_STUB_EXIT", "1")

    result = run_binary(bin_dir / "llama-quantize", [], capture_output=True, text=True)

    assert result.returncode == 1


def test_stub_stdout_reaches_caller(tmp_path, monkeypatch):
    bin_dir = make_stubs(tmp_path / "bin")
    stdout_file = tmp_path / "canned.json"
    stdout_file.write_text('[{"n_prompt": 512}]')
    monkeypatch.setenv("KA_STUB_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setenv("KA_STUB_STDOUT", str(stdout_file))
    monkeypatch.delenv("KA_STUB_EXIT", raising=False)

    result = run_binary(bin_dir / "llama-bench", ["-o", "json"], capture_output=True, text=True)

    assert result.returncode == 0
    assert '"n_prompt": 512' in result.stdout
