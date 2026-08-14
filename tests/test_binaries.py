"""Tests for kleidi_advisor.binaries resolution (F2.S1.T1, D-06)."""

from __future__ import annotations

import pytest

from kleidi_advisor.binaries import BinaryResolutionError, resolve_binaries


def test_missing_binaries_reported_together(tmp_path):
    (tmp_path / "llama-bench").write_text("#!/bin/sh\n")
    with pytest.raises(BinaryResolutionError) as exc_info:
        resolve_binaries(
            ["llama-imatrix", "llama-quantize", "llama-cli", "llama-bench"],
            llama_bin_dir=str(tmp_path),
        )
    message = str(exc_info.value)
    assert "llama-imatrix" in message
    assert "llama-quantize" in message
    assert "llama-cli" in message
    assert "llama-bench" not in message


def test_all_present_resolves_without_error(tmp_path):
    for name in ["llama-imatrix", "llama-quantize", "llama-cli", "llama-bench"]:
        (tmp_path / name).write_text("#!/bin/sh\n")
    resolved = resolve_binaries(
        ["llama-imatrix", "llama-quantize", "llama-cli", "llama-bench"],
        llama_bin_dir=str(tmp_path),
    )
    assert set(resolved) == {"llama-imatrix", "llama-quantize", "llama-cli", "llama-bench"}
    assert all(path.exists() for path in resolved.values())


def test_llama_bin_dir_env_var_used_when_flag_absent(tmp_path, monkeypatch):
    (tmp_path / "llama-bench").write_text("#!/bin/sh\n")
    monkeypatch.setenv("LLAMA_BIN_DIR", str(tmp_path))
    resolved = resolve_binaries(["llama-bench"])
    assert resolved["llama-bench"] == tmp_path / "llama-bench"


def test_flag_takes_precedence_over_env_var(tmp_path, monkeypatch):
    flag_dir = tmp_path / "flag"
    env_dir = tmp_path / "env"
    flag_dir.mkdir()
    env_dir.mkdir()
    (flag_dir / "llama-bench").write_text("#!/bin/sh\n")
    (env_dir / "llama-bench").write_text("#!/bin/sh\n")
    monkeypatch.setenv("LLAMA_BIN_DIR", str(env_dir))
    resolved = resolve_binaries(["llama-bench"], llama_bin_dir=str(flag_dir))
    assert resolved["llama-bench"] == flag_dir / "llama-bench"
