"""Tests for kleidi_advisor.binaries resolution (F2.S1.T1, D-06)."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from kleidi_advisor.binaries import (
    BinaryResolutionError,
    BinaryTimeout,
    resolve_binaries,
    run_binary,
)


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


# --- Hang prevention: stdin handling and timeouts ----------------------------
#
# Confirmed on the box: `fix`'s smoke stage blocked forever in
# subprocess.communicate because `llama-cli -p ... -n ...` without `-st` opens
# an interactive chat turn and waits on an inherited stdin that never closes.
# Both halves of the fix are enforced here: every child gets a closed stdin by
# default, and a caller can bound a stage so a stall fails instead of hanging.

# A shell *builtin* read loop -- no `cat`, deliberately. A stub that spawns a
# grandchild leaves it holding the stdout pipe after the timeout kills the
# shell, and the post-kill drain then blocks, which would hang the very suite
# these tests exist to keep unhangable.
_STDIN_READER = "#!/bin/sh\nwhile IFS= read -r _line; do : ; done\nprintf 'REACHED_EOF\\n'\n"

_EXITS_IMMEDIATELY = "#!/bin/sh\nprintf 'done\\n'\n"


def _write_stub(path: Path, body: str) -> Path:
    path.write_bytes(body.encode("utf-8"))
    os.chmod(path, 0o755)
    return path


def test_stdin_is_devnull_by_default(tmp_path, monkeypatch):
    """Asserts the contract, not the symptom.

    The observable version of this check — run a stdin-reading stub and see
    that it reaches EOF — passes even with the default removed, because pytest
    hands the suite an already-empty stdin to inherit. It would only catch the
    bug when run from a real terminal, which is exactly when nobody runs it.
    """
    captured = {}

    def fake_run(argv, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_binary(tmp_path / "does-not-need-to-exist", [])

    assert captured.get("stdin") is subprocess.DEVNULL


def test_a_reading_child_reaches_eof_rather_than_blocking(tmp_path):
    stub = _write_stub(tmp_path / "reader", _STDIN_READER)

    result = run_binary(stub, [], capture_output=True, text=True, timeout=30)

    assert result.returncode == 0
    assert "REACHED_EOF" in result.stdout


def test_input_kwarg_is_not_broken_by_the_stdin_default(tmp_path):
    # subprocess treats `input=` and `stdin=` as mutually exclusive, so a
    # blanket stdin default turns every input= call into a ValueError.
    stub = _write_stub(tmp_path / "reader", _STDIN_READER)

    result = run_binary(stub, [], input="a line\n", capture_output=True, text=True, timeout=30)

    assert result.returncode == 0
    assert "REACHED_EOF" in result.stdout


def test_a_child_blocked_on_an_open_stdin_hits_the_timeout_instead_of_hanging(tmp_path):
    """The box bug, reproduced exactly, then bounded.

    An open pipe nobody writes to is what an inherited terminal stdin looks
    like to the child: a read that never returns. Without the timeout this test
    would hang the suite forever, which is precisely what it is here to stop.
    """
    stub = _write_stub(tmp_path / "reader", _STDIN_READER)
    read_fd, write_fd = os.pipe()  # never written, never closed -> child blocks
    started = time.monotonic()
    try:
        with pytest.raises(BinaryTimeout) as excinfo:
            run_binary(stub, [], stdin=read_fd, capture_output=True, text=True, timeout=1)
    finally:
        os.close(read_fd)
        os.close(write_fd)

    elapsed = time.monotonic() - started
    assert elapsed < 30, f"timeout did not fire promptly ({elapsed:.1f}s)"
    assert excinfo.value.name == "reader"
    assert "timeout" in str(excinfo.value)


def test_no_timeout_means_no_timeout(tmp_path):
    # imatrix and perplexity passes legitimately run for half an hour; the
    # default must not be a limit that kills them.
    stub = _write_stub(tmp_path / "quick", _EXITS_IMMEDIATELY)

    result = run_binary(stub, [], capture_output=True, text=True)

    assert result.returncode == 0 and "done" in result.stdout
