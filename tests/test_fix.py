"""Tests for the fix pipeline happy path (F2.S2.T1, Spec F2 rule 2)."""

from __future__ import annotations

import os
import time

from gguf_writer import write_gguf
from stubs.make_stubs import make_stubs

from kleidi_advisor import fix
from kleidi_advisor.cli import main


def _write_f16_source(path):
    write_gguf(
        path,
        version=3,
        kvs={"general.architecture": "llama", "general.file_type": 1},
        tensors=[("blk.0.attn_q.weight", [4096, 4096], 1)],  # ggml type 1 = F16
    )


def test_fix_happy_path_argv_order_and_flags(tmp_path, monkeypatch):
    bin_dir = make_stubs(tmp_path / "bin")
    monkeypatch.setenv("KA_STUB_LOG", str(tmp_path / "calls.log"))
    monkeypatch.delenv("KA_STUB_EXIT", raising=False)
    monkeypatch.delenv("KA_STUB_STDOUT", raising=False)

    source = tmp_path / "source-f16.gguf"
    _write_f16_source(source)
    calib = tmp_path / "calib.txt"
    calib.write_text("hello world\n")
    output = tmp_path / "out-q4_0.gguf"
    imatrix_path = tmp_path / "out-q4_0.imatrix"

    exit_code = main(
        [
            "fix", str(source), "--calib", str(calib), "-o", str(output),
            "--llama-bin-dir", str(bin_dir),
        ]
    )

    assert exit_code == 0
    log_lines = (tmp_path / "calls.log").read_text().strip().splitlines()
    assert len(log_lines) == 3
    imatrix_call, quantize_call, cli_call = log_lines

    assert "llama-imatrix" in imatrix_call
    assert imatrix_call.split()[1:] == ["-m", str(source), "-f", str(calib), "-o", str(imatrix_path)]

    assert "llama-quantize" in quantize_call
    assert quantize_call.split()[1:] == [
        "--imatrix", str(imatrix_path), str(source), str(output), "Q4_0",
    ]

    assert "llama-cli" in cli_call
    # -st (single-turn) is not optional: without it llama-cli finishes the
    # generation and then opens an interactive turn, blocking on stdin forever.
    assert cli_call.split()[1:] == ["-m", str(output), "-p", "The", "-n", "8", "-st"]


def test_fix_no_imatrix_skips_imatrix_stage_and_warns(tmp_path, monkeypatch, capsys):
    bin_dir = make_stubs(tmp_path / "bin")
    monkeypatch.setenv("KA_STUB_LOG", str(tmp_path / "calls.log"))
    monkeypatch.delenv("KA_STUB_EXIT", raising=False)
    monkeypatch.delenv("KA_STUB_STDOUT", raising=False)

    source = tmp_path / "source-f16.gguf"
    _write_f16_source(source)
    output = tmp_path / "out-q4_0.gguf"

    exit_code = main(
        ["fix", str(source), "-o", str(output), "--no-imatrix", "--llama-bin-dir", str(bin_dir)]
    )

    assert exit_code == 0
    log_lines = (tmp_path / "calls.log").read_text().strip().splitlines()
    assert len(log_lines) == 2  # quantize, cli — imatrix stage skipped
    assert "llama-quantize" in log_lines[0]
    assert "--imatrix" not in log_lines[0]
    assert "llama-cli" in log_lines[1]
    assert "WARN" in capsys.readouterr().err


# --- The box hang, at the `fix` level ----------------------------------------

# Busy-loops in the shell itself. A `sleep` would be an external binary, i.e. a
# grandchild that outlives the kill and holds the stdout pipe open -- the same
# trap documented in test_binaries.py.
_NEVER_EXITS = "#!/bin/sh\nwhile : ; do : ; done\n"


def test_smoke_stage_stall_times_out_with_a_clear_message_and_exit_4(
    tmp_path, monkeypatch, capsys
):
    """Reproduces the confirmed box failure: `fix` reached the smoke stage and
    never came back. It must now fail as a named stall, not hang an unattended
    run forever.
    """
    bin_dir = make_stubs(tmp_path / "bin")
    stuck_cli = bin_dir / "llama-cli"
    stuck_cli.write_bytes(_NEVER_EXITS.encode("utf-8"))
    os.chmod(stuck_cli, 0o755)

    monkeypatch.setenv("KA_STUB_LOG", str(tmp_path / "calls.log"))
    monkeypatch.delenv("KA_STUB_EXIT", raising=False)
    monkeypatch.delenv("KA_STUB_STDOUT", raising=False)
    monkeypatch.setattr(fix, "SMOKE_TIMEOUT_SECONDS", 1)

    source = tmp_path / "source-f16.gguf"
    _write_f16_source(source)
    output = tmp_path / "out-q4_0.gguf"

    started = time.monotonic()
    exit_code = main(
        ["fix", str(source), "-o", str(output), "--no-imatrix", "--llama-bin-dir", str(bin_dir)]
    )
    elapsed = time.monotonic() - started

    assert exit_code == 4, "a stalled stage is a subprocess stage failure (exit 4)"
    assert elapsed < 30, f"fix did not give up promptly ({elapsed:.1f}s)"

    err = capsys.readouterr().err
    assert "smoke generation timed out" in err, err
    # The artifact survived the stall; the message must say so rather than
    # leaving the operator to guess whether the quantization is usable.
    assert str(output) in err
    assert "scan" in err


def test_smoke_timeout_default_is_generous_enough_for_a_real_generation():
    # Eight tokens on a slow CPU-only box is seconds, not minutes; the bound
    # exists to catch a stall, not to race a legitimately slow machine.
    assert fix.SMOKE_TIMEOUT_SECONDS >= 300


def test_smoke_invocation_carries_the_single_turn_flag():
    assert "-st" in fix.SMOKE_ARGS, "without -st llama-cli opens an interactive turn and blocks"
