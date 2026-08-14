"""Tests for the fix pipeline happy path (F2.S2.T1, Spec F2 rule 2)."""

from __future__ import annotations

from gguf_writer import write_gguf
from stubs.make_stubs import make_stubs

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
    assert cli_call.split()[1:] == ["-m", str(output), "-p", "The", "-n", "8"]


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
