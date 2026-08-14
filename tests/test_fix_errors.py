"""Tests for fix error paths (F2.S2.T2, Spec F2 rules 3-4, D-10)."""

from __future__ import annotations

from gguf_writer import write_gguf
from stubs.make_stubs import make_stubs

from kleidi_advisor.cli import main


def _write_f16_source(path):
    write_gguf(
        path,
        version=3,
        kvs={"general.architecture": "llama", "general.file_type": 1},
        tensors=[("blk.0.attn_q.weight", [4096, 4096], 1)],
    )


def test_missing_calib_without_no_imatrix_exits_2(tmp_path, capsys):
    source = tmp_path / "source-f16.gguf"
    _write_f16_source(source)
    output = tmp_path / "out.gguf"

    exit_code = main(["fix", str(source), "-o", str(output)])

    assert exit_code == 2
    assert "--calib" in capsys.readouterr().err


def test_k_quant_source_refused_exits_2_mentioning_f16(gguf_q4_k_only, tmp_path, capsys):
    output = tmp_path / "out.gguf"

    exit_code = main(
        ["fix", str(gguf_q4_k_only), "--calib", str(tmp_path / "nonexistent.txt"), "-o", str(output)]
    )

    assert exit_code == 2
    assert "F16" in capsys.readouterr().err


def test_stage_failure_exits_4_naming_stage(tmp_path, monkeypatch, capsys):
    bin_dir = make_stubs(tmp_path / "bin")
    monkeypatch.setenv("KA_STUB_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setenv("KA_STUB_EXIT", "1")
    monkeypatch.delenv("KA_STUB_STDOUT", raising=False)

    source = tmp_path / "source-f16.gguf"
    _write_f16_source(source)
    calib = tmp_path / "calib.txt"
    calib.write_text("hello\n")
    output = tmp_path / "out.gguf"

    exit_code = main(
        [
            "fix", str(source), "--calib", str(calib), "-o", str(output),
            "--llama-bin-dir", str(bin_dir),
        ]
    )

    assert exit_code == 4
    assert "llama-imatrix" in capsys.readouterr().err
