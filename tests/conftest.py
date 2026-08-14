"""Shared pytest fixtures — GGUF file factories for offline testing.

Every fixture writes a real (byte-valid) GGUF file to tmp_path via
tests/gguf_writer.py so parsing tests never depend on real model files.
ggml type ids and value-type ids are taken verbatim from REFERENCE.md §2/§3.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fixture_server import FixtureHTTPServer
from gguf_writer import write_gguf

# ggml type ids — REFERENCE.md §3
GGML_F32 = 0
GGML_Q4_0 = 2
GGML_Q4_K = 12


def _weight_tensors(ggml_type: int, n: int = 3):
    return [(f"blk.{i}.attn_q.weight", [4096, 4096], ggml_type) for i in range(n)]


def _norm_tensors(n: int = 3):
    return [(f"blk.{i}.attn_norm.weight", [4096], GGML_F32) for i in range(n)]


@pytest.fixture
def gguf_q4_0_only(tmp_path: Path) -> Path:
    """All weight tensors are Q4_0 -> should classify OK_KLEIDIAI."""
    path = tmp_path / "q4_0_only.gguf"
    write_gguf(
        path,
        version=3,
        kvs={"general.architecture": "llama", "general.file_type": 2},
        tensors=_weight_tensors(GGML_Q4_0),
    )
    return path


@pytest.fixture
def gguf_q4_k_only(tmp_path: Path) -> Path:
    """All weight tensors are Q4_K -> should classify NOT_KLEIDIAI_PATH."""
    path = tmp_path / "q4_k_only.gguf"
    write_gguf(
        path,
        version=3,
        kvs={"general.architecture": "llama", "general.file_type": 15},
        tensors=_weight_tensors(GGML_Q4_K),
    )
    return path


@pytest.fixture
def gguf_mixed_q4k_f32(tmp_path: Path) -> Path:
    """Q4_K 2-D weight tensors plus F32 1-D norm tensors -> dominant type is q4_K."""
    path = tmp_path / "mixed_q4k_f32.gguf"
    write_gguf(
        path,
        version=3,
        kvs={"general.architecture": "llama", "general.file_type": 15},
        tensors=_weight_tensors(GGML_Q4_K) + _norm_tensors(),
    )
    return path


@pytest.fixture
def gguf_large_vocab(tmp_path: Path) -> Path:
    """A >2 MiB vocab array, forcing scan --url's 2 MiB -> 16 MiB escalation (F7.S1)."""
    path = tmp_path / "large_vocab.gguf"
    vocab = [f"tok_{i:05d}_" + ("x" * 440) for i in range(5000)]
    write_gguf(
        path,
        version=3,
        kvs={
            "general.architecture": "llama",
            "general.file_type": 2,
            "tokenizer.ggml.tokens": vocab,
        },
        tensors=_weight_tensors(GGML_Q4_0),
    )
    return path


@pytest.fixture
def gguf_contradicting_file_type(tmp_path: Path) -> Path:
    """Tensors are Q4_0 but general.file_type claims Q4_K_M (15) -> disagreement fixture."""
    path = tmp_path / "contradicting_file_type.gguf"
    write_gguf(
        path,
        version=3,
        kvs={"general.architecture": "llama", "general.file_type": 15},
        tensors=_weight_tensors(GGML_Q4_0),
    )
    return path


@pytest.fixture
def gguf_truncated(tmp_path: Path) -> Path:
    """A well-formed file cut in half -> parsing must fail with a clear offset."""
    path = tmp_path / "truncated.gguf"
    write_gguf(
        path,
        version=3,
        kvs={"general.architecture": "llama", "general.file_type": 2},
        tensors=_weight_tensors(GGML_Q4_0),
    )
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    return path


@pytest.fixture
def gguf_bad_magic(tmp_path: Path) -> Path:
    """First 4 bytes are not 'GGUF' -> parsing must fail immediately."""
    path = tmp_path / "bad_magic.gguf"
    write_gguf(
        path,
        version=3,
        kvs={"general.architecture": "llama", "general.file_type": 2},
        tensors=_weight_tensors(GGML_Q4_0),
    )
    data = bytearray(path.read_bytes())
    data[0:4] = b"BADM"
    path.write_bytes(bytes(data))
    return path


@pytest.fixture
def http_fixture_server():
    """A fresh FixtureHTTPServer per test, on an ephemeral port in a thread."""
    server = FixtureHTTPServer()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def gguf_version1(tmp_path: Path) -> Path:
    """version=1 must be rejected by name (only 2 and 3 are accepted)."""
    path = tmp_path / "version1.gguf"
    write_gguf(
        path,
        version=1,
        kvs={"general.architecture": "llama", "general.file_type": 2},
        tensors=_weight_tensors(GGML_Q4_0),
    )
    return path
