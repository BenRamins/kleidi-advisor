"""Tests for kleidi_advisor.gguf header/kv/tensor parsing (F1.S2.T1)."""

from __future__ import annotations

import pytest

from kleidi_advisor.gguf import GGUFError, read_gguf


def test_q4_0_only_parses(gguf_q4_0_only):
    info = read_gguf(gguf_q4_0_only)
    assert info.version == 3
    assert info.tensor_count == 3
    assert all(t.ggml_type == 2 for t in info.tensors)
    assert info.kvs["general.file_type"] == 2


def test_q4_k_only_parses(gguf_q4_k_only):
    info = read_gguf(gguf_q4_k_only)
    assert info.tensor_count == 3
    assert all(t.ggml_type == 12 for t in info.tensors)
    assert info.kvs["general.file_type"] == 15


def test_mixed_parses(gguf_mixed_q4k_f32):
    info = read_gguf(gguf_mixed_q4k_f32)
    assert info.tensor_count == 6
    assert {t.ggml_type for t in info.tensors} == {12, 0}


def test_large_vocab_parses(gguf_large_vocab):
    info = read_gguf(gguf_large_vocab)
    assert len(info.kvs["tokenizer.ggml.tokens"]) == 5000
    assert info.tensor_count == 3


def test_contradicting_file_type_parses(gguf_contradicting_file_type):
    info = read_gguf(gguf_contradicting_file_type)
    assert info.kvs["general.file_type"] == 15
    assert all(t.ggml_type == 2 for t in info.tensors)


def test_bad_magic_raises_with_offset(gguf_bad_magic):
    with pytest.raises(GGUFError) as exc_info:
        read_gguf(gguf_bad_magic)
    assert exc_info.value.offset == 0
    assert "0" in str(exc_info.value)


def test_truncated_raises_with_offset(gguf_truncated):
    with pytest.raises(GGUFError) as exc_info:
        read_gguf(gguf_truncated)
    assert isinstance(exc_info.value.offset, int)
    assert str(exc_info.value.offset) in str(exc_info.value)


def test_version1_raises_with_offset(gguf_version1):
    with pytest.raises(GGUFError) as exc_info:
        read_gguf(gguf_version1)
    assert exc_info.value.offset == 4
    assert "version 1" in str(exc_info.value)


def test_bytes_buffer_input_matches_path_input(gguf_q4_0_only):
    from_path = read_gguf(gguf_q4_0_only)
    from_bytes = read_gguf(gguf_q4_0_only.read_bytes())
    assert from_path.tensor_count == from_bytes.tensor_count
    assert [t.ggml_type for t in from_path.tensors] == [t.ggml_type for t in from_bytes.tensors]
