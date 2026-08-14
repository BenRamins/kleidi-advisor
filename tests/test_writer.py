"""Round-trip checks for tests/gguf_writer.py using raw struct reads.

F1.S1 ships before F1.S2's reader exists, so this file parses bytes directly
rather than importing kleidi_advisor.gguf. Deeper error-path assertions
(GGUFError with byte offsets) belong to F1.S2's tests/test_gguf.py.
"""

from __future__ import annotations

import struct
from pathlib import Path


def _read_gguf_string(buf: bytes, offset: int):
    (length,) = struct.unpack_from("<Q", buf, offset)
    offset += 8
    value = buf[offset : offset + length].decode("utf-8")
    offset += length
    return value, offset


def _skip_kv_value(buf: bytes, offset: int, value_type: int) -> int:
    scalar_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if value_type == 8:  # STRING
        _, offset = _read_gguf_string(buf, offset)
        return offset
    if value_type == 9:  # ARRAY
        (element_type,) = struct.unpack_from("<I", buf, offset)
        offset += 4
        (count,) = struct.unpack_from("<Q", buf, offset)
        offset += 8
        for _ in range(count):
            offset = _skip_kv_value(buf, offset, element_type)
        return offset
    return offset + scalar_sizes[value_type]


def _read_header(path: Path):
    buf = path.read_bytes()
    magic = buf[0:4]
    (version,) = struct.unpack_from("<I", buf, 4)
    (tensor_count,) = struct.unpack_from("<Q", buf, 8)
    (kv_count,) = struct.unpack_from("<Q", buf, 16)
    offset = 24
    for _ in range(kv_count):
        _key, offset = _read_gguf_string(buf, offset)
        (value_type,) = struct.unpack_from("<I", buf, offset)
        offset += 4
        offset = _skip_kv_value(buf, offset, value_type)
    first_tensor_type = None
    if tensor_count:
        _name, offset = _read_gguf_string(buf, offset)
        (n_dims,) = struct.unpack_from("<I", buf, offset)
        offset += 4
        offset += 8 * n_dims
        (first_tensor_type,) = struct.unpack_from("<I", buf, offset)
    return magic, version, tensor_count, kv_count, first_tensor_type


def test_q4_0_only_round_trips(gguf_q4_0_only):
    magic, version, tensor_count, kv_count, first_type = _read_header(gguf_q4_0_only)
    assert magic == b"GGUF"
    assert version == 3
    assert tensor_count == 3
    assert kv_count == 2
    assert first_type == 2


def test_q4_k_only_round_trips(gguf_q4_k_only):
    magic, version, tensor_count, kv_count, first_type = _read_header(gguf_q4_k_only)
    assert magic == b"GGUF"
    assert version == 3
    assert tensor_count == 3
    assert kv_count == 2
    assert first_type == 12


def test_mixed_round_trips(gguf_mixed_q4k_f32):
    magic, version, tensor_count, kv_count, first_type = _read_header(gguf_mixed_q4k_f32)
    assert magic == b"GGUF"
    assert version == 3
    assert tensor_count == 6
    assert kv_count == 2
    assert first_type == 12


def test_large_vocab_round_trips(gguf_large_vocab):
    magic, version, tensor_count, kv_count, first_type = _read_header(gguf_large_vocab)
    assert magic == b"GGUF"
    assert version == 3
    assert tensor_count == 3
    assert kv_count == 3
    assert first_type == 2
    assert gguf_large_vocab.stat().st_size > 2 * 1024 * 1024


def test_contradicting_file_type_round_trips(gguf_contradicting_file_type):
    magic, version, tensor_count, kv_count, first_type = _read_header(gguf_contradicting_file_type)
    assert magic == b"GGUF"
    assert version == 3
    assert tensor_count == 3
    assert kv_count == 2
    assert first_type == 2


def test_version1_round_trips(gguf_version1):
    magic, version, tensor_count, kv_count, first_type = _read_header(gguf_version1)
    assert magic == b"GGUF"
    assert version == 1
    assert tensor_count == 3
    assert kv_count == 2
    assert first_type == 2


def test_bad_magic_is_not_gguf(gguf_bad_magic):
    magic, *_rest = _read_header(gguf_bad_magic)
    assert magic != b"GGUF"


def test_truncated_is_nonempty_and_short(gguf_truncated):
    assert gguf_truncated.stat().st_size > 0
