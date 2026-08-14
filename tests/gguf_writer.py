"""Minimal GGUF v2/v3 writer for offline test fixtures.

Follows REFERENCE.md §1 (file layout) and §2 (metadata value types) exactly.
This is not a general-purpose GGUF writer — only what the test suite needs
to round-trip and to build deliberately malformed fixtures.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Union

GGUF_MAGIC = b"GGUF"

# Metadata value_type enum — REFERENCE.md §2
TYPE_UINT8 = 0
TYPE_INT8 = 1
TYPE_UINT16 = 2
TYPE_INT16 = 3
TYPE_UINT32 = 4
TYPE_INT32 = 5
TYPE_FLOAT32 = 6
TYPE_BOOL = 7
TYPE_STRING = 8
TYPE_ARRAY = 9
TYPE_UINT64 = 10
TYPE_INT64 = 11
TYPE_FLOAT64 = 12

TensorSpec = Tuple[str, List[int], int]

_SCALAR_PACKERS = {
    TYPE_UINT8: "<B",
    TYPE_INT8: "<b",
    TYPE_UINT16: "<H",
    TYPE_INT16: "<h",
    TYPE_UINT32: "<I",
    TYPE_INT32: "<i",
    TYPE_FLOAT32: "<f",
    TYPE_UINT64: "<Q",
    TYPE_INT64: "<q",
    TYPE_FLOAT64: "<d",
}


def _pack_string(s: str) -> bytes:
    data = s.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _pack_scalar(value_type: int, value: Any) -> bytes:
    if value_type == TYPE_BOOL:
        return struct.pack("<B", 1 if value else 0)
    if value_type == TYPE_STRING:
        return _pack_string(value)
    fmt = _SCALAR_PACKERS.get(value_type)
    if fmt is None:
        raise ValueError(f"unsupported value_type for writer: {value_type}")
    return struct.pack(fmt, value)


def _infer_value_type(value: Any) -> int:
    if isinstance(value, bool):
        return TYPE_BOOL
    if isinstance(value, str):
        return TYPE_STRING
    if isinstance(value, int):
        return TYPE_UINT32
    if isinstance(value, float):
        return TYPE_FLOAT32
    raise TypeError(f"cannot infer GGUF value_type for {type(value)!r}")


def _pack_kv_value(value: Any) -> bytes:
    """Pack `value_type` + payload for one kv value.

    Supports STRING, UINT32, FLOAT32, BOOL scalars and ARRAY-of-STRING —
    the set F1.S1.T2 requires. Nothing else is needed by this build's fixtures.
    """
    if isinstance(value, (list, tuple)):
        if not all(isinstance(v, str) for v in value):
            raise TypeError("array kv values must be all-string (only ARRAY-of-STRING is supported)")
        payload = struct.pack("<I", TYPE_STRING) + struct.pack("<Q", len(value))
        for item in value:
            payload += _pack_string(item)
        return struct.pack("<I", TYPE_ARRAY) + payload
    value_type = _infer_value_type(value)
    return struct.pack("<I", value_type) + _pack_scalar(value_type, value)


def write_gguf(
    path: Union[str, Path],
    *,
    version: int = 3,
    kvs: dict,
    tensors: Iterable[TensorSpec],
) -> None:
    """Write a minimal-but-byte-valid GGUF file for test fixtures.

    kvs: mapping of key -> str | int | float | bool | list[str]
    tensors: iterable of (name, dims, ggml_type_id)
    """
    tensors = list(tensors)
    alignment = kvs.get("general.alignment", 32)

    body = bytearray()
    body += GGUF_MAGIC
    body += struct.pack("<I", version)
    body += struct.pack("<Q", len(tensors))
    body += struct.pack("<Q", len(kvs))

    for key, value in kvs.items():
        body += _pack_string(key)
        body += _pack_kv_value(value)

    for name, dims, ggml_type in tensors:
        body += _pack_string(name)
        body += struct.pack("<I", len(dims))
        for d in dims:
            body += struct.pack("<Q", d)
        body += struct.pack("<I", ggml_type)
        body += struct.pack("<Q", 0)  # offset — fixtures never read tensor data

    pad = (-len(body)) % alignment
    body += b"\x00" * pad

    # Zero-filled placeholder tensor data section. Real sizes are irrelevant:
    # REFERENCE.md §1 notes everything the scanner needs precedes tensor data.
    body += b"\x00" * 64

    Path(path).write_bytes(bytes(body))
