"""GGUF v2/v3 parsing — REFERENCE.md §1 (file layout) and §2 (value types).

`read_gguf` accepts either a local path or an in-memory bytes buffer so
F7.S1's head-only remote fetch can reuse it unchanged: both code paths parse
the same kind of buffer once the bytes are in hand.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

GGUF_MAGIC = b"GGUF"
MIN_SUPPORTED_VERSION = 2
MAX_SUPPORTED_VERSION = 3

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

_SCALAR_STRUCTS = {
    TYPE_UINT8: ("<B", 1),
    TYPE_INT8: ("<b", 1),
    TYPE_UINT16: ("<H", 2),
    TYPE_INT16: ("<h", 2),
    TYPE_UINT32: ("<I", 4),
    TYPE_INT32: ("<i", 4),
    TYPE_FLOAT32: ("<f", 4),
    TYPE_BOOL: ("<B", 1),
    TYPE_UINT64: ("<Q", 8),
    TYPE_INT64: ("<q", 8),
    TYPE_FLOAT64: ("<d", 8),
}

# ggml type ids — REFERENCE.md §3 [STAMP]: believed correct as of llama.cpp
# master mid-2026, but this table drifts across versions. Any id not present
# here is not a parse error — callers map it to UNKNOWN_VERIFY_ON_DEVICE.
GGML_TYPES: Dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
}

# general.file_type kv values — REFERENCE.md §5 [GUESS]. Corroboration only;
# the tensor scan in compute_dominant_type always wins on disagreement.
FILE_TYPE_NAMES: Dict[int, str] = {
    0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1",
    7: "MOSTLY_Q8_0", 8: "MOSTLY_Q5_0", 9: "MOSTLY_Q5_1", 10: "MOSTLY_Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
}


class GGUFError(Exception):
    """Any malformed/unsupported GGUF input. Always carries the byte offset
    of the problem so a human can find it in seconds on the box."""

    def __init__(self, message: str, offset: int):
        self.offset = offset
        super().__init__(f"{message} (at byte offset {offset})")


@dataclass
class TensorInfo:
    name: str
    n_dims: int
    dims: List[int]
    ggml_type: int
    offset: int


@dataclass
class GGUFInfo:
    version: int
    tensor_count: int
    kv_count: int
    kvs: Dict[str, Any]
    tensors: List[TensorInfo]


@dataclass
class DominantType:
    ggml_type_id: Optional[int]
    ggml_type_name: str
    file_type_kv: Optional[int]
    file_type_name: Optional[str]


class _Reader:
    """Bounds-checked little-endian cursor over an in-memory GGUF buffer."""

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def _require(self, nbytes: int) -> None:
        end = self.pos + nbytes
        if end > len(self.buf):
            # Offset is deliberately the position the read would have ended
            # at (always > len(buf) here), not where it started: this is the
            # precise, checkable "ran past the buffer end" signal F7.S1's
            # remote reader uses to decide whether to escalate its fetch
            # budget, as opposed to guessing from a truncated-looking parse.
            raise GGUFError(f"unexpected end of data reading {nbytes} byte(s) at {self.pos}", end)

    def bytes_(self, n: int) -> bytes:
        self._require(n)
        value = self.buf[self.pos : self.pos + n]
        self.pos += n
        return value

    def scalar(self, value_type: int) -> Any:
        spec = _SCALAR_STRUCTS.get(value_type)
        if spec is None:
            raise GGUFError(f"unknown scalar value_type {value_type}", self.pos)
        fmt, size = spec
        self._require(size)
        (value,) = struct.unpack_from(fmt, self.buf, self.pos)
        self.pos += size
        return bool(value) if value_type == TYPE_BOOL else value

    def string(self) -> str:
        offset = self.pos
        length = self.scalar(TYPE_UINT64)
        raw = self.bytes_(length)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError(f"invalid UTF-8 in gguf_string: {exc}", offset) from exc

    def value(self, value_type: int) -> Any:
        if value_type == TYPE_STRING:
            return self.string()
        if value_type == TYPE_ARRAY:
            element_type = self.scalar(TYPE_UINT32)
            count = self.scalar(TYPE_UINT64)
            return [self.value(element_type) for _ in range(count)]
        if value_type in _SCALAR_STRUCTS:
            return self.scalar(value_type)
        raise GGUFError(f"unknown metadata value_type {value_type}", self.pos)


def _load_bytes(source: Union[str, Path, bytes]) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    return Path(source).read_bytes()


def read_gguf(source: Union[str, Path, bytes]) -> GGUFInfo:
    """Parse a GGUF v2/v3 header, all metadata kvs, and all tensor infos.

    `source` is a local path or an in-memory bytes buffer (F7.S1 passes the
    latter after a head-only ranged HTTP fetch) — both parse identically,
    since everything this function reads precedes the tensor data section.
    """
    buf = _load_bytes(source)
    r = _Reader(buf)

    magic = r.bytes_(4)
    if magic != GGUF_MAGIC:
        raise GGUFError(f"not a GGUF file: bad magic {magic!r}", 0)

    version_offset = r.pos
    version = r.scalar(TYPE_UINT32)
    if not (MIN_SUPPORTED_VERSION <= version <= MAX_SUPPORTED_VERSION):
        raise GGUFError(
            f"unsupported GGUF version {version} (need {MIN_SUPPORTED_VERSION} or {MAX_SUPPORTED_VERSION})",
            version_offset,
        )

    tensor_count = r.scalar(TYPE_UINT64)
    kv_count = r.scalar(TYPE_UINT64)

    kvs: Dict[str, Any] = {}
    for _ in range(kv_count):
        key = r.string()
        value_type = r.scalar(TYPE_UINT32)
        kvs[key] = r.value(value_type)

    tensors: List[TensorInfo] = []
    for _ in range(tensor_count):
        name = r.string()
        n_dims = r.scalar(TYPE_UINT32)
        dims = [r.scalar(TYPE_UINT64) for _ in range(n_dims)]
        ggml_type = r.scalar(TYPE_UINT32)
        offset = r.scalar(TYPE_UINT64)
        tensors.append(
            TensorInfo(name=name, n_dims=n_dims, dims=dims, ggml_type=ggml_type, offset=offset)
        )

    return GGUFInfo(version=version, tensor_count=tensor_count, kv_count=kv_count, kvs=kvs, tensors=tensors)


def compute_dominant_type(info: GGUFInfo) -> DominantType:
    """Spec F1 rule 2: corroborate with `general.file_type`, but the most
    frequent ggml type among 2-D tensors named `*.weight` always wins on
    disagreement — both fields are always returned so a mismatch is visible.
    """
    counts: Dict[int, int] = {}
    for t in info.tensors:
        if t.n_dims == 2 and t.name.endswith(".weight"):
            counts[t.ggml_type] = counts.get(t.ggml_type, 0) + 1

    dominant_id = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0] if counts else None
    dominant_name = GGML_TYPES.get(dominant_id, f"UNKNOWN_{dominant_id}") if dominant_id is not None else "NONE"

    file_type_kv = info.kvs.get("general.file_type")
    file_type_name = FILE_TYPE_NAMES.get(file_type_kv) if isinstance(file_type_kv, int) else None

    return DominantType(
        ggml_type_id=dominant_id,
        ggml_type_name=dominant_name,
        file_type_kv=file_type_kv,
        file_type_name=file_type_name,
    )
