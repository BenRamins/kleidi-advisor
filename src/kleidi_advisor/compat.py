"""Compat verdict classification — REFERENCE.md §4 (D-04). Reason strings are
copied verbatim; do not paraphrase them even to fix style.

Rewritten 2026-08-14: the box run on llama.cpp b10431 (Neoverse N2) falsified
the original two-outcome premise. There are three load paths, not two —
KleidiAI's own buffer, ggml's aarch64 CPU_REPACK buffer, and neither — so
K-quants moved out of FALLBACK_GENERIC into NOT_KLEIDIAI_PATH.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .gguf import GGML_TYPES

OK_KLEIDIAI = "OK_KLEIDIAI"
NOT_KLEIDIAI_PATH = "NOT_KLEIDIAI_PATH"
FALLBACK_GENERIC = "FALLBACK_GENERIC"
NOT_APPLICABLE = "NOT_APPLICABLE"
UNKNOWN_VERIFY_ON_DEVICE = "UNKNOWN_VERIFY_ON_DEVICE"

# Every class that is a KleidiAI miss, i.e. everything `--fail-on-miss` exits 3
# on and everything `audit` counts. K-quants reach ggml's own aarch64 repack
# path, IQ types reach neither — different fallbacks, same missed fast path.
KLEIDIAI_MISS_VERDICTS = frozenset({NOT_KLEIDIAI_PATH, FALLBACK_GENERIC})

_REASON_OK_KLEIDIAI = "Q4_0 weights are repacked at load time into Arm-optimised kernels (i8mm/dotprod)."
_REASON_NOT_KLEIDIAI_PATH = (
    "K-quant weights are repacked by ggml's own aarch64 path (CPU_REPACK), not KleidiAI's "
    "i8mm kernels; measured 1.61x slower at pp512 on Neoverse N2."
)
_REASON_FALLBACK_GENERIC = (
    "No CPU_KLEIDIAI and no CPU_REPACK model buffer observed for this weight type; "
    "inference runs the generic kernels."
)
_REASON_NOT_APPLICABLE = "Not a Q4_0-repack candidate; no kernel-miss to report for this weight type."
_REASON_UNKNOWN = "Unrecognised weight type for this table version; run scan --verify on the target machine."

_NEXT_FIX_TEMPLATE = "kleidi-advisor fix <source-f16.gguf> --calib <corpus.txt> -o <out.gguf>"

# REFERENCE.md §4, measured on b10431: K-quants land in a CPU_REPACK buffer.
# Only Q4_K_M was actually loaded on the box; the rest of the family is placed
# here by family, not by measurement — `scan --verify` is what falsifies it.
_NOT_KLEIDIAI_PATH_NAMES = {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K"}

# "all IQ*" per REFERENCE.md §4, derived from GGML_TYPES itself so this set
# can never silently drift out of sync with the id->name table it depends on.
# Unobserved on the box: no IQ model was loaded, so this is the "neither
# buffer" bucket by inference from the K-quant result, not by measurement.
_FALLBACK_GENERIC_NAMES = {name for name in GGML_TYPES.values() if name.startswith("IQ")}


@dataclass
class CompatResult:
    verdict: str
    reason: str
    next: Optional[str]


def classify(ggml_type_id: Optional[int]) -> CompatResult:
    """Map a ggml type id to a D-04 compat class.

    Ids outside `GGML_TYPES` — including the legacy pre-packed Arm formats
    31/33 and anything this table version doesn't recognise — always return
    UNKNOWN_VERIFY_ON_DEVICE, never a crash.
    """
    name = GGML_TYPES.get(ggml_type_id) if ggml_type_id is not None else None

    if name is None:
        return CompatResult(UNKNOWN_VERIFY_ON_DEVICE, _REASON_UNKNOWN, None)
    if name == "Q4_0":
        return CompatResult(OK_KLEIDIAI, _REASON_OK_KLEIDIAI, None)
    if name in _NOT_KLEIDIAI_PATH_NAMES:
        return CompatResult(NOT_KLEIDIAI_PATH, _REASON_NOT_KLEIDIAI_PATH, _NEXT_FIX_TEMPLATE)
    if name in _FALLBACK_GENERIC_NAMES:
        return CompatResult(FALLBACK_GENERIC, _REASON_FALLBACK_GENERIC, _NEXT_FIX_TEMPLATE)
    return CompatResult(NOT_APPLICABLE, _REASON_NOT_APPLICABLE, None)
