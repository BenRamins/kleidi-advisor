"""Compat verdict classification — REFERENCE.md §4 (D-04). Reason strings are
copied verbatim; do not paraphrase them even to fix style.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .gguf import GGML_TYPES

OK_KLEIDIAI = "OK_KLEIDIAI"
FALLBACK_GENERIC = "FALLBACK_GENERIC"
NOT_APPLICABLE = "NOT_APPLICABLE"
UNKNOWN_VERIFY_ON_DEVICE = "UNKNOWN_VERIFY_ON_DEVICE"

_REASON_OK_KLEIDIAI = "Q4_0 weights are repacked at load time into Arm-optimised kernels (i8mm/dotprod)."
_REASON_FALLBACK_GENERIC = "K-quant/IQ weights have no Arm repack path; inference runs the generic kernels."
_REASON_NOT_APPLICABLE = "Not a Q4_0-repack candidate; no kernel-miss to report for this weight type."
_REASON_UNKNOWN = "Unrecognised weight type for this table version; run scan --verify on the target machine."

_NEXT_FIX_TEMPLATE = "kleidi-advisor fix <source-f16.gguf> --calib <corpus.txt> -o <out.gguf>"

# "all IQ*" per REFERENCE.md §4, derived from GGML_TYPES itself so this set
# can never silently drift out of sync with the id->name table it depends on.
_FALLBACK_GENERIC_NAMES = {name for name in GGML_TYPES.values() if name.startswith("IQ")} | {
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
}


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
    if name in _FALLBACK_GENERIC_NAMES:
        return CompatResult(FALLBACK_GENERIC, _REASON_FALLBACK_GENERIC, _NEXT_FIX_TEMPLATE)
    return CompatResult(NOT_APPLICABLE, _REASON_NOT_APPLICABLE, None)
