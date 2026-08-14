"""Tests for kleidi_advisor.compat classification (F1.S3.T1, REFERENCE.md §4)."""

from __future__ import annotations

import pytest

from kleidi_advisor.compat import (
    FALLBACK_GENERIC,
    KLEIDIAI_MISS_VERDICTS,
    NOT_APPLICABLE,
    NOT_KLEIDIAI_PATH,
    OK_KLEIDIAI,
    UNKNOWN_VERIFY_ON_DEVICE,
    classify,
)
from kleidi_advisor.gguf import GGML_TYPES

# Independently transcribed from REFERENCE.md §4 — verbatim, and deliberately
# NOT imported from kleidi_advisor.compat, so a typo in that module would
# still be caught here rather than the test trivially agreeing with itself.
REASON_OK_KLEIDIAI = "Q4_0 weights are repacked at load time into Arm-optimised kernels (i8mm/dotprod)."
REASON_NOT_KLEIDIAI_PATH = (
    "K-quant weights are repacked by ggml's own aarch64 path (CPU_REPACK), not KleidiAI's "
    "i8mm kernels; measured 1.61x slower at pp512 on Neoverse N2."
)
REASON_FALLBACK_GENERIC = (
    "No CPU_KLEIDIAI and no CPU_REPACK model buffer observed for this weight type; "
    "inference runs the generic kernels."
)
REASON_NOT_APPLICABLE = "Not a Q4_0-repack candidate; no kernel-miss to report for this weight type."
REASON_UNKNOWN = "Unrecognised weight type for this table version; run scan --verify on the target machine."

EXPECTED_VERDICTS = {
    "F32": NOT_APPLICABLE, "F16": NOT_APPLICABLE, "Q4_0": OK_KLEIDIAI, "Q4_1": NOT_APPLICABLE,
    "Q5_0": NOT_APPLICABLE, "Q5_1": NOT_APPLICABLE, "Q8_0": NOT_APPLICABLE, "Q8_1": NOT_APPLICABLE,
    "Q2_K": NOT_KLEIDIAI_PATH, "Q3_K": NOT_KLEIDIAI_PATH, "Q4_K": NOT_KLEIDIAI_PATH,
    "Q5_K": NOT_KLEIDIAI_PATH, "Q6_K": NOT_KLEIDIAI_PATH, "Q8_K": NOT_KLEIDIAI_PATH,
    "IQ2_XXS": FALLBACK_GENERIC, "IQ2_XS": FALLBACK_GENERIC, "IQ3_XXS": FALLBACK_GENERIC,
    "IQ1_S": FALLBACK_GENERIC, "IQ4_NL": FALLBACK_GENERIC, "IQ3_S": FALLBACK_GENERIC,
    "IQ2_S": FALLBACK_GENERIC, "IQ4_XS": FALLBACK_GENERIC, "I8": NOT_APPLICABLE,
    "I16": NOT_APPLICABLE, "I32": NOT_APPLICABLE, "I64": NOT_APPLICABLE, "F64": NOT_APPLICABLE,
    "IQ1_M": FALLBACK_GENERIC, "BF16": NOT_APPLICABLE, "TQ1_0": NOT_APPLICABLE, "TQ2_0": NOT_APPLICABLE,
}

REASONS_BY_VERDICT = {
    OK_KLEIDIAI: REASON_OK_KLEIDIAI,
    NOT_KLEIDIAI_PATH: REASON_NOT_KLEIDIAI_PATH,
    FALLBACK_GENERIC: REASON_FALLBACK_GENERIC,
    NOT_APPLICABLE: REASON_NOT_APPLICABLE,
    UNKNOWN_VERIFY_ON_DEVICE: REASON_UNKNOWN,
}


def test_every_reference_table_id_is_classified_correctly():
    # Guards against REFERENCE.md §3 drifting out from under this test unnoticed.
    assert set(GGML_TYPES) == {
        0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 34, 35,
    }
    for type_id, name in GGML_TYPES.items():
        result = classify(type_id)
        assert result.verdict == EXPECTED_VERDICTS[name], f"id {type_id} ({name})"
        assert result.reason == REASONS_BY_VERDICT[result.verdict]


@pytest.mark.parametrize("unknown_id", [31, 33, 999])
def test_unknown_and_legacy_ids_are_unknown_verify_on_device(unknown_id):
    result = classify(unknown_id)
    assert result.verdict == UNKNOWN_VERIFY_ON_DEVICE
    assert result.reason == REASON_UNKNOWN
    assert result.next is None


def test_both_miss_classes_carry_the_next_step():
    fix_command = "kleidi-advisor fix <source-f16.gguf> --calib <corpus.txt> -o <out.gguf>"

    k_quant = classify(12)  # Q4_K — measured onto ggml's CPU_REPACK path, not KleidiAI's
    assert k_quant.verdict == NOT_KLEIDIAI_PATH
    assert k_quant.next == fix_command

    iq = classify(23)  # IQ4_XS — neither buffer observed
    assert iq.verdict == FALLBACK_GENERIC
    assert iq.next == fix_command


def test_ok_kleidiai_and_not_applicable_have_no_next_step():
    assert classify(2).next is None  # Q4_0
    assert classify(0).next is None  # F32


def test_miss_set_is_exactly_the_two_non_kleidiai_accelerable_classes():
    # The set `--fail-on-miss` and `audit` both count off. Q4_0 reaching
    # KleidiAI and F16 having no repack story are not misses.
    assert KLEIDIAI_MISS_VERDICTS == {NOT_KLEIDIAI_PATH, FALLBACK_GENERIC}
    assert classify(2).verdict not in KLEIDIAI_MISS_VERDICTS  # Q4_0
    assert classify(1).verdict not in KLEIDIAI_MISS_VERDICTS  # F16
    assert classify(12).verdict in KLEIDIAI_MISS_VERDICTS  # Q4_K
    assert classify(23).verdict in KLEIDIAI_MISS_VERDICTS  # IQ4_XS
