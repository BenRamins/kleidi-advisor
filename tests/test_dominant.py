"""Tests for dominant-type detection (F1.S2.T2, Spec F1 rule 2)."""

from __future__ import annotations

from kleidi_advisor.gguf import compute_dominant_type, read_gguf


def test_mixed_fixture_dominant_is_q4_k(gguf_mixed_q4k_f32):
    info = read_gguf(gguf_mixed_q4k_f32)
    dominant = compute_dominant_type(info)
    assert dominant.ggml_type_id == 12
    assert dominant.ggml_type_name == "Q4_K"


def test_contradicting_file_type_reports_both_fields(gguf_contradicting_file_type):
    info = read_gguf(gguf_contradicting_file_type)
    dominant = compute_dominant_type(info)
    # Tensor scan wins the verdict; the file_type kv is still reported so the
    # disagreement is visible rather than silently resolved (Spec F1 rule 2).
    assert dominant.ggml_type_name == "Q4_0"
    assert dominant.file_type_kv == 15
    assert dominant.file_type_name == "Q4_K_M"
    assert dominant.ggml_type_name != dominant.file_type_name
