"""Tests for kleidi_advisor.remote head-only fetch + escalation (F7.S1.T2, D-12)."""

from __future__ import annotations

import pytest

from kleidi_advisor.compat import classify
from kleidi_advisor.gguf import compute_dominant_type, read_gguf
from kleidi_advisor.remote import ESCALATED_BUDGET, INITIAL_BUDGET, RemoteScanError, fetch_and_read


def test_remote_verdict_matches_on_disk_scan(gguf_q4_k_only, http_fixture_server):
    on_disk_info = read_gguf(gguf_q4_k_only)
    on_disk_verdict = classify(compute_dominant_type(on_disk_info).ggml_type_id).verdict

    http_fixture_server.routes["/m.gguf"] = gguf_q4_k_only.read_bytes()
    result = fetch_and_read(http_fixture_server.url_for("/m.gguf"))
    remote_verdict = classify(compute_dominant_type(result.info).ggml_type_id).verdict

    assert remote_verdict == on_disk_verdict
    assert result.bytes_fetched <= INITIAL_BUDGET


def test_large_vocab_fixture_escalates_and_still_parses(gguf_large_vocab, http_fixture_server):
    content = gguf_large_vocab.read_bytes()
    assert len(content) > INITIAL_BUDGET  # otherwise this test wouldn't exercise escalation
    http_fixture_server.routes["/m.gguf"] = content

    result = fetch_and_read(http_fixture_server.url_for("/m.gguf"))

    assert INITIAL_BUDGET < result.bytes_fetched <= ESCALATED_BUDGET
    assert len(result.info.kvs["tokenizer.ggml.tokens"]) == 5000


def test_no_range_server_aborts_instead_of_streaming(gguf_q4_k_only, http_fixture_server):
    http_fixture_server.routes["/m.gguf"] = gguf_q4_k_only.read_bytes()
    http_fixture_server.support_range = False

    with pytest.raises(RemoteScanError) as exc_info:
        fetch_and_read(http_fixture_server.url_for("/m.gguf"))
    assert "Range" in str(exc_info.value)


def test_404_raises_remote_scan_error(http_fixture_server):
    with pytest.raises(RemoteScanError):
        fetch_and_read(http_fixture_server.url_for("/missing.gguf"))
