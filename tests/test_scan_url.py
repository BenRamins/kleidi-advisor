"""Tests for `scan --url` wiring (F7.S1.T3)."""

from __future__ import annotations

import json

from kleidi_advisor.cli import main


def test_scan_url_fail_on_miss_exits_3(gguf_q4_k_only, http_fixture_server, capsys):
    http_fixture_server.routes["/m.gguf"] = gguf_q4_k_only.read_bytes()
    url = http_fixture_server.url_for("/m.gguf")

    exit_code = main(["scan", "--url", url, "--fail-on-miss"])

    assert exit_code == 3
    assert "NOT_KLEIDIAI_PATH" in capsys.readouterr().out


def test_scan_url_json_matches_on_disk_verdict(gguf_q4_k_only, http_fixture_server, capsys):
    http_fixture_server.routes["/m.gguf"] = gguf_q4_k_only.read_bytes()
    url = http_fixture_server.url_for("/m.gguf")

    exit_code = main(["scan", "--url", url, "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "NOT_KLEIDIAI_PATH"
    assert payload["source"] == url
