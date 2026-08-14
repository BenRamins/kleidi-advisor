"""Tests for tests/fixture_server.py (F7.S1.T1)."""

from __future__ import annotations

import urllib.error
import urllib.request


def test_range_request_returns_206_with_exact_byte_count(http_fixture_server):
    content = b"x" * 4096
    http_fixture_server.routes["/m.gguf"] = content
    req = urllib.request.Request(
        http_fixture_server.url_for("/m.gguf"), headers={"Range": "bytes=0-1023"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206
        body = resp.read()
    assert len(body) == 1024
    assert body == content[:1024]


def test_no_range_mode_ignores_range_header_and_returns_full_body(http_fixture_server):
    content = b"y" * 2048
    http_fixture_server.routes["/m.gguf"] = content
    http_fixture_server.support_range = False
    req = urllib.request.Request(
        http_fixture_server.url_for("/m.gguf"), headers={"Range": "bytes=0-1023"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        body = resp.read()
    assert len(body) == 2048


def test_unregistered_path_is_404(http_fixture_server):
    try:
        urllib.request.urlopen(http_fixture_server.url_for("/missing.gguf"))
        assert False, "expected HTTPError"
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
