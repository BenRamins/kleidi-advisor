"""Tests for the `audit` command (F7.S2.T1, Spec F7 rules 1-2 and 4).

--out/--md always point into tmp_path here: CLAUDE.md forbids this build
from ever creating a real AUDIT.md (that's the operator's RUNBOOK step 2b).
"""

from __future__ import annotations

import json

from kleidi_advisor.cli import main


def test_three_url_list_with_one_404_yields_three_rows(
    gguf_q4_k_only, gguf_q4_0_only, http_fixture_server, tmp_path, capsys
):
    http_fixture_server.routes["/a.gguf"] = gguf_q4_k_only.read_bytes()
    http_fixture_server.routes["/b.gguf"] = gguf_q4_0_only.read_bytes()
    # /c.gguf is deliberately never registered -> 404

    list_file = tmp_path / "list.txt"
    list_file.write_text(
        "\n".join(
            [
                f"model-a {http_fixture_server.url_for('/a.gguf')}",
                f"model-b {http_fixture_server.url_for('/b.gguf')}",
                f"model-c {http_fixture_server.url_for('/c.gguf')}",
            ]
        )
    )
    out_path = tmp_path / "audit.json"
    md_path = tmp_path / "AUDIT.md"

    exit_code = main(
        ["audit", "--list", str(list_file), "--out", str(out_path), "--md", str(md_path), "--delay", "0"]
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    rows = payload["rows"]
    assert len(rows) == 3

    errored = [r for r in rows if r["error"] is not None]
    ok = [r for r in rows if r["error"] is None]
    assert len(errored) == 1
    assert len(ok) == 2
    assert {r["verdict"] for r in ok} == {"FALLBACK_GENERIC", "OK_KLEIDIAI"}

    misses = sum(1 for r in ok if r["verdict"] == "FALLBACK_GENERIC")
    assert payload["summary"] == f"{misses} of 3 audited GGUFs fall back to generic kernels"

    printed_summary = capsys.readouterr().out.strip()
    assert printed_summary == payload["summary"]
    assert md_path.exists()
    assert payload["summary"] in md_path.read_text()


def test_malformed_list_line_is_a_usage_error(tmp_path, capsys):
    list_file = tmp_path / "bad.txt"
    list_file.write_text("only-one-field-no-url\n")

    exit_code = main(["audit", "--list", str(list_file)])

    assert exit_code == 2
    assert "malformed" in capsys.readouterr().err
