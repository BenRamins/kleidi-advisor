"""Tests for the `audit` command (F7.S2.T1, Spec F7 rules 1-2 and 4).

--out/--md always point into tmp_path here: CLAUDE.md forbids this build
from ever creating a real AUDIT.md (that's REPRODUCE.md step 3, run by a human).
"""

from __future__ import annotations

import json

from kleidi_advisor.audit import EM_DASH, AuditRow, compute_counts, summary_line
from kleidi_advisor.cli import main
from kleidi_advisor.compat import KLEIDIAI_MISS_VERDICTS as MISS_VERDICTS


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
    assert {r["verdict"] for r in ok} == {"NOT_KLEIDIAI_PATH", "OK_KLEIDIAI"}

    misses = sum(1 for r in ok if r["verdict"] in {"NOT_KLEIDIAI_PATH", "FALLBACK_GENERIC"})
    assert payload["summary"] == (
        f"{misses} of 2 successfully scanned GGUFs never reach KleidiAI's kernels "
        f"(1 URL unreachable, listed below)."
    )

    printed = capsys.readouterr().out.strip().splitlines()
    assert printed[0] == payload["summary"]
    assert printed[1].startswith("Classified "), "the byte total is printed alongside the summary"
    assert md_path.exists()
    assert payload["summary"] in md_path.read_text()


def test_malformed_list_line_is_a_usage_error(tmp_path, capsys):
    list_file = tmp_path / "bad.txt"
    list_file.write_text("only-one-field-no-url\n")

    exit_code = main(["audit", "--list", str(list_file)])

    assert exit_code == 2
    assert "malformed" in capsys.readouterr().err


def test_summary_counts_only_rows_that_actually_scanned(
    gguf_q4_k_only, gguf_q4_0_only, http_fixture_server, tmp_path, capsys
):
    """The denominator is the headline claim's credibility.

    Three unreachable URLs are not three models that reach KleidiAI, and not
    three that miss it — they are three non-observations, and folding them
    into the denominator understates the miss rate against evidence that was
    never collected.
    """
    http_fixture_server.routes["/hit.gguf"] = gguf_q4_k_only.read_bytes()
    http_fixture_server.routes["/ok.gguf"] = gguf_q4_0_only.read_bytes()

    lines = [
        f"miss-a {http_fixture_server.url_for('/hit.gguf')}",
        f"ok-a {http_fixture_server.url_for('/ok.gguf')}",
        f"dead-a {http_fixture_server.url_for('/nope-1.gguf')}",
        f"dead-b {http_fixture_server.url_for('/nope-2.gguf')}",
        f"dead-c {http_fixture_server.url_for('/nope-3.gguf')}",
    ]
    list_file = tmp_path / "list.txt"
    list_file.write_text("\n".join(lines), encoding="utf-8")
    out_path = tmp_path / "audit.json"

    assert main(["audit", "--list", str(list_file), "--out", str(out_path), "--delay", "0"]) == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    non_null_verdicts = [r for r in payload["rows"] if r["verdict"] is not None]
    misses = [r for r in non_null_verdicts if r["verdict"] in MISS_VERDICTS]

    assert payload["scanned_count"] == len(non_null_verdicts) == 2
    assert payload["error_count"] == 3
    assert payload["miss_count"] == len(misses) == 1
    # The arithmetic the JSON exists to make checkable.
    assert payload["scanned_count"] + payload["error_count"] == len(payload["rows"]) == 5
    assert payload["miss_count"] <= payload["scanned_count"]

    assert payload["summary"] == (
        "1 of 2 successfully scanned GGUFs never reach KleidiAI's kernels "
        "(3 URLs unreachable, listed below)."
    )
    assert "of 5" not in payload["summary"], "attempted rows must not be the denominator"


def test_no_replacement_character_in_generated_markdown(
    gguf_q4_k_only, http_fixture_server, tmp_path
):
    """AUDIT.md was once written in the platform default encoding, turning the
    em-dash in every error row into U+FFFD when read back as UTF-8.
    """
    http_fixture_server.routes["/hit.gguf"] = gguf_q4_k_only.read_bytes()
    list_file = tmp_path / "list.txt"
    list_file.write_text(
        "\n".join(
            [
                f"scanned {http_fixture_server.url_for('/hit.gguf')}",
                f"dead {http_fixture_server.url_for('/gone.gguf')}",
            ]
        ),
        encoding="utf-8",
    )
    md_path = tmp_path / "AUDIT.md"

    assert main(["audit", "--list", str(list_file), "--md", str(md_path), "--delay", "0"]) == 0

    raw = md_path.read_bytes()
    raw.decode("utf-8")  # raises if the writer used a platform default encoding
    text = raw.decode("utf-8")

    assert "\ufffd" not in text, "U+FFFD in generated markdown"
    assert "None" not in text, "a null cell leaked into the table as 'None'"
    # The dead row renders em-dashes for the fields it never produced.
    dead_row = next(line for line in text.splitlines() if line.startswith("| dead "))
    assert dead_row.count(EM_DASH) == 2, dead_row
    assert "| error |" in dead_row, "unreachable rows must stay findable in the table"


def test_bytes_line_reports_fetched_total_and_refuses_to_estimate_full_size(
    gguf_q4_k_only, http_fixture_server, tmp_path
):
    http_fixture_server.routes["/a.gguf"] = gguf_q4_k_only.read_bytes()
    list_file = tmp_path / "list.txt"
    list_file.write_text(f"a {http_fixture_server.url_for('/a.gguf')}\n", encoding="utf-8")
    md_path = tmp_path / "AUDIT.md"

    assert main(["audit", "--list", str(list_file), "--md", str(md_path), "--delay", "0"]) == 0
    text = md_path.read_text(encoding="utf-8")

    assert "Classified 1 models by fetching" in text
    # No "vs. N GB if downloaded in full" clause: the audit never sees a
    # Content-Length for the whole file, so that number is not ours to state.
    assert "downloading them in full" not in text
    assert "GB" not in text


def test_counts_helper_handles_an_all_error_run():
    rows = [AuditRow(label="a", url="u1", error="404"), AuditRow(label="b", url="u2", error="404")]
    counts = compute_counts(rows)
    assert (counts.scanned, counts.errors, counts.misses, counts.bytes_fetched) == (0, 2, 0, 0)
    assert summary_line(rows) == (
        "0 of 0 successfully scanned GGUFs never reach KleidiAI's kernels "
        "(2 URLs unreachable, listed below)."
    )


def test_clean_run_summary_omits_the_error_clause():
    rows = [AuditRow(label="a", url="u", verdict="OK_KLEIDIAI", dominant_type="Q4_0", bytes_fetched=10)]
    assert summary_line(rows) == "0 of 1 successfully scanned GGUFs never reach KleidiAI's kernels."
