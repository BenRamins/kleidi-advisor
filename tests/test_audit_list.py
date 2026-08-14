"""Tests for data/hf-top-gguf.txt (F7.S2.T2, REFERENCE.md §12)."""

from __future__ import annotations

from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "hf-top-gguf.txt"


def _entry_lines():
    return [
        line
        for line in DATA_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_at_least_twenty_candidate_entries():
    assert len(_entry_lines()) >= 20


def test_every_entry_has_two_fields_and_an_https_url():
    for line in _entry_lines():
        parts = line.strip().split(maxsplit=1)
        assert len(parts) == 2, f"malformed line (need '<label> <url>'): {line!r}"
        _label, url = parts
        assert url.startswith("https://"), f"not an https URL: {line!r}"
