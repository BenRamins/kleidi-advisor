"""Automated regression coverage for `report` CLI wiring (bundled into F4.S1).

-o/--results-dir/--plot always point into tmp_path so routine `pytest -q`
runs never write RESULTS.md into the repo root.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from kleidi_advisor.cli import main
from kleidi_advisor.report import ATTRIBUTION_LINE

DATA_DIR = Path(__file__).resolve().parent / "data"


def test_report_cli_writes_markdown_with_two_rows(tmp_path, capsys):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    shutil.copy(DATA_DIR / "results-baseline.json", results_dir / "results-baseline.json")
    shutil.copy(DATA_DIR / "results-fixed.json", results_dir / "results-fixed.json")
    output = tmp_path / "RESULTS.md"

    exit_code = main(
        [
            "report", "--results-dir", str(results_dir), "-o", str(output), "--instance", "c8g.8xlarge",
        ]
    )

    assert exit_code == 0
    text = output.read_text()
    assert text.count("|") > 0
    assert "baseline" in text and "fixed" in text
    assert ATTRIBUTION_LINE in text
    assert f"wrote {output}" in capsys.readouterr().out
