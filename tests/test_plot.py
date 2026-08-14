"""Tests for the report plot (F4.S1.T3, Spec F4 rules 2 and 4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from kleidi_advisor.report import ResultEntry, render_plot

DATA_DIR = Path(__file__).resolve().parent / "data"


def _entries():
    baseline = ResultEntry(json.loads((DATA_DIR / "results-baseline.json").read_text()))
    fixed = ResultEntry(json.loads((DATA_DIR / "results-fixed.json").read_text()))
    return [baseline, fixed]


def test_plot_file_exists_and_is_nontrivial_when_matplotlib_available(tmp_path):
    pytest.importorskip("matplotlib")  # skips cleanly in a matplotlib-less environment

    output = render_plot(_entries(), tmp_path / "plot.png")

    assert output is not None
    assert output.exists()
    assert output.stat().st_size > 1024


def test_plot_title_carries_the_ppl_delta(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # render_plot closes its figure when done (correct, to avoid leaking
    # figures across calls) — keep it open here so the title is inspectable.
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)

    render_plot(_entries(), tmp_path / "plot.png")

    assert "ppl" in plt.gcf().axes[0].get_title()
    plt.close("all")


def test_plot_skips_gracefully_when_matplotlib_import_fails(monkeypatch, tmp_path, capsys):
    # sys.modules[name] = None makes the next `import name` raise ImportError
    # (documented CPython behaviour) — simulates an environment without
    # matplotlib without needing to actually uninstall it.
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    output = render_plot(_entries(), tmp_path / "plot.png")

    assert output is None
    assert "WARN" in capsys.readouterr().err
