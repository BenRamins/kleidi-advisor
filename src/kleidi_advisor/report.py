"""Report table, headline, and plot — Spec F4.

D-14: any throughput figure is printed adjacent to its perplexity delta in
one line, never bare. When the pairing isn't possible, the writer prints
`ppl: not measured` (or `speedup: not measured`) instead of a bare number —
never both a number and a missing caveat.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ATTRIBUTION_LINE = (
    "Speedup comes from Arm's KleidiAI kernels; this tool detects the miss and measures the delta."
)


@dataclass
class ResultEntry:
    data: Dict[str, Any]

    @property
    def model(self) -> str:
        return self.data.get("model", "?")

    @property
    def tag(self) -> str:
        return self.data.get("tag", "?")

    @property
    def threads(self) -> Any:
        return self.data.get("threads", "?")

    @property
    def instance(self) -> Optional[str]:
        instance = self.data.get("instance")
        return instance if instance and instance != "TODO(box)" else None

    def metric(self, name: str) -> Optional[Dict[str, Any]]:
        return (self.data.get("metrics") or {}).get(name)

    @property
    def ppl_value(self) -> Optional[float]:
        ppl = self.data.get("ppl")
        return ppl.get("value") if ppl else None

    @property
    def ppl_corpus(self) -> Optional[str]:
        ppl = self.data.get("ppl")
        return ppl.get("corpus") if ppl else None


def load_results(results_dir: Path) -> List[ResultEntry]:
    """Read every *.json in results_dir; anything without "schema": 1 —
    including unparseable JSON — is skipped silently (Spec F4 rule 1)."""
    entries: List[ResultEntry] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or data.get("schema") != 1:
            continue
        entries.append(ResultEntry(data))
    return entries


def _sort_key(entry: ResultEntry) -> Tuple[int, str]:
    return (0 if entry.tag == "baseline" else 1, entry.tag)


def _fmt_metric(entry: ResultEntry, name: str) -> str:
    metric = entry.metric(name)
    if not metric:
        return "—"
    return f"{metric['median']:.1f} ± {metric['stdev']:.2f}"


def build_table(entries: List[ResultEntry]) -> str:
    """Spec F4 rule 1: model, tag, threads, pp512 med±sd, tg128 med±sd, ppl or —, baseline-first."""
    lines = [
        "| model | tag | threads | pp512 (tok/s) | tg128 (tok/s) | ppl |",
        "|---|---|---|---|---|---|",
    ]
    for entry in sorted(entries, key=_sort_key):
        ppl_cell = f"{entry.ppl_value:.4f}" if entry.ppl_value is not None else "—"
        lines.append(
            f"| {entry.model} | {entry.tag} | {entry.threads} | "
            f"{_fmt_metric(entry, 'pp512')} | {_fmt_metric(entry, 'tg128')} | {ppl_cell} |"
        )
    return "\n".join(lines)


def find_baseline_and_candidate(
    entries: List[ResultEntry],
) -> Tuple[Optional[ResultEntry], Optional[ResultEntry]]:
    baseline = next((e for e in entries if e.tag == "baseline"), None)
    candidate = next((e for e in entries if e.tag != "baseline"), None)
    return baseline, candidate


def headline_line(entries: List[ResultEntry], *, instance: str) -> str:
    """Spec F4 rule 3 (D-14): pair speedup with its ppl cost, one sentence."""
    baseline, candidate = find_baseline_and_candidate(entries)
    base_pp = baseline.metric("pp512") if baseline else None
    cand_pp = candidate.metric("pp512") if candidate else None
    has_speedup = bool(base_pp and cand_pp and base_pp.get("median"))
    has_ppl = bool(
        baseline and candidate and baseline.ppl_value is not None and candidate.ppl_value is not None
    )

    if has_speedup and has_ppl:
        speedup = cand_pp["median"] / base_pp["median"]
        delta = candidate.ppl_value - baseline.ppl_value
        n_runs = len(base_pp.get("runs", [])) or "?"
        return f"{speedup:.1f}× pp512 at {delta:+.2f} ppl (WikiText-2, n={n_runs}, {instance})"

    if has_speedup and not has_ppl:
        # D-14: a throughput number with no ppl neighbour is a bug — the ×
        # figure is withheld entirely, not just left "unpaired".
        return "ppl: not measured (throughput figure withheld without its quality cost, per D-14)"

    if has_ppl and not has_speedup:
        delta = candidate.ppl_value - baseline.ppl_value
        return f"speedup: not measured at {delta:+.2f} ppl"

    return "speedup: not measured — ppl: not measured"


def render_markdown(entries: List[ResultEntry], *, instance: Optional[str] = None) -> str:
    instance = instance or "TODO(box)"
    lines = [
        headline_line(entries, instance=instance),
        ATTRIBUTION_LINE,
        "",
        build_table(entries),
        "",
        f"Instance: {instance}",
    ]
    return "\n".join(lines) + "\n"


def render_plot(entries: List[ResultEntry], output_path: Path) -> Optional[Path]:
    """Spec F4 rules 2 and 4: grouped pp512/tg128 bars with stdev error bars,
    baseline vs the first non-baseline tag; title always carries the ppl
    delta so the chart can't be screenshotted without its caveat. Absent
    matplotlib (or missing baseline/candidate data) degrades to a WARN and
    no file, never a failure.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARN: matplotlib not installed; skipping plot.", file=sys.stderr)
        return None

    baseline, candidate = find_baseline_and_candidate(entries)
    if not baseline or not candidate:
        print("WARN: need a baseline and a non-baseline result to plot; skipping.", file=sys.stderr)
        return None

    metrics = ["pp512", "tg128"]
    base_medians = [(baseline.metric(m) or {}).get("median", 0) for m in metrics]
    base_stdevs = [(baseline.metric(m) or {}).get("stdev", 0) for m in metrics]
    cand_medians = [(candidate.metric(m) or {}).get("median", 0) for m in metrics]
    cand_stdevs = [(candidate.metric(m) or {}).get("stdev", 0) for m in metrics]

    positions = list(range(len(metrics)))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(
        [p - width / 2 for p in positions], base_medians, width,
        yerr=base_stdevs, label=baseline.tag, capsize=4,
    )
    ax.bar(
        [p + width / 2 for p in positions], cand_medians, width,
        yerr=cand_stdevs, label=candidate.tag, capsize=4,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("tok/s")
    ax.legend()

    ppl_caption = "ppl: not measured"
    if baseline.ppl_value is not None and candidate.ppl_value is not None:
        ppl_caption = f"ppl {candidate.ppl_value - baseline.ppl_value:+.2f}"
    ax.set_title(f"{baseline.model} vs {candidate.model} ({ppl_caption})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
