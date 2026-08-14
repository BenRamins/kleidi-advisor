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
        # The literal placeholder is still tolerated on read: results files
        # written before the field became nullable carry it.
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

    @property
    def ppl_chunks(self) -> Optional[int]:
        ppl = self.data.get("ppl")
        return ppl.get("chunks") if ppl else None


def load_results(results_dir: Path) -> List[ResultEntry]:
    """Read every *.json in results_dir; anything that is not a bench result —
    including unparseable JSON — is skipped (Spec F4 rule 1).

    "schema": 1 alone is not enough to identify a bench result: `audit --out`
    writes its own schema-1 document into the same directory by default, and
    without the `metrics` check it lands in the results table as a row of
    question marks.
    """
    entries: List[ResultEntry] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or data.get("schema") != 1:
            continue
        if not isinstance(data.get("metrics"), dict):
            continue
        entries.append(ResultEntry(data))
    return entries


def _sort_key(entry: ResultEntry) -> Tuple[int, str]:
    return (0 if entry.tag == "baseline" else 1, entry.tag)


def _fmt_metric(entry: ResultEntry, name: str) -> str:
    metric = entry.metric(name)
    if not metric:
        return "—"
    # Two decimals, not one: these are medians of five runs whose stdev is
    # ~0.04-0.12 tok/s, and rounding to 0.1 erases differences the report is
    # being read to judge.
    return f"{metric['median']:.2f} ± {metric['stdev']:.2f}"


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
    *,
    headline_tag: Optional[str] = None,
) -> Tuple[Optional[ResultEntry], Optional[ResultEntry]]:
    """Pick the two entries the headline compares.

    With exactly one non-baseline row, "the other one" is unambiguous. With
    several it is not, and falling back to whichever sorted first would let the
    headline change meaning as files are added to the directory. `headline_tag`
    names the comparison explicitly; without it the old behaviour stands, and
    the caller is warned when the choice was arbitrary.
    """
    baseline = next((e for e in entries if e.tag == "baseline"), None)
    others = [e for e in entries if e.tag != "baseline"]

    if headline_tag is not None:
        candidate = next((e for e in others if e.tag == headline_tag), None)
        if candidate is None:
            available = ", ".join(sorted(e.tag for e in others)) or "(none)"
            raise ValueError(
                f"no result tagged {headline_tag!r} to headline; available tags: {available}"
            )
        return baseline, candidate

    if len(others) > 1:
        chosen = others[0]
        print(
            f"WARN: {len(others)} non-baseline results present; headlining {chosen.tag!r}. "
            "Pass --headline TAG to choose deliberately.",
            file=sys.stderr,
        )
    return baseline, others[0] if others else None


def headline_line(
    entries: List[ResultEntry], *, instance: str, headline_tag: Optional[str] = None
) -> str:
    """Spec F4 rule 3 (D-14): pair speedup with its ppl cost, one sentence."""
    baseline, candidate = find_baseline_and_candidate(entries, headline_tag=headline_tag)
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
        # D-15: the chunk count belongs next to the ppl delta, because a
        # truncated corpus pass is not comparable to a full one and the number
        # is meaningless without it. It is recorded in the schema; use it.
        chunks = candidate.ppl_chunks
        corpus = f"WikiText-2, {chunks} chunks" if chunks else "WikiText-2"
        return f"{speedup:.2f}× pp512 at {delta:+.3f} ppl ({corpus}, n={n_runs}, {instance})"

    if has_speedup and not has_ppl:
        # D-14: a throughput number with no ppl neighbour is a bug — the ×
        # figure is withheld entirely, not just left "unpaired".
        return "ppl: not measured (throughput figure withheld without its quality cost, per D-14)"

    if has_ppl and not has_speedup:
        delta = candidate.ppl_value - baseline.ppl_value
        return f"speedup: not measured at {delta:+.2f} ppl"

    return "speedup: not measured — ppl: not measured"


def render_markdown(
    entries: List[ResultEntry],
    *,
    instance: Optional[str] = None,
    headline_tag: Optional[str] = None,
) -> str:
    instance = instance or "not recorded"
    lines = [
        headline_line(entries, instance=instance, headline_tag=headline_tag),
        ATTRIBUTION_LINE,
        "",
        build_table(entries),
        "",
        f"Instance: {instance}",
    ]
    return "\n".join(lines) + "\n"


def render_plot(
    entries: List[ResultEntry], output_path: Path, *, headline_tag: Optional[str] = None
) -> Optional[Path]:
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

    baseline, candidate = find_baseline_and_candidate(entries, headline_tag=headline_tag)
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
