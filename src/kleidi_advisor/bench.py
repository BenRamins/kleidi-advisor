"""llama-bench runner and stats — Spec F3 rule 1, REFERENCE.md §6 and §10.

Parsing follows §6 defensively (the highest-risk parse in the build): rows
are selected by n_prompt/n_gen, never positionally, and a missing-key error
always names the keys actually present so a human can fix it on the box.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .binaries import resolve_binaries, run_binary
from .ppl import parse_ppl

REQUIRED_BENCH_BINARIES = ["llama-bench"]


class BenchParseError(Exception):
    """llama-bench output couldn't be parsed into pp512/tg128 stats."""


@dataclass
class MetricStats:
    runs: List[float]
    median: float
    stdev: float
    unit: str = "tok/s"

    def to_dict(self) -> Dict[str, Any]:
        return {"runs": self.runs, "median": self.median, "stdev": self.stdev, "unit": self.unit}


def _row_samples(row: Dict[str, Any]) -> List[float]:
    """REFERENCE.md §6 rule 2: prefer samples_ts; fall back to avg_ts; else
    raise, naming the keys actually present."""
    if "samples_ts" in row:
        return list(row["samples_ts"])
    if "avg_ts" in row:
        return [row["avg_ts"]]
    raise BenchParseError(
        f"llama-bench row has neither 'samples_ts' nor 'avg_ts'; keys present: {sorted(row.keys())}"
    )


def _select_row(rows: List[Dict[str, Any]], *, prompt: bool) -> Dict[str, Any]:
    """REFERENCE.md §6 rule 1/3: select by n_prompt/n_gen, never positionally
    and never assuming row count."""
    for row in rows:
        n_prompt = row.get("n_prompt", 0)
        n_gen = row.get("n_gen", 0)
        if prompt and n_prompt > 0 and n_gen == 0:
            return row
        if not prompt and n_gen > 0 and n_prompt == 0:
            return row
    kind = "pp512 (n_prompt>0, n_gen==0)" if prompt else "tg128 (n_gen>0, n_prompt==0)"
    raise BenchParseError(f"no llama-bench row matches {kind}")


def _stats(samples: List[float]) -> MetricStats:
    return MetricStats(
        runs=list(samples),
        median=statistics.median(samples),
        stdev=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    )


def parse_bench_json(raw: str) -> Dict[str, MetricStats]:
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as exc:
        snippet = raw[:200] + ("..." if len(raw) > 200 else "")
        raise BenchParseError(f"llama-bench did not print valid JSON ({exc}): {snippet!r}") from exc
    return {
        "pp512": _stats(_row_samples(_select_row(rows, prompt=True))),
        "tg128": _stats(_row_samples(_select_row(rows, prompt=False))),
    }


@dataclass
class BenchResult:
    metrics: Dict[str, MetricStats]
    argv: List[str]
    results_path: Path
    ppl_value: Optional[float] = None


def detect_llama_cpp_commit(binary_path: Path) -> Optional[str]:
    """Best-effort short SHA of the llama.cpp checkout a binary came from.

    Walks up from the binary looking for a git repository — `build/bin/llama-bench`
    sits two levels below the checkout root in the documented build. Returns None
    on any failure at all: an unrecorded commit is honest, a guessed one is not,
    and a bench run must never fail because provenance lookup did.
    """
    try:
        start = Path(binary_path).resolve()
    except OSError:
        return None
    for candidate in start.parents:
        if not (candidate / ".git").exists():
            continue
        try:
            result = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    return None


def run_bench(
    gguf: Path,
    *,
    threads: int,
    tag: str,
    repeats: int = 5,
    results_dir: Path,
    llama_bin_dir: Optional[str] = None,
    perplexity: bool = False,
    calib: Optional[Path] = None,
    chunks: Optional[int] = None,
    instance: Optional[str] = None,
) -> BenchResult:
    """D-07 bench matrix: pp512 + tg128, JSON out, median + population stdev.
    Spec F3 rule 2: with `perplexity=True`, also runs llama-perplexity and
    stores the result under `ppl` — the pairing D-14 requires downstream.
    D-15: --chunks caps llama-perplexity's corpus pass; the value is recorded
    in the results schema's `ppl.chunks` field so README can cite it next to
    the ppl delta.
    """
    required = list(REQUIRED_BENCH_BINARIES)
    if perplexity:
        required.append("llama-perplexity")
    binaries = resolve_binaries(required, llama_bin_dir=llama_bin_dir)

    bench_args = [
        "-m", str(gguf), "-p", "512", "-n", "128", "-t", str(threads), "-r", str(repeats), "-o", "json",
    ]
    result = run_binary(binaries["llama-bench"], bench_args, capture_output=True, text=True)
    if result.returncode != 0:
        raise BenchParseError(f"llama-bench exited {result.returncode}: {result.stderr.strip()}")

    metrics = parse_bench_json(result.stdout)
    argv = ["llama-bench", *bench_args]

    ppl_payload: Optional[Dict[str, Any]] = None
    ppl_value: Optional[float] = None
    if perplexity:
        ppl_args = ["-m", str(gguf), "-f", str(calib), "-t", str(threads)]
        if chunks is not None:
            ppl_args += ["--chunks", str(chunks)]
        ppl_result = run_binary(binaries["llama-perplexity"], ppl_args, capture_output=True, text=True)
        if ppl_result.returncode != 0:
            raise BenchParseError(
                f"llama-perplexity exited {ppl_result.returncode}: {ppl_result.stderr.strip()}"
            )
        ppl_value = parse_ppl(ppl_result.stdout + "\n" + ppl_result.stderr)
        ppl_payload = {"value": ppl_value, "corpus": str(calib), "chunks": chunks}

    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{Path(gguf).stem}-{tag}.json"
    payload = {
        "schema": 1,
        "model": Path(gguf).name,
        "tag": tag,
        "threads": threads,
        # Recorded, not placeheld. `instance` comes from --instance and the
        # commit from the llama.cpp checkout the binary was built in; either is
        # null when genuinely unavailable, which reads the same way `ppl` does
        # when perplexity was not measured. A literal placeholder string here
        # is what put an unfilled-slot token into shipped result files once
        # already; report.py still tolerates that legacy value on read.
        "instance": instance,
        "llama_cpp_commit": detect_llama_cpp_commit(binaries["llama-bench"]),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "argv": argv,
        "metrics": {name: stats.to_dict() for name, stats in metrics.items()},
        "ppl": ppl_payload,
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return BenchResult(metrics=metrics, argv=argv, results_path=results_path, ppl_value=ppl_value)
