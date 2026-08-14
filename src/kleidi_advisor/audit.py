"""Ecosystem audit — Spec F7. Runs the head-only remote scan over a list of
GGUF URLs and reports how often the ecosystem silently misses Arm's
KleidiAI kernel path. One dead URL must never abort the whole run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from .compat import FALLBACK_GENERIC, classify
from .gguf import compute_dominant_type
from .remote import RemoteScanError, fetch_and_read

DEFAULT_DELAY_SECONDS = 1.0


@dataclass
class AuditRow:
    label: str
    url: str
    verdict: Optional[str] = None
    dominant_type: Optional[str] = None
    bytes_fetched: Optional[int] = None
    error: Optional[str] = None


def parse_list_file(path: Union[str, Path]) -> List[Tuple[str, str]]:
    """Parse `<label> <url>` lines, skipping blank lines and `#` comments."""
    entries = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"malformed audit list line (need '<label> <url>'): {raw_line!r}")
        label, url = parts
        entries.append((label, url))
    return entries


def run_audit(
    entries: List[Tuple[str, str]],
    *,
    delay: float = DEFAULT_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> List[AuditRow]:
    """Scan every (label, url) pair, single-threaded, tolerating per-entry
    failure — a dead URL becomes an `error` row, never an aborted run.
    """
    rows: List[AuditRow] = []
    for i, (label, url) in enumerate(entries):
        if i > 0 and delay > 0:
            sleep(delay)
        try:
            result = fetch_and_read(url)
            dominant = compute_dominant_type(result.info)
            verdict = classify(dominant.ggml_type_id).verdict
            rows.append(
                AuditRow(
                    label=label,
                    url=url,
                    verdict=verdict,
                    dominant_type=dominant.ggml_type_name,
                    bytes_fetched=result.bytes_fetched,
                )
            )
        except RemoteScanError as exc:
            rows.append(AuditRow(label=label, url=url, error=str(exc)))
    return rows


def summary_line(rows: List[AuditRow]) -> str:
    misses = [r for r in rows if r.verdict == FALLBACK_GENERIC]
    return f"{len(misses)} of {len(rows)} audited GGUFs fall back to generic kernels"


def to_json(rows: List[AuditRow]) -> dict:
    return {
        "schema": 1,
        "summary": summary_line(rows),
        "rows": [
            {
                "label": r.label,
                "url": r.url,
                "verdict": r.verdict,
                "dominant_type": r.dominant_type,
                "bytes_fetched": r.bytes_fetched,
                "error": r.error,
            }
            for r in rows
        ],
    }


def to_markdown(rows: List[AuditRow]) -> str:
    def sort_key(r: AuditRow):
        if r.error is not None:
            return (2, r.label)
        return (0, r.label) if r.verdict == FALLBACK_GENERIC else (1, r.label)

    lines = [
        "# Ecosystem audit",
        "",
        summary_line(rows),
        "",
        "| label | verdict | dominant type | bytes fetched | url |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=sort_key):
        if r.error is not None:
            lines.append(f"| {r.label} | error | — | — | {r.url} |")
        else:
            lines.append(f"| {r.label} | {r.verdict} | {r.dominant_type} | {r.bytes_fetched} | {r.url} |")
    lines.append("")
    return "\n".join(lines)
