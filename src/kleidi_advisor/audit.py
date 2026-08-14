"""Ecosystem audit — Spec F7. Runs the head-only remote scan over a list of
GGUF URLs and reports how often the ecosystem silently misses Arm's
KleidiAI kernel path. One dead URL must never abort the whole run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from .compat import KLEIDIAI_MISS_VERDICTS, classify
from .gguf import compute_dominant_type
from .remote import RemoteScanError, fetch_and_read

DEFAULT_DELAY_SECONDS = 1.0

# Rendered for any null cell. Kept as a named constant because it is the one
# non-ASCII character these writers emit, and writing it to a file opened in
# the platform default encoding is exactly how AUDIT.md ended up full of
# U+FFFD — every write here passes encoding="utf-8" explicitly.
EM_DASH = "—"


@dataclass
class AuditRow:
    label: str
    url: str
    verdict: Optional[str] = None
    dominant_type: Optional[str] = None
    bytes_fetched: Optional[int] = None
    error: Optional[str] = None

    @property
    def scanned(self) -> bool:
        """A row that produced a classification. An unreachable URL did not,
        and must never count toward the denominator of the headline claim.
        """
        return self.verdict is not None


@dataclass
class AuditCounts:
    scanned: int
    errors: int
    misses: int
    bytes_fetched: int


def compute_counts(rows: List[AuditRow]) -> AuditCounts:
    scanned = [r for r in rows if r.scanned]
    return AuditCounts(
        scanned=len(scanned),
        errors=len(rows) - len(scanned),
        misses=sum(1 for r in scanned if r.verdict in KLEIDIAI_MISS_VERDICTS),
        bytes_fetched=sum(r.bytes_fetched or 0 for r in rows),
    )


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
    """The headline claim, and therefore the line most worth getting right.

    The denominator is **rows that actually scanned**, not rows attempted: an
    unreachable URL is not evidence that a model reaches KleidiAI or that it
    misses, so counting it either way overstates what was measured. Errors are
    reported alongside rather than folded in.

    A miss is any weight type that does not reach KleidiAI's kernels — the
    K-quants that take ggml's own CPU_REPACK path and the IQ types that take
    neither. Saying "fall back to generic kernels" was measured wrong on
    b10431 for K-quants (REFERENCE.md §4).
    """
    counts = compute_counts(rows)
    line = (
        f"{counts.misses} of {counts.scanned} successfully scanned GGUFs "
        f"never reach KleidiAI's kernels"
    )
    if counts.errors:
        plural = "URL" if counts.errors == 1 else "URLs"
        line += f" ({counts.errors} {plural} unreachable, listed below)"
    return line + "."


def bytes_line(rows: List[AuditRow]) -> str:
    """The ecosystem-measurement claim in one number: classifying a model costs
    a header read, not a download.

    Deliberately no "vs. N GB if downloaded in full" comparison. The audit
    never fetches past the header, so it never sees a Content-Length for the
    whole file, and inferring one from the quant type and parameter count
    would be an estimate dressed as a measurement. A missing number beats a
    wrong one.
    """
    counts = compute_counts(rows)
    megabytes = counts.bytes_fetched / 1_000_000
    return f"Classified {counts.scanned} models by fetching {megabytes:.1f} MB."


def to_json(rows: List[AuditRow]) -> dict:
    counts = compute_counts(rows)
    return {
        "schema": 1,
        "summary": summary_line(rows),
        # Explicit so the headline's arithmetic is checkable without counting
        # table rows: scanned_count + error_count == len(rows), and
        # miss_count <= scanned_count.
        "scanned_count": counts.scanned,
        "error_count": counts.errors,
        "miss_count": counts.misses,
        "bytes_fetched_total": counts.bytes_fetched,
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


def _cell(value: Optional[object]) -> str:
    """Every markdown cell goes through here, so a null can never reach the
    table as the string "None" or as a mojibake artefact.
    """
    return EM_DASH if value is None else str(value)


def to_markdown(rows: List[AuditRow]) -> str:
    def sort_key(r: AuditRow):
        if not r.scanned:
            return (2, r.label)
        return (0, r.label) if r.verdict in KLEIDIAI_MISS_VERDICTS else (1, r.label)

    lines = [
        "# Ecosystem audit",
        "",
        summary_line(rows),
        "",
        bytes_line(rows),
        "",
        "| label | verdict | dominant type | bytes fetched | url |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=sort_key):
        # Unreachable rows keep the word "error" in the verdict column — the
        # summary says they are listed below, so they have to be findable.
        verdict = "error" if not r.scanned else r.verdict
        lines.append(
            f"| {_cell(r.label)} | {_cell(verdict)} | {_cell(r.dominant_type)} "
            f"| {_cell(r.bytes_fetched)} | {_cell(r.url)} |"
        )
    lines.append("")
    return "\n".join(lines)
