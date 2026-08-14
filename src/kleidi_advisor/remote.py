"""Head-only remote GGUF scan over HTTP Range — REFERENCE.md §1, D-12.

GGUF metadata lives at byte 0, so classification needs only the head of a
file. Fetches via stdlib urllib with a Range header, starting at 2 MiB and
escalating once to 16 MiB if the tensor infos were truncated. A server that
ignores Range (200 instead of 206) is a distinct, non-retryable failure:
retrying a bigger budget against a server that always sends the whole file
would defeat the entire point of a head-only scan, so we abort instead of
streaming it.

Tested exclusively against the in-process stdlib http.server fixture
(tests/fixture_server.py) — this module never fetches a real remote host.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

from .gguf import GGUFError, GGUFInfo, read_gguf

INITIAL_BUDGET = 2 * 1024 * 1024  # 2 MiB
ESCALATED_BUDGET = 16 * 1024 * 1024  # 16 MiB
USER_AGENT = "kleidi-advisor/0.1 head-only-gguf-scanner"


class RemoteScanError(Exception):
    """Any failure fetching or parsing a remote GGUF head. Always a clear message."""


@dataclass
class RemoteScanResult:
    info: GGUFInfo
    bytes_fetched: int


def _fetch_range(url: str, budget: int) -> Tuple[int, bytes]:
    """GET `url` with `Range: bytes=0-<budget-1>`. Returns (status, body)."""
    req = urllib.request.Request(
        url, headers={"Range": f"bytes=0-{budget - 1}", "User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req) as resp:
        status = resp.status
        body = resp.read(budget)
    return status, body


def fetch_and_read(url: str) -> RemoteScanResult:
    """Fetch just enough of `url` to classify it, escalating once on truncation."""
    last_error: Optional[GGUFError] = None

    for budget in (INITIAL_BUDGET, ESCALATED_BUDGET):
        try:
            status, body = _fetch_range(url, budget)
        except OSError as exc:
            raise RemoteScanError(f"could not fetch {url}: {exc}") from exc

        if status == 200:
            raise RemoteScanError(
                f"{url}: server ignored the Range request (got 200, not 206) — "
                "refusing to stream the whole file for a head-only scan"
            )
        if status != 206:
            raise RemoteScanError(f"{url}: unexpected HTTP status {status}")

        try:
            info = read_gguf(body)
            return RemoteScanResult(info=info, bytes_fetched=len(body))
        except GGUFError as exc:
            if exc.offset <= len(body):
                # The parser stopped within bytes we actually have (bad
                # magic, unknown type, ...) — a bigger budget won't fix it.
                raise RemoteScanError(f"{url}: {exc}") from exc
            last_error = exc  # offset lands past the buffer end: truncated, escalate

    raise RemoteScanError(
        f"{url}: GGUF metadata did not fit in {ESCALATED_BUDGET} bytes ({last_error})"
    )
