"""Perplexity parsing and quality gate — Spec F3 rules 2-3, REFERENCE.md §7.

D-14: a throughput figure is never printed without its quality cost next to
it. This module is the honesty check that pairing depends on.
"""

from __future__ import annotations

import re

_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9]+\.?[0-9]*)")


class PPLParseError(Exception):
    """llama-perplexity output didn't contain a parseable PPL line."""


class QualityGateError(Exception):
    """The ppl delta against baseline exceeded --max-delta. Always names both numbers."""

    def __init__(self, candidate: float, baseline: float, max_delta: float):
        self.candidate = candidate
        self.baseline = baseline
        self.delta = abs(candidate - baseline)
        self.max_delta = max_delta
        super().__init__(
            f"perplexity gate failed: candidate={candidate} baseline={baseline} "
            f"delta={self.delta:.4f} exceeds max_delta={max_delta}"
        )


def parse_ppl(raw: str) -> float:
    """REFERENCE.md §7: 'Final estimate: PPL = <float>', tolerating a
    '+/- <err>' suffix, which the regex simply never captures."""
    match = _PPL_RE.search(raw)
    if match is None:
        last_lines = raw.strip().splitlines()[-5:]
        raise PPLParseError(
            "could not find 'Final estimate: PPL = <float>' in llama-perplexity output; "
            f"last lines seen: {last_lines}"
        )
    return float(match.group(1))


def check_gate(candidate: float, baseline: float, max_delta: float) -> None:
    """Spec F3 rule 3. Raises QualityGateError (both numbers included) on FAIL."""
    if abs(candidate - baseline) > max_delta:
        raise QualityGateError(candidate, baseline, max_delta)
