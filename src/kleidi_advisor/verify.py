"""On-device verify mode — REFERENCE.md §8 (D-13).

The static scan verdict is a prediction; this cross-checks it against
llama-cli's real load log so the classification table is falsifiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .binaries import run_binary
from .compat import FALLBACK_GENERIC, OK_KLEIDIAI

AGREE = "AGREE"
DISAGREE = "DISAGREE"
INCONCLUSIVE = "INCONCLUSIVE"

# Unconfirmed as of 2026-08-13; confirm via RUNBOOK step 3. These strings
# drift between llama.cpp versions — that drift is exactly why INCONCLUSIVE
# is a first-class, non-error outcome rather than a forced AGREE/DISAGREE.
VERIFY_PATTERNS: List[str] = [
    "repack", "kleidi", "aarch64", "extra_buffer_type", "i8mm", "sve",
]


@dataclass
class VerifyResult:
    outcome: str
    matched_patterns: List[str]
    static_verdict: str


def _matched_patterns(text: str) -> List[str]:
    lowered = text.lower()
    return [pattern for pattern in VERIFY_PATTERNS if pattern in lowered]


def classify_verify_outcome(static_verdict: str, matched_patterns: List[str]) -> str:
    """The five outcome rules from REFERENCE.md §8, implemented as written."""
    hit = bool(matched_patterns)
    if hit and static_verdict == OK_KLEIDIAI:
        return AGREE
    if hit and static_verdict == FALLBACK_GENERIC:
        return DISAGREE
    if not hit and static_verdict == FALLBACK_GENERIC:
        return AGREE
    if not hit and static_verdict == OK_KLEIDIAI:
        return INCONCLUSIVE
    return INCONCLUSIVE


def run_verify(gguf_path: Path, static_verdict: str, llama_cli_path: Path) -> VerifyResult:
    """Run `llama-cli -m <gguf> -n 1`, capturing stdout and stderr both —
    llama.cpp logs dispatch info to stderr, but D-13 says not to rely on that.
    """
    result = run_binary(
        llama_cli_path, ["-m", str(gguf_path), "-n", "1"], capture_output=True, text=True
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    matched = _matched_patterns(combined)
    outcome = classify_verify_outcome(static_verdict, matched)
    return VerifyResult(outcome=outcome, matched_patterns=matched, static_verdict=static_verdict)
