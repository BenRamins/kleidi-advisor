"""On-device verify mode — REFERENCE.md §8 (D-13).

The static scan verdict is a prediction; this cross-checks it against
llama.cpp's real load log so the classification table is falsifiable.

Rewritten 2026-08-14 against build 1692f9e50 (b10431). The previous
substring list ("repack", "kleidi", ...) was measured to fire for *both*
Q4_0 and Q4_K_M and would have produced false AGREEs — `repack:` lines are
emitted by ggml's own aarch64 path too, and the line "cannot be used with
preferred buffer type CPU_KLEIDIAI, using CPU instead" contains the token
CPU_KLEIDIAI while meaning the opposite. The signal is which *model buffer*
the tensors landed in, which only appears under `-v`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .binaries import run_binary
from .compat import FALLBACK_GENERIC, NOT_KLEIDIAI_PATH, OK_KLEIDIAI

AGREE = "AGREE"
DISAGREE = "DISAGREE"
INCONCLUSIVE = "INCONCLUSIVE"

# Case-insensitive substrings. "buffer size" is load-bearing: it is what
# excludes the "cannot be used with preferred buffer type CPU_KLEIDIAI" decoy
# that both formats emit. Measured on b10431; confirm again after a llama.cpp
# bump via REPRODUCE.md step 4.
KLEIDIAI_BUFFER_MARKER = "cpu_kleidiai model buffer size"
REPACK_BUFFER_MARKER = "cpu_repack model buffer size"
ANY_BUFFER_MARKER = "model buffer size"

# The buffer lines are verbose-only, and llama-cli with no prompt goes
# interactive and hangs — so verify drives llama-bench with a minimal
# 8-token prompt, no generation, one repeat.
VERIFY_BINARY = "llama-bench"
VERIFY_ARGS = ["-p", "8", "-n", "0", "-r", "1", "-v"]


@dataclass
class VerifySignals:
    """What the load log actually showed, independent of any verdict."""

    kleidiai_buffer: bool
    repack_buffer: bool
    any_buffer_line: bool

    def as_dict(self) -> dict:
        return {
            "cpu_kleidiai_buffer": self.kleidiai_buffer,
            "cpu_repack_buffer": self.repack_buffer,
            "buffer_lines_seen": self.any_buffer_line,
        }

    def describe(self) -> str:
        if not self.any_buffer_line:
            return "no 'model buffer size' line in the log (was -v passed?)"
        parts: List[str] = [
            f"CPU_KLEIDIAI buffer: {'present' if self.kleidiai_buffer else 'absent'}",
            f"CPU_REPACK buffer: {'present' if self.repack_buffer else 'absent'}",
        ]
        return ", ".join(parts)


@dataclass
class VerifyResult:
    outcome: str
    signals: VerifySignals
    static_verdict: str


def detect_signals(text: str) -> VerifySignals:
    lowered = text.lower()
    return VerifySignals(
        kleidiai_buffer=KLEIDIAI_BUFFER_MARKER in lowered,
        repack_buffer=REPACK_BUFFER_MARKER in lowered,
        any_buffer_line=ANY_BUFFER_MARKER in lowered,
    )


def classify_verify_outcome(static_verdict: str, signals: VerifySignals) -> str:
    """The seven outcome rules from REFERENCE.md §8, in the order written."""
    # 1. The log carried no buffer-type line at all — nothing was observed, so
    #    neither presence nor absence of CPU_KLEIDIAI means anything here.
    if not signals.any_buffer_line:
        return INCONCLUSIVE

    if signals.kleidiai_buffer:
        if static_verdict == OK_KLEIDIAI:
            return AGREE  # 2
        if static_verdict in (NOT_KLEIDIAI_PATH, FALLBACK_GENERIC):
            return DISAGREE  # 3
        return INCONCLUSIVE  # 7

    # From here the log had buffer lines but no CPU_KLEIDIAI buffer, so absence
    # is evidence — the whole reason -v is mandatory.
    if static_verdict == OK_KLEIDIAI:
        return DISAGREE  # 4
    if static_verdict == NOT_KLEIDIAI_PATH:
        return AGREE if signals.repack_buffer else DISAGREE  # 5
    if static_verdict == FALLBACK_GENERIC:
        return DISAGREE if signals.repack_buffer else AGREE  # 6
    return INCONCLUSIVE  # 7


def run_verify(gguf_path: Path, static_verdict: str, llama_bench_path: Path) -> VerifyResult:
    """Run `llama-bench -m <gguf> -p 8 -n 0 -r 1 -v`, capturing stdout and
    stderr both — llama.cpp logs dispatch info to stderr, but D-13 says not to
    rely on that.
    """
    result = run_binary(
        llama_bench_path,
        ["-m", str(gguf_path), *VERIFY_ARGS],
        capture_output=True,
        text=True,
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    signals = detect_signals(combined)
    outcome = classify_verify_outcome(static_verdict, signals)
    return VerifyResult(outcome=outcome, signals=signals, static_verdict=static_verdict)
