"""Fix pipeline — Spec F2. Requantizes an F16/BF16 source to Q4_0 (optionally
imatrix-guided) so the model reaches Arm's KleidiAI repack path. We requant
to the format that reaches Arm's kernels; the kernels themselves are Arm's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .binaries import BinaryTimeout, resolve_binaries, run_binary
from .gguf import compute_dominant_type, read_gguf

REQUIRED_BINARIES = ["llama-imatrix", "llama-quantize", "llama-cli"]

# `-st` (single-turn) makes llama-cli answer the prompt and exit. Without it,
# a `-p ... -n ...` invocation finishes generating and then opens an
# interactive chat turn, blocking on stdin forever — confirmed hanging on the
# box. run_binary's DEVNULL stdin already prevents the block; this flag means
# we are not relying on that alone, on a build where the flag might be spelled
# differently or the behaviour might change.
SMOKE_ARGS = ["-p", "The", "-n", "8", "-st"]

# A generation of 8 tokens that has not finished in five minutes is stuck, not
# slow — even on a small CPU-only instance. Bounded so a stall surfaces as a
# named failure instead of an unattended run that never returns.
SMOKE_TIMEOUT_SECONDS = 300

# D-10: requanting a K-quant directly would double-quantize and poison the
# quality story, so fix only accepts an F16/BF16 source.
_ACCEPTABLE_SOURCE_TYPES = {"F16", "BF16"}


class FixInputError(Exception):
    """A usage-level problem with fix's input: D-10 refusal, missing --calib, ..."""


class FixStageError(Exception):
    """One pipeline stage exited nonzero. Always names the stage."""

    def __init__(self, stage: str, returncode: int, stderr: str):
        self.stage = stage
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"{stage} failed (exit {returncode}): {stderr.strip() or '(no stderr)'}")


class FixStageTimeout(FixStageError):
    """A stage stalled rather than failed.

    Subclasses FixStageError so it keeps the same exit code (4, subprocess
    stage failure) while reporting a stall as a stall — the two have very
    different causes and the message must not blur them.
    """

    def __init__(self, stage: str, timeout: float, hint: str):
        self.stage = stage
        self.returncode = None
        self.stderr = ""
        Exception.__init__(
            self, f"{stage} timed out after {timeout:g}s and was killed. {hint}"
        )


@dataclass
class FixResult:
    output_path: Path
    used_imatrix: bool
    warnings: List[str] = field(default_factory=list)


def _check_source_is_f16_or_bf16(source: Path) -> None:
    info = read_gguf(source)
    dominant = compute_dominant_type(info)
    if dominant.ggml_type_name not in _ACCEPTABLE_SOURCE_TYPES:
        raise FixInputError(
            f"{source}: fix needs an F16 or BF16 source GGUF, got {dominant.ggml_type_name} "
            "(requanting a K-quant directly would double-quantize it)"
        )


def run_fix(
    source: Path,
    calib: Optional[Path],
    output: Path,
    *,
    no_imatrix: bool = False,
    llama_bin_dir: Optional[str] = None,
    chunks: Optional[int] = None,
) -> FixResult:
    """Spec F2 rule 2 happy path, rules 3-4 for --no-imatrix and D-10 refusal.
    D-15: --chunks caps llama-imatrix's corpus pass so an 8-core box doesn't
    spend hours on the full calibration corpus.
    """
    _check_source_is_f16_or_bf16(source)

    if not no_imatrix and calib is None:
        raise FixInputError("fix needs --calib unless --no-imatrix is given")

    binaries = resolve_binaries(REQUIRED_BINARIES, llama_bin_dir=llama_bin_dir)
    warnings: List[str] = []
    imatrix_path = output.parent / f"{output.stem}.imatrix"
    used_imatrix = not no_imatrix

    if used_imatrix:
        imatrix_args = ["-m", str(source), "-f", str(calib), "-o", str(imatrix_path)]
        if chunks is not None:
            imatrix_args += ["--chunks", str(chunks)]
        result = run_binary(
            binaries["llama-imatrix"],
            imatrix_args,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise FixStageError("llama-imatrix", result.returncode, result.stderr)
    else:
        warnings.append("WARN: --no-imatrix given; quantizing without an importance matrix.")

    quantize_args = ["--imatrix", str(imatrix_path)] if used_imatrix else []
    result = run_binary(
        binaries["llama-quantize"],
        [*quantize_args, str(source), str(output), "Q4_0"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FixStageError("llama-quantize", result.returncode, result.stderr)

    try:
        result = run_binary(
            binaries["llama-cli"],
            ["-m", str(output), *SMOKE_ARGS],
            capture_output=True,
            text=True,
            timeout=SMOKE_TIMEOUT_SECONDS,
        )
    except BinaryTimeout:
        raise FixStageTimeout(
            "llama-cli smoke generation",
            SMOKE_TIMEOUT_SECONDS,
            f"The quantized model was written to {output} — only the smoke check stalled, so "
            "the artifact is probably fine; verify it with `kleidi-advisor scan` before use.",
        ) from None
    if result.returncode != 0:
        raise FixStageError("llama-cli", result.returncode, result.stderr)

    return FixResult(output_path=output, used_imatrix=used_imatrix, warnings=warnings)
