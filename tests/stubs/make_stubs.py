"""Writes REFERENCE.md §9 stub binaries into a tmp bin dir for offline testing.

Every stub is byte-identical except for its `# stub: <name>` comment line;
behaviour at run time is driven entirely by the KA_STUB_* environment
variables the caller sets before invoking one (REFERENCE.md §9).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

STUB_NAMES: List[str] = [
    "llama-imatrix", "llama-quantize", "llama-cli", "llama-bench", "llama-perplexity",
]

_STUB_TEMPLATE = (
    "#!/bin/sh\n"
    "# stub: __NAME__\n"
    "printf '%s\\n' \"$0 $*\" >> \"$KA_STUB_LOG\"\n"
    "[ -n \"$KA_STUB_STDOUT\" ] && cat \"$KA_STUB_STDOUT\"\n"
    "[ -n \"$KA_STUB_STDERR\" ] && cat \"$KA_STUB_STDERR\" >&2\n"
    "exit \"${KA_STUB_EXIT:-0}\"\n"
)


def make_stubs(bin_dir: Path) -> Path:
    """Write all five REFERENCE.md §9 stub binaries into `bin_dir`, mode 0o755."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in STUB_NAMES:
        path = bin_dir / name
        path.write_bytes(_STUB_TEMPLATE.replace("__NAME__", name).encode("utf-8"))
        os.chmod(path, 0o755)
    return bin_dir
