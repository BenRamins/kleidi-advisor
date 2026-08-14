"""Binary resolution and invocation for llama.cpp tools.

D-06 resolution order: `--llama-bin-dir` flag > `$LLAMA_BIN_DIR` > PATH.
Whichever source is configured is authoritative for every requested name;
missing binaries are always reported together, never one at a time.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class BinaryResolutionError(Exception):
    """Raised with every missing binary named in one message."""


class BinaryTimeout(Exception):
    """A binary was killed after exceeding its timeout.

    Carries the name and the limit so the caller can build a message that says
    which stage stalled, rather than surfacing a bare TimeoutExpired.
    """

    def __init__(self, name: str, timeout: float):
        self.name = name
        self.timeout = timeout
        super().__init__(f"{name} exceeded its {timeout:g}s timeout and was killed")


def resolve_binaries(names: Iterable[str], *, llama_bin_dir: Optional[str] = None) -> Dict[str, Path]:
    """Resolve each name in `names` to an executable path.

    `llama_bin_dir` (the --llama-bin-dir flag) takes precedence over the
    `LLAMA_BIN_DIR` environment variable, which takes precedence over PATH.
    Whichever one is configured is checked for every requested name; on any
    misses, one BinaryResolutionError names all of them.
    """
    names = list(names)
    source = llama_bin_dir or os.environ.get("LLAMA_BIN_DIR")

    found: Dict[str, Optional[Path]] = {}
    if source is not None:
        base = Path(source)
        for name in names:
            candidate = base / name
            if not candidate.exists():
                candidate = base / f"{name}.exe"  # dev-time convenience on Windows
            found[name] = candidate if candidate.exists() else None
    else:
        for name in names:
            which = shutil.which(name)
            found[name] = Path(which) if which else None

    missing = [name for name, path in found.items() if path is None]
    if missing:
        where = source if source is not None else "PATH"
        raise BinaryResolutionError(f"missing required binaries in {where}: {', '.join(missing)}")

    return {name: path for name, path in found.items() if path is not None}


def _shebang_interpreter(path: Path) -> Optional[str]:
    """Return the interpreter named on a `#!` line, or None for a native binary.

    Real llama.cpp binaries are ELF executables with no shebang, so this is a
    no-op for them. The REFERENCE.md §9 test stubs are POSIX shell scripts;
    their OS-level loader (Linux) already honours the shebang directly, but
    some dev/test loaders (e.g. Windows) cannot exec a shebang file, so
    `run_binary` resolves the interpreter itself rather than relying on that.
    """
    try:
        with open(path, "rb") as f:
            first_line = f.readline(256)
    except OSError:
        return None
    if not first_line.startswith(b"#!"):
        return None
    interpreter_line = first_line[2:].decode("utf-8", "replace").strip()
    if not interpreter_line:
        return None
    interpreter_name = Path(interpreter_line.split()[0]).name
    return shutil.which(interpreter_name) or interpreter_name


def run_binary(
    binary_path: Path,
    args: List[str],
    *,
    timeout: Optional[float] = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Invoke a resolved binary. Every fix/bench/verify subprocess call goes
    through this one adapter, so shebang-bridging and stdin handling live in
    exactly one place.

    **stdin defaults to DEVNULL.** Several llama.cpp tools drop into an
    interactive prompt when they are given work but no terminal instruction to
    stop — `llama-cli -p ... -n ...` without `-st` is the known case — and an
    inherited stdin means they then block on a read that will never return,
    hanging the caller forever with no output. Handing every child an
    already-closed stdin turns that class of hang into an immediate EOF. A
    caller that genuinely needs to write to a child passes `stdin=` explicitly.

    `timeout` is opt-in per call rather than a default, because the legitimate
    runtimes here differ by orders of magnitude: an imatrix or perplexity pass
    can run for half an hour, while a smoke generation that has not finished in
    minutes is stuck. On expiry the child is killed and `BinaryTimeout` is
    raised naming the binary.

    Boundary worth knowing: the timeout kills the process we spawned. A child
    that forks helpers of its own can leave one holding the stdout pipe open,
    and the post-kill drain then waits on it. Every llama.cpp binary here is
    invoked directly as a native executable, so the process we spawn is the
    process that does the work — but a shell wrapper in between would
    reintroduce that gap.
    """
    argv = [str(binary_path), *args]
    interpreter = _shebang_interpreter(binary_path)
    if interpreter is not None:
        argv = [interpreter, *argv]
    # `input=` is subprocess's own way of supplying stdin and is mutually
    # exclusive with `stdin=`; defaulting unconditionally would turn every such
    # call into a ValueError.
    if "stdin" not in kwargs and "input" not in kwargs:
        kwargs["stdin"] = subprocess.DEVNULL
    try:
        return subprocess.run(argv, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise BinaryTimeout(binary_path.name, timeout) from exc
