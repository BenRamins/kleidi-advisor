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


def run_binary(binary_path: Path, args: List[str], **kwargs) -> subprocess.CompletedProcess:
    """Invoke a resolved binary. Every fix/bench/verify subprocess call goes
    through this one adapter so shebang-bridging lives in exactly one place.
    """
    argv = [str(binary_path), *args]
    interpreter = _shebang_interpreter(binary_path)
    if interpreter is not None:
        argv = [interpreter, *argv]
    return subprocess.run(argv, **kwargs)
