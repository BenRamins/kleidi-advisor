#!/bin/sh
# End-to-end smoke test for the whole offline loop (F5.S3) — proves scan,
# scan --url, scan --verify, audit, fix, bench, and report all compose
# correctly against stubs and the in-process fixture server. No real
# network, no real llama.cpp binary, no real model — exactly the same
# offline guarantees the rest of this build was verified against.
#
# Usage: bash scripts/smoke.sh   (needs .venv active / kleidi_advisor importable)
set -eu

cd "$(dirname "$0")/.."

python3 - <<'PYEOF'
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, "tests")

from fixture_server import FixtureHTTPServer  # noqa: E402
from gguf_writer import write_gguf  # noqa: E402
from stubs.make_stubs import make_stubs  # noqa: E402

from kleidi_advisor.cli import main  # noqa: E402


def fail(message):
    print(f"SMOKE FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def check(condition, message):
    if not condition:
        fail(message)


def call(args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(args)
    return code, buf.getvalue()


work = Path(tempfile.mkdtemp(prefix="kleidi-advisor-smoke-"))
results_dir = work / "results"
results_dir.mkdir(parents=True)

bin_dir = make_stubs(work / "bin")
os.environ["KA_STUB_LOG"] = str(work / "calls.log")
for var in ("KA_STUB_EXIT", "KA_STUB_STDOUT", "KA_STUB_STDERR"):
    os.environ.pop(var, None)

server = FixtureHTTPServer()
server.start()

try:
    # A q4_K fixture: NOT_KLEIDIAI_PATH, the miss this whole tool exists to find.
    q4k_path = work / "q4k.gguf"
    write_gguf(
        q4k_path,
        version=3,
        kvs={"general.architecture": "llama", "general.file_type": 15},
        tensors=[("blk.0.attn_q.weight", [4096, 4096], 12)],
    )

    code, _out = call(["scan", str(q4k_path), "--fail-on-miss"])
    check(code == 3, f"scan --fail-on-miss expected exit 3, got {code}")

    code, out = call(["scan", str(q4k_path), "--json"])
    check(code == 0, f"scan --json expected exit 0, got {code}")
    local_verdict = json.loads(out)["verdict"]

    server.routes["/q4k.gguf"] = q4k_path.read_bytes()
    code, out = call(["scan", "--url", server.url_for("/q4k.gguf"), "--json"])
    check(code == 0, f"scan --url expected exit 0, got {code}")
    remote_verdict = json.loads(out)["verdict"]
    check(
        remote_verdict == local_verdict,
        f"scan --url verdict {remote_verdict!r} != on-disk verdict {local_verdict!r}",
    )

    code, out = call(["scan", str(q4k_path), "--verify", "--llama-bin-dir", str(bin_dir)])
    check(code == 0, f"scan --verify expected exit 0, got {code}")
    check("verify:" in out, "scan --verify output missing a 'verify:' line")

    server.routes["/a.gguf"] = q4k_path.read_bytes()
    server.routes["/b.gguf"] = q4k_path.read_bytes()
    list_path = work / "list.txt"
    list_path.write_text(
        "\n".join(
            [
                f"smoke-a {server.url_for('/a.gguf')}",
                f"smoke-b {server.url_for('/b.gguf')}",
                f"smoke-c {server.url_for('/missing.gguf')}",
            ]
        )
    )
    code, _out = call(
        [
            "audit", "--list", str(list_path),
            "--out", str(work / "audit.json"), "--md", str(work / "AUDIT.md"), "--delay", "0",
        ]
    )
    check(code == 0, f"audit expected exit 0, got {code}")

    f16_path = work / "source-f16.gguf"
    write_gguf(
        f16_path,
        version=3,
        kvs={"general.architecture": "llama", "general.file_type": 1},
        tensors=[("blk.0.attn_q.weight", [4096, 4096], 1)],
    )
    calib_path = work / "calib.txt"
    calib_path.write_text("hello world\n")
    fixed_path = work / "fixed-q4_0.gguf"
    code, _out = call(
        [
            "fix", str(f16_path), "--calib", str(calib_path), "-o", str(fixed_path),
            "--llama-bin-dir", str(bin_dir),
        ]
    )
    check(code == 0, f"fix expected exit 0, got {code}")

    os.environ["KA_STUB_STDOUT"] = str(Path("tests/data/llama-bench-baseline.json").resolve())
    code, _out = call(
        [
            "bench", str(q4k_path), "--threads", "4", "--tag", "baseline",
            "--results-dir", str(results_dir), "--llama-bin-dir", str(bin_dir),
        ]
    )
    check(code == 0, f"bench baseline expected exit 0, got {code}")

    os.environ["KA_STUB_STDOUT"] = str(Path("tests/data/llama-bench-fixed.json").resolve())
    code, _out = call(
        [
            "bench", str(fixed_path), "--threads", "4", "--tag", "fixed",
            "--results-dir", str(results_dir), "--llama-bin-dir", str(bin_dir),
        ]
    )
    check(code == 0, f"bench fixed expected exit 0, got {code}")
    os.environ.pop("KA_STUB_STDOUT", None)

    results_md = work / "RESULTS.md"
    code, _out = call(
        ["report", "--results-dir", str(results_dir), "-o", str(results_md), "--instance", "smoke-test"]
    )
    check(code == 0, f"report expected exit 0, got {code}")

    text = results_md.read_text()
    table_lines = [line for line in text.splitlines() if line.startswith("|")]
    check(len(table_lines) == 4, f"expected 2 header + 2 data table rows, got {len(table_lines)}: {table_lines}")
    check("ppl" in text.splitlines()[0], "RESULTS.md's first line doesn't look like a headline line")
    check(
        "Speedup comes from Arm's KleidiAI kernels; this tool detects the miss and measures the delta." in text,
        "RESULTS.md missing the attribution sentence",
    )
finally:
    server.stop()
    shutil.rmtree(work, ignore_errors=True)

print("SMOKE OK")
PYEOF
