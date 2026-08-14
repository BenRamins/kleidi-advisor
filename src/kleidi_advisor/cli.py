"""kleidi-advisor command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import __version__
from .audit import (
    DEFAULT_DELAY_SECONDS,
    bytes_line,
    parse_list_file,
    run_audit,
    summary_line,
    to_json,
    to_markdown,
)
from .bench import BenchParseError, run_bench
from .binaries import BinaryResolutionError, resolve_binaries
from .compat import KLEIDIAI_MISS_VERDICTS, classify
from .fix import FixInputError, FixStageError, run_fix
from .gguf import DominantType, GGUFError, compute_dominant_type, read_gguf
from .ppl import PPLParseError, QualityGateError, check_gate
from .remote import RemoteScanError, fetch_and_read
from .report import load_results, render_markdown, render_plot
from .verify import VERIFY_BINARY, run_verify


def _not_implemented(args: argparse.Namespace) -> int:
    print("not implemented", file=sys.stderr)
    return 2


def _scan_payload(source: str, dominant: DominantType) -> Dict[str, Any]:
    result = classify(dominant.ggml_type_id)
    return {
        "source": source,
        "dominant_type": dominant.ggml_type_name,
        "file_type_kv": dominant.file_type_kv,
        "verdict": result.verdict,
        "reason": result.reason,
        "next": result.next,
    }


def _print_scan_text(payload: Dict[str, Any], dominant: DominantType) -> None:
    lines = [
        f"verdict: {payload['verdict']}",
        f"source: {payload['source']}",
        f"dominant type: {payload['dominant_type']}",
    ]
    if dominant.file_type_kv is not None:
        lines.append(f"file_type kv: {dominant.file_type_kv} ({dominant.file_type_name or 'unmapped'})")
    lines.append(f"reason: {payload['reason']}")
    if payload["next"]:
        lines.append(f"next: {payload['next']}")
    verify = payload.get("verify")
    if verify is not None:
        lines.append(f"verify: {verify['outcome']} ({verify['detail']})")
    print("\n".join(lines))


def _cmd_scan(args: argparse.Namespace) -> int:
    if args.url and args.verify:
        print(
            "error: --verify needs a local file, not --url (on-device verify requires a local file)",
            file=sys.stderr,
        )
        return 2
    if not args.url and not args.source:
        print("error: scan needs a source path or --url", file=sys.stderr)
        return 2

    source_label = args.url if args.url else args.source

    try:
        info = fetch_and_read(args.url).info if args.url else read_gguf(args.source)
    except GGUFError as exc:
        print(f"error: {source_label}: {exc}", file=sys.stderr)
        return 2
    except RemoteScanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot read {source_label}: {exc}", file=sys.stderr)
        return 2

    dominant = compute_dominant_type(info)
    payload = _scan_payload(source_label, dominant)
    payload["verify"] = None

    if args.verify:
        try:
            resolved = resolve_binaries([VERIFY_BINARY], llama_bin_dir=args.llama_bin_dir)
        except BinaryResolutionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        verify_result = run_verify(Path(args.source), payload["verdict"], resolved[VERIFY_BINARY])
        payload["verify"] = {
            "outcome": verify_result.outcome,
            "signals": verify_result.signals.as_dict(),
            "detail": verify_result.signals.describe(),
        }

    if args.json:
        print(json.dumps(payload))
    else:
        _print_scan_text(payload, dominant)

    if args.fail_on_miss and payload["verdict"] in KLEIDIAI_MISS_VERDICTS:
        return 3
    return 0


def _cmd_fix(args: argparse.Namespace) -> int:
    source = Path(args.gguf)
    output = Path(args.output)
    calib = Path(args.calib) if args.calib else None

    try:
        result = run_fix(
            source,
            calib,
            output,
            no_imatrix=args.no_imatrix,
            llama_bin_dir=args.llama_bin_dir,
            chunks=args.chunks,
        )
    except FixInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (GGUFError, OSError) as exc:
        print(f"error: {source}: {exc}", file=sys.stderr)
        return 2
    except BinaryResolutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FixStageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4

    for warning in result.warnings:
        print(warning, file=sys.stderr)
    print(f"artifact: {result.output_path}")
    print(f"now run: kleidi-advisor scan {result.output_path}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    if args.perplexity and not args.calib:
        print("error: --perplexity needs --calib", file=sys.stderr)
        return 2
    if args.gate and not args.perplexity:
        print("error: --gate needs --perplexity (nothing to compare)", file=sys.stderr)
        return 2

    try:
        result = run_bench(
            Path(args.gguf),
            threads=args.threads,
            tag=args.tag,
            repeats=args.repeats,
            results_dir=Path(args.results_dir),
            llama_bin_dir=args.llama_bin_dir,
            perplexity=args.perplexity,
            calib=Path(args.calib) if args.calib else None,
            chunks=args.chunks,
            instance=args.instance,
        )
    except BinaryResolutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (BenchParseError, PPLParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4

    print(f"wrote {result.results_path}")

    if args.gate:
        try:
            baseline_data = json.loads(Path(args.gate).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"error: cannot read {args.gate}: {exc}", file=sys.stderr)
            return 2
        baseline_ppl = (baseline_data.get("ppl") or {}).get("value")
        if baseline_ppl is None:
            print(f"error: {args.gate}: baseline has no ppl.value to gate against", file=sys.stderr)
            return 2
        try:
            check_gate(result.ppl_value, baseline_ppl, args.max_delta)
        except QualityGateError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 5
        print(f"gate: PASS (candidate={result.ppl_value} baseline={baseline_ppl} max_delta={args.max_delta})")

    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    entries = load_results(Path(args.results_dir))
    rendered = render_markdown(entries, instance=args.instance)
    Path(args.output).write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")

    if args.plot:
        render_plot(entries, Path(args.plot))

    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    try:
        entries = parse_list_file(args.list)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows = run_audit(entries, delay=args.delay)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(to_json(rows), indent=2), encoding="utf-8")
    if args.md:
        Path(args.md).write_text(to_markdown(rows), encoding="utf-8")

    print(summary_line(rows))
    print(bytes_line(rows))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kleidi-advisor",
        description=(
            "Diagnose, measure, and fix the KleidiAI kernel-path miss in GGUF models on Arm. "
            "We detect the miss and measure the ecosystem; the kernels are Arm's."
        ),
    )
    parser.add_argument("--version", action="version", version=f"kleidi-advisor {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    scan = subparsers.add_parser(
        "scan",
        help="Classify a GGUF file's Arm KleidiAI kernel-path compatibility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: kleidi-advisor scan model.gguf --fail-on-miss",
    )
    scan.add_argument("source", nargs="?", default=None, help="Path to a local GGUF file.")
    scan.add_argument("--url", default=None, help="HTTP(S) URL to a GGUF file; fetched head-only.")
    scan.add_argument("--json", action="store_true", help="Emit a single JSON object instead of text.")
    scan.add_argument(
        "--fail-on-miss",
        action="store_true",
        help="Exit 3 when the model misses KleidiAI (NOT_KLEIDIAI_PATH or FALLBACK_GENERIC).",
    )
    scan.add_argument(
        "--verify",
        action="store_true",
        help="Cross-check the verdict against llama-bench -v's load log (local files only).",
    )
    scan.add_argument("--llama-bin-dir", default=None, help="Directory containing llama.cpp binaries.")
    scan.set_defaults(func=_cmd_scan)

    fix = subparsers.add_parser(
        "fix",
        help="Requantize a model so it reaches Arm's KleidiAI repack path.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example: kleidi-advisor fix source-f16.gguf --calib wiki.train.raw -o model-q4_0.gguf"
        ),
    )
    fix.add_argument("gguf", help="Path to the F16/BF16 source GGUF.")
    fix.add_argument("--calib", default=None, help="Calibration corpus for llama-imatrix.")
    fix.add_argument("-o", "--output", required=True, help="Output GGUF path.")
    fix.add_argument(
        "--no-imatrix", action="store_true", help="Skip the imatrix step (plain Q4_0, no calib needed)."
    )
    fix.add_argument("--llama-bin-dir", default=None, help="Directory containing llama.cpp binaries.")
    fix.add_argument(
        "--chunks", type=int, default=None, help="Cap llama-imatrix's corpus pass (D-15)."
    )
    fix.set_defaults(func=_cmd_fix)

    bench = subparsers.add_parser(
        "bench",
        help="Run llama-bench and record median/stdev throughput.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: kleidi-advisor bench model-q4_0.gguf --threads 16 --tag fixed",
    )
    bench.add_argument("gguf", help="Path to the GGUF file to benchmark.")
    bench.add_argument(
        "--threads", type=int, required=True, help="Thread count (physical cores; no autodetect)."
    )
    bench.add_argument("--tag", default="run", help="Tag in the results filename, e.g. baseline/fixed.")
    bench.add_argument("-r", "--repeats", type=int, default=5, help="Repeats per metric.")
    bench.add_argument("--results-dir", default="results", help="Directory to write the results JSON into.")
    bench.add_argument("--llama-bin-dir", default=None, help="Directory containing llama.cpp binaries.")
    bench.add_argument(
        "--perplexity", action="store_true", help="Also run llama-perplexity and store the result."
    )
    bench.add_argument("--calib", default=None, help="Corpus file for --perplexity.")
    bench.add_argument("--gate", default=None, help="Baseline results JSON to gate this run's ppl against.")
    bench.add_argument(
        "--max-delta", type=float, default=0.3, help="Max allowed |candidate ppl - baseline ppl|."
    )
    bench.add_argument(
        "--chunks", type=int, default=None, help="Cap llama-perplexity's corpus pass (D-15)."
    )
    bench.add_argument(
        "--instance",
        default=None,
        help="Machine label recorded in the results file, e.g. 'Azure Standard_E8ps_v6 (Cobalt 100, Neoverse N2), 8 threads'.",
    )
    bench.set_defaults(func=_cmd_bench)

    report = subparsers.add_parser(
        "report",
        help="Render a results table and plot from bench output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: kleidi-advisor report --instance c8g.8xlarge --plot results/plot.png",
    )
    report.add_argument("--results-dir", default="results", help="Directory of bench results JSON files.")
    report.add_argument("-o", "--output", default="RESULTS.md", help="Path to write the markdown report to.")
    report.add_argument("--plot", default=None, help="Path to write a grouped-bar PNG plot to.")
    report.add_argument("--instance", default=None, help="Instance type label, e.g. c8g.8xlarge.")
    report.set_defaults(func=_cmd_report)

    audit = subparsers.add_parser(
        "audit",
        help="Scan a list of remote GGUF URLs and summarize kernel-path misses across the ecosystem.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: kleidi-advisor audit --list data/hf-top-gguf.txt --md AUDIT.md",
    )
    audit.add_argument("--list", required=True, help="Path to a '<label> <url>' per line file.")
    audit.add_argument("--out", default=None, help="Write JSON results to this path.")
    audit.add_argument("--md", default=None, help="Write a markdown table to this path.")
    audit.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY_SECONDS, help="Seconds between requests."
    )
    audit.set_defaults(func=_cmd_audit)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func: Optional[Callable[[argparse.Namespace], int]] = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
