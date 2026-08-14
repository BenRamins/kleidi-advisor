# kleidi-advisor

Diagnose, measure, and fix the KleidiAI kernel-path miss in GGUF models on Arm.

Built for the Arm AI Optimization Challenge, Track 2 (Cloud AI). `kleidi-advisor` is a CLI —
`scan`, `fix`, `bench`, `report`, `audit` — that tells you whether a GGUF model actually reaches
Arm's KleidiAI kernels, measures how often the wider ecosystem misses that path, and gives you a
reproducible way to fix a model that does.

## 1. The Finding

**TODO(box)**: `N of 20` widely-published GGUF quantizations audited in this repo fall back to
generic kernels instead of reaching Arm's KleidiAI Q4_0 repack path (`bash scripts/run-audit.sh`
→ `AUDIT.md` has the row-by-row detail; RUNBOOK.md step 2b is how this number gets filled in).

Speedup comes from Arm's KleidiAI kernels; this tool detects the miss and measures the delta.

## 2. Why Nobody Notices

The miss is silent. A K-quant (or IQ-quant) GGUF loads fine, runs fine, and produces correct
output on an Arm box — it just never reaches the repacked Q4_0 kernel path, so it's slower than it
could be with no error, no warning, and no log line most people would ever read closely enough to
catch. `general.file_type` in the file doesn't tell you either — corroborating metadata, not a
compatibility signal. The only way to know is to look at the actual tensor types and check them
against what the current llama.cpp build's dispatcher does at load time, which is exactly what
`scan` and `scan --verify` do.

## 3. What This Is / Is Not

| This is | This is not |
|---|---|
| A detector: `scan` classifies any local or remote GGUF's kernel-path compatibility in one read. | A new kernel. Every kernel involved is Arm's KleidiAI work. |
| An ecosystem measurement: `audit` quantifies how often widely-published quantizations miss the path. | A speed claim about anything we built — see the attribution line above, everywhere. |
| A reproducible requant path: `fix` gets a model onto the fast path, `bench`/`report` prove it honestly, paired with its quality cost. | A guess — `scan --verify` cross-checks every static prediction against the real on-device load log. |

## 4. Quickstart

Five commands, on an Arm box, ending in a verified scan:

```bash
pip install -e ".[dev]"
kleidi-advisor scan your-model.gguf
kleidi-advisor fix your-model-f16.gguf --calib wiki.train.raw -o fixed-q4_0.gguf
kleidi-advisor bench fixed-q4_0.gguf --threads "$(nproc)" --tag fixed
kleidi-advisor scan fixed-q4_0.gguf --verify --llama-bin-dir path/to/llama.cpp/build/bin
```

Full attended walkthrough — instance selection, building llama.cpp with KleidiAI, the audit, the
quality gate, and submission steps — is in [`RUNBOOK.md`](RUNBOOK.md).

## 5. Results

**TODO(box)**: headline (`<X.X>× pp512 at <+0.0X> ppl`), results table, plot, instance type, and
llama.cpp commit SHA all land here after RUNBOOK.md steps 4-7 run on the real box — `kleidi-advisor
report --instance <type> --plot results/plot.png` generates the table and chart from
`results/*.json` directly; this section is that command's output, pasted in. Every throughput
figure below is printed next to its perplexity delta on the same line, never alone — if it isn't,
that's a bug in `report`, not a stylistic choice.

## 6. How It Works

`scan` reads a GGUF's header and tensor metadata (locally, or head-only over HTTP for `--url`),
finds the dominant tensor quantization, and classifies it:

| Verdict | Triggering types | Meaning |
|---|---|---|
| `OK_KLEIDIAI` | `Q4_0` | Repacked at load time into Arm-optimised (i8mm/dotprod) kernels. |
| `FALLBACK_GENERIC` | `Q2_K`…`Q6_K`, `Q8_K`, all `IQ*` | No Arm repack path exists for these; inference runs the generic kernels. |
| `NOT_APPLICABLE` | `F32`, `F16`, `BF16`, `Q8_0`, `Q4_1`, `Q5_0`/`Q5_1`, integer types, `TQ*` | Not a Q4_0-repack candidate at all — no miss to report. |
| `UNKNOWN_VERIFY_ON_DEVICE` | anything this table version doesn't recognise | Run `scan --verify` on the target machine rather than trust a stale table. |

Why Q4_0 specifically: llama.cpp used to ship *pre-packed* Arm formats (`Q4_0_4_4`, `Q4_0_4_8`,
`Q4_0_8_8`) that made the fast path visible in the file itself. Those were removed upstream in
favor of repacking plain `Q4_0` *at load time* — which is strictly better for the ecosystem (one
quant format, not three) but means the fast path left no trace in the GGUF file. K-quants and
IQ-quants were never part of that repack story at all; they have their own (generic) kernels
regardless of llama.cpp version. That invisibility is the entire reason this tool exists.

## 7. Verify It Yourself

The table above is a prediction from a table version stamped to one llama.cpp commit — it can be
wrong on a different build. `scan --verify` runs `llama-cli -m <gguf> -n 1`, greps its load log for
repack/KleidiAI dispatch lines, and reports `AGREE`, `DISAGREE`, or `INCONCLUSIVE` (the honest
answer when a log line's absence isn't proof of anything). Real output from both the baseline and
fixed models, pasted verbatim, with the llama.cpp commit SHA it was checked against:

```
TODO(box) — RUNBOOK.md step 3 pastes both `kleidi-advisor scan <model> --verify` outputs here
```

## 8. Limitations

- The `OK_KLEIDIAI`/`FALLBACK_GENERIC`/`NOT_APPLICABLE` classification table is static and
  version-pinned to the llama.cpp commit recorded in §7 — a different build can disagree, which is
  exactly what `--verify` is for.
- Bench numbers are measured on one instance family (see §5) — the KleidiAI uplift is real Arm
  work, and its magnitude varies with which CPU features (i8mm, SVE, dotprod) are present.
- Perplexity is measured on one corpus (WikiText-2 raw) — a quality delta on a different corpus or
  task may differ.
- `data/hf-top-gguf.txt` is a hand-assembled list of widely-published GGUF repositories, not a
  download-ranked "top 20" — no popularity or download-count claim is made anywhere in this repo.

## 9. Future Work

Cross-device corroboration — running `scan --verify` across more Arm CPU families than the one
this submission measured, to see how far the static table travels before it needs a second row.

## 10. License

MIT — see [`LICENSE`](LICENSE).
