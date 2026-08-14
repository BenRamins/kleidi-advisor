# kleidi-advisor

Diagnose, measure, and fix the KleidiAI kernel-path miss in GGUF models on Arm.

Built for the Arm AI Optimization Challenge, Track 2 (Cloud AI). `kleidi-advisor` is a CLI —
`scan`, `fix`, `bench`, `report`, `audit` — that tells you whether a GGUF model actually reaches
Arm's KleidiAI kernels, measures how often the wider ecosystem misses that path, and gives you a
reproducible way to fix a model that does.

## 1. The Finding

**All 7 Q4_K_M files in our audit list miss Arm's KleidiAI kernels — every one, across all 5
publishers that ship one: bartowski (Llama-3.1, Phi-3.5, Gemma-2), lmstudio-community, TheBloke,
unsloth, and Qwen's own repo.** Not one gets it right, which is the point: this is not a packaging
mistake anyone made, it is the default outcome of the format everyone converged on.

In total, **13 of 17 successfully scanned GGUFs never reach KleidiAI's kernels** (3 URLs were
unreachable and are excluded from the denominator rather than counted either way). Classifying all
17 took 211.8 MB of traffic, because `scan` reads the GGUF header and stops — see [`AUDIT.md`](AUDIT.md)
for the row-by-row table and `results/audit.json` for the explicit scanned/error/miss counts.

Both common quantization formats are accelerated on Arm — but only one of them reaches Arm's
KleidiAI kernels. Qwen2.5-7B-Instruct on Azure Standard_E8ps_v6 (Cobalt 100, Arm Neoverse-N2),
llama.cpp build `1692f9e50` (b10431), 8 threads:

| format | pp512 (tok/s) | tg128 (tok/s) | PPL (WikiText-2, 100 chunks) | load path taken |
|---|---|---|---|---|
| Q4_K_M | 44.47 ± 0.05 | 15.85 ± 0.03 | 8.1728 ± 0.14245 | `CPU_REPACK` — ggml's own aarch64 repack (`q4_K_8x8`) |
| Q4_0 | 71.48 ± 0.12 | 17.56 ± 0.03 | 8.2215 ± 0.14170 | `CPU_KLEIDIAI` — Arm's KleidiAI i8mm kernels |

**1.61× prompt processing at +0.049 perplexity — a 0.6% quality cost that sits inside the error bars
of both measurements.** That is a statement about resolution, not equivalence: this run cannot
distinguish the two models' quality, which is not the same as showing they are identical. Token
generation gains 1.11×, less than prompt processing because it is memory-bandwidth-bound rather than
compute-bound.

The interesting part is *not* that Q4_0 is faster. It's that the widely-repeated shorthand — "K-quants
are unaccelerated on Arm" — is false, and it was this repo's own starting assumption too. K-quants
take a real Arm repack path; it just isn't KleidiAI's. See §8 for what we got wrong and how the
measurement caught it.

Speedup comes from Arm's KleidiAI kernels; this tool detects the miss and measures the delta.

## 2. Why Nobody Notices

The miss is silent, and it is silent in a way that defeats casual inspection twice over. A K-quant
(or IQ-quant) GGUF loads fine, runs fine, and produces correct output on an Arm box — it just never
reaches KleidiAI's kernels, with no error, no warning, and nothing in the file to indicate it.
`general.file_type` doesn't tell you either — corroborating metadata, not a compatibility signal.

Worse, the load log *looks* like it says the fast path was taken. On b10431 a Q4_K_M model prints
`repack: repack tensor blk.N.attn_q.weight with q4_K_8x8` — real repacking, just ggml's own, not
KleidiAI's — and both formats print `cannot be used with preferred buffer type CPU_KLEIDIAI, using
CPU instead`, a line containing the word KleidiAI that means the opposite of what it looks like.
Grepping the log for "repack" or "kleidi" tells you a model is accelerated when it isn't. The signal
that actually separates the paths is which model buffer the tensors land in, and it only prints
under `-v`. That is what `scan` and `scan --verify` check.

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
quality gate, and the cold re-run — is in [`REPRODUCE.md`](REPRODUCE.md).

## 5. Results

**1.61× pp512 at +0.049 ppl (WikiText-2, 100 chunks), Azure Standard_E8ps_v6 (Cobalt 100, Neoverse
N2), 8 threads.**

| model | tag | threads | pp512 (tok/s) | tg128 (tok/s) | PPL |
|---|---|---|---|---|---|
| qwen2.5-7b-instruct-q4_k_m | baseline | 8 | 44.47 ± 0.05 | 15.85 ± 0.03 | 8.1728 ± 0.14245 |
| qwen2.5-7b-instruct-q4_0 | fixed | 8 | 71.48 ± 0.12 | 17.56 ± 0.03 | 8.2215 ± 0.14170 |

| | pp512 | tg128 | PPL |
|---|---|---|---|
| **fixed ÷ baseline** | **1.61×** | **1.11×** | **+0.049 (+0.6%)** |

Environment: llama.cpp `1692f9e50` (b10431), built `-DGGML_CPU_KLEIDIAI=ON -DGGML_NATIVE=ON`.
Perplexity is WikiText-2 raw at `--chunks 100`, the same corpus and chunk count on both sides.

Read the quality column carefully. The +0.049 perplexity gap is smaller than the ±0.14 uncertainty
on either measurement, so this run **cannot resolve** a quality difference between the two models —
which is a limit on what was measured, not a demonstration that the models are equivalent. A longer
corpus pass could separate them. This section is what `kleidi-advisor report --instance "Azure
Standard_E8ps_v6 (Cobalt 100, Neoverse N2), 8 threads" --plot results/plot.png` renders from
`results/*.json`; `report` will not emit a throughput figure without its perplexity neighbour.

## 6. How It Works

`scan` reads a GGUF's header and tensor metadata (locally, or head-only over HTTP for `--url`),
finds the dominant tensor quantization, and classifies it:

| Verdict | Triggering types | Meaning |
|---|---|---|
| `OK_KLEIDIAI` | `Q4_0` | Lands in a `CPU_KLEIDIAI` buffer — Arm's i8mm/dotprod kernels. |
| `NOT_KLEIDIAI_PATH` | `Q2_K`…`Q6_K`, `Q8_K` | Accelerated, but by ggml's own aarch64 `CPU_REPACK` path, not KleidiAI's. Measured 1.61× slower at pp512, at a +0.049 ppl cost for switching (§1). |
| `FALLBACK_GENERIC` | all `IQ*` | Neither buffer observed; inference runs the generic kernels. |
| `NOT_APPLICABLE` | `F32`, `F16`, `BF16`, `Q8_0`, `Q4_1`, `Q5_0`/`Q5_1`, integer types, `TQ*` | Not a Q4_0-repack candidate at all — no miss to report. |
| `UNKNOWN_VERIFY_ON_DEVICE` | anything this table version doesn't recognise | Run `scan --verify` on the target machine rather than trust a stale table. |

Both `NOT_KLEIDIAI_PATH` and `FALLBACK_GENERIC` are KleidiAI misses — `--fail-on-miss` exits 3 on
either, and `audit` counts both. They differ in what the model falls back *to*, which is worth
knowing: a K-quant is not sitting on unoptimised scalar code, it's on a slower optimised path.

Why Q4_0 specifically: llama.cpp used to ship *pre-packed* Arm formats (`Q4_0_4_4`, `Q4_0_4_8`,
`Q4_0_8_8`) that made the fast path visible in the file itself. Those were removed upstream in
favor of repacking plain `Q4_0` *at load time* — which is strictly better for the ecosystem (one
quant format, not three) but means the fast path left no trace in the GGUF file. That invisibility
is the entire reason this tool exists.

The class boundary is drawn from measurement on one build: Q4_0 and Q4_K_M were loaded on the box
and their buffers observed directly. The remaining K-quants are grouped with Q4_K_M by family, and
no `IQ*` model was loaded at all — so `FALLBACK_GENERIC` is now the *unobserved* bucket. Both
generalisations are one `scan --verify` away from being falsified, which is the point of §7.

## 7. Verify It Yourself

The table above is a prediction from a table version stamped to one llama.cpp commit — it can be
wrong on a different build, and it *has been* wrong on this one (§8). `scan --verify` runs
`llama-bench -m <gguf> -p 8 -n 0 -r 1 -v` and checks which model buffer the tensors actually landed
in — `CPU_KLEIDIAI`, `CPU_REPACK`, or neither — then reports `AGREE`, `DISAGREE`, or `INCONCLUSIVE`.

Three details in that command are load-bearing, all of them learned the hard way:

- **`-v` is required.** The `load_tensors: ... model buffer size` lines are verbose-only. Without
  them there is nothing to read, which `--verify` reports as `INCONCLUSIVE` rather than guessing.
- **`llama-bench`, not `llama-cli`.** `llama-cli` with no prompt goes interactive and hangs.
- **The buffer line, not the word "repack".** See §2 and §8 for why the obvious grep is a trap.

Dispatch evidence from both models, pasted verbatim from the `-v` load logs on llama.cpp
`1692f9e50` (b10431):

```
Q4_0:   load_tensors: CPU_KLEIDIAI model buffer size = 3500.45 MiB
        kleidiai: primary q4 kernel feature I8MM

Q4_K_M: load_tensors:   CPU_REPACK model buffer size = 4166.82 MiB
        kleidiai: no kernel for tensor type q4_K, not accelerated by KleidiAI
```

That is the whole finding in four lines. The Q4_0 model gets a KleidiAI buffer and an i8mm kernel;
the Q4_K_M model is told in as many words that no KleidiAI kernel exists for its tensor type, and
lands in ggml's own repack buffer instead. Both models load, run, and answer correctly either way —
nothing above is an error, which is exactly why the miss goes unnoticed.

## 8. What We Got Wrong

This repo was built on a premise the box falsified, and the falsification is recorded here rather
than quietly patched out — a static classification table that has never been contradicted is a
table that has never been tested.

**The original assumption:** two outcomes. Q4_0 reaches Arm's repack path; every K-quant and
IQ-quant "has no Arm repack path" and runs generic kernels. The predicted uplift was the
`~2.5–2.9×` figure published in llama.cpp PR #9921 for Q4_0 repack vs generic on Graviton3.

**What b10431 actually does:** three outcomes. Q4_0 gets a `CPU_KLEIDIAI` buffer; Q4_K_M gets a
`CPU_REPACK` buffer and per-tensor `q4_K_8x8` repacking — ggml's own aarch64 path, added since that
PR — and everything else gets neither. The measured Q4_0-vs-Q4_K_M gap on Neoverse-N2 is 1.61×
pp512, not 2.5–2.9×, because the comparison is no longer optimised-vs-generic. It's
optimised-vs-differently-optimised. The PR #9921 anchor is kept in this repo only as historical
context and is explicitly superseded by the measurement in §1.

**What that broke in this tool, and how it was caught:** the original `--verify` matched substrings
including `repack`, `kleidi` and `aarch64`. On this build all three appear in *both* formats' load
logs, so the pattern-hit branch was always taken and the outcome was decided entirely by the static
verdict being checked — `AGREE` whenever the table said `OK_KLEIDIAI`, `DISAGREE` whenever it said
`FALLBACK_GENERIC`. A step whose whole purpose is to falsify the table would instead have rubber-
stamped it, and its `DISAGREE` on every K-quant would have read as a genuine signal pointing at the
one verdict that was arguably right. It was caught by reading the raw `-v` load logs of both formats
side by side *before* trusting the tool's own output — which is why REPRODUCE.md step 4 has you read
the raw logs first and record verbatim excerpts, not just the `AGREE`/`DISAGREE` line.

The fix is in `REFERENCE.md` §4 and §8: a new class (`NOT_KLEIDIAI_PATH`) and buffer-type detection
in place of substring matching, with "absence of a `CPU_KLEIDIAI` buffer" promoted from
inconclusive to real evidence now that `-v` guarantees the line would be there if it applied.

## 9. Limitations

Every number in §1 and §5 comes from **one machine, one model, one build, one corpus sample**. That
is enough to establish the kernel-path finding, which is a fact about dispatch, and it is not enough
to establish how large the gap is in general.

- **One machine.** A single 8-vCPU Arm Neoverse-N2 VM (Azure Standard_E8ps_v6, Cobalt 100). No
  second instance, no second CPU family, no repeat on different hardware. llama.cpp reported this
  machine's capabilities as:
  ```
  NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | SVE = 1 | DOTPROD = 1 |
  SVE_CNT = 16 | OPENMP = 1 | KLEIDIAI = 1 | REPACK = 1
  ```
  `MATMUL_INT8 = 1` is why the i8mm microkernel was selected. A CPU family without it (e.g. Ampere
  Altra / Neoverse-N1) will not reproduce 1.61×, in either direction.
- **One model.** Qwen2.5-7B-Instruct only. Architecture and tensor shapes affect how much of the
  runtime is in the matmuls KleidiAI accelerates, so the ratio is not portable to other models.
- **One build.** llama.cpp `1692f9e50` (b10431). The classification table is static and pinned to
  that commit — a different build can disagree, which is exactly what `--verify` is for, and §8 is
  what it looks like when that actually happens.
- **One corpus sample.** Perplexity is WikiText-2 raw at `--chunks 100`, not the full corpus
  (imatrix uses `--chunks 50`), so an 8-core machine finishes in reasonable time. Absolute PPL here
  is therefore not comparable to figures published over the full corpus, and the ±0.14 uncertainty
  is wider than the +0.049 delta it is being used to judge — see §5.
- **Two quant types actually loaded.** Only Q4_0 and Q4_K_M had their buffers observed directly. The
  other K-quants are classified by family and no `IQ*` model was loaded at all; those rows are
  predictions awaiting their own `--verify` run.
- **The audit list is hand-assembled, not download-ranked.** `data/hf-top-gguf.txt` is 20
  widely-published GGUF repositories chosen by hand. It is not a "top 20" by downloads, popularity,
  or any other ranking — no such claim is made anywhere in this repo. 3 of its 20 URLs were
  unreachable at audit time and are excluded from the denominator in §1.

## 10. Future Work

Cross-device corroboration — running `scan --verify` across more Arm CPU families than the one
this submission measured, to see how far the static table travels before it needs a second row.
§8 is the argument for doing it: one build already moved a whole quant family between classes, so
the interesting question is not whether the table drifts but where it drifts first.

## 11. License

MIT — see [`LICENSE`](LICENSE).
