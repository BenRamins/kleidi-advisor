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
17 took 211.8 MB of traffic, because `scan` reads the GGUF header and stops — see
[`results/audit.json`](results/audit.json) for the row-by-row table and the explicit
scanned/error/miss counts.

Both common quantization formats are accelerated on Arm — but only one of them reaches Arm's
KleidiAI kernels. Qwen2.5-7B-Instruct on Azure Standard_E8ps_v6 (Cobalt 100, Arm Neoverse-N2),
llama.cpp build `1692f9e50` (b10431), 8 threads. Both files are Qwen's own published builds from
one repository, so the quantization type is the only variable between them (§5):

| format | pp512 (tok/s) | tg128 (tok/s) | PPL (WikiText-2, 100 chunks) | load path taken |
|---|---|---|---|---|
| Q4_K_M | 44.47 ± 0.04 | 15.79 ± 0.01 | 8.1728 ± 0.14245 | `CPU_REPACK` — ggml's own aarch64 repack (`q4_K_8x8`) |
| Q4_0 | 71.60 ± 0.06 | 17.61 ± 0.04 | 8.2215 ± 0.14170 | `CPU_KLEIDIAI` — Arm's KleidiAI i8mm kernels |

**1.61× prompt processing at +0.049 perplexity — a 0.6% quality cost that sits inside the error bars
of both measurements.** That is a statement about resolution, not equivalence: this run cannot
distinguish the two models' quality, which is not the same as showing they are identical. Token
generation gains less, 1.12×, because it is memory-bandwidth-bound rather than compute-bound;
the +0.049 ppl above is the quality cost for the pair.

The interesting part is *not* that Q4_0 is faster. It's that the widely-repeated shorthand — "K-quants
are unaccelerated on Arm" — is false, and it was this repo's own starting assumption too. K-quants
take a real Arm repack path; it just isn't KleidiAI's. See §8 for what we got wrong and how the
measurement caught it.

Speedup comes from Arm's KleidiAI kernels; this tool detects the miss and measures the delta.

## 2. Why Nobody Notices

**llama.cpp tells you.** Loading a Q4_K_M model on this build logs a warning that is clear, correct,
and exactly on point:

```
kleidiai: no kernel for tensor type q4_K, not accelerated by KleidiAI
```

So this is not a case of information being withheld, and this tool is not here to surface a hidden
fact. The problem is *when* that warning arrives, *where* it sits, and what it leaves out:

- **It arrives after the decision it should have informed.** The warning is emitted at load time —
  which is after you chose a quantization, downloaded several GB of it, and started a run. The
  choice it bears on was made hours earlier, from a repository file listing that says nothing about
  kernel paths.
- **It sits in the middle of startup output.** It scrolls past among the loader's KV-pair dump and
  tensor summaries, on the way to the first token. Nothing about it is louder than the lines around
  it, and by the time output appears the model is already running correctly — which is precisely
  when a person stops reading logs.
- **It quantifies nothing.** "not accelerated by KleidiAI" is true and gives you no way to judge
  whether it matters. Is the other path 2% slower or 2× slower? Worth a requantization or not? The
  warning cannot say, because it does not know what the alternative costs on this CPU. On Neoverse-N2
  the answer turned out to be 1.61× at pp512 for +0.049 ppl (§1) — a number that had to be
  measured, not read.
- **The surrounding log punishes the obvious shortcut.** Grepping for `kleidi` matches the warning
  above *and* the line `cannot be used with preferred buffer type CPU_KLEIDIAI, using CPU instead`,
  which both formats emit and which means the opposite of how it reads. Grepping for `repack` matches
  `repack: repack tensor blk.N.attn_q.weight with q4_K_8x8` — real repacking, just ggml's own rather
  than KleidiAI's. Either grep reports acceleration on a model that never reached KleidiAI. The
  signal that actually separates the paths is which model buffer the tensors land in, and it prints
  only under `-v`.

Nor does the file itself help: `general.file_type` is corroborating metadata, not a compatibility
signal, so nothing you can inspect before downloading answers the question.

That is the gap `scan` fills. It answers the same question **from a URL, before the download** — a
few MB of HTTP range requests against the GGUF header rather than a few GB of model — and attaches
the measured cost of the path it finds. `scan --verify` then checks that prediction against the real
load log on the machine in front of you (§7). Same fact llama.cpp already knows, moved to the point
where it can still change what you download.

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
kleidi-advisor fix your-model-f16.gguf --calib wiki.test.raw -o fixed-q4_0.gguf
kleidi-advisor bench fixed-q4_0.gguf --threads "$(nproc)" --tag fixed
kleidi-advisor scan fixed-q4_0.gguf --verify --llama-bin-dir path/to/llama.cpp/build/bin
```

Full attended walkthrough — instance selection, building llama.cpp with KleidiAI, the audit, the
quality gate, and the cold re-run — is in [`REPRODUCE.md`](REPRODUCE.md).

## 5. Results

Rendered by `kleidi-advisor report --results-dir results --headline published-q4_0 --instance "Azure
Standard_E8ps_v6 (Cobalt 100, Neoverse N2), 8 threads"`, from the three result files in
[`results/`](results/):

```
1.61× pp512 at +0.049 ppl (WikiText-2, 100 chunks, n=5, Azure Standard_E8ps_v6 (Cobalt 100, Neoverse N2), 8 threads)
Speedup comes from Arm's KleidiAI kernels; this tool detects the miss and measures the delta.

| model                  | tag             | threads | pp512 (tok/s) | tg128 (tok/s) | ppl    |
|------------------------|-----------------|---------|---------------|---------------|--------|
| qwen-q4km.gguf         | baseline        | 8       | 44.47 ± 0.04  | 15.79 ± 0.01  | 8.1728 |
| qwen-imatrix-q4_0.gguf | imatrix-fix     | 8       | 66.65 ± 0.12  | 17.13 ± 0.04  | 8.1525 |
| qwen-q4_0-ref.gguf     | published-q4_0  | 8       | 71.60 ± 0.06  | 17.61 ± 0.04  | 8.2215 |

Instance: Azure Standard_E8ps_v6 (Cobalt 100, Neoverse N2), 8 threads
```

Derived from those medians:

| vs. baseline | pp512 | tg128 | ppl |
|---|---|---|---|
| **`published-q4_0`** — the headline | **1.61×** | **1.12×** | **+0.049 (+0.6%)** |
| `imatrix-fix` — our own `fix` output | 1.50× | 1.08× | −0.020, see caveats below |

**The headline is the two Qwen-published builds, deliberately.** `published-q4_0` is
`qwen2.5-7b-instruct-q4_0.gguf` from the same repository as the Q4_K_M baseline — one publisher, one
fp16 lineage, quantization type the only variable, and nothing calibrated by us anywhere in it. That
is the honest ecosystem claim: this is what a user gets by downloading the other file already on the
page. The `imatrix-fix` row is *our* artifact and carries two caveats the headline does not.

**Caveat 1 — the −0.020 ppl is contaminated and is not evidence that imatrix improves quality.** The
importance matrix was calibrated on `wiki.test.raw` and perplexity was then evaluated on
`wiki.test.raw`: the same file. Calibrating on the evaluation set very likely flatters that number,
so it must not be read as a quality gain. `wiki.train.raw` exists and calibrating on it would remove
the overlap entirely; there was not time to re-run before submission. Note also that −0.020 is an
order of magnitude smaller than the ±0.14 uncertainty on the two published builds' perplexity
measurements, so even without the contamination this run could not resolve a difference that size.

**Caveat 2 — unexplained: our imatrix Q4_0 is 6.9% slower at pp512 than Qwen's published Q4_0**
(66.65 vs 71.60 tok/s), although both are Q4_0 and both land in a `CPU_KLEIDIAI` buffer. **We did not
investigate this.** Candidates worth checking, named as candidates and not as conclusions: per-tensor
precision choices made under imatrix guidance, and how the output and token-embedding tensors were
handled during quantization. Anyone reproducing this work should treat the gap as open — it is the
first thing we would look at with more time, and it means `fix` should not yet be presented as
matching a well-made published Q4_0 on speed.

The quality gate passed on its own terms: candidate 8.1525 against baseline 8.1728, `--max-delta 0.3`
— but see caveat 1 before reading anything into the direction of that difference.

**Harness cross-check.** The `published-q4_0` model was also benchmarked by hand with raw
`llama-bench`, before this harness existed, and the two runs agree: pp512 71.48 hand-run vs 71.6045
through `bench` (0.2%), tg128 17.56 vs 17.6122 (0.3%), perplexity identical at 8.2215. The same
comparison on the baseline row agrees to 0.007% at pp512 (44.47 vs 44.467) and 0.4% at tg128 (15.85
vs 15.7903). `kleidi-advisor bench` reports what the underlying tool reports; it parses and
aggregates, it does not adjust. The residual differences are run-to-run variance, and they are larger
on token generation than on prompt processing in both rows.

Environment: llama.cpp `1692f9e50` (b10431), built `-DGGML_CPU_KLEIDIAI=ON -DGGML_NATIVE=ON`.
Perplexity is WikiText-2 raw at `--chunks 100`, the same corpus and chunk count on all three rows.
The per-run perplexity uncertainties (±0.14245 baseline, ±0.14170 published) come from
`llama-perplexity`'s own output; the results schema records the value only, so they do not appear in
the rendered table above.

Read the quality column carefully. The headline's +0.049 perplexity gap is smaller than the ±0.14
uncertainty on either measurement, so this run **cannot resolve** a quality difference between the
two published builds — a limit on what was measured, not a demonstration that they are equivalent. A
longer corpus pass could separate them.

The three files this section renders from are committed under [`results/`](results/), each recording
the instance and the llama.cpp commit it was measured on, so the table above can be regenerated and
checked rather than taken on trust.

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
lands in ggml's own repack buffer instead.

Note the fourth line: llama.cpp is not hiding this. It states the miss plainly, at warning level.
What it cannot tell you is what the miss costs, or tell you early enough to pick a different file —
which is the whole of §2, and the whole reason `scan` reads headers over HTTP instead.

## 8. What We Got Wrong

This repo was built on a premise the box falsified, and the falsification is recorded here rather
than quietly patched out — a static classification table that has never been contradicted is a
table that has never been tested.

**The original assumption:** two outcomes. Q4_0 reaches Arm's repack path; every K-quant and
IQ-quant "has no Arm repack path" and runs generic kernels. The predicted uplift was the
`~2.5–2.9×` figure published in llama.cpp PR #9921 for Q4_0 repack vs generic on Graviton3.

**What b10431 actually does:** three outcomes. Q4_0 gets a `CPU_KLEIDIAI` buffer; Q4_K_M gets a
`CPU_REPACK` buffer and per-tensor `q4_K_8x8` repacking — ggml's own aarch64 path, added since that
PR — and everything else gets neither. The measured Q4_0-vs-Q4_K_M gap on Neoverse-N2 is
1.61× pp512 at +0.049 ppl, not 2.5–2.9×, because the comparison is no longer
optimised-vs-generic. It's optimised-vs-differently-optimised. The PR #9921 anchor is kept in this repo only as historical
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
  Altra / Neoverse-N1) will not reproduce 1.61× at +0.049 ppl, in either direction.
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
