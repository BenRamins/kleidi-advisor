# REFERENCE — the facts this project must not guess

Every byte layout, enum value, output shape, and command `kleidi-advisor` depends on is written out
here, so no part of the implementation has to reconstruct one from memory. **A fact that is not in
this file is not to be invented: derive it from one that is, or make the code tolerant of being
wrong and have it report what it actually saw.**

Confidence is marked on every section:
- **[SPEC]** — format definition; treat as authoritative, build against it.
- **[STAMP]** — believed correct as of llama.cpp master in mid 2026, but versions drift. Build against it,
  make the code degrade gracefully if reality differs, and flag it for on-device confirmation.
- **[GUESS]** — plausible but unverified. Code must tolerate being wrong; the operator confirms on the box.

---

## 1. GGUF file layout **[SPEC]**

All integers little-endian. GGUF v2 and v3 are byte-identical for our purposes (v3 only added
big-endian support); **accept version 2 and 3, reject 1 and anything >3** with a clear message.

```
HEADER
  magic            4 bytes   ASCII "GGUF" (0x47 0x47 0x55 0x46)
  version          uint32    2 or 3
  tensor_count     uint64
  kv_count         uint64

METADATA (kv_count times)
  key              gguf_string
  value_type       uint32    (enum in §2)
  value            per value_type

TENSOR INFO (tensor_count times)
  name             gguf_string
  n_dims           uint32
  dims             uint64 × n_dims
  ggml_type        uint32    (enum in §3)
  offset           uint64    relative to start of tensor data

PADDING            zero bytes until the file offset is a multiple of `general.alignment`
                   (uint32 kv, default 32 when absent)

TENSOR DATA        opaque; we never read it
```

`gguf_string` = `uint64 length` followed by exactly that many UTF-8 bytes. **No null terminator.**
Lengths are byte counts, not character counts.

**Why this matters for us**: everything the scanner needs (metadata + all tensor infos) sits before the
tensor data, so reading the first few MiB of a multi-GB file is sufficient. That is the whole basis of
`scan --url` and `audit`.

## 2. Metadata value type enum **[SPEC]**

| id | type | encoding |
|---|---|---|
| 0 | UINT8 | 1 byte |
| 1 | INT8 | 1 byte |
| 2 | UINT16 | 2 bytes |
| 3 | INT16 | 2 bytes |
| 4 | UINT32 | 4 bytes |
| 5 | INT32 | 4 bytes |
| 6 | FLOAT32 | 4 bytes IEEE-754 |
| 7 | BOOL | 1 byte, 0 or 1 |
| 8 | STRING | gguf_string |
| 9 | ARRAY | `uint32 element_type`, `uint64 count`, then count elements of element_type |
| 10 | UINT64 | 8 bytes |
| 11 | INT64 | 8 bytes |
| 12 | FLOAT64 | 8 bytes |

Arrays may contain strings (each its own gguf_string) and may be long — the tokenizer vocab is an array
of ~128k strings and is the reason a 2 MiB read sometimes isn't enough. Arrays never nest.
An unknown value_type is unrecoverable (you cannot know the width to skip): raise, naming the byte offset.

## 3. ggml type IDs **[STAMP]**

| id | name | id | name | id | name |
|---|---|---|---|---|---|
| 0 | F32 | 12 | Q4_K | 24 | I8 |
| 1 | F16 | 13 | Q5_K | 25 | I16 |
| 2 | Q4_0 | 14 | Q6_K | 26 | I32 |
| 3 | Q4_1 | 15 | Q8_K | 27 | I64 |
| 6 | Q5_0 | 16 | IQ2_XXS | 28 | F64 |
| 7 | Q5_1 | 17 | IQ2_XS | 29 | IQ1_M |
| 8 | Q8_0 | 18 | IQ3_XXS | 30 | BF16 |
| 9 | Q8_1 | 19 | IQ1_S | 34 | TQ1_0 |
| 10 | Q2_K | 20 | IQ4_NL | 35 | TQ2_0 |
| 11 | Q3_K | 21 | IQ3_S | | |
| | | 22 | IQ2_S | | |
| | | 23 | IQ4_XS | | |

4 and 5 (Q4_2, Q4_3) are removed and must not appear. **31–33 were `Q4_0_4_4`, `Q4_0_4_8`, `Q4_0_8_8`**
— pre-packed Arm formats that llama.cpp *removed* in favour of repacking plain `Q4_0` at load time. If
you meet 31–33, classify as `UNKNOWN_VERIFY_ON_DEVICE` with the reason "legacy pre-packed Arm format,
removed upstream; reconvert from F16". That history is also the reason the project exists: the fast path
is now invisible — it happens at load time with no file-format signal.

**Any id not in this table → `UNKNOWN_VERIFY_ON_DEVICE`, never a crash.** Keep the mapping in one dict
named `GGML_TYPES` with a comment stating it is version-stamped and a pointer to `scan --verify`.

## 4. Verdict table **[MEASURED 2026-08-14]**

**This table was rewritten on 2026-08-14 after the box run falsified the original [STAMP] version.**
Measured on Azure Standard_E8ps_v6 (Cobalt 100, Neoverse N2), llama.cpp build `1692f9e50` (b10431),
8 threads. The load log shows **three** dispatch paths, not two: KleidiAI's own model buffer
(`CPU_KLEIDIAI`), ggml's own aarch64 repack buffer (`CPU_REPACK`), and neither. K-quants are *not*
unaccelerated — they are accelerated by a different path that is not KleidiAI's, and they were
measured 1.61× slower at pp512 than Q4_0 on that machine, at a +0.049 perplexity cost for switching
(WikiText-2, 100 chunks — smaller than the ±0.14 error bars on either measurement, so the run cannot
resolve a quality difference; that is a limit on the measurement, not a claim of equivalence). The
no-bare-throughput rule applies to the speed figure everywhere it is quoted. See §8 for how the
paths are told apart.

| Class | Types | Reason string (use verbatim) |
|---|---|---|
| `OK_KLEIDIAI` | Q4_0 | "Q4_0 weights are repacked at load time into Arm-optimised kernels (i8mm/dotprod)." |
| `NOT_KLEIDIAI_PATH` | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | "K-quant weights are repacked by ggml's own aarch64 path (CPU_REPACK), not KleidiAI's i8mm kernels; measured 1.61x slower at pp512 on Neoverse N2 (+0.049 ppl, WikiText-2 100 chunks)." |
| `FALLBACK_GENERIC` | all IQ* | "No CPU_KLEIDIAI and no CPU_REPACK model buffer observed for this weight type; inference runs the generic kernels." |
| `NOT_APPLICABLE` | F32, F16, BF16, F64, Q8_0, Q4_1, Q5_0, Q5_1, Q8_1, I8/I16/I32/I64, TQ* | "Not a Q4_0-repack candidate; no kernel-miss to report for this weight type." |
| `UNKNOWN_VERIFY_ON_DEVICE` | everything else, incl. 31–33 | "Unrecognised weight type for this table version; run scan --verify on the target machine." |

`next` field: for `NOT_KLEIDIAI_PATH` **and** `FALLBACK_GENERIC` → `kleidi-advisor fix <source-f16.gguf> --calib <corpus.txt> -o <out.gguf>`. Otherwise `null`. Both classes are a KleidiAI miss; only the fallback path differs.

**What was actually measured, and what is generalised** — the box run covered Q4_0 and Q4_K_M only:
- Q4_0 → `CPU_KLEIDIAI` buffer, confirmed. Q4_K_M → `CPU_REPACK` buffer, no `CPU_KLEIDIAI`, confirmed.
- The other K-quants (Q2_K/Q3_K/Q5_K/Q6_K/Q8_K) are placed in `NOT_KLEIDIAI_PATH` by family, not by
  measurement. The IQ types were never loaded on the box at all, so `FALLBACK_GENERIC` is now the
  *unobserved* bucket rather than a measured one.
- Both generalisations are falsifiable in one command per model: `scan --verify` (§8). A `DISAGREE`
  from it beats this table.

## 5. `general.file_type` values **[GUESS]**

Corroboration only — **the tensor scan always wins** and disagreement is reported, never resolved silently.

| 0 | ALL_F32 | 1 | MOSTLY_F16 | 2 | MOSTLY_Q4_0 | 3 | MOSTLY_Q4_1 |
| 7 | MOSTLY_Q8_0 | 8 | MOSTLY_Q5_0 | 9 | MOSTLY_Q5_1 | 10 | MOSTLY_Q2_K |
| 11 | Q3_K_S | 12 | Q3_K_M | 13 | Q3_K_L | 14 | Q4_K_S |
| 15 | Q4_K_M | 16 | Q5_K_S | 17 | Q5_K_M | 18 | Q6_K |

Unknown id → store the integer, render as `file_type=<n> (unmapped)`. Never crash, never guess a name.

## 6. `llama-bench -o json` output shape **[STAMP]**

Top-level is a JSON **array**, one object per benchmark row. Relevant keys:

```json
[
  {
    "build_commit": "a1b2c3d", "model_filename": "llama-3.1-8b-q4_k_m.gguf",
    "model_size": 4920000000, "n_threads": 16,
    "n_prompt": 512, "n_gen": 0,
    "avg_ns": 1234567890, "stddev_ns": 12345678,
    "avg_ts": 415.23, "stddev_ts": 4.11,
    "samples_ts": [413.9, 415.1, 416.0, 415.4, 415.7]
  },
  { "n_prompt": 0, "n_gen": 128, "avg_ts": 28.44, "stddev_ts": 0.31, "samples_ts": [28.1, 28.5, 28.6, 28.4, 28.6] }
]
```

Parsing rules — **write them defensively, this is the highest-risk parse in the build**:
1. Row with `n_prompt > 0 and n_gen == 0` → the **pp512** row. Row with `n_gen > 0 and n_prompt == 0` → **tg128**.
2. Prefer `samples_ts` when present (compute median/stdev yourself with `statistics.median` / `statistics.pstdev`);
   fall back to `avg_ts` with `stddev_ts`; if neither exists, raise a clear error that **prints the keys actually
   present** so the operator can fix it in thirty seconds on the box.
3. Never index rows positionally. Never assume row count.
4. Unknown extra keys are normal — ignore them.

Commit `tests/data/llama-bench-baseline.json` and `tests/data/llama-bench-fixed.json` containing exactly
the shape above (baseline pp 415.23 / tg 28.44; fixed pp 1163.8 / tg 30.12 — **these are invented shape
fixtures for tests only and must never appear in README, RESULTS.md, or any prose**).

## 7. `llama-perplexity` output **[STAMP]**

Final line, one of these two shapes:

```
Final estimate: PPL = 6.7841 +/- 0.04012
Final estimate: PPL = 6.7841
```

Regex: `Final estimate:\s*PPL\s*=\s*([0-9]+\.?[0-9]*)` — capture group 1, ignore the error term.
No match → error naming the last 5 lines of output. Commit both shapes as `tests/data/ppl-with-err.txt`
and `tests/data/ppl-no-err.txt`.

## 8. Kernel-dispatch detection **[MEASURED 2026-08-14]** — buffer type, not the word "repack"

**The original `VERIFY_PATTERNS` list was wrong and would have produced false `AGREE`s.** On build
`1692f9e50` (b10431) the substring `repack` appears in the load log for **both** Q4_0 and Q4_K_M, and
so does `CPU_KLEIDIAI` — via the line `cannot be used with preferred buffer type CPU_KLEIDIAI, using
CPU instead`, which is emitted for both formats and means the *opposite* of what a bare `kleidi`
substring match would conclude. Matching on either token classifies every model as KleidiAI-accelerated.

The signal that actually separates the paths is **which model buffer the tensors landed in**:

```
Q4_0:     kleidiai: primary q4 kernel feature I8MM
          load_tensors: CPU_KLEIDIAI model buffer size =  3500.45 MiB
Q4_K_M:   load_tensors: CPU_REPACK model buffer size =  4166.82 MiB
          repack: repack tensor blk.N.attn_q.weight with q4_K_8x8
both:     ... cannot be used with preferred buffer type CPU_KLEIDIAI, using CPU instead
```

Markers (case-insensitive substrings). The word "buffer size" is load-bearing — it is what excludes
the `cannot be used with preferred buffer type CPU_KLEIDIAI` decoy:

```python
KLEIDIAI_BUFFER_MARKER = "cpu_kleidiai model buffer size"   # measured b10431, 2026-08-14
REPACK_BUFFER_MARKER   = "cpu_repack model buffer size"
ANY_BUFFER_MARKER      = "model buffer size"                # did the log carry buffer lines at all?
```

**Command** — `-v` is **required** (the buffer lines are verbose-only), and it must be `llama-bench`,
not `llama-cli`: llama-cli with no prompt goes interactive and hangs.

```bash
llama-bench -m MODEL.gguf -p 8 -n 0 -r 1 -v
```

Outcome rules, in order:
1. No `ANY_BUFFER_MARKER` anywhere in the output → `INCONCLUSIVE` — the log carries no buffer-type
   line at all, so nothing was observed. Most likely `-v` was dropped or the build doesn't print them.
2. `KLEIDIAI_BUFFER_MARKER` present **and** verdict is `OK_KLEIDIAI` → `AGREE`
3. `KLEIDIAI_BUFFER_MARKER` present **and** verdict is `NOT_KLEIDIAI_PATH` or `FALLBACK_GENERIC` → `DISAGREE`
4. `KLEIDIAI_BUFFER_MARKER` absent **and** verdict is `OK_KLEIDIAI` → `DISAGREE` — with `-v` on a build
   that prints buffer lines, absence *is* evidence, unlike under the old pattern scheme.
5. `KLEIDIAI_BUFFER_MARKER` absent **and** verdict is `NOT_KLEIDIAI_PATH` → `AGREE` if
   `REPACK_BUFFER_MARKER` is present, else `DISAGREE` (the class asserts a CPU_REPACK buffer exists).
6. `KLEIDIAI_BUFFER_MARKER` absent **and** verdict is `FALLBACK_GENERIC` → `DISAGREE` if
   `REPACK_BUFFER_MARKER` is present (it's really `NOT_KLEIDIAI_PATH`), else `AGREE`.
7. Any other static verdict (`NOT_APPLICABLE`, `UNKNOWN_VERIFY_ON_DEVICE`) → `INCONCLUSIVE`.

`INCONCLUSIVE` exits 0 and prints which buffers were seen. It is an honest answer, not a failure.

## 9. Stub binary template **[SPEC]**

Every stub is a shell script in a temp bin dir, mode 0o755:

```sh
#!/bin/sh
# stub: <name>
printf '%s\n' "$0 $*" >> "$KA_STUB_LOG"
[ -n "$KA_STUB_STDOUT" ] && cat "$KA_STUB_STDOUT"
[ -n "$KA_STUB_STDERR" ] && cat "$KA_STUB_STDERR" >&2
exit "${KA_STUB_EXIT:-0}"
```

Env vars are set per-test, so one template serves every binary. Tests assert on `$KA_STUB_LOG` contents.
For `llama-bench`, `KA_STUB_STDOUT` points at a `tests/data/llama-bench-*.json`. To simulate a failing
stage, set `KA_STUB_EXIT=1`.

## 10. `results/<stem>-<tag>.json` schema **[SPEC]** — write this into `results/README.md` verbatim

```json
{
  "schema": 1,
  "model": "llama-3.1-8b-q4_k_m.gguf",
  "tag": "baseline",
  "threads": 16,
  "instance": "Azure Standard_E8ps_v6 (Cobalt 100, Neoverse N2), 8 threads",
  "llama_cpp_commit": "1692f9e50",
  "timestamp_utc": "2026-08-14T09:12:33Z",
  "argv": ["llama-bench", "-m", "...", "-p", "512", "-n", "128", "-r", "5", "-o", "json"],
  "metrics": {
    "pp512": {"runs": [413.9, 415.1], "median": 414.5, "stdev": 0.6, "unit": "tok/s"},
    "tg128": {"runs": [28.1, 28.5], "median": 28.3, "stdev": 0.2, "unit": "tok/s"}
  },
  "ppl": {"value": 6.7841, "corpus": "wikitext-2-raw", "chunks": null}
}
```

`ppl` is `null` when not measured, and so are `instance`/`llama_cpp_commit` until supplied.
`report` reads every `*.json` in the results dir, skipping files without `"schema": 1`.

## 11. Known-good commands **[STAMP]**

These are the exact invocations `REPRODUCE.md` is built from. Copy them rather than composing your
own; where a value depends on your machine it is written as an obvious `<placeholder>`.

Target machine: any Arm64 Linux instance with >=8 cores, >=32 GB RAM, >=100 GB disk. The commands
below were exercised on Azure Standard_E8ps_v6 (Cobalt 100, Neoverse N2), Ubuntu Server 24.04 arm64.

**Build llama.cpp with KleidiAI:**
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_KLEIDIAI=ON -DGGML_NATIVE=ON
cmake --build build --config Release -j"$(nproc)"
# Stale-binary trap (llama.cpp issue #12701): if you change cmake flags, `rm -rf build` first.
```

**Confirm the toolchain saw the right CPU features** — i8mm and sve2 are expected present on
Neoverse N2, but this must be verified on the machine, not asserted in prose. (Confirmed 2026-08-14:
the Q4_0 load log printed `kleidiai: primary q4 kernel feature I8MM`; sve2 remains unclaimed.)
```bash
lscpu | grep -oE 'i8mm|sve2|sve|asimddp|bf16'
./build/bin/llama-bench --help | head -5      # binary runs at all
```

**Bench** (1 vCPU = 1 physical core on this SKU, no SMT, so `-t $(nproc)` is correct):
```bash
./build/bin/llama-bench -m MODEL.gguf -p 512 -n 128 -r 5 -t "$(nproc)" -o json
```

**Download the model** (Qwen/Qwen2.5-7B-Instruct-GGUF, Apache-2.0, ungated):
```bash
pip install -U huggingface_hub
hf download Qwen/Qwen2.5-7B-Instruct-GGUF --include "*fp16*"   --local-dir .
hf download Qwen/Qwen2.5-7B-Instruct-GGUF --include "*q4_k_m*" --local-dir .
```

**Merge the split fp16 GGUF** (llama-quantize needs a single file; the fp16 ships as 4 shards):
```bash
llama-gguf-split --merge qwen2.5-7b-instruct-fp16-00001-of-00004.gguf qwen-fp16.gguf
```

**imatrix + quantize** (chunk budget: `--chunks 50` for imatrix, the same value for baseline and fixed runs):
```bash
./build/bin/llama-imatrix -m SOURCE-F16.gguf -f calib.txt -o model.imatrix -ngl 0 --chunks 50
./build/bin/llama-quantize --imatrix model.imatrix SOURCE-F16.gguf OUT-Q4_0.gguf Q4_0
```

**Perplexity** (chunk budget: `--chunks 100`, the same value for baseline and fixed — the quality gate
compares them directly, so a mismatched chunk count makes the comparison meaningless):
```bash
./build/bin/llama-perplexity -m MODEL.gguf -f wikitext-2-raw/wiki.test.raw -t "$(nproc)" --chunks 100
```

**Calibration corpus:** WikiText-2 raw.
```
https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
```
Unzips to `wikitext-2-raw/wiki.test.raw`.

## 12. Hugging Face URL pattern for `data/hf-top-gguf.txt` **[GUESS]**

Resolve pattern: `https://huggingface.co/<org>/<repo>/resolve/main/<filename>.gguf`

**You cannot verify these exist — you have no network.** Write 20 best-effort candidate lines from
well-known GGUF publishers (e.g. `bartowski`, `TheBloke`, `Qwen`, `unsloth`, `lmstudio-community`),
covering a spread of K-quants, Q4_0 and IQ types so the audit has variety. Then:

1. Put this header comment at the top of the file, verbatim:
   `# Candidate list assembled 2026-08-13 without network access. URLs are UNVERIFIED.`
   `# REPRODUCE.md step 3 runs the audit; any 404 rows must be corrected before the result is quoted.`
2. The `audit` command must record 404s as `error` rows and keep going (Spec F7 rule 1) — that is
   precisely what makes an unverified list safe to ship.
3. Never state a download count, ranking, or popularity claim anywhere. The list is "widely-published
   GGUF repositories", not "the top 20".
