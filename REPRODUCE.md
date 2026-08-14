# REPRODUCE — measuring the KleidiAI kernel-path miss on Arm

This is the attended, end-to-end procedure behind every number in [`README.md`](README.md): build
llama.cpp with KleidiAI, confirm which kernel path each quantization format actually takes, then
measure the difference with its quality cost attached.

Everything in this repository's test suite runs offline against stubs and an in-process fixture
server. This document is the part that does not — it needs a real Arm machine, a real llama.cpp
build, and real model files.

**Hardware**: any Arm64 Linux instance with >=8 cores, >=32 GB RAM, >=100 GB disk; results below
were measured on Azure Standard_E8ps_v6 (Cobalt 100, Neoverse N2).

Steps are in dependency order. Each names the exact command to paste, an estimated wall-clock, and
what "done" looks like. Total: roughly 2.5–4 hours, most of it download and quantization time.

---

## 1. Connect and check CPU features (5-10 min)

Start a `tmux` (or `screen`) session and do all work inside it — a dropped SSH connection must not
lose a multi-hour build or bench run.

```bash
tmux new -s kleidi
```

Confirm the CPU features this machine actually has before anything downstream depends on the
assumption:

```bash
lscpu | grep -oE 'i8mm|sve2|sve|asimddp|bf16'
```

Expected on Neoverse-N2 (Cobalt 100): `i8mm` and `sve2` both present. **If `i8mm` is absent**, the
kernel-path finding still holds — the miss is still real and still detectable — but the measured
delta will differ from the one in `README.md`, because which KleidiAI microkernel gets selected
depends on exactly these features. Record what you actually see; it is a required caveat on any
number you publish from this run.

## 2. Build llama.cpp with KleidiAI (15-25 min)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_KLEIDIAI=ON -DGGML_NATIVE=ON
cmake --build build --config Release -j"$(nproc)"
```

`-DGGML_CPU_KLEIDIAI=ON` is what makes the KleidiAI kernel path exist at all; `-DGGML_NATIVE=ON` is
what gets native `-mcpu=...` codegen for this CPU. Both are required.

**Stale-binary trap** (llama.cpp issue #12701): if you change any cmake flag after a first build,
`rm -rf build` before reconfiguring. A stale partial build silently keeps the old codegen and will
quietly invalidate every benchmark number that follows.

Confirm the binary runs, and record the commit SHA — step 4 needs it:

```bash
./build/bin/llama-bench --help | head -5
git rev-parse HEAD
```

Put the build output on `PATH` (or pass `--llama-bin-dir` on every `kleidi-advisor` invocation
below):

```bash
export LLAMA_BIN_DIR="$PWD/build/bin"
cd ..
```

## 3. Run the ecosystem audit (~10 min)

This step needs nothing but outbound network — no Arm machine required, so it can run anywhere,
including before the instance exists.

```bash
bash scripts/run-audit.sh
```

This scans every entry in `data/hf-top-gguf.txt` head-only — a few MB total, never a full model
download, because GGUF metadata lives at byte 0 — and writes an `AUDIT.md` table plus
`results/audit.json`. Read the printed summary line: it is the ecosystem-measurement claim in one
sentence.

If a URL 404s, that row is recorded as `error` and the run continues; per-entry tolerance is what
makes shipping a hand-assembled candidate list safe. Fix the entry in `data/hf-top-gguf.txt` and
re-run rather than quoting a silently-short audit. Note the total bytes fetched printed alongside —
auditing twenty models for a few tens of MB is itself part of the result.

## 4. Read the raw load logs, then verify dispatch (30-60 min, mostly download time)

**This step is the methodological core of the project. Do not skip to `--verify`.**

1. Download the model files. Qwen2.5-7B-Instruct-GGUF is used here because it is ungated and
   Apache-2.0, and because fp16, q4_k_m *and* q4_0 ship in the same repository — so the baseline
   and the fixed model derive from one lineage with no publisher confound between them:
   ```bash
   pip install -U huggingface_hub
   hf download Qwen/Qwen2.5-7B-Instruct-GGUF --include "*fp16*"   --local-dir .
   hf download Qwen/Qwen2.5-7B-Instruct-GGUF --include "*q4_k_m*" --local-dir .
   ```
   fp16 is ~15.2 GB across 4 shards; q4_k_m is ~4.7 GB across 2 shards.

2. **Merge the fp16 shards** — `llama-quantize` (step 6) needs a single file, not a split set:
   ```bash
   llama-gguf-split --merge qwen2.5-7b-instruct-fp16-00001-of-00004.gguf qwen-fp16.gguf
   ```
   The q4_k_m baseline is only 2 shards; merge it the same way if `bench`/`scan` need one file, or
   check whether the binaries you built accept split GGUFs directly.

3. **Read the raw load logs before running `--verify` at all.** Run each format once and look at
   which model buffer the tensors landed in:
   ```bash
   ./build/bin/llama-bench -m <q4_k_m-baseline.gguf> -p 8 -n 0 -r 1 -v 2>&1 | grep -Ei 'buffer size|kleidiai|repack'
   ./build/bin/llama-bench -m <q4_0.gguf>            -p 8 -n 0 -r 1 -v 2>&1 | grep -Ei 'buffer size|kleidiai|repack'
   ```

   Expected on build b10431 (Neoverse N2):

   ```
   Q4_0:     kleidiai: primary q4 kernel feature I8MM
             load_tensors: CPU_KLEIDIAI model buffer size =  3500.45 MiB
   Q4_K_M:   load_tensors: CPU_REPACK model buffer size =  4166.82 MiB
             repack: repack tensor blk.N.attn_q.weight with q4_K_8x8
   both:     ... cannot be used with preferred buffer type CPU_KLEIDIAI, using CPU instead
   ```

   **Why this is not optional ceremony.** This project was originally built on the assumption that
   there are two load paths — Q4_0 reaches Arm's optimized kernels, everything else runs generic
   ones. There are three. Q4_K_M gets no `CPU_KLEIDIAI` buffer, but it *does* get a `CPU_REPACK`
   buffer and per-tensor `q4_K_8x8` repacking: ggml's own aarch64 path, which is real acceleration
   that simply isn't KleidiAI's. Reading these two logs side by side is what falsified that
   assumption, and it worked only because the logs were read directly rather than through the
   tool's own verdict.

   Two traps are visible in the excerpt above, and both defeat the obvious approach of grepping the
   log for a keyword:
   - `repack` appears for **both** formats. It is emitted by ggml's aarch64 path too, so it proves
     nothing about KleidiAI.
   - `cannot be used with preferred buffer type CPU_KLEIDIAI, using CPU instead` also appears for
     both formats. It contains the string `CPU_KLEIDIAI` while meaning the *opposite* of what a
     substring match would conclude.

   The signal that actually separates the paths is which model buffer the tensors landed in —
   `CPU_KLEIDIAI model buffer size` versus `CPU_REPACK model buffer size` — and those lines print
   only under `-v`. If what you see differs from the excerpt above, the classification table in
   [`REFERENCE.md`](REFERENCE.md) §4 is wrong for your build, and **the log wins**.

4. Now run the on-device verify on both models. Keep the grep excerpts from 4.3 alongside its
   output: the excerpts are the evidence, the verdict line is only the summary.
   ```bash
   kleidi-advisor scan <q4_k_m-baseline.gguf> --verify --llama-bin-dir "$LLAMA_BIN_DIR"
   kleidi-advisor scan qwen-fp16.gguf --verify --llama-bin-dir "$LLAMA_BIN_DIR"
   ```
   `--verify` drives `llama-bench -p 8 -n 0 -r 1 -v` itself. `-v` is required for the buffer lines,
   and `llama-cli` is not usable here — with no prompt it goes interactive and hangs.

5. Record the llama.cpp commit SHA from step 2 next to both results, whatever the outcome.
   - `DISAGREE` means the log contradicts the static table. Trust the log: fix the buffer markers in
     `src/kleidi_advisor/verify.py` and/or the classification table in `REFERENCE.md` §4, note it,
     and re-run before trusting anything downstream of `scan`.
   - `INCONCLUSIVE` means the log carried no `model buffer size` line at all. Check that `-v`
     survived, then note it and move on — it is honest and non-blocking.

## 5. Bench the baseline, with perplexity (25-40 min)

Fetch the WikiText-2 calibration corpus now; it is needed here for baseline perplexity and again in
step 6 for the importance matrix:

```
https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
```

It unzips to `wikitext-2-raw/wiki.test.raw`.

Bench the baseline exactly as downloaded, pinning thread count to physical cores. On an 8-vCPU
Arm SKU with no SMT, `-t $(nproc)` is exactly the physical core count.

**Chunk budget**: a full WikiText-2 pass takes hours on 8 cores. Use `--chunks 100` for perplexity
and *the same value* on both the baseline and the fixed model — step 7's quality gate compares this
file's `ppl.value` directly, so a mismatched chunk count makes the comparison meaningless. Measure
baseline perplexity now; without it the gate has nothing to compare against:

```bash
kleidi-advisor bench <q4_k_m-baseline.gguf> --threads "$(nproc)" --tag baseline -r 5 \
  --perplexity --calib wikitext-2-raw/wiki.test.raw --chunks 100 \
  --llama-bin-dir "$LLAMA_BIN_DIR"
```

This writes `results/<stem>-baseline.json`. Its `pp512`/`tg128` medians are the "before" half of the
headline; its `ppl.value` is what step 7 gates against.

## 6. Requantize to the format that reaches the kernels (20-40 min)

Run the whole chain — importance matrix, quantize to Q4_0, then a smoke generation — in one
command. Pass `--chunks 50` so the imatrix pass doesn't run the full corpus on 8 cores:

```bash
kleidi-advisor fix qwen-fp16.gguf --calib wikitext-2-raw/wiki.test.raw -o qwen-imatrix-q4_0.gguf \
  --chunks 50 --llama-bin-dir "$LLAMA_BIN_DIR"
```

It prints the artifact path and the exact next command (`kleidi-advisor scan <out>`). Run that now
and confirm the verdict flips to `OK_KLEIDIAI`.

**Timebox**: if imatrix is still running past ~30 min, kill it and either drop to `--chunks 25` or
fall back to Qwen2.5-3B-Instruct, restarting from step 4 with the smaller model.

**Optional, for a stricter comparison**: the downloaded q4_k_m baseline and the fixed model already
derive from the same fp16 source, but you can also self-quantize Q4_K_M from your merged file (plain
`llama-quantize qwen-fp16.gguf qwen-q4_k_m-self.gguf Q4_K_M`, no imatrix) so that the quantization
type is the *only* variable between baseline and fixed — including which exact fp16 bytes each came
from.

## 7. Bench the fix and gate on quality (15-20 min)

Same `--chunks 100` as step 5, for the reason given there:

```bash
kleidi-advisor bench qwen-imatrix-q4_0.gguf --threads "$(nproc)" --tag fixed -r 5 \
  --perplexity --calib wikitext-2-raw/wiki.test.raw --chunks 100 \
  --gate results/<baseline-stem>-baseline.json --max-delta 0.3 \
  --llama-bin-dir "$LLAMA_BIN_DIR"
```

Exit 0 means the gate passed: the fixed model is both faster and within the quality budget.

**If the gate fails (exit 5)**, do not discard the run. The command prints both perplexity numbers,
and the honest move is to report the result as "`<N>`× throughput at `+<X>` ppl" rather than to
claim a gate that didn't pass. A speed number without its quality cost is not a result. To print
both numbers again at any time:

```bash
python3 -c "import json,sys; a=json.load(open(sys.argv[1]));b=json.load(open(sys.argv[2]));print(a['ppl'],b['ppl'])" \
  results/<baseline-stem>-baseline.json results/<fixed-stem>-fixed.json
```

**Timebox**: if perplexity is still running past ~30 min, kill it and drop to `--chunks 25` —
re-running *both* the baseline and the fixed model at the new value so they stay comparable — or
fall back to Qwen2.5-3B-Instruct.

## 8. Report (10 min)

```bash
kleidi-advisor report --instance <actual-instance-type> --plot results/plot.png
```

This reads every `results/*.json` and renders `RESULTS.md` plus a grouped-bar plot. Every throughput
figure it emits is printed adjacent to its perplexity delta, or to `ppl: not measured` — it will not
print a bare speedup, by design. Name the actual silicon you measured on, not just the instance
type: the magnitude of the difference depends on which CPU features step 1 found.

## 9. Cold reproduce (15-20 min)

Open a fresh shell — new SSH connection, new `tmux` pane, no warmed-up state — and re-run both
headline benches from scratch:

```bash
kleidi-advisor bench <baseline-q4_k_m.gguf>    --threads "$(nproc)" --tag baseline-cold -r 5 --llama-bin-dir "$LLAMA_BIN_DIR"
kleidi-advisor bench <model-imatrix-q4_0.gguf> --threads "$(nproc)" --tag fixed-cold    -r 5 --llama-bin-dir "$LLAMA_BIN_DIR"
```

Confirm both medians land within one standard deviation of the numbers you already recorded. If
they don't, investigate before publishing anything — a headline that doesn't reproduce cold is not
a headline.
