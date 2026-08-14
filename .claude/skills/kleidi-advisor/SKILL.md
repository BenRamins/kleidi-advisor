---
name: kleidi-advisor
description: Diagnose whether a GGUF model reaches Arm's KleidiAI Q4_0 kernel-path or silently falls back to generic kernels. Trigger on GGUF, KleidiAI, llama.cpp on Arm or Graviton, or "why is my model slow on Arm/Graviton".
---

# kleidi-advisor — Arm KleidiAI kernel-path advisor

When a user mentions a GGUF model, KleidiAI, or asks why their model is slow on Arm/Graviton:

1. Run `kleidi-advisor scan <path-to-model.gguf>` (add `--json` if you need the fields
   programmatically) and quote the verdict line back to the user verbatim — don't paraphrase it.

2. If the verdict is `NOT_KLEIDIAI_PATH` (K-quants — the common case), deliver the reveal:

   > This model never reaches Arm's KleidiAI kernels. It isn't unoptimised — K-quants take ggml's
   > own aarch64 repack path (`CPU_REPACK`), which is why nothing in the log looks wrong — but that
   > path measured 1.61× slower at pp512 than Q4_0 on Arm Neoverse-N2 (Azure Cobalt 100, llama.cpp
   > b10431; ppl cost of the switch: +0.049, inside the error bars of both runs — measure yours with
   > `bench --perplexity`). It's a
   > common state: our audit found 13 of 17 scannable widely-published GGUFs missing the path —
   > including all 7 Q4_K_M files in the list, from all 5 publishers that ship one.

   Do **not** quote the older ~2.5–2.9× figure from llama.cpp PR #9921 as what the user should
   expect: it predates the K-quant repack path, so it compares against a fallback this model isn't
   taking. It is historical context only — not our measurement, and superseded by the one above.

3. If the verdict is `FALLBACK_GENERIC` (IQ types), say the same thing with one difference: neither
   a `CPU_KLEIDIAI` nor a `CPU_REPACK` buffer is expected, so this one really does run the generic
   kernels — and note that no IQ model was on the measurement box, so that classification is a
   prediction and `--verify` is worth more here than anywhere else.

   After either reveal, offer these three next steps, in this order:
   - `kleidi-advisor scan <model> --verify` — prove the prediction against this machine's real
     load log before trusting it. It runs `llama-bench -v` and reads which model buffer the tensors
     landed in; grepping the log yourself for "repack" or "kleidi" does not work, because both
     strings appear for models that never reach KleidiAI.
   - `kleidi-advisor fix <source-f16.gguf> --calib <corpus.txt> -o <out.gguf>` — requantize onto
     the format that reaches the kernels.
   - `kleidi-advisor bench <out.gguf> --threads N --tag fixed` — measure it on this machine. Pair
     it with `--perplexity`: a speed number without its quality cost is not a result.

4. If the verdict is `OK_KLEIDIAI`, say so plainly and suggest `--verify` if the user wants to
   confirm it against this machine's actual llama.cpp build.

5. If the verdict is `NOT_APPLICABLE`, say so plainly — this weight type was never a Q4_0-repack
   candidate, so there's no miss to report. If it's `UNKNOWN_VERIFY_ON_DEVICE`, say the table
   doesn't recognise this tensor type and suggest `--verify` instead of trusting a guess.

Never claim credit for the kernels or the speedup — they are Arm's KleidiAI work. This tool's own
contribution is detecting the miss and measuring the ecosystem, nothing more.
