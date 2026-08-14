---
name: kleidi-advisor
description: Diagnose whether a GGUF model reaches Arm's KleidiAI Q4_0 kernel-path or silently falls back to generic kernels. Trigger on GGUF, KleidiAI, llama.cpp on Arm or Graviton, or "why is my model slow on Arm/Graviton".
---

# kleidi-advisor — Arm KleidiAI kernel-path advisor

When a user mentions a GGUF model, KleidiAI, or asks why their model is slow on Arm/Graviton:

1. Run `kleidi-advisor scan <path-to-model.gguf>` (add `--json` if you need the fields
   programmatically) and quote the verdict line back to the user verbatim — don't paraphrase it.

2. If the verdict is `FALLBACK_GENERIC`, deliver the reveal:

   > This model never reaches Arm's KleidiAI kernels — it silently runs the generic path. That's
   > common: our audit found `TODO(box)` of 20 widely-downloaded GGUFs in the same state.
   > Published uplifts for the fixed format reach ~2.5–2.9× prompt processing on Graviton-class CPUs (llama.cpp PR #9921 — not our measurement; run `bench` for yours).

   Then offer these three next steps, in this order:
   - `kleidi-advisor scan <model> --verify` — prove the prediction against this machine's real
     llama-cli load log before trusting it (the classification table is version-pinned to one
     llama.cpp commit and can be wrong on a different build).
   - `kleidi-advisor fix <source-f16.gguf> --calib <corpus.txt> -o <out.gguf>` — requantize onto
     the format that reaches the kernels.
   - `kleidi-advisor bench <out.gguf> --threads N --tag fixed` — measure it for real on this
     machine rather than repeat the published anchor number above.

3. If the verdict is `OK_KLEIDIAI`, say so plainly and suggest `--verify` if the user wants to
   confirm it against this machine's actual llama.cpp build.

4. If the verdict is `NOT_APPLICABLE`, say so plainly — this weight type was never a Q4_0-repack
   candidate, so there's no miss to report. If it's `UNKNOWN_VERIFY_ON_DEVICE`, say the table
   doesn't recognise this tensor type and suggest `--verify` instead of trusting a guess.

Never claim credit for the kernels or the speedup — they are Arm's KleidiAI work. This tool's own
contribution is detecting the miss and measuring the ecosystem, nothing more.
