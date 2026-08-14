#!/bin/sh
# Runs the real ecosystem audit against data/hf-top-gguf.txt.
#
# Needs open network access to Hugging Face. This is REPRODUCE.md step 3 —
# the operator runs it by hand. The build agent that wrote this repo never
# executes this script and never creates AUDIT.md itself (CLAUDE.md
# guardrail): data/hf-top-gguf.txt is an unverified candidate list, and
# audit's per-entry error tolerance (Spec F7 rule 1) is exactly what makes
# running it safe the first time, 404s and all.
set -eu

cd "$(dirname "$0")/.."
kleidi-advisor audit \
  --list data/hf-top-gguf.txt \
  --out results/audit.json \
  --md AUDIT.md \
  --delay 1

echo "Audit complete. See AUDIT.md and results/audit.json."
