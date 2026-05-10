#!/usr/bin/env bash
# Run a method's repro.py with canonical settings on the lab server.
# Usage: bash scripts/run_phase1_method.sh GPTQ [--smoke]
#
# Assumes: conda env quant-<method> already created via env_lab.sh
# and `pip install -e .` already run inside it.

set -euo pipefail

METHOD="${1:?Usage: $0 <GPTQ|AWQ|BiLLM|KIVI> [extra-repro-args...]}"
shift || true

ENV_NAME="quant-$(echo "$METHOD" | tr '[:upper:]' '[:lower:]')"
SUBDIR="$(dirname "$0")/../$METHOD"

echo "Activating $ENV_NAME"
# `conda activate` requires shell hook
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$SUBDIR"
echo "Running repro.py in $(pwd)"
python repro.py \
    --model meta-llama/Llama-2-7b-hf \
    --config w4g128 \
    --eval ppl,zeroshot,memory \
    --out results/ \
    "$@"
