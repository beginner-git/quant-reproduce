#!/usr/bin/env bash
# Set up a method's conda env on the Linux lab server.
# Run from repo root: bash scripts/env_lab.sh GPTQ
#
# Lab convention: HF_HOME points to a shared cache dir
# (set via /etc/profile.d/ or per-user .bashrc, not here).

set -euo pipefail

METHOD="${1:?Usage: $0 <GPTQ|AWQ|BiLLM|KIVI>}"
ENV_NAME="quant-$(echo "$METHOD" | tr '[:upper:]' '[:lower:]')"
YML_PATH="$(dirname "$0")/../$METHOD/env.yml"

if [[ ! -f "$YML_PATH" ]]; then
    echo "No env.yml at $YML_PATH" >&2
    exit 1
fi

echo "Creating conda env $ENV_NAME from $YML_PATH"
conda env create -f "$YML_PATH"

echo "Activate then install common:"
echo "  conda activate $ENV_NAME"
echo "  pip install -e ."

if [[ -z "${HF_HOME:-}" ]]; then
    echo "Tip: export HF_HOME=/shared/path/huggingface_cache to share model weights"
fi
