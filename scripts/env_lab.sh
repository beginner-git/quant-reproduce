#!/usr/bin/env bash
# Set up a method's conda env on the GPU server (yiminl50@<host>).
# Run from repo root: bash scripts/env_lab.sh AWQ
#
# Server pre-config (already in ~/.bashrc, do not override):
#   HF_HOME=/shared/yiminl50/hf_cache    （模型权重共享缓存）
#   ~/.condarc envs_dirs → /shared/yiminl50/conda_envs/   （env 走大盘）
# 所以创建出来的 env 自动落在 /shared，不挤 /home。

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

cat <<EOF

Done. Next:
  conda activate $ENV_NAME

Phase 1 不需要在 method env 里装本仓库 common/（common 是 Phase 3 资产）。
直接进 $METHOD/third_party/<UPSTREAM>/ 跑上游脚本即可。

跑长任务（量化 + eval >5 min）记得开 tmux：
  tmux new -s ${ENV_NAME}-run

多用户 GPU：开跑前 \`gpu\` 看哪几张空闲，CUDA_VISIBLE_DEVICES 显式指定，别一把抓 4 张。
EOF
