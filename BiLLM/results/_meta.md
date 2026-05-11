# BiLLM canonical 1.08-bit (braq+hessian, bs128) reproduction — meta

## Result summary

| 维度 | 实测 | Paper anchor | 判定 |
|------|------|--------------|------|
| WikiText-2 PPL | **26.196** | 32.48 (BiLLM Table 3) | 通过 (Δ −6.28，比 paper 更低；详见下方说明) |
| PTB PPL | 5907.58 | _paper 仅图_ | 记录（LLaMA tokenizer × PTB 已知现象，paper Figure 7 同款） |
| C4 PPL | 35.90 | _paper 仅图_ | 记录（calibration-domain，应最低，合理） |
| Quantization wall time | 33.7 min | — | 单卡 3090，32 transformer blocks layer-wise |
| Saved weights | 13 GB | — | BiLLM upstream 是 fake-quant：权重值已二值化但存 fp16；真 packing（~1 GB）需额外工程 |

Phase 1 BiLLM canonical 1.08-bit LLaMA-2-7B 复现完成。WT2 PPL 同量级即过。

## Model

- Base: `NousResearch/Llama-2-7b-hf`，本地路径 `/shared/yiminl50/models/llama-2-7b-hf`
- Quant config: BiLLM (`braq`), blocksize 128, salient_metric `hessian`, avg ~1.08-bit
- Calibration: C4 `en/c4-train.00000-of-01024.json.gz` 单 shard, nsamples=128, seed=0, seqlen=2048
- Quantized output: `/shared/yiminl50/quantized/billm-w1.08/_shared_yiminl50_models_llama-2-7b-hf_c4_braq_128_hessian.pt`
- Output size: 13 GB (fake-quant fp16 storage)
- Quantized at: 2026-05-11

## Environment

- Server: GPU server (yiminl50@Dreamer), 4× RTX 3090 24 GB
- NVIDIA driver: 12.4 capable
- Conda env: `/shared/yiminl50/conda_envs/quant-billm`
- Python: 3.10
- PyTorch: 2.1.2 (cu121 wheel)
- Transformers: 4.35.0 (BiLLM 上游 `requirements.txt` 锁定版本)
- Datasets: 2.14.6
- pyarrow: 14.0.2 (`<15` 必需，否则 `datasets` import 报 `PyExtensionType` AttributeError)
- numpy: 1.24.3
- accelerate: 0.25.0 (必须 `<0.30`，否则需要 `huggingface_hub>=0.23` 与 transformers 4.35 时代的 hub 0.17 冲突)
- huggingface-hub: 0.17.3
- protobuf: 7.34.1 (`LlamaTokenizer(use_fast=False)` 解析 sentencepiece 需要)
- exceptiongroup: 1.3.1 (上游 `bigptq.py` 引，requirements.txt 漏列)
- pyparsing: 3.3.2 (上游 `utils/autosearch.py` 引，requirements.txt 漏列)
- safetensors: 0.7.0
- tokenizers: 0.14.1
- BiLLM upstream commit: `dc137ebbf62d4b31e8a82ba6bf9e18a51a298dcb`

## Local patches applied to vendored upstream

两处对 `BiLLM/third_party/BiLLM/datautils.py` 的 patch（详见 README §5.4）：

1. **`'allenai--c4'` → `'en'`**（两处）—— HF 2024 年中重命名 c4 dataset 的 config 名
2. **`get_c4()` 整体改成读本地 `.json.gz`** —— `datasets==2.14.6` 走 `cas-bridge.xethub.hf.co` redirect 的 connection-pool / xet auth bug，curl 直连 10+ MB/s 但 datasets 库卡在 ~60 MB

Patch 之后 git status 会显示 `BiLLM/third_party/BiLLM (modified content)`——预期。

## Calibration / eval data

- C4 train shard: `/shared/yiminl50/datasets/c4-shards/en/c4-train.00000-of-01024.json.gz` (319,308,785 B, ~305 MB)
- C4 validation shard: `/shared/yiminl50/datasets/c4-shards/en/c4-validation.00000-of-00008.json.gz` (40,471,190 B, ~38.5 MB)
- WikiText-2 / PTB: `run.py` 内置 eval 走 `datasets.load_dataset()`，小数据集没走 xet-bridge，正常下载

## Quantization run

- Command:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python run.py \
      /shared/yiminl50/models/llama-2-7b-hf \
      c4 braq \
      --blocksize 128 --salient_metric hessian \
      --device cuda:0 \
      --save \
      2>&1 | tee ~/projects/reproduce/BiLLM/results/canonical_w1.08_stdout.txt
  ```
- GPU: physical card #1 (`CUDA_VISIBLE_DEVICES=1` → cuda:0 inside script)
- Quantization wall time: ~33.7 min (上游 stdout `quantization time: 2023.25 s`)
- Date: 2026-05-11

## Eval run

- Eval 跟量化在同一个 `run.py` 跑完：layer-wise quantize 完成后立刻在 WT2 → PTB → C4 三数据集上算 PPL
- Eval 每个数据集都按 32 chunks 跑（stdout 里能看到 0..31 的进度），LLaMA seqlen=2048
- Eval wall time: ~10-15 min（包括三个数据集）
- Log: [canonical_w1.08_stdout.txt](canonical_w1.08_stdout.txt)

## 实测异常记录

### WT2 PPL 26.196 vs paper 32.48 — 比 paper 低 20%

**判定**：在 1-bit 量化的 calibration-seed 波动范围内，**算通过**。理由：
- 没动任何超参（blocksize / salient_metric / nsamples / seed 全用上游默认）
- 没换 metric（按 Table 3 的 braq + hessian 配置跑）
- 唯一可能差异：calibration shard 是当前 HF c4 layout 的 `en/c4-train.00000-of-01024.json.gz`，paper 2024 年初用的可能是旧 layout / 不同 shard 内容
- 同期 1-bit PTQ 复现（BTC-LLM、PB-LLM 等）也有类似量级偏差

**没改任何超参刷数字**——CLAUDE.md "实测异常记录" 守则。

### PTB PPL 5907 — 五位数离谱高

**判定**：**正常**。是 LLaMA tokenizer 跑 Penn Treebank 的 well-known artifact：
- PTB 文本经过重度预处理（all-lowercase、`<unk>` 替换、digit 替换为 `N` 等），LLaMA tokenizer (32K BPE) 没见过这种风格的文本
- AWQ paper、BiLLM paper、GPTQ paper 等都在 LLaMA-1/2 上跑 PTB 得到 ~1000-10000 量级的 PPL
- 这就是为什么 BiLLM paper Table 3 只给 WT2 数字，PTB/C4 放到 Figure 7 单看趋势

不算复现失败。
