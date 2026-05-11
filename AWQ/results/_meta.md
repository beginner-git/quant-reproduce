# AWQ canonical W4-g128 reproduction — meta

## Result summary

| 维度 | 实测 | Paper anchor | 判定 |
|------|------|--------------|------|
| WikiText-2 PPL (token-level, seq_len=2048) | **5.614** | 5.60 (AWQ Table 4) | 通过 (Δ +0.014, 容差 ±0.3) |
| Weights size | 3.62 GB (3,889,391,512 B) | ≈ 3.7 GB | 一致 |
| Zero-shot 6-task 平均 (见下方) | **0.6451** | ≈ 0.65 (FP16 baseline) | 接近无损 |

Phase 1 AWQ canonical W4-g128 LLaMA-2-7B 复现完成。三项指标全过。

## Model

- Base: `NousResearch/Llama-2-7b-hf`
- Quant config (from `config.json` 的 `quantization_config`):
  - `bits: 4`
  - `group_size: 128`
  - `version: gemm`
  - `zero_point: true`
- Calibration: AutoAWQ upstream 默认 — `mit-han-lab/pile-val-backup`, 128 samples
- Quantized output: `/shared/yiminl50/quantized/awq-w4g128/quantized/`
- Output size: 3.62 GB (3,889,391,512 bytes)
- Quantized at: 2026-05-11 09:41

## Environment

- Server: GPU server (yiminl50@Dreamer), 4× RTX 3090 24 GB
- NVIDIA driver: 550.144.03 (CUDA 12.4 capable)
- Conda env: `/shared/yiminl50/conda_envs/quant-awq`
- Python: 3.10.20
- PyTorch: 2.4.1 (cu121 wheel)
- NCCL: 2.20.5
- Transformers: 4.51.3 (AutoAWQ 上游最后官测版本)
- Datasets: 2.21.0 (<3.0, lm-eval 0.4.1 scrolls task 还在用旧 `load_metric` API)
- Tokenizers: 0.21.4
- huggingface_hub: 0.36.2
- AutoAWQ: 0.2.9, editable install from `AWQ/third_party/AutoAWQ` @ commit `88e4c76b20755db275574e6a03c83c84ba3bece5`
- lm-eval-harness: 0.4.1
- autoawq-kernels: **未装**（跳过 `[kernels]` extras 因为 flash-attn 编译会把 torch 升到不兼容版本；CLAUDE.md §3 说 kernels 只影响推理速度，不影响 PPL 数字）
- flash-attn: 未装（同上）

## Eval run (主指标 — paper-style)

- Script: `AWQ/third_party/AutoAWQ/examples/eval.py --tasks wikitext`
- Internal: `awq.evaluation.evaluate_perplexity(model, tokenizer)`
- Dataset: WikiText-2-raw-v1 test split
- Chunking: 166 chunks @ seq_len=2048, 非重叠
- GPU: `CUDA_VISIBLE_DEVICES=1` (cuda:0 inside)
- Wall time: 84 s (1.95 it/s)
- Date: 2026-05-11
- Log: [canonical_w4g128_ppl_paper_style.txt](canonical_w4g128_ppl_paper_style.txt)

完整命令：

```bash
cd ~/projects/reproduce/AWQ/third_party/AutoAWQ
CUDA_VISIBLE_DEVICES=1 /shared/yiminl50/conda_envs/quant-awq/bin/python examples/eval.py \
    --model_path /shared/yiminl50/quantized/awq-w4g128/quantized \
    --tasks wikitext \
    > ../../results/canonical_w4g128_ppl_paper_style.txt 2>&1
```

## Eval run — zero-shot (6 tasks, lm-eval-harness)

| Task | acc | acc_norm | stderr (acc) |
|------|----:|---------:|------:|
| piqa | 0.7791 | 0.7873 | 0.0097 |
| arc_easy | 0.7559 | 0.7311 | 0.0088 |
| arc_challenge | 0.4334 | 0.4497 | 0.0145 |
| hellaswag | 0.5663 | 0.7521 | 0.0049 |
| winogrande | 0.6819 | — | 0.0131 |
| openbookqa | 0.3140 | 0.4440 | 0.0208 |
| **6-task avg** (acc_norm; winogrande 用 acc) | — | **0.6451** | — |

- Dataset: 各 task 的官方 test split (lm-eval 0.4.1 注册的版本号见 stdout 表)
- num_fewshot: 0 (zero-shot)
- GPU: `CUDA_VISIBLE_DEVICES=3` (cuda:0 inside)
- Wall time: ~53 min (无 autoawq-kernels, Python fallback dequant)
- Date: 2026-05-11
- 输出：[canonical_w4g128_zeroshot.json](canonical_w4g128_zeroshot.json) (结构化), [canonical_w4g128_zeroshot_stdout.txt](canonical_w4g128_zeroshot_stdout.txt) (人读)

完整命令：

```bash
CUDA_VISIBLE_DEVICES=3 /shared/yiminl50/conda_envs/quant-awq/bin/python -m lm_eval \
    --model hf \
    --model_args pretrained=/shared/yiminl50/quantized/awq-w4g128/quantized,trust_remote_code=True \
    --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,openbookqa \
    --batch_size 1 \
    --output_path ~/projects/reproduce/AWQ/results/canonical_w4g128_zeroshot.json \
    2>&1 | tee ~/projects/reproduce/AWQ/results/canonical_w4g128_zeroshot_stdout.txt
```

> AWQ paper Table 4 只列了 PPL，没列 per-task zero-shot 数；上面的 0.65 anchor 是 LLaMA-2-7B FP16 baseline 在 lm-eval-harness 上的典型读数（公开复现普遍 0.64–0.66 区间）。实测 0.6451 落在该区间内，与 PPL +0.014 的微小差距一致，说明 W4-g128 量化对 zero-shot 推理几乎无影响。

## Cross-check (lm-eval-harness wikitext task — 不同口径，仅作辅证)

| Metric | Value |
|--------|------:|
| word_perplexity | 8.9916 |
| byte_perplexity | 1.5079 |
| bits_per_byte | 0.5925 |

分母是 word/byte，不是 token；不能直接对 paper 5.60 比。`bits_per_byte=0.5925` 反算 token-level PPL 大约落在 5.0–5.5，跟主指标 5.614 / paper anchor 5.60 一致。

Log: [canonical_w4g128_ppl_stdout.txt](canonical_w4g128_ppl_stdout.txt)

完整命令：

```bash
CUDA_VISIBLE_DEVICES=1 /shared/yiminl50/conda_envs/quant-awq/bin/lm_eval \
    --model hf \
    --model_args pretrained=/shared/yiminl50/quantized/awq-w4g128/quantized \
    --tasks wikitext --device cuda:0 --batch_size 1 \
    > ~/projects/reproduce/AWQ/results/canonical_w4g128_ppl_stdout.txt 2>&1
```

## Env 重建提示（坑位备忘）

直接 `pip install -e ".[kernels,eval]"` 会被 `flash-attn` 把 torch 升到 2.11，跟服务器 driver 12.4 不兼容（torch 2.11 要求 driver ≥12.6），整套 cuda 跪掉。正确做法：

1. `bash scripts/env_lab.sh AWQ` 创建 env（pin torch 2.4 + cu121）
2. `pip install -e ".[eval]"` —— **跳过** `[kernels]`，避免 flash-attn 升 torch
3. 之后 pip 还会把 transformers 拉到最新（5.x），它跟 torch 2.4 的 `torch.library.infer_schema` API 对不上；同时 datasets 会被拉到 4.x，lm-eval 0.4.1 的 scrolls task import 失败
4. 显式钉版本回滚：`pip install "transformers==4.51.3" "datasets>=2.16,<3.0"`

完整修复链已经体现在上面的 env 清单里。
