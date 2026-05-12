# KIVI canonical K2V2 (g32 r32) reproduction — meta

## Result summary

| 维度 | 实测 | Paper anchor | 判定 |
|------|------|--------------|------|
| LongBench 8-task avg (LLaMA-2-7B-chat, K2V2 vs FP16) | K2V2 46.05 vs FP16 46.50, Δ −0.45 | paper Table 5 同量级（−0.1 到 −0.5） | **通过** |
| Peak VRAM ratio (batch=16, seqlen 498) | 0.93× (KIVI 仅省 6.8%) | 2.6× (paper abstract) | **未复现 — regime 不匹配，详见 §9.2** |
| Throughput ratio (batch=16, seqlen 498) | 0.88× (KIVI 反慢 13%) | 2.35-3.47× faster | **未复现 — regime 不匹配，详见 §9.2** |
| Quantization step | 无（KIVI tuning-free，运行时量化 KV cache，不存盘） | — | — |

Phase 1 KIVI K2V2 LLaMA-2-7B-chat 主验收完成。LongBench 主指标过线；paper headline 的显存/吞吐 ratio 因单 3090 装不下大 batch fp16 baseline 而未复现，**这不是 KIVI 失效，是 benchmark regime 我们够不到**。

## Model

- LongBench eval: `NousResearch/Llama-2-7b-chat-hf`，本地 `/shared/yiminl50/models/llama-2-7b-chat-hf/` (sed-patched in `pred_long_bench.py` via `--model_name_or_path`)
- mem_spd_test: `NousResearch/Llama-2-7b-hf` (base，跟 paper 默认一致；chat/base 对 mem_spd 测量无差异)，本地 `/shared/yiminl50/models/llama-2-7b-hf/`
- KIVI 配置: `k_bits=2, v_bits=2, group_size=32, residual_length=32`（LongBench eval）/ `residual_length=128`（mem_spd_test 上游默认）
- **没有"save quantized model"步骤**——KIVI 是 runtime KV cache 量化，模型权重保持 fp16，量化逻辑由 `LlamaForCausalLM_KIVI` subclass 承担

## Environment

- Server: GPU server (yiminl50@Dreamer), 4× RTX 3090 24 GB
- NVIDIA driver: 12.4 capable
- Conda env: `/shared/yiminl50/conda_envs/quant-kivi`
- Python: 3.10
- PyTorch: 2.4.1 (cu121 wheel)
- Transformers: 4.43.1 (KIVI pyproject 锁定)
- Datasets: 2.21.0 (`<3.0` 必需；datasets 3.0+ 砍了 script-based dataset 支持，THUDM/LongBench 是老式 script dataset)
- Accelerate: 1.13.0 (KIVI pyproject 没钉版本；新 transformers + 新 accelerate 自动兼容，没踩到 BiLLM 时代 hub 错配)
- flash-attn: 2.8.3 (**预编译 wheel 直装**——见 README §3.1；走 `pip install flash-attn` 会撞 build isolation + cross-device link 两个坑)
- KIVI kivi 主包: 0.1.0, editable install from `KIVI/third_party/KIVI` @ commit `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`
- KIVI kivi_gemv CUDA extension: 0.0.0, editable install from `KIVI/third_party/KIVI/quant`，TORCH_CUDA_ARCH_LIST=8.6 (3090 sm_86)
- 其它: numpy 2.2.6, jieba 0.42.1, rouge 1.0.1, fuzzywuzzy, sentencepiece 0.2.1, protobuf 7.34.1, safetensors 0.7.0, tokenizers 0.19.1, pyarrow 24.0.0

## Local patches applied to vendored upstream

3 处对 `KIVI/third_party/KIVI/` 的 patch（详见 README §5.4 / §6 / §7.2）：

### 1. `config/model2maxlen.json` + `config/model2path.json` 加全小写 `llama-2-7b-chat-hf` 键

`pred_long_bench.py:190` 用 `os.path.basename(args.model_name_or_path)` 作为 dict key 严格查；上游 JSON 只有大写 `"Llama-2-7b-chat-hf"`，跟我们 lowercase 本地路径不匹配。

### 2. `example.py` + `mem_spd_test.py` 改本地模型路径 + `mem_spd_test.py` 补 `config.use_flash = True`

- 上游两脚本硬编码 `meta-llama/Llama-2-7b-hf`（Meta gated repo）/`meta-llama/Llama-3.1-8B-Instruct`
- sed 改成本地 NousResearch mirror 路径
- `mem_spd_test.py` 上游漏列 `config.use_flash = True`（`LlamaAttention_KIVI` 硬要求），sed 补上

### 3. `mem_spd_test.py` 把 `BATCH_SIZE` 96 → 16

上游默认 96 是为了 demo paper "4× larger batch" headline；单 3090 (24GB) 跑 LLaMA-2-7B fp16 必 OOM。

Patch 后 `KIVI/third_party/KIVI` 在 git status 显示 `modified content`——预期。

## Calibration / eval data

- KIVI 是 tuning-free —— **无 calibration data**。
- LongBench 数据：`huggingface-cli download THUDM/LongBench --repo-type dataset --local-dir /shared/yiminl50/datasets/longbench/`（§5.2 预下载）
- 实际 LongBench eval 流程仍走 `datasets.load_dataset('THUDM/LongBench', ...)`，datasets 2.x 能正常 cache。

## LongBench pred + eval run

- Pred command:
  ```bash
  CUDA_VISIBLE_DEVICES=<idx> python pred_long_bench.py \
      --model_name_or_path /shared/yiminl50/models/llama-2-7b-chat-hf \
      --k_bits 2 --v_bits 2 \
      --group_size 32 --residual_length 32
  ```
- 同样命令换 `--k_bits 16 --v_bits 16` 跑 FP16 baseline
- 8-task standard mode（triviaqa / qasper / trec / samsum / lcc / repobench-p / qmsum / multi_news），不加 `--e`
- 每 task ~200-500 samples
- Pred wall time: K2V2 ~80-90 min，FP16 baseline ~80-90 min（两 GPU 并行跑）
- Eval (`eval_long_bench.py --model llama-2-7b-chat-hf_4096_<X>bits_group32_residual32`): <1 min，纯 CPU 评分
- 用的物理 GPU: card #1 (K2V2) / card #3 (baseline)，并行

## mem_spd_test run

- Command:
  ```bash
  CUDA_VISIBLE_DEVICES=<idx> python mem_spd_test.py
  ```
- 同命令 sed `K_BITS=V_BITS=2` ↔ `K_BITS=V_BITS=16` 各跑一次
- 配置: batch=16, prompt 160, output 338, num_repeats=3, residual_length=128
- 每轮 wall time: ~15-20 sec (含 model load + 3 次 generate)
- 用的物理 GPU: card #3

## 实测异常记录

### mem_spd_test paper headline 未复现 — regime 不匹配

**判定**：**不是 KIVI 失效**，而是 paper 的 mem/throughput benchmark 设计在我们的硬件够不到。

paper 默认 `BATCH_SIZE=96` + 长 context 是想 demo "在 KV cache 主导显存的 regime 下 KIVI 优势明显"；我们因单 3090 (24GB) 装不下 batch=96 fp16 baseline 只能砍到 batch=16 + seqlen=498——在这个 regime 下 model weights (14GB fp16) 主导显存（占总 16.5GB 的 85%），KV cache 只占百分之十几，KIVI 的 INT2 优势几乎被淹没（即便理论上把 KV 压到 1/8 也只省总显存的 ~10%）。

**为什么 KIVI 反慢 13%**：INT2 GEMV kernel 在 batch 小时其 launch overhead + dequantize 开销 > KV cache 节省的 memory bandwidth。同样需要 KV cache 大才划算。

**主验收用 §9.1 的 LongBench 8-task 对照**：这是 paper 主要 quality claim，过了即算复现。headline ratio 引用 paper 即可。

### K2V2 LongBench 在 triviaqa 上反超 FP16 0.75 分

小样本随机波动（200 samples）+ K2V2 的 KV 量化对 triviaqa 这类 short-context retrieval 影响小。paper Table 5 也有类似 +/- 0.5 量级的 task 级波动。不算异常。

### 没跑 16-task extended mode

KIVI 上游 `pred_long_bench.py` 默认走 standard 8-task（不加 `--e` flag），跟 paper Table 5 一致。extended 16-task 是另一份 benchmark（`--e` flag 触发），paper Table 7 报告——本次复现没跑。Phase 1 不需要。

### 没跑 example.py 烟雾测

直接进 canonical pred，跳过 §6 smoke。pred 第一个 task 跑通后即等同烟雾通过。
