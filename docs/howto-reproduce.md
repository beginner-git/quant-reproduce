# 复现操作手册（How to Reproduce）

> 适用：Plan A 起的所有 Phase 1 + Phase 2 工作。读完这份就能开干。
> 配套阅读：[设计文档](superpowers/specs/2026-05-09-quant-reproduce-design.md) · [Plan A](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md)

## 0. 总览

每个量化方法走同一个 6 步循环。一次只做一个方法（GPTQ → AWQ → BiLLM → KIVI）：

```
┌─ 1. 子目录骨架        (README + env.yml + requirements.txt + repro.py 占位)
├─ 2. 建 conda env      (.\scripts\env_local.ps1 <METHOD>)
├─ 3. 写 repro.py 算法侧 (调官方库做 quantize → 用 common/eval 出数字)
├─ 4. 本地烟雾跑        (TinyLlama-1.1B / calib=32 / seq=1024，5–10 min)
├─ 5. Lab canonical 跑  (LLaMA-2-7B / calib=128 / seq=2048，30–90 min)
└─ 6. Phase 2 源码笔记  (docs/reports/<method>.md，五节模板)
```

**不变量**（所有方法都靠 `common/` 保证横向可比）：

- 评测数据：`load_wikitext2_test` (GPTQ 协议) + `load_c4_calibration(seed=42)`
- PPL 公式：`compute_ppl(seq_len=2048, stride=2048)` (GPTQ 论文 `loss × seq_len`)
- Zero-shot：`evaluate_zeroshot()` 默认 6 项 (piqa / arc_e / arc_c / hella / wino / obqa)
- 内存：`measure_weight_memory()` 读真实 state_dict 字节数
- 元数据：每跑一次都自动写 `meta_<config>.json` (pkg 版本、HF SHA、CLI、GPU、时间)

---

## 1. 一次性准备（整个项目只做一次）

### 1.1 HuggingFace token

LLaMA-2-7B 需要授权访问。先到 <https://huggingface.co/meta-llama/Llama-2-7b-hf> 同意 license，然后：

```powershell
huggingface-cli login
# 粘贴 token (Settings → Access Tokens → Read 权限就够)
```

### 1.2 共享模型 cache（避免下载 4 次）

设置 `HF_HOME` 到一个固定目录，所有 conda env 共用。建议放到大盘：

```powershell
# 临时（当前 PowerShell 会话）
$env:HF_HOME = "D:\hf_cache"

# 永久（用户级环境变量）
[Environment]::SetEnvironmentVariable("HF_HOME", "D:\hf_cache", "User")
# 重开 PowerShell 才生效
```

Lab Linux 服务器上：

```bash
# 加到 ~/.bashrc
export HF_HOME=/shared/path/huggingface_cache
```

### 1.3 PowerShell 执行策略（首次跑 .ps1 脚本）

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# 一次性，之后跑 scripts\env_local.ps1 不再报错
```

---

## 2. 通用工作流（每方法 6 步）

### 2.1 写子目录骨架

每个方法子目录都长这样：

```
GPTQ/                       (后续 AWQ/ BiLLM/ KIVI/ 同结构)
├── README.md               # 跑法 + 数字表 + 论文对比 + troubleshooting
├── env.yml                 # conda env 定义（method 专属依赖）
├── requirements.txt        # pip 依赖（与 env.yml 同步）
├── repro.py                # 唯一入口，CLI 参数 → quantize → eval → 写 results
├── third_party/            # vendor 或 submodule（仅 BiLLM / KIVI 必要）
└── results/                # 数字 JSON + meta JSON + 每片 NLL CSV
```

模板见 Plan A Task 8。

### 2.2 建 conda env

```powershell
# 仓库根
.\scripts\env_local.ps1 GPTQ
conda activate quant-gptq
pip install -e .                    # 装 common 到这个 env
```

> 当前活跃 base env 装的是 `torch 2.11.0+cpu`（无 CUDA）。每个方法的 `env.yml` 应该指定 `pytorch-cuda=12.1` channel 装 CUDA 版，避开污染 base。
>
> 如果发现 env 里 `import torch; torch.cuda.is_available()` 是 `False`，重装 CUDA 版：
> ```
> pip uninstall torch -y
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 2.3 写 repro.py（这是你写算法的地方）

`repro.py` 的固定 CLI 契约（四方法统一，方便后面 unified pipeline 复用）：

```bash
python repro.py \
  --model meta-llama/Llama-2-7b-hf \
  --config w4g128            \   # 该方法预设组合
  --calib-samples 128        \
  --seq-len 2048             \
  --eval ppl,zeroshot,memory \
  --device cuda              \
  --save-quant ./quantized_w4g128/ \
  --out results/             \
  [--smoke]                      # 烟雾模式 tag
```

内部 6 步骨架：

```python
# repro.py 主流程伪代码
from common.data import load_c4_calibration, load_wikitext2_test
from common.eval.memory import measure_weight_memory, peak_gpu_memory
from common.eval.ppl import compute_ppl
from common.eval.zeroshot import evaluate_zeroshot
from common.models import load_tokenizer

def main():
    args = parse_args()                              # 1. 解析 CLI
    tokenizer = load_tokenizer(args.model)           # 2. 加载 tokenizer
    calib = load_c4_calibration(tokenizer,
                                n_samples=args.calib_samples,
                                seq_len=args.seq_len, seed=args.seed)
                                                      # 3. C4 calibration（KIVI 跳过）

    with peak_gpu_memory(args.device) as peak:
        # ===  方法特定的算法在这里  =====================
        # 例如 GPTQ:
        #   quant_config = BaseQuantizeConfig(bits=4, group_size=128, ...)
        #   m = AutoGPTQForCausalLM.from_pretrained(args.model, quant_config)
        #   m.quantize(calib)
        #   m.save_quantized(save_dir)
        #   m = AutoGPTQForCausalLM.from_quantized(save_dir)
        # ================================================
        quantized_model = ...                        # 4. 量化 + 重载

    results = {}                                     # 5. 评测
    if "ppl" in eval_set:
        wt2 = load_wikitext2_test(tokenizer)
        results["ppl_wikitext2"] = compute_ppl(quantized_model, wt2,
                                               seq_len=args.seq_len,
                                               stride=args.seq_len,
                                               device=args.device)
    if "zeroshot" in eval_set:
        results["zeroshot"] = evaluate_zeroshot(quantized_model, tokenizer)
    if "memory" in eval_set:
        results["memory"] = measure_weight_memory(quantized_model)

    write_json(out_dir / f"results_{args.config}.json", results)  # 6. 落盘
    write_json(out_dir / f"meta_{args.config}.json",
               collect_meta(args, t_start, t_end, peak.bytes))
```

完整模板见 Plan A Task 9–10。

### 2.4 本地烟雾跑（必做，验证 repro.py 不挂）

```powershell
conda activate quant-gptq
cd GPTQ
python repro.py `
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
    --config w4g128 `
    --calib-samples 32 `
    --seq-len 1024 `
    --eval ppl,memory `
    --out results/smoke/ `
    --smoke
```

预期：5–15 分钟跑完，生成 `results/smoke/results_w4g128.json` + `meta_w4g128.json`。

**判定**：能跑出 JSON，结构正确，PPL > 0、memory > 0。**不**判定数字落在哪个范围 —— 这一步只验证代码不挂。

> 默认 `.gitignore` 已屏蔽 `*/results/smoke/`，烟雾结果不进 git。

### 2.5 Lab canonical 跑（出真数字）

把代码同步到实验室服务器（git push 或 rsync），然后：

```bash
# 在 lab 上
cd ~/quant-reproduce
bash scripts/env_lab.sh GPTQ
conda activate quant-gptq
pip install -e .

# 预下载模型（避免计入跑时）
python -c "from common.models import load_hf_model; load_hf_model('meta-llama/Llama-2-7b-hf')"

# Canonical 跑
cd GPTQ
python repro.py \
    --model meta-llama/Llama-2-7b-hf \
    --config w4g128 \
    --calib-samples 128 --seq-len 2048 \
    --eval ppl,zeroshot,memory --out results/
```

预期：30–90 分钟。生成 `results/results_w4g128.json` + `meta_w4g128.json`。

把 `results/` 拉回本地：

```bash
# Lab 上
git add GPTQ/results/results_w4g128.json GPTQ/results/meta_w4g128.json
git commit -m "data(gptq): canonical W4-g128 results on LLaMA-2-7B"
git push
```

```powershell
# 本地
git pull
```

### 2.6 数字判定 + 写 README

打开 `GPTQ/results/results_w4g128.json`，把数字填到 `GPTQ/README.md` 的 "实测 vs 论文" 表里：

| 容差判定（[设计文档 §3.4](superpowers/specs/2026-05-09-quant-reproduce-design.md#34-复现完成判定与论文同量级)） |
|---|
| GPTQ / AWQ / KIVI: PPL ±0.3 内算"同量级"  |
| BiLLM: PPL 落 20–60 都算"同量级"（二值化容忍度大） |

数字打不到 → 见本文 §4 排查流程。

### 2.7 Phase 2 源码笔记（最有学习价值的一步）

```bash
# 找官方代码位置
python -c "import auto_gptq, os; print(os.path.dirname(auto_gptq.__file__))"
# 例：~/anaconda3/envs/quant-gptq/lib/python3.10/site-packages/auto_gptq
```

按设计文档 §3.5 的五节模板写 `docs/reports/gptq.md`：

1. **算法回顾**（1 段，假定读者懂 PTQ 但没读过这篇）
2. **官方代码地图**（入口函数、主要 class、调用栈树 — 哪行真正动权重）
3. **关键实现选择**（论文没明说但代码关键的细节）
4. **硬件相关注释**（累加器位宽、kernel 是 fake-quant 还是 packed-int、NPU/SRAM 友好度）
5. **如果让我再写一遍**（你会怎么改 — 这一节是 Phase 3 unified pipeline spec 的真正素材）

> 这一步和数字一样重要。建议 1–2 个工作日。

---

## 3. 各方法的特异点

### 3.1 GPTQ（先做这个）

- **官方实现选 `auto-gptq`**（社区 fork，LLaMA-2 支持成熟），原版 IST-DASLab/gptq 主要支持 OPT。
- 装法：`pip install auto-gptq>=0.7,<1.0`（写进 `GPTQ/requirements.txt`）。
- 关键 API：`AutoGPTQForCausalLM.from_pretrained(model_id, quant_config)`，然后 `.quantize(calib_examples)`，最后 `save_quantized` + `from_quantized` 重载。
- Calibration 数据形状：要把 `common/data.load_c4_calibration` 返回的 `list[LongTensor]` 转成 auto-gptq 期望的 `list[{"input_ids": ..., "attention_mask": ...}]`。
- 预设 `PRESETS = {"w4g128": {"bits": 4, "group_size": 128, "desc_act": False, "sym": True}, ...}`。
- 论文 anchor: LLaMA-2-7B / W4-g128 / WT2 PPL ≈ 5.69（baseline ≈ 5.47，容差 ±0.3）。

### 3.2 AWQ

- 装：`pip install autoawq>=0.2`（事实标准，pre-built wheel 可用）。
- 关键 API：`AutoAWQForCausalLM.from_pretrained` → `.quantize(tokenizer, quant_config)`。注意 AWQ 的 `quant_config` 字典 key 与 GPTQ 略有差异（`q_group_size` 而不是 `group_size`，`zero_point` 而不是 `sym`）。
- AWQ 算法本身不需要太多 calibration sample，但仍按统一协议 128 个走。
- 论文 anchor: LLaMA-2-7B / W4-g128 / WT2 PPL ≈ 5.60（容差 ±0.3）。
- 与 GPTQ 共享大半 `repro.py` 骨架，只换算法块。

### 3.3 BiLLM

- 官方 repo 不在 PyPI，必须 vendor：
  ```bash
  cd BiLLM
  git submodule add https://github.com/Aaronhuang-778/BiLLM third_party/BiLLM
  pip install -e third_party/BiLLM
  ```
- 二值化（≈1 bit）对很多层敏感，论文里有 salient-residual 分解。要按官方 repo 的入口跑，不要自己改阈值。
- 论文 anchor: LLaMA-2-7B 实测 PPL 落 20–60 都算"同量级"（baseline 5.47，二值化容忍度大）。
- 跑时间最长（论文里说几十分钟到小时级）。
- **可能需要 patch**：repo 里 hardcode 的 transformers 版本可能跟你的 env 冲突，按报错改。

### 3.4 KIVI（流程最不同）

- KV cache 量化在 **inference time**，**不需要 calibration**（`repro.py` 里 calib 那步跳过）。
- 必须 vendor + 编译 CUDA extension：
  ```bash
  cd KIVI
  git submodule add https://github.com/jy-yuan/KIVI third_party/KIVI
  pip install -e third_party/KIVI
  cd third_party/KIVI/quant && python setup_cuda.py install
  ```
- 编译需要 CUDA toolkit；Lab 服务器一般有，本地 12GB 机器要装 CUDA 12.1+ + nvcc。
- 关键概念：per-channel keys + per-token values，2bit。
- `repro.py` 里需要把 KIVI 的自定义 attention monkey-patch 到 HF model（替换原 `LlamaAttention.forward`）。
- **`common/eval/memory` 当前缺 `measure_kv_cache_bytes`** —— 这是 Plan D 才补的。Plan D 写 KIVI 时一并加。
- 论文 anchor: LLaMA-2-7B + KV-2bit / WT2 PPL 几乎不掉（容差 ±0.3）。

---

## 4. 数字打不到目标怎么办

按顺序排查（所有方法通用）：

1. **模型 ID 与 HF commit SHA**：`meta_<config>.json` 里 `model` 字段对照 HF 模型卡的 commit；HF 偶尔会换权重。LLaMA-2-7B 用 `meta-llama/Llama-2-7b-hf`（不是 NousResearch 镜像，那个 tokenizer 略有不同）。
2. **Calibration 切片**：seed=42、n_samples=128、seq_len=2048 都固定？换 seed 重跑一次看波动多大。
3. **官方 repo issue 区**：搜 `"reproduce"` / `"PPL difference"`，看作者怎么回。auto-gptq / autoawq 都很活跃。
4. **官方 eval 脚本对照**：把 `auto_gptq/examples/quantization/quant_with_alpaca.py` 或 AWQ repo 的 `evaluation/eval.py` 拉过来跑同一份模型 + 同一份 calibration，对比 PPL —— 差异在你的 eval 实现还是在量化算法本身？
5. **如实记录**：差距确实存在但找不到原因 → 在子目录 `README.md` 末尾加 `## 实测异常记录` 一节，**如实**写实测 vs 论文 + 已排查的 5 步。**不要刷参数。** 作品集真正可信度在这里。

---

## 5. 完成判定（每方法）

每方法的 "✅ 完成" = 同时满足：

- [ ] `GPTQ/results/results_w4g128.json` 存在，含 `ppl_wikitext2` + `zeroshot` (6 项) + `memory`
- [ ] `GPTQ/results/meta_w4g128.json` 含完整元数据
- [ ] `GPTQ/README.md` 实测 vs 论文表填好（数字达 §3.4 容差 OR 写明排查记录）
- [ ] `docs/reports/gptq.md` 五节都写完（不只是模板）
- [ ] git log 干净（每个里程碑一个 commit）

四方法都 ✅ 后：

- [ ] 更新顶层 `README.md` 把方法状态改成 "✅ 复现 + 笔记"
- [ ] 创建 `docs/results/summary.md` 把四方法数字汇总成大表
- [ ] git tag `phase1-all-done`
- [ ] **此时才**开 Phase 3 spec：`docs/superpowers/specs/YYYY-MM-DD-phase3-unified-pipeline-design.md`

---

## 6. 常见坑

| 现象 | 原因 | 解 |
|------|------|----|
| `pip install auto-gptq` 报 `torch not found` 之类 | conda env 没装 CUDA torch | 先 `pip install torch --index-url https://download.pytorch.org/whl/cu121` 再装 auto-gptq |
| `torch.cuda.is_available()` 返回 False | 装了 CPU 版的 torch | 卸掉重装，或者 env.yml 里 pin `pytorch-cuda=12.1` |
| `.\scripts\env_local.ps1` 报 "running scripts is disabled" | PowerShell 执行策略默认不允许 | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 一次性 |
| `huggingface_hub` warn "no symlinks" Windows | Windows 默认不开发者模式 | 不影响，warning 可忽略；或开 Settings → Developer Mode |
| LLaMA-2 下载报 401/403 | 没授权或没 login | `huggingface-cli login` + 同意 license 页面 |
| `conda activate` 在 .sh 里失败 | 非交互 shell 没加载 conda 函数 | 我们的 `run_phase1_method.sh` 已 `source conda.sh`；自定义脚本要照做 |
| Lab 上 push results 被拒 | 默认分支保护或 git config 不全 | `git config user.email`、`git config user.name`，然后 `git push origin master` |
| `ppl_wikitext2` NaN | 模型在 cuda 上但 tokens 在 cpu | `compute_ppl` 内部已 `.to(device)`，但需要传对 `device` 参数 |
| BiLLM 跑到一半 OOM | 二值化中间张量很大 | 减 batch / 用 96GB lab 机器，不要在 12GB 上跑 canonical |
| KIVI CUDA 编译失败 | nvcc 版本不匹配 | 检查 `nvcc --version` 与 PyTorch CUDA 版一致 |

---

## 7. 接下来你具体做什么

按 Plan A Tasks 8–15 顺序，你的 GPTQ 第一份产出大概是这样的 commit 历史：

```
* docs(reports): add Phase 2 source-code study for GPTQ
* docs(gptq): fill in canonical numbers vs paper anchor
* data(gptq): canonical W4-g128 results on LLaMA-2-7B
* test(gptq): smoke run with TinyLlama-1.1B passes
* feat(gptq): implement end-to-end quantize → eval → save pipeline
* feat(gptq): add CLI argparse + quantization presets
* scaffold(gptq): create subdir with env.yml, requirements, README template
```

每个 commit 的内容、TDD 步骤、命令在 [Plan A](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md) Tasks 8–14 里都给好了。

跑出第一份 GPTQ 数字后告诉我，我写 Plan B (AWQ)。
