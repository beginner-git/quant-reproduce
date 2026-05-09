---
project: LLM Quantization Paper Reproduction Portfolio
phases_covered: Phase 1（复现） + Phase 2（源码研读）
phase_3_status: 占位 — Phase 1+2 完成后单独写设计
created: 2026-05-09
language: zh-CN
status: draft（待 user review）
---

# LLM 量化论文复现项目 — 设计文档（Phase 1 + 2）

## 0. 背景与目标

### 0.1 项目目的

作为申请 NPU / 端侧推理方向研究生作品集的核心工程作品，分阶段复现四篇 LLM 量化代表论文：

- **GPTQ**（Frantar et al., ICLR 2023）— Hessian-based 权重 PTQ，W4 起家
- **AWQ**（Lin et al., MLSys 2024）— Activation-aware 权重 PTQ，channel scaling
- **BiLLM**（Huang et al., ICML 2024）— 推到约 1bit 的极限二值化
- **KIVI**（Liu et al., ICML 2024）— KV cache 2bit 量化（per-channel keys + per-token values）

### 0.2 三阶段策略

| Phase | 内容 | 本文档覆盖 |
|------|------|-----------|
| 1 | 用各论文官方 repo + 薄共享层，跑出**与论文同量级**的数字 | ✅ |
| 2 | 读懂四份官方实现，写四份"算法地图 + 关键实现选择"研读笔记 | ✅ |
| 3 | 自己写一个 unified pipeline，**同时支持权重 + KV 量化叠加** | ❌ — 见 §5 |

> Phase 3 不在本文档详细设计。理由：未读源码就抽象，几乎一定抽象错。Phase 1+2 完成后单独写 spec。

### 0.3 硬件三档

| 档位 | 配置 | 用途 |
|------|------|------|
| 本地 | 12GB 单卡 | 烟雾测试 / 小模型 / 调试 |
| 实验室 | 96GB 服务器 | 论文可比的 LLaMA-2-7B canonical 跑 |
| Colab Pro | 备用 | lab 不可用时单方法补跑 |

### 0.4 评测范围

**In scope（Phase 1）**：

- Perplexity on WikiText-2（test split）+ C4（validation split）
- Zero-shot accuracy（lm-eval-harness 6 项：piqa / arc_easy / arc_challenge / hellaswag / winogrande / openbookqa）
- 内存占用：模型权重字节数 + KV cache 在给定 seqlen 下的字节数

**Out of scope（Phase 1）**：

- 推理 latency / throughput（需自定义 INT4 kernel，独立 ~2 周工作）
- 多 batch / 长 context benchmark
- 任何"自己实现量化算法"的尝试 — 留给 Phase 3

### 0.5 模型策略

- **Phase 1 canonical**：LLaMA-2-7B-hf（四篇论文的共同 anchor）
- **Phase 1 smoke**：TinyLlama-1.1B 或 Qwen2-1.5B（12GB 本地可跑）
- **Phase 3 自由模型**：完成 Phase 1+2 后再选（候选：LLaMA-3-8B / Qwen2.5-1.5B-3B / Phi-3.5-mini，倾向小模型贴近边缘场景）

### 0.6 工作语言

中文为主：spec、README、Phase 2 研读笔记。代码注释允许中英混合。短期内无 US-facing GitHub 公开计划。

---

## 1. 仓库结构与责任分层

```
F:\CODE\Quant\reproduce\
├── .git/                       # 起始就 git init
├── .gitignore                  # 屏蔽 *.bin / *.safetensors / __pycache__ / .venv / models/ / 各 conda env 工作目录
├── README.md                   # 项目入口；汇总四方法复现数字与到论文的对比
├── pyproject.toml              # common/ 包定义 + 顶层 dev tooling（pytest / ruff）
│
├── common/                     # 【方案 B 轻量共享层 — 仅 Phase 1+2 用】
│   ├── __init__.py
│   ├── data.py                 # WikiText-2 / C4 加载器，固定 seq=2048 / stride=2048 / seed=42
│   ├── eval/
│   │   ├── ppl.py              # 统一 PPL 实现，保证四方法横向可比
│   │   ├── zeroshot.py         # lm-eval-harness 包装
│   │   └── memory.py           # weight + KV cache 字节数 profiler
│   └── models.py               # HF 模型加载（统一 dtype / device / cache_dir）
│
├── GPTQ/                       # Phase 1 #1 — 子目录自带 conda env
│   ├── README.md               # 跑法 + 实测数字 + 论文对比 + troubleshooting
│   ├── requirements.txt        # 锁住 torch / auto-gptq 版本
│   ├── env.yml                 # （可选）conda env 完整快照
│   ├── repro.py                # 唯一入口：调官方代码 → common/eval → 写 results/
│   ├── third_party/auto-gptq/  # git submodule（Phase 2 切到 -e 安装时启用）
│   └── results/
│       ├── ppl_w4g128.csv
│       ├── zeroshot_w4g128.json
│       └── memory_w4g128.json
├── AWQ/   · BiLLM/   · KIVI/   # 同结构
│
├── docs/
│   ├── superpowers/specs/2026-05-09-quant-reproduce-design.md   # 本文档
│   ├── reports/                # Phase 2 输出
│   │   ├── gptq.md
│   │   ├── awq.md
│   │   ├── billm.md
│   │   └── kivi.md
│   └── results/
│       └── summary.md          # 四方法数字汇总大表
│
└── scripts/
    ├── env_local.sh            # 12GB 本地：建四个 conda env
    ├── env_lab.sh              # 96GB lab：同上，CUDA / HF_HOME 路径不同
    └── run_phase1_method.sh    # ./run_phase1_method.sh GPTQ — 切 env 跑 repro.py
```

### 1.1 责任边界

| 目录 | Phase | 角色 |
|------|------|------|
| `common/` | 1 + 2 | 评测 / 数据 / 模型加载 — 算法无关，唯一上游基础件 |
| `GPTQ/` `AWQ/` `BiLLM/` `KIVI/` | 1 主体 / 2 研读对象 | 各自调官方算法实现，靠 `common/eval` 出可比数字 |
| `docs/reports/*.md` | 2 输出 | 每方法五节研读笔记 |
| `docs/results/summary.md` | 1 输出 | 四方法横向比对表 |

### 1.2 关键设计选择

- **每方法独立 conda env**：四个官方仓库的 torch / cuda / triton 版本会打架，靠隔离唯一可靠。
- **`common/` 是纯 Python 包**（torch + datasets + lm_eval），任何 env 都能 `pip install -e .`。
- **`common/` 按需扩张**：跑 GPTQ 时只写当下需要的最小子集；AWQ/BiLLM/KIVI 跑到时按需补。不"先写完整套基础件再开工"。
- **不预创 `pipeline/` 目录**：Phase 3 spec 决定。

---

## 2. `common/` 模块接口契约

> 所有数字横向可比的前提：四方法用**同一份** PPL 实现、**同一份** calibration 切片、**同一份**内存量法。本节把这些协议定死。

### 2.1 `common/data.py`

```python
def load_wikitext2_test(tokenizer) -> torch.LongTensor:
    """GPTQ 论文协议：test split 全段 concat with '\\n\\n' 后整段 tokenize，返回单一 1D LongTensor。"""

def load_c4_calibration(tokenizer, n_samples=128, seq_len=2048, seed=42) -> list[torch.LongTensor]:
    """从 C4 validation 随机抽 n_samples 段、每段 seq_len token，作为 PTQ calibration 集。"""
```

**协议锁死**：seq_len = 2048、stride = 2048（非重叠切片）、seed = 42、n_samples = 128。GPTQ / AWQ / BiLLM 论文都按这个跑；KIVI 不需要 calibration 但 PPL eval 用同一份 WikiText-2。

### 2.2 `common/eval/ppl.py`

```python
def compute_ppl(model, tokens: torch.LongTensor, seq_len=2048, stride=2048, device="cuda") -> float:
    """滑窗 NLL 平均后 exp()。FP16 / fake-quant / real-quant 模型走完全相同的 forward 路径。"""
```

副产物：每片 NLL 写到 `results/<config>_ppl_raw.csv`，方便复盘哪几片异常。

### 2.3 `common/eval/zeroshot.py`

```python
DEFAULT_TASKS = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "openbookqa"]

def evaluate_zeroshot(model, tokenizer, tasks=DEFAULT_TASKS, num_fewshot=0, batch_size=1) -> dict[str, float]:
    """包一层 lm_eval.simple_evaluate；返回 {task_name: accuracy} + 写 JSON 到 results/。"""
```

### 2.4 `common/eval/memory.py`

```python
def measure_weight_memory(model) -> dict:                     # {"weights_bytes", "buffers_bytes"}
def measure_kv_cache_bytes(model, seq_len, batch=1) -> int    # dummy forward 后看 past_key_values
class peak_gpu_memory:                                        # context manager，包 torch.cuda.max_memory_allocated
```

**诚实标尺**：`measure_weight_memory` 直接读 `state_dict` 真实字节数。fake-quant 时报告就是 FP16 大小（这是事实）；只有跑了真 INT4 kernel 的产物才能报"压缩后"。**每方法 README 必须写清自己用哪种**。

### 2.5 `common/models.py`

```python
def load_hf_model(model_id, dtype=torch.float16, device_map="auto", cache_dir=None) -> PreTrainedModel
def load_tokenizer(model_id) -> PreTrainedTokenizer
```

唯一职责：四方法用同一个加载入口，避免出现 "GPTQ 跑的是 LLaMA-2-7b-hf 而 AWQ 跑的是 NousResearch 镜像" 这种数字不可比的情况。`cache_dir` 默认 `~/.cache/huggingface`，可被 `HF_HOME` 覆盖。

### 2.6 不放在 `common/` 里的东西

- 量化算法本身（Hessian 求逆 / channel scaling 网格搜索 / salient-residual 分解 / per-channel-key 插入）— **全部留在各自子目录**。它们是 Phase 2 研读对象、Phase 3 综合素材。
- 任何 method 间"统一抽象基类" — 留给 Phase 3 spec 决定。

---

## 3. Phase 1 各方法子目录 + 完成判定

### 3.1 子目录通用模板

```
GPTQ/
├── README.md               # 怎么跑 / 实测数字 / 论文对比表 / troubleshooting
├── requirements.txt        # 锁住版本（含 torch、auto-gptq、pip install -e ../ 安装 common）
├── env.yml                 # （可选）conda env 完整快照
├── repro.py                # 唯一入口
├── third_party/auto-gptq/  # git submodule，仅在源码研读 / 编译时启用
└── results/
    ├── ppl_w4g128.csv
    ├── zeroshot_w4g128.json
    └── memory_w4g128.json
```

### 3.2 `repro.py` 命令行契约（四方法统一）

```bash
python repro.py \
  --model meta-llama/Llama-2-7b-hf \
  --config w4g128                  \   # 该方法预设组合，方法各自定义
  --calib-samples 128              \
  --eval ppl,zeroshot,memory       \
  --device cuda                    \
  --save-quant ./quantized_w4g128/ \
  --out results/
```

内部固定流程：load model → calibration（KIVI 跳过）→ 调官方算法 API quantize → 保存 → 重载 → 三项 eval → 写 CSV/JSON。

可调超参全部通过 `--config` 字符串选预设（`w4g128` / `w3g128` / `w2g128` / `binary` / `kv2bit` 等），避免 CLI 参数爆炸。

**出错 fail-fast**：不做"failed gracefully fall back to baseline"。复现失败本身就是要看见的。

### 3.3 各方法 install 策略

| 方法 | 安装 | 依据 |
|------|------|------|
| **GPTQ** | `pip install auto-gptq>=0.7` | 社区 fork 极成熟，原版 IST-DASLab/gptq 主要支持 OPT；auto-gptq 直接吃 LLaMA-2 |
| **AWQ** | `pip install autoawq>=0.2` | casper-hansen/AutoAWQ 是事实标准，pre-built wheel 可用 |
| **BiLLM** | git submodule + `pip install -e third_party/BiLLM` | 官方 repo 不在 PyPI，需要 patch 兼容性 |
| **KIVI** | git submodule + `pip install -e third_party/KIVI` | 自定义 CUDA extension 必须本地编译 |

> Phase 2 研读 GPTQ / AWQ 时再补加 submodule，把 install 切到 `-e` 模式以便修改追踪。

### 3.4 复现"完成"判定（与论文同量级）

| 方法 | 模型 | 配置 | 论文 PPL（WT2，参考） | FP16 baseline | 容差判定 |
|------|------|------|---------------------|---------------|---------|
| GPTQ | LLaMA-2-7B | W4-g128 | ≈ 5.69 | ≈ 5.47 | **±0.3 PPL** |
| AWQ  | LLaMA-2-7B | W4-g128 | ≈ 5.60 | ≈ 5.47 | **±0.3 PPL** |
| BiLLM| LLaMA-2-7B | ≈ 1.08 bit | ≈ 32–35 | ≈ 5.47 | **同量级**：实测落 20–60 即可 |
| KIVI | LLaMA-2-7B | KV-2bit | ≈ 5.55 | ≈ 5.47 | **±0.3 PPL** |

> 表中是初始 anchor，实际复现以官方 README / 论文表格中具体数字为准。每方法 `README.md` 写"实测 X.YZ vs 论文 A.BC"两栏对照。
>
> **12GB 本地 smoke 跑**用 TinyLlama-1.1B 或 Qwen2-1.5B —— 这一档没有论文数字可比，只判定 **量化前后 PPL 单调正确**：量化后略升、量化越激进升越多。

### 3.5 Phase 2 研读笔记 `docs/reports/<method>.md` 大纲

每篇五个固定小节，避免写到一半失焦：

1. **算法回顾**（1 段） — 假定读者懂 PTQ 但没读过这篇，一段话总结。
2. **官方代码地图** — 入口函数、主要 class、调用栈树（哪行真正动权重）。
3. **关键实现选择** — 论文没明说但代码关键的细节：
   - GPTQ：列序选取 + Cholesky 分解 vs 直接求逆
   - AWQ：scale 网格搜索粒度与启发式
   - BiLLM：salient mask 怎么定，残差怎么二值化
   - KIVI：per-channel-key / per-token-value 在 attention forward 哪一步插入
4. **硬件相关注释** — 累加器位宽、kernel 是 fake-quant 还是 packed-int、对 NPU / SRAM 友好度评论。
5. **如果让我再写一遍** — 你会怎么改 / 不要复制官方 repo 的什么决定。**这一节是 Phase 3 spec 的真正素材。**

---

## 4. 环境管理 / 烟雾测试 / 失败处理 / 完成判定

### 4.1 环境隔离与共享 cache

```bash
# scripts/env_local.sh / env_lab.sh 内做的事
conda create -n quant-gptq  python=3.10 -y && conda activate quant-gptq
pip install -r GPTQ/requirements.txt
pip install -e .                 # 从仓库根装 common
# 同样四份 quant-{awq, billm, kivi}
```

- **`HF_HOME` 跨 env 共享**（默认 `~/.cache/huggingface`，lab 服务器指向共享盘）— LLaMA-2-7B 14GB 不重复下载四次。
- **`env_local.sh` 与 `env_lab.sh` 唯一差异**是 CUDA 版本和 `HF_HOME` 路径；其余尽量一致，便于"在本地复现 lab 上的失败"。
- **Colab 不写脚本**，仅作为 lab 不可用时的单方法补跑通道；不属于主路径。

### 4.2 烟雾测试 vs 正式跑

两档配置预设：

| 档位 | 模型 | calib | seq_len | zeroshot tasks | 用途 |
|------|------|-------|---------|----------------|------|
| `--smoke` | TinyLlama-1.1B | 32 样本 | 1024 | 仅 piqa | 12GB 本地 5–10 分钟跑通端到端 |
| `--canonical` | LLaMA-2-7b-hf | 128 样本 | 2048 | 全 6 项 | 96GB lab，论文可比 |

**约束**：每改动 `repro.py` 必须先 `--smoke` 通过；否则不提交去 lab 跑 `--canonical`。避免把 30 分钟的服务器时间烧在低级 bug 上。

### 4.3 元数据落盘

每次跑完，`results/<config>_meta.json` 自动写入：

- torch / transformers / auto-gptq（或对应方法库）等关键 pkg 版本号
- HF model commit SHA
- calibration seed
- 完整 CLI argv
- GPU 型号与峰值显存
- 起止时间戳

> 三个月后数字突然不可复现时，这是查"是 transformers 升级了 attention 实现还是 HF 换了模型权重"的唯一依据。**不可省**。

### 4.4 数字打不到目标的 escalation 流程

按顺序排查（写进各 method README 末尾 troubleshooting 节）：

1. 模型 ID 与 commit SHA 是否对得上论文（HF 上有时换权重）；
2. Calibration 数据切片是否跟官方一致（C4 split / WikiText 版本号）；
3. 官方 repo issue 区搜 "reproduce" / "PPL difference"，看作者怎么回复；
4. 把官方 repo 自己的 eval 脚本拉过来跑一遍，对照 `common/eval/ppl` 输出 — 差异在你的 eval 实现还是在量化算法本身；
5. 仍不通：在 README 里**如实写**实测 vs 论文 + 已排查的步骤。**不要为了"好看"调参刷数字** — 作品集真正可信度在这里。

### 4.5 Phase 1+2 完成定义（spec 整体的 done）

- [ ] 四个子目录各自 `repro.py --canonical` 跑通，`results/` 数字齐
- [ ] 四份 README 数字达 §3.4 容差，或写明已排查的步骤
- [ ] `docs/results/summary.md` 一张大表汇总四方法 PPL / zero-shot / memory
- [ ] 四份 `docs/reports/<method>.md` 按 §3.5 五节模板写完
- [ ] 顶层 README 链接到上述全部内容
- [ ] git log 干净（每个里程碑一个 commit）
- [ ] 此时**才**开新 spec：`docs/superpowers/specs/YYYY-MM-DD-phase3-unified-pipeline-design.md`

---

## 5. Phase 3 — 占位

待 §4.5 全部 ✅ 后，新建 Phase 3 设计文档；**届时已知信息**：

- 四方法在 LLaMA-2-7B 上的实测数字与对论文差距
- 四份源码研读笔记 §3.5(5) "如果让我再写一遍" 节的反思
- 各方法在 HF transformers 中真实的 hook / 替换点
- 哪些抽象有实质内容、哪些是空架子
- Phase 3 自己 pipeline 该选什么新模型

那时再设计：unified API 形状、kernel 复用策略、新模型选择、是否纳入 latency benchmark。

**关键约束**：Phase 3 的 unified pipeline 目标是**同时支持权重量化（GPTQ / AWQ / BiLLM 任一）+ KV cache 量化（KIVI）叠加**，能跑 `--weight awq:w4g128 --kv kivi:k=ch2_v=tok2` 这种组合。这是用户 brainstorming 阶段明确的设计目标。

---

## 附录 A — 设计选择记录

| # | 选择 | 替代方案 | 取舍 |
|---|------|----------|------|
| A.1 | 方案 B：`common/` 轻量共享 | A 松散布局 / C 重抽象 | 数字横向可比（A 不行）；不预先抽象算法（C 易抽象错） |
| A.2 | Phase 3 占位、不预设计 | 一份 spec 涵盖三阶段 | 未读源码就抽象几乎一定错；Phase 1+2 完成后信息更全 |
| A.3 | Phase 1 评测不含 latency | 跑 latency 数字 | 自定义 INT4 kernel ~2 周独立项目，与"同量级数字 + 源码理解"目标错位 |
| A.4 | GPTQ → AWQ → BiLLM → KIVI 顺序 | 难度 / 兴趣 / 其他顺序 | 先权重族（共享 calibration & eval 路径），后 KV；难度递增；代码复用最大化 |
| A.5 | 每方法独立 conda env | 单一大 env | 四官方 repo 的 torch / cuda / triton 版本互不兼容 |
| A.6 | Phase 1 用 LLaMA-2-7B canonical | 用更新模型 | 四篇论文都用它，是论文可比的唯一 anchor |
| A.7 | 中文 spec / 英文代码 | 全英文 | 短期无 US-facing 公开计划；写作效率优先；申请前可补英文 README |
| A.8 | smoke / canonical 两档 | 单档 / 三档 | 12GB 本地必须能跑通（防止 30 分钟 lab 时间被 bug 浪费）；正式数字必须 7B 才可比 |

## 附录 B — 后续待补

- 各方法 `requirements.txt` 具体版本号 — 在 Phase 1 实施时写入对应 PR
- `scripts/env_*.sh` 完整脚本 — 同上
- `docs/results/summary.md` 表格模板 — 第一份方法跑出数字时定型
