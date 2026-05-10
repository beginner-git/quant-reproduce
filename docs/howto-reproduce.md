# 复现操作手册（How to Reproduce）

> 这是项目的**第一份操作文档**。从工具链安装开始，到跑出第一份 GPTQ 数字为止，整条链路都覆盖。
> 假定读者：会用命令行，但不假定本机已有 Python / CUDA / Conda。
> 配套文档：[设计文档](superpowers/specs/2026-05-09-quant-reproduce-design.md) · [Plan A](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md)

---

## 目录

- [§0 总览](#0-总览)
- [§1 工具链安装（一次性）](#1-工具链安装一次性)
- [§2 仓库与账号 setup（一次性）](#2-仓库与账号-setup一次性)
- [§3 验证共享层（quant-common 跑通 pytest）](#3-验证共享层quant-common-跑通-pytest)
- [§4 通用方法工作流（每方法 6 步）](#4-通用方法工作流每方法-6-步)
- [§5 GPTQ 完整 walkthrough（第一次）](#5-gptq-完整-walkthrough第一次)
- [§6 各方法特异点（AWQ / BiLLM / KIVI）](#6-各方法特异点awq--billm--kivi)
- [§7 数字打不到目标的排查流程](#7-数字打不到目标的排查流程)
- [§8 完成判定](#8-完成判定)
- [§9 Colab 备用通道](#9-colab-备用通道)
- [§10 常见坑速查](#10-常见坑速查)

---

## §0 总览

### 0.1 三阶段 + 单方法循环

```
Phase 1 复现 ────► Phase 2 源码研读 ────► Phase 3 自写统一 pipeline
   ↑                  ↑                       ↑
  本文档覆盖         本文档覆盖              （Phase 1+2 完后另写 spec）
```

每个量化方法走同一个 6 步循环，**一次只做一个方法**：

```
GPTQ → AWQ → BiLLM → KIVI
 │
 ├─ 1. 子目录骨架        (README + env.yml + requirements.txt + repro.py 占位)
 ├─ 2. 建 conda env      (.\scripts\env_local.ps1 <METHOD>)
 ├─ 3. 写 repro.py 算法侧 (调官方库做 quantize → 用 common/eval 出数字)
 ├─ 4. 本地烟雾跑        (TinyLlama-1.1B / calib=32 / seq=1024，5–10 min)
 ├─ 5. Lab canonical 跑  (LLaMA-2-7B / calib=128 / seq=2048，30–90 min)
 └─ 6. Phase 2 源码笔记  (docs/reports/<method>.md，五节模板)
```

### 0.2 不变量（所有方法都靠 `common/` 保证横向可比）

- 评测数据：`load_wikitext2_test` (GPTQ 协议) + `load_c4_calibration(seed=42)`
- PPL 公式：`compute_ppl(seq_len=2048, stride=2048)` (GPTQ 论文 `loss × seq_len`)
- Zero-shot：`evaluate_zeroshot()` 默认 6 项 (piqa / arc_easy / arc_challenge / hellaswag / winogrande / openbookqa)
- 内存：`measure_weight_memory()` 读真实 state_dict 字节数
- 元数据：每跑一次都自动写 `meta_<config>.json` (pkg 版本、HF SHA、CLI、GPU、时间)

### 0.3 三档硬件分工

| 档位 | 配置 | 用途 |
|------|------|------|
| 本地 | 12GB GPU 单卡 | **烟雾跑** + 写 / 调试 repro.py + 单元测试 |
| 实验室 | 96GB 服务器 | **Canonical 跑**（LLaMA-2-7B 论文可比） |
| Colab Pro | A100 40GB | lab 不可用时备用（详见 §9） |

---

## §1 工具链安装（一次性）

> 本节假定你的本机是 Windows 11，lab 是 Linux。两侧都需要走一遍。

### 1.1 Git

**检查**：

```powershell
git --version
# 期望: git version 2.x.x
```

**没装就装**：<https://git-scm.com/download/win>，安装时勾选 "Git Bash"（提供 Bash shell，跑 .sh 用得上）。

**Linux lab 上**：通常已预装；没有就 `sudo apt install git` 或 `module load git`（依集群）。

**首次配身份**：

```powershell
git config --global user.name "你的名字"
git config --global user.email "yiminl.edu@gmail.com"
```

> 我们项目里**不会**改 git config，但首次本机配身份是必要的。

### 1.2 Anaconda 或 Miniconda

**检查**：

```powershell
conda --version
# 期望: conda 23.x 或更高
```

**没装就装**：

- Anaconda（带 GUI，~1 GB）：<https://www.anaconda.com/download>
- **Miniconda（推荐，~70 MB，命令行）**：<https://www.anaconda.com/docs/getting-started/miniconda/install>

安装时勾选 "Add Miniconda to PATH"（不推荐 Anaconda 默认设置）。装完**重开 PowerShell** 才能用 `conda`。

**初始化 PowerShell hook**（让 `conda activate` 在 PowerShell 里能用）：

```powershell
conda init powershell
# 重开 PowerShell 后，命令提示符前会出现 (base)
```

**Linux lab 上**：

```bash
# 如果没有
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 同意 license, 选安装路径, 同意 init bash
source ~/.bashrc
conda --version
```

### 1.3 NVIDIA 驱动 + CUDA Toolkit

**前置**：你的 GPU 是 NVIDIA，且驱动版本要够新。

**检查驱动**：

```powershell
nvidia-smi
# 看右上角 "CUDA Version: 12.x"，需要 ≥ 12.1（PyTorch 2.x 的 CUDA 要求）
```

如果 `nvidia-smi` 找不到或 CUDA Version < 12.1：去 <https://www.nvidia.com/Download/index.aspx> 选你的 GPU 型号下载最新 Game Ready Driver 或 Studio Driver 装上。

**CUDA Toolkit 是否需要装？** 分情况：

- **只跑 GPTQ / AWQ**（用 pip 装 auto-gptq / autoawq 的 pre-built wheel）：**不需要本机有 CUDA Toolkit**，PyTorch 自带 runtime 就够。只要 `nvidia-smi` 能跑。
- **跑 BiLLM / KIVI**（需要本地编译自定义 CUDA extension）：**必须装 CUDA Toolkit 12.1+** 和 nvcc 编译器。

**装 CUDA Toolkit（KIVI 时再做也来得及）**：

- Windows: <https://developer.nvidia.com/cuda-12-1-1-download-archive> → Windows → x86_64 → 11 → exe (local)
- 装完检查：
  ```powershell
  nvcc --version
  # 期望: Cuda compilation tools, release 12.1, V12.1.x
  ```

**Linux lab**：通常集群已预装多个 CUDA 版本，用 `module load cuda/12.1` 加载（依集群文档）。

### 1.4 PowerShell 准备（Windows-specific）

**检查版本**：

```powershell
$PSVersionTable.PSVersion
# 期望: Major 5+ (Windows PowerShell 5.1 内置就够；7.x 也行)
```

**首次跑 .ps1 脚本要放开执行策略**：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# 弹窗选 [Y]es
# 之后跑 .\scripts\env_local.ps1 不再报 "running scripts is disabled"
```

> `RemoteSigned` 表示：本机写的 .ps1 直接跑；从网上下载的需要签名。比 `Bypass` 安全。

### 1.5 编辑器（建议 VS Code）

- 下载：<https://code.visualstudio.com/>
- 必装扩展：
  - Python（Microsoft 官方）— 支持 venv 切换、单元测试 UI、debug
  - Pylance — 类型检查
  - Ruff — 项目里已用的 linter
  - GitLens — 看 commit history 直观
- 推荐：把 VS Code 终端设为 PowerShell（默认就是），方便跟 `.\scripts\` 配合。

---

## §2 仓库与账号 setup（一次性）

### 2.1 仓库本地位置

```
F:\CODE\Quant\reproduce\
```

**情况 A：你之前已经跟我一起搭好框架（commit `b5b4663` 或之后）**：跳到 §3。

**情况 B：换台机器，从远端 clone**：

```powershell
cd F:\CODE\Quant
git clone <你的仓库 URL> reproduce
cd reproduce
git log --oneline | Select-Object -First 5
# 应能看到 framework 各 commit
```

### 2.2 HuggingFace 账号 + token

LLaMA-2-7B 需要授权才能下载。

**步骤**：

1. 去 <https://huggingface.co/join> 注册账号（免费）。
2. 去 <https://huggingface.co/meta-llama/Llama-2-7b-hf> 同意使用条款（点 "Agree and access repository"，等审批通常 < 1 小时）。
3. 去 <https://huggingface.co/settings/tokens> 创建一个 **Read 权限**的 token（不需要 Write）。复制 token（形如 `hf_xxxxxxxxxxx`）。
4. 在本机命令行登录：
   ```powershell
   pip install --upgrade huggingface_hub  # 如果尚未装
   huggingface-cli login
   # 粘贴 token，回车
   ```
5. 验证：
   ```powershell
   huggingface-cli whoami
   # 期望输出你的用户名
   ```

> 同步在 lab 服务器走一遍 step 4。

### 2.3 设置 `HF_HOME`（避免 14GB 模型下载 4 次）

`HF_HOME` 是 HuggingFace cache 根目录。每个 conda env 都用它读模型，所以**设到一个固定的大盘路径**，跨 env 共享。

**Windows**：

```powershell
# 永久（用户级），重开 PowerShell 才生效
[Environment]::SetEnvironmentVariable("HF_HOME", "D:\hf_cache", "User")

# 或临时（仅当前会话）
$env:HF_HOME = "D:\hf_cache"
```

**Linux lab**：

```bash
# 加到 ~/.bashrc 末尾
export HF_HOME=/shared/path/huggingface_cache
# 重开 shell 或 source ~/.bashrc 生效
echo $HF_HOME
```

> **检查空间**：LLaMA-2-7B 约 14 GB，再加 WikiText-2 + C4 + 量化产物，预留 30 GB 起步。
> **不要把 HF_HOME 设到 SSD 系统盘**（OneDrive 同步路径也避免 — 否则 14GB 会被同步上传）。

---

## §3 验证共享层（`quant-common` 跑通 pytest）

> 跑 4 个方法之前，先做一遍这个"对账"，确认 `common/` 在你环境里能用。

### 3.1 创建专用 dev env

不要污染 anaconda `base`。建一个 `quant-common`：

```powershell
conda create -n quant-common python=3.10 -y
conda activate quant-common
```

**为什么 Python 3.10**：项目 `pyproject.toml` 写了 `requires-python = ">=3.10"`。3.10 / 3.11 / 3.12 都行；3.13 也能跑但有些上游库（auto-gptq、lm-eval）的 wheel 还没全跟上，3.10 最稳。

### 3.2 装 framework 依赖

```powershell
cd F:\CODE\Quant\reproduce
pip install -e .[dev]
```

预期：

- 装 torch、transformers、datasets、accelerate、lm-eval、pytest 等
- 总下载 1–3 GB，2–5 分钟
- 最后一行 `Successfully installed ... quant-reproduce-common-0.1.0 ...`

> 此时 torch 应是 GPU 版（pip 自动选）。验证：
> ```powershell
> python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
> # 期望: 2.x.x+cu121 True
> ```
> 如果是 CPU 版（`+cpu`）或 `False`：先跑
> ```powershell
> pip uninstall torch -y
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 3.3 跑单元测试

```powershell
pytest tests/ -v -m "not slow"
```

预期：

```
tests/test_data.py::test_load_wikitext2_returns_1d_long_tensor PASSED
tests/test_data.py::test_load_wikitext2_deterministic PASSED
tests/test_data.py::test_load_c4_calibration_returns_list_of_long_tensors PASSED
tests/test_data.py::test_load_c4_calibration_seeded_deterministic PASSED
tests/test_data.py::test_load_c4_calibration_seed_changes_output PASSED
tests/test_eval_memory.py::test_measure_weight_memory_returns_dict_with_bytes PASSED
tests/test_eval_memory.py::test_measure_weight_memory_dtype_scaling PASSED
tests/test_eval_memory.py::test_peak_gpu_memory_records_increase PASSED   ← CUDA 版才会过；CPU 版 SKIPPED
tests/test_eval_ppl.py::test_compute_ppl_finite_and_positive PASSED        ← CUDA 版才会过
tests/test_eval_ppl.py::test_compute_ppl_deterministic PASSED              ← 同上
tests/test_eval_ppl.py::test_compute_ppl_short_input_raises PASSED         ← 同上
tests/test_models.py::test_load_tokenizer_returns_tokenizer PASSED
tests/test_models.py::test_load_hf_model_returns_model PASSED
tests/test_models.py::test_load_hf_model_respects_dtype PASSED

==================== 14 passed (or 11 passed + 3 skipped on CPU) ====================
```

第一次跑会下载几个小 dataset（WT2 ~5MB、C4 几个文档 ~1MB、tiny-gpt2 ~5MB），15–30 秒。

**通过 = 共享层在你环境里 OK**，可以开始 §5（GPTQ）了。

> Q：lm-eval 的 smoke 测试呢？
> A：默认不跑（标记为 `slow`）。等真做 GPTQ canonical 时再跑：
> ```powershell
> pytest tests/test_eval_zeroshot.py -v -m slow
> ```

---

## §4 通用方法工作流（每方法 6 步）

下面是**抽象描述**，§5 用 GPTQ 把它具象化。

### Step 1 — 写子目录骨架

```
GPTQ/                    （AWQ/ BiLLM/ KIVI/ 同结构）
├── README.md            # 跑法 + 数字表 + 论文对比 + troubleshooting（先填模板，跑完再填数字）
├── env.yml              # conda env 定义（这是"该方法的"专属依赖）
├── requirements.txt     # pip 依赖（与 env.yml pip 段同步）
├── repro.py             # 唯一入口：调官方算法 → 用 common/eval → 写 results
├── third_party/         # vendor 或 git submodule（仅 BiLLM / KIVI 需要）
└── results/             # 数字 JSON、meta JSON、ppl_raw CSV（canonical 跑后落这里）
```

**模板内容**全在 [Plan A Task 8](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md#task-8-创建-gptq-子目录骨架readme--envyml--requirementstxt--空-repropy)。

### Step 2 — 建 conda env

```powershell
.\scripts\env_local.ps1 <METHOD>     # Windows 本地
# 或 Linux lab:
bash scripts/env_lab.sh <METHOD>
```

脚本做三件事：
1. 找 `<METHOD>/env.yml` 是否存在
2. `conda env create -f <METHOD>/env.yml`，建出 `quant-<method>` env
3. 提示你 `conda activate quant-<method>` 然后 `pip install -e .` 装 common

跑完手动接两步：

```powershell
conda activate quant-gptq
cd F:\CODE\Quant\reproduce
pip install -e .
```

**验证**：

```powershell
python -c "import torch; print(torch.cuda.is_available())"   # True
python -c "import auto_gptq; print(auto_gptq.__version__)"   # 0.7.x
python -c "from common.eval.ppl import compute_ppl; print('OK')"
```

### Step 3 — 写 repro.py（你的活儿）

这是 Phase 1 你写代码最多的地方。`repro.py` 的固定 CLI 契约（四方法统一）：

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
  [--smoke]
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
    args = parse_args()                              # 1. 解析 CLI（含 PRESETS）
    tokenizer = load_tokenizer(args.model)           # 2. 加载 tokenizer
    calib = load_c4_calibration(tokenizer,
                                n_samples=args.calib_samples,
                                seq_len=args.seq_len, seed=args.seed)
                                                      # 3. C4 calibration（KIVI 跳过此步）

    with peak_gpu_memory(args.device) as peak:
        # ===  方法特定的算法（只有这一段每方法不同）  =====
        # 例如 GPTQ:
        #   quant_config = BaseQuantizeConfig(bits=4, group_size=128, ...)
        #   m = AutoGPTQForCausalLM.from_pretrained(args.model, quant_config)
        #   m.quantize(calib_in_auto_gptq_format)
        #   m.save_quantized(save_dir)
        #   m = AutoGPTQForCausalLM.from_quantized(save_dir)
        # ====================================================
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

完整 argparse 模板 + main flow 在 [Plan A Task 9](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md#task-9-实现-gptqrepropy-的-argparse--presetstdd) 和 [Task 10](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md#task-10-实现-gptqrepropy-端到端-main-flow无单元测试smoke-验证)。

### Step 4 — 本地烟雾跑（5–10 min）

```powershell
conda activate quant-gptq
cd F:\CODE\Quant\reproduce\GPTQ
python repro.py `
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
    --config w4g128 `
    --calib-samples 32 `
    --seq-len 1024 `
    --eval ppl,memory `
    --out results/smoke/ `
    --smoke
```

**判定**：能跑出 JSON、结构正确、PPL > 0、memory > 0 → 通过。**不**判定数字范围 — 这一步只验证代码能端到端跑，不挂。

### Step 5 — Lab canonical 跑（30–90 min）

把代码同步到 lab，跑 LLaMA-2-7B canonical，把 `results/` 拉回。详细命令在 §5.5。

### Step 6 — Phase 2 源码笔记

读 `auto-gptq` / `autoawq` / `BiLLM` / `KIVI` 官方代码，按 §3.5 五节模板写 `docs/reports/<method>.md`。

---

## §5 GPTQ 完整 walkthrough（第一次）

下面是**第一次做 GPTQ** 的完整命令序列。AWQ / BiLLM / KIVI 后面三次都是这套流程，只换算法块。

### §5.1 写 `GPTQ/` 子目录骨架

按 [Plan A Task 8](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md#task-8-创建-gptq-子目录骨架readme--envyml--requirementstxt--空-repropy)，5 个文件：

| 文件 | 作用 |
|------|------|
| `GPTQ/README.md` | 跑法、数字表、对论文对比、troubleshooting 节 |
| `GPTQ/env.yml` | conda env 定义（含 pytorch-cuda=12.1 + auto-gptq） |
| `GPTQ/requirements.txt` | pip 依赖（auto-gptq>=0.7,<1.0 + transformers + ...） |
| `GPTQ/repro.py` | 占位 stub（Task 9–10 才填实） |
| `GPTQ/results/.gitkeep` | 让空目录进 git |

**完成后**：

```powershell
cd F:\CODE\Quant\reproduce
git add GPTQ/
git commit -m "scaffold(gptq): create subdir with env.yml, requirements, README template"
```

### §5.2 建 `quant-gptq` conda env

```powershell
cd F:\CODE\Quant\reproduce
.\scripts\env_local.ps1 GPTQ
```

预期输出：

```
Creating conda env quant-gptq from F:\CODE\Quant\reproduce\scripts\..\GPTQ\env.yml
Collecting package metadata (repodata.json): done
Solving environment: done
... (10–60 秒，取决于网络) ...
Activate then install common:
  conda activate quant-gptq
  pip install -e .
```

接两步：

```powershell
conda activate quant-gptq
pip install -e .
```

**验证**：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.x.x+cu121 True

python -c "import auto_gptq; print(auto_gptq.__version__)"
# 期望: 0.7.x

python -c "from common.eval.ppl import compute_ppl; print('OK')"
# 期望: OK
```

### §5.3 写 `repro.py`（分两阶段 TDD）

**阶段 A — argparse + PRESETS（[Task 9](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md#task-9-实现-gptqrepropy-的-argparse--presetstdd)）**：

1. 先写 `GPTQ/test_repro_cli.py` 三个测试（presets 含 w4g128、parse_args 默认值、未知 config 拒绝）
2. 跑测试看 fail（没 `PRESETS` 属性）
3. 实现 `repro.py`（暂时 main 只 raise `NotImplementedError`）
4. 跑测试看 pass
5. Commit `feat(gptq): add CLI argparse + quantization presets`

**阶段 B — 端到端 main flow（[Task 10](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md#task-10-实现-gptqrepropy-端到端-main-flow无单元测试smoke-验证)）**：

把 main() 实成完整 6 步流程（伪代码见 §4 Step 3）。auto-gptq 的关键 API：

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quant_config = BaseQuantizeConfig(
    bits=preset["bits"],          # 4
    group_size=preset["group_size"],  # 128
    desc_act=preset["desc_act"],  # False
    sym=preset["sym"],            # True
)
m = AutoGPTQForCausalLM.from_pretrained(args.model, quant_config, torch_dtype=torch.float16)
m.quantize(calib_examples)        # calib_examples 要转成 list[{"input_ids", "attention_mask"}]
m.save_quantized(save_dir)
del m; torch.cuda.empty_cache()
m = AutoGPTQForCausalLM.from_quantized(save_dir, device=args.device)
```

**Calibration 数据格式适配**（auto-gptq 期望 dict 格式）：

```python
calib_tokens = load_c4_calibration(tokenizer, n_samples=args.calib_samples, seq_len=args.seq_len, seed=args.seed)
calib_examples = [
    {
        "input_ids": t.unsqueeze(0).to(args.device),
        "attention_mask": torch.ones_like(t).unsqueeze(0).to(args.device),
    }
    for t in calib_tokens
]
```

完整 main 模板见 [Task 10 Step 1 完整代码](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md#task-10-实现-gptqrepropy-端到端-main-flow无单元测试smoke-验证)。

### §5.4 本地烟雾跑（[Task 12](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md#task-12-在-12gb-本地用-tinyllama-烟雾跑-gptq)）

```powershell
conda activate quant-gptq
cd F:\CODE\Quant\reproduce\GPTQ
python repro.py `
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
    --config w4g128 `
    --calib-samples 32 `
    --seq-len 1024 `
    --eval ppl,memory `
    --out results/smoke/ `
    --smoke
```

**预期 stdout**（大致）：

```
[1/6] Loading tokenizer for TinyLlama/TinyLlama-1.1B-Chat-v1.0…
[2/6] Loading C4 calibration (32 × 1024)…
[3/6] Quantizing with GPTQ w4g128…
... (auto-gptq 进度条，2–5 min) ...
[4/6] Saving & reloading quantized model…
[5/6] Evaluating: ['memory', 'ppl']
  WikiText2 PPL = 12.4521        (任意正数即可，TinyLlama 不是论文模型)
  weights = 768.4 MB, buffers = 1.2 MB
[6/6] Writing results to results/smoke/
Done in 387.2s. GPU peak = 4.83 GB.
```

**核对产出**：

```powershell
type results\smoke\results_w4g128.json
```

应是 JSON，含 `ppl_wikitext2`（数值）和 `memory.weights_bytes`（数值）。

> **重要**：smoke PPL 数字本身不要用来判断"复现成功"。只判断是否报错、JSON 结构正确。
> Smoke 结果默认不进 git（`.gitignore` 屏蔽 `*/results/smoke/`）。

### §5.5 Lab canonical 跑（[Task 13](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md#task-13-在-96gb-实验室-gpu-跑-llama-2-7b-canonical人工任务)）

#### §5.5.1 把代码同步到 lab

**Option A — git push/pull**（推荐，commit 历史清晰）：

```powershell
# 本地：先 commit 你的 GPTQ 代码
cd F:\CODE\Quant\reproduce
git add GPTQ/
git commit -m "feat(gptq): implement repro.py end-to-end"

# 推到远端（需要先有远端 repo，比如 GitHub private）
git remote add origin git@github.com:你的账号/quant-reproduce.git  # 一次性
git push -u origin master
```

```bash
# Lab 上
cd ~
git clone git@github.com:你的账号/quant-reproduce.git
cd quant-reproduce
```

**Option B — rsync**（无远端时）：

```powershell
# 本地（Git Bash 或 PowerShell + WSL）
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='*/results/smoke' \
  F:/CODE/Quant/reproduce/  user@lab.example.edu:~/quant-reproduce/
```

#### §5.5.2 Lab 上准备 env

```bash
cd ~/quant-reproduce
bash scripts/env_lab.sh GPTQ
conda activate quant-gptq
pip install -e .

# 一次性预下载 LLaMA-2-7B（避免计入跑时）
huggingface-cli login    # 粘贴你的 HF token
python -c "from common.models import load_hf_model; load_hf_model('meta-llama/Llama-2-7b-hf')"
# 14GB 下载到 $HF_HOME
```

#### §5.5.3 跑 canonical

```bash
cd ~/quant-reproduce/GPTQ
python repro.py \
    --model meta-llama/Llama-2-7b-hf \
    --config w4g128 \
    --calib-samples 128 --seq-len 2048 \
    --eval ppl,zeroshot,memory --out results/

# 30–90 分钟。建议挂 nohup 或 tmux：
nohup python repro.py ... > run.log 2>&1 &
tail -f run.log
```

**完成标志**：`results/results_w4g128.json` 和 `results/meta_w4g128.json` 都生成。

#### §5.5.4 把 `results/` 拉回本地

```bash
# Lab 上
git add GPTQ/results/results_w4g128.json GPTQ/results/meta_w4g128.json
git commit -m "data(gptq): canonical W4-g128 results on LLaMA-2-7B"
git push
```

```powershell
# 本地
cd F:\CODE\Quant\reproduce
git pull
type GPTQ\results\results_w4g128.json
# 看到真实数字了
```

### §5.6 数字判定 + 写 README

打开 `GPTQ/results/results_w4g128.json`，把数字填到 `GPTQ/README.md`：

| Config | Model | Metric | 实测 | 论文 anchor | 判定 |
|--------|-------|--------|------|------------|------|
| w4g128 | LLaMA-2-7B | WT2 PPL | **5.71** | ≈ 5.69 | ✅ 在 ±0.3 内 |
| w4g128 | LLaMA-2-7B | piqa | 0.781 | ≈ 0.78 | ✅ |
| w4g128 | LLaMA-2-7B | weights | 3.71 GB | ≈ 3.7 GB | ✅ 真 INT4 packed |

> 数字打不到 ±0.3 → 跳到 §7 排查流程。**不要为了好看刷参数。**

Commit：

```powershell
git add GPTQ/README.md
git commit -m "docs(gptq): fill in canonical numbers vs paper anchor"
```

### §5.7 Phase 2 源码笔记（[Task 14](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md#task-14-写-phase-2-gptq-源码研读笔记)）

```powershell
conda activate quant-gptq
python -c "import auto_gptq, os; print(os.path.dirname(auto_gptq.__file__))"
# 例：C:\Users\xxx\miniconda3\envs\quant-gptq\Lib\site-packages\auto_gptq
```

关键文件（auto-gptq 0.7+）：
- `auto_gptq/quantization/gptq.py` — 核心 `GPTQ` 类的 `add_batch` / `fasterquant`
- `auto_gptq/modeling/_base.py` — `BaseGPTQForCausalLM.quantize` orchestrator
- `auto_gptq/modeling/llama.py` — LLaMA 适配（哪些 nn.Linear 被替换）

按 [设计文档 §3.5](superpowers/specs/2026-05-09-quant-reproduce-design.md#35-phase-2-研读笔记-docsreportsmethodmd-大纲) 五节模板写：

1. **算法回顾**（1 段）
2. **官方代码地图**（入口、主 class、调用栈树 — 哪行真正动权重）
3. **关键实现选择**（论文没明说但代码关键的：列序选取、Cholesky vs 直接求逆、阻尼 percdamp、group_size、sym/asym）
4. **硬件相关注释**（累加器位宽、kernel fake-quant vs packed-int、对 NPU/SRAM 友好度）
5. **如果让我再写一遍**（这一节是 Phase 3 unified pipeline spec 的真正素材）

Commit：

```powershell
git add docs/reports/gptq.md
git commit -m "docs(reports): add Phase 2 source-code study for GPTQ"
```

### §5.8 收尾

更新顶层 README + 创建 summary，打 tag：

```powershell
# 顶层 README.md 把 GPTQ 行改成 "✅ 复现 + 笔记"
# 创建 docs/results/summary.md（GPTQ 一行）

git add README.md docs/results/summary.md
git commit -m "docs: link GPTQ as completed in top README + initialize summary table"
git tag -a phase1-gptq-done -m "Phase 1 milestone: GPTQ reproduction + Phase 2 writeup complete"
```

🎉 **第一份方法完成。Plan B (AWQ) 我会照这次 GPTQ 实战的反馈写。**

---

## §6 各方法特异点（AWQ / BiLLM / KIVI）

GPTQ 之后的三个方法都是 §5 流程的变体。**只换 §5.3（algorithm 块）和 §5.6（数字 anchor）**，其余全等。

### §6.1 AWQ

- **装法**：`pip install autoawq>=0.2`（写进 `AWQ/requirements.txt`）。
- **关键 API**：
  ```python
  from awq import AutoAWQForCausalLM
  m = AutoAWQForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
  m.quantize(tokenizer, quant_config={"q_group_size": 128, "w_bit": 4, "zero_point": True, "version": "GEMM"})
  m.save_quantized(save_dir)
  m = AutoAWQForCausalLM.from_quantized(save_dir, device="cuda")
  ```
- **注意**：AWQ 的 `quant_config` key 与 GPTQ 略不同：
  - `q_group_size` ≠ GPTQ 的 `group_size`
  - `w_bit` ≠ `bits`
  - `zero_point: True` ≠ `sym: False`
  - 多一个 `version`（推理 kernel 类型："GEMM"、"GEMV" 等）
- **Calibration**：AWQ 自己内部做 calibration（`m.quantize(tokenizer, ...)` 接 tokenizer 而不是 calib examples）。我们仍按统一协议用 `load_c4_calibration` 出 128 个，传 AWQ 的内部接口（具体看官方 examples）。
- **论文 anchor**: LLaMA-2-7B / W4-g128 / WT2 PPL ≈ 5.60（baseline 5.47，容差 ±0.3）。
- **跑时间**：~30 min on lab 96GB。

### §6.2 BiLLM

- **官方 repo 不在 PyPI**，必须 vendor：
  ```bash
  cd BiLLM
  git submodule add https://github.com/Aaronhuang-778/BiLLM third_party/BiLLM
  cd third_party/BiLLM
  pip install -r requirements.txt
  cd ../..
  pip install -e .   # 让 BiLLM 模块可 import
  ```
  > 注：BiLLM 的 `requirements.txt` 经常 pin 老版本 transformers，跟我们 lm-eval 的版本可能冲突。这种情况：用 BiLLM 自己的 venv（不与其它方法共享 common 也行）；或 patch BiLLM 的代码兼容新 transformers。
- **二值化（≈1 bit）**：salient-residual 分解。盐分敏感：照 BiLLM 论文 / repo 默认参数跑，不要自己改阈值。
- **论文 anchor**: LLaMA-2-7B / 实测 PPL **20–60 都算"同量级"**（baseline 5.47，二值化容忍度大）。这是为什么 §3.4 给 BiLLM 的判定用"同量级"而不是 ±0.3。
- **跑时间**：1–4 小时（最长）；**不要在 12GB 本地跑 canonical**，OOM。
- **重点研读**：BiLLM 的 salient mask 怎么定（Phase 2 笔记 §3 节）、binarization+残差的具体形式。

### §6.3 KIVI（流程最不同）

- **不需要 calibration**。KIVI 是 inference-time KV cache 量化（per-channel keys + per-token values, 2bit），跟权重量化是两个层面。
- **vendor + 编译 CUDA extension**：
  ```bash
  cd KIVI
  git submodule add https://github.com/jy-yuan/KIVI third_party/KIVI
  cd third_party/KIVI
  pip install -r requirements.txt
  cd quant && python setup_cuda.py install   # 编译自定义 CUDA kernel
  cd ../../..
  ```
  - **前置**：本机有 CUDA Toolkit + nvcc（§1.3）。Lab 服务器上一般有。
  - **常见编译失败**：nvcc 版本与 PyTorch 自带 CUDA runtime 不匹配。检查：
    ```bash
    nvcc --version          # 编译器
    python -c "import torch; print(torch.version.cuda)"   # PyTorch 期望
    ```
    两者主版本号要一致（都是 12.1 或都是 11.8）。
- **`repro.py` 改造**：
  - Step 3（calibration）跳过
  - Step 4（量化）变成把 KIVI 的自定义 attention monkey-patch 到 HF model：
    ```python
    from kivi.kivi_llama import enable_kivi_for_llama
    enable_kivi_for_llama(model, k_bits=2, v_bits=2, k_group_size=32, v_group_size=32)
    # 之后 model.forward(...) 时 KV cache 自动 2-bit
    ```
- **关键评测差异**：还要测 KV cache 显存（不只权重）。**`common/eval/memory.py` 当前缺 `measure_kv_cache_bytes`** —— 这是 Plan D 才补的（[设计文档 Plan D](superpowers/specs/2026-05-09-quant-reproduce-design.md#5-phase-3--占位)）。Plan D 会一并加 KV cache profiler。
- **论文 anchor**: LLaMA-2-7B + KV-2bit / WT2 PPL 几乎不掉（容差 ±0.3）。

---

## §7 数字打不到目标的排查流程

按顺序排查（所有方法通用）：

### Step 1 — 模型 ID 与 HF commit SHA

打开 `meta_<config>.json` 看 `model` 字段，对照 HuggingFace 模型卡的当前 commit。HF 偶尔会换权重。LLaMA-2-7B **必须用** `meta-llama/Llama-2-7b-hf`（不是 NousResearch 等镜像，那个 tokenizer 略有差异）。

### Step 2 — Calibration 切片

确认 `meta` 里 `seed=42`、`calib_samples=128`、`seq_len=2048`。换 seed 重跑一次看波动多大：

```bash
python repro.py --model ... --config w4g128 --seed 7 --out results_seed7/
```

差异 < 0.1 PPL = 正常波动；差异 > 0.5 PPL = calibration 路径有问题。

### Step 3 — 官方 repo 的 issue 区

```
https://github.com/AutoGPTQ/AutoGPTQ/issues?q=reproduce
https://github.com/casper-hansen/AutoAWQ/issues?q=reproduce
```

搜 `"reproduce"` / `"PPL difference"`，看作者怎么回。两个 repo 都很活跃，类似问题大概率有人问过。

### Step 4 — 用官方 eval 脚本对照

把官方 `auto_gptq/examples/quantization/quant_with_alpaca.py`（或 AWQ repo 的 `evaluation/eval.py`）拉过来跑同一份模型 + 同一份 calibration，对比 PPL。

差异原因：

- 在你的 `compute_ppl` 实现上 → 检查 stride、loss × seq_len 的细节
- 在量化算法本身 → auto-gptq / autoawq 自己的 bug 或版本回归

### Step 5 — 如实记录

差距找不到原因 → 在子目录 `README.md` 末尾加：

```markdown
## 实测异常记录

| 配置 | 实测 PPL | 论文 anchor | 差距 | 已排查步骤 |
|------|---------|------------|------|-----------|
| w4g128 | 6.21 | 5.69 | +0.52 | 1.✓ HF SHA 对得上 2.✓ seed/calib/seq 都按协议 3.✗ 官方 issue 暂无相同问题 4.✓ 官方 eval 脚本同环境跑也得 6.18 |

**结论**：差距与官方实现自身一致。可能是 auto-gptq 0.7.1 与论文版本（GPTQ 原版）的 Cholesky 实现差异。可在 Phase 2 笔记里展开。
```

> **不要为了好看刷参数**。作品集真正可信度在这里 —— 申请委员会更看重你"诚实地排查并记录"的能力，而不是拼凑出与论文严格一致的数字。

---

## §8 完成判定

### 单方法完成（Phase 1 + Phase 2 of 该方法）

`✅` 所有条目都满足：

- [ ] `<METHOD>/results/results_<config>.json` 存在，含 `ppl_wikitext2` + `zeroshot` (6 项) + `memory`
- [ ] `<METHOD>/results/meta_<config>.json` 含完整元数据
- [ ] `<METHOD>/README.md` "实测 vs 论文" 表填好（数字达 §3.4 容差，或写明 §7 的排查记录）
- [ ] `docs/reports/<method>.md` 五节都写完（不是模板）
- [ ] git log 干净（每个里程碑一个 commit）
- [ ] 打 tag `phase1-<method>-done`

### Plan A (Phase 1+2 of GPTQ) 完成

GPTQ 单方法完成 + 顶层 README 把 GPTQ 标 ✅ + `docs/results/summary.md` 有 GPTQ 一行。

### Phase 1+2 全部完成（四方法都 ✅ 后）

- [ ] 顶层 `README.md` 4 个方法都 ✅
- [ ] `docs/results/summary.md` 大表 4 行
- [ ] git tag `phase1-all-done`
- [ ] **此时才**写 Phase 3 spec：`docs/superpowers/specs/YYYY-MM-DD-phase3-unified-pipeline-design.md`

---

## §9 Colab 备用通道

什么时候用：
- Lab 服务器排队 / 暂时不可用
- 本地 12GB 想跑稍大的模型（比如 GPTQ TinyLlama 已经过了，想顺手试 LLaMA-2-7B 但本地 12GB OOM）

### §9.1 在 Colab 跑 GPTQ canonical

新建 Colab notebook，选 **A100 GPU runtime**（Pro 才有）：

```python
# Cell 1: 装 git + clone 仓库
!apt-get install -y git
!git clone https://github.com/你的账号/quant-reproduce.git
%cd quant-reproduce

# Cell 2: 装依赖（Colab 自带 CUDA torch，省 1GB 下载）
!pip install -e .[dev]
!pip install -r GPTQ/requirements.txt

# Cell 3: HF login
from huggingface_hub import login
login("hf_xxxxxxxx")  # 你的 token

# Cell 4: 预下载 LLaMA-2-7B
import os
os.environ["HF_HOME"] = "/content/hf_cache"
from common.models import load_hf_model
load_hf_model("meta-llama/Llama-2-7b-hf")

# Cell 5: 跑 canonical
%cd GPTQ
!python repro.py \
    --model meta-llama/Llama-2-7b-hf \
    --config w4g128 \
    --calib-samples 128 --seq-len 2048 \
    --eval ppl,zeroshot,memory --out results/

# Cell 6: 把 results 下载到本地（保险措施，Colab session 会断）
from google.colab import files
files.download("results/results_w4g128.json")
files.download("results/meta_w4g128.json")
```

### §9.2 注意事项

- **Colab session 限时**：免费 12 小时、Pro 24 小时，断后丢失 cache。每次跑长任务**及时下载** results JSON。
- **挂载 Google Drive 持久化 cache**：
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"   # 14GB 模型存到 Drive
  ```
- **Colab 的 GPU 型号会浮动**：T4 / V100 / A100 / L4。LLaMA-2-7B canonical 至少要 V100 16GB；A100 40GB 最稳。
- **Colab 不算"主路径"**：拿到的数字进 git 时，`meta.json` 会记录是 Colab 跑的（GPU name = "Tesla A100-SXM4-40GB" 等），README 里如实写"在 A100 / Colab Pro 复跑"。

---

## §10 常见坑速查

| # | 现象 | 原因 | 解决 / 提前预防 |
|---|------|------|----------------|
| 1 | `pip install auto-gptq` 报 `torch not found` | conda env 没装 CUDA torch | 先 `pip install torch --index-url https://download.pytorch.org/whl/cu121` 再装 auto-gptq；或在 `env.yml` pin `pytorch-cuda=12.1` |
| 2 | `torch.cuda.is_available()` 返回 False | 装了 CPU 版 torch | 卸掉重装：`pip uninstall torch -y; pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| 3 | `.\scripts\env_local.ps1` 报 `running scripts is disabled` | PS 执行策略默认禁 | 一次性：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| 4 | `huggingface_hub` 报 `no symlinks` warning（Windows） | Windows 默认不开发者模式 | 不影响功能，warning 可忽略；或 Settings → Developer Mode 开 |
| 5 | LLaMA-2 下载 401/403 | 没授权 / 没登录 | `huggingface-cli login` + 同意 license 页面（§2.2） |
| 6 | `conda activate` 在 .sh 里失败 | 非交互 shell 没加载 conda 函数 | 我们的 `run_phase1_method.sh` 已 `source conda.sh`；自定义脚本要照做 |
| 7 | Lab 上 git push 被拒 | 默认分支保护或缺 git 身份 | `git config --global user.email ...` + `git config --global user.name ...` |
| 8 | `ppl_wikitext2` 是 NaN 或 inf | 模型在 cuda 上但 tokens 在 cpu / 数值溢出 | `compute_ppl` 内部已 `.to(device)`；如果 NaN，看 model dtype（fp16 在长 seq 容易溢出，强制 fp32） |
| 9 | BiLLM 跑到一半 OOM | 二值化中间张量大 | 减 batch；canonical 必须用 96GB lab，不要在 12GB 本地跑 |
| 10 | KIVI CUDA 编译失败 | nvcc 与 PyTorch CUDA 版不匹配 | `nvcc --version` 与 `python -c "import torch; print(torch.version.cuda)"` 主版本号要一致 |
| 11 | `pytest --collect-only` 报 `Unknown mark` | 标记没在 pyproject 注册 | 已在 pyproject.toml 注册 `slow` 和 `require_cuda`；自定义新标记要追加 |
| 12 | `pip install -e .` 在 Windows 报路径太长 | Windows 默认 MAX_PATH 260 | 注册表开 LongPathsEnabled，或装到短路径（D:\Q\repro） |
| 13 | `lm-eval-harness` 第一次跑卡很久不动 | 在下 dataset（PIQA ~10MB） | 等 / 设代理；走完一次后 cache 在 `~/.cache/huggingface/datasets` |
| 14 | Lab 上跑 `repro.py` 报 `no module named common` | 该 env 里 `pip install -e .` 没做 | 在 lab 那个 conda env 里再走一次 `pip install -e .` |
| 15 | git push 出现 `LF will be replaced by CRLF` warning | Windows 默认 `core.autocrlf=true` | 不影响（git 内部存 LF，Linux 拉下来正常）；Warning 可忽略 |
| 16 | Colab session 断了 results 没保存 | runtime 超时 | 下次记得 §9.1 Cell 6 下载 results JSON；或挂 Drive 持久化 |

---

## §11 接下来你具体做什么

1. **现在**（已经完成）：framework 全部就绪，所有 `common/` 测试通过。
2. **下一步**：从 [Plan A Task 8](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md#task-8-创建-gptq-子目录骨架readme--envyml--requirementstxt--空-repropy) 开始，按 §5 走 GPTQ 完整流程。
3. **第一里程碑**：跑出 GPTQ canonical 数字（PPL ≈ 5.69 ± 0.3）+ Phase 2 笔记 → tag `phase1-gptq-done`。
4. **告诉我数字** → 我写 Plan B (AWQ)。AWQ 流程跟 GPTQ 95% 重叠，复用你 GPTQ 时积累的所有 muscle memory 和 troubleshooting 经验。

GPTQ 做完之后的预期 commit 历史：

```
* (tag: phase1-gptq-done) docs: link GPTQ as completed in top README + initialize summary table
* docs(reports): add Phase 2 source-code study for GPTQ
* docs(gptq): fill in canonical numbers vs paper anchor
* data(gptq): canonical W4-g128 results on LLaMA-2-7B
* test(gptq): smoke run with TinyLlama-1.1B passes
* feat(gptq): implement end-to-end quantize → eval → save pipeline
* feat(gptq): add CLI argparse + quantization presets
* scaffold(gptq): create subdir with env.yml, requirements, README template
```

每个 commit 对应 [Plan A](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md) 的一个 Task。卡住的步骤把数字 / 报错贴回来，我帮 debug，但不替你写算法。
