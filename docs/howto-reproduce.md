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

每个量化方法走同一个 5 步循环，**一次只做一个方法**：

```
GPTQ → AWQ → BiLLM → KIVI
 │
 ├─ 1. Vendor 上游 + 子目录骨架   (git submodule add <upstream>; 写 README + env.yml)
 ├─ 2. 建 conda env             (.\scripts\env_local.ps1 <METHOD> + 装上游 requirements)
 ├─ 3. 本地烟雾跑               (跑上游 example/quantize.py + TinyLlama-1.1B，5–10 min)
 ├─ 4. Lab canonical 跑         (跑同一个 example + LLaMA-2-7B，30–90 min)
 └─ 5. 抽数字 + 写 README       (从上游 stdout 抠 PPL / accuracy → 实测 vs 论文表)

之后再做 Phase 2: 源码笔记 (docs/reports/<method>.md，五节模板)
```

> **不是** 写自己的 `repro.py` / argparse / orchestration。**是** 跑上游 repo 自己的复现脚本，把数字捞出来。`common/` 是 Phase 3 资产。

### 0.2 评测协议（按各论文/上游 repo 默认）

不同方法的上游 example 用的 PPL 实现 / zero-shot tasks 可能略有差异，**Phase 1 不强求统一** —— 目标是各自跟自己论文的数字对得上。横向可比是 Phase 3 的事（用 `common/eval` 自己跑）。

每方法跑出的 stdout 文件保留进 git，三个月后想复盘 / 对照时知道当时跑了什么。

### 0.3 三档硬件分工

| 档位 | 配置 | 用途 |
|------|------|------|
| 本地 | 12GB GPU 单卡 | **烟雾跑**（跑上游 example + TinyLlama 验证 env） |
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

### 3.4 各方法 conda env 要求总览

每个方法**独立 conda env**（命名 `quant-<method>`），不与其他方法或 base 混。原因：

- transformers 版本互斥（BiLLM/KIVI 跟着论文当时 pin 在 4.36.x；AWQ 要 ≥4.42）
- torch 主版本不同（BiLLM/KIVI 用 2.1.x；AWQ 要 ≥2.2；GPTQ 较灵活）
- 自定义 CUDA kernel（KIVI）编译时与 torch 自带 CUDA runtime 绑死，错配直接编译失败

下表是上游官方 repo 实际要求的归纳（截至 2026-01；具体以你 vendor 后看到的 `requirements.txt` 为准）：

| 方法 | conda env name | Python | PyTorch | CUDA | 主依赖（pip） | 装法 | 上游来源 |
|------|---------------|--------|---------|------|--------------|------|---------|
| **GPTQ** | `quant-gptq`  | 3.10 | 2.2.x | 12.1 | `auto-gptq>=0.7,<1.0`、`transformers>=4.40` | pip pre-built wheel | <https://github.com/AutoGPTQ/AutoGPTQ> |
| **AWQ**  | `quant-awq`   | 3.10 | 2.3.x | 12.1 | `autoawq>=0.2`、`autoawq-kernels`、`transformers>=4.42` | pip pre-built wheel | <https://github.com/casper-hansen/AutoAWQ> |
| **BiLLM**| `quant-billm` | 3.10 | 2.1.x | 12.1 | (vendor) `transformers~=4.36`、`accelerate` | git submodule + `pip install -e third_party/BiLLM` | <https://github.com/Aaronhuang-778/BiLLM> |
| **KIVI** | `quant-kivi`  | 3.10 | 2.1.x | 12.1 | (vendor) `transformers~=4.36`、`flash-attn>=2.4`、`triton` | git submodule + 编译 `setup_cuda.py` | <https://github.com/jy-yuan/KIVI> |

> ⚠️ 这些版本号是**起步模板**，不是最终钦点。每方法 vendor 或装完之后，对照上游 README / `requirements.txt` 检查一遍：上游可能又改了 pin。如果 vendor 后 `pip install -r third_party/<repo>/requirements.txt` 报版本冲突，**优先按 vendor repo 的 pin** 调你的 `<METHOD>/env.yml`。

### 3.5 每个方法的 env 都要的"公共底座"

不管哪个方法，下面这套必须在 env 里：

```yaml
# 每个 <METHOD>/env.yml 都长这样
name: quant-<method>
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pytorch=2.x.*           # 各方法不同，见 §3.4 表
  - pytorch-cuda=12.1       # 与上面 torch CUDA 版必须一致
  - pip:
    - -r requirements.txt   # 该方法的 pip 依赖
```

```text
# 每个 <METHOD>/requirements.txt 都至少含这些（公共部分）
transformers>=4.40           # 各方法可能 pin 不同的版本
accelerate>=0.27
datasets>=2.18
sentencepiece                # LLaMA / Mistral tokenizer 必须
lm-eval>=0.4.2               # 用于 zero-shot 评测
# ↓ 加各方法专属的 pip 包 ↓
```

**Phase 1 不需要在每个方法 env 里装 `common/`** —— Phase 1 跑的是上游 repo 自己的脚本，不 import `common/`。`pip install -e .` 这一步留给 Phase 3（那时你的 unified pipeline 才会 `from common.* import`）。

> 真正需要在每个方法 env 里装的是该方法 vendor 的上游 repo 自己的 `requirements.txt`：
> ```bash
> conda activate quant-<method>
> cd <METHOD>/third_party/<UPSTREAM_REPO>
> pip install -r requirements.txt   # 跟着上游 README
> ```
>
> 共享的不是 `env`（每个 env 都独立），共享的是 **`HF_HOME` 的模型 cache**（§2.3）—— 模型权重不重复下载。

---

## §4 通用方法工作流（每方法 5 步）

> **核心理念**：Phase 1 不写自己的 Python orchestration（`repro.py`、argparse、main flow 都不写）。直接跑上游 repo 自己提供的复现脚本。`common/`（已搭好的 5 个模块）是 Phase 3 资产，Phase 1 不用。

### Step 1 — Vendor 上游 repo + 写子目录骨架

```
GPTQ/                            （AWQ/ BiLLM/ KIVI/ 同结构）
├── README.md                    # 跑法（粘贴上游命令） + 数字表 + 与论文对比 + troubleshooting
├── env.yml                      # conda env 定义（pytorch + 该方法依赖）
├── third_party/<UPSTREAM_REPO>/ # git submodule，整个上游 repo 在这里
└── results/                     # 上游脚本的输出（stdout 重定向、JSON、checkpoint 路径等）
    ├── smoke/                   # 本地小模型烟雾跑（gitignored）
    ├── canonical_w4g128_stdout.txt
    ├── canonical_w4g128_meta.md  # 你手记的：HF SHA、GPU 型号、torch 版本、命令
    └── ...
```

**没有 `repro.py`，没有 `requirements.txt` 单独维护**（pip 依赖直接走 `third_party/<repo>/requirements.txt`）。

### Step 2 — 建 conda env + 装上游

```powershell
# Windows 本地
.\scripts\env_local.ps1 <METHOD>
conda activate quant-<method>

# 装上游：先按 third_party 里 requirements.txt
cd F:\CODE\Quant\reproduce\<METHOD>\third_party\<UPSTREAM_REPO>
pip install -r requirements.txt
pip install -e .                         # 让上游可以 import（如果是 vendor 型）
# 或 pip pre-built wheel 型：
# pip install auto-gptq>=0.7,<1.0  # 不需要 vendor 也行
```

> Lab 端用 `bash scripts/env_lab.sh <METHOD>` 替代第一行。

**验证上游能跑**：每个上游 repo 都有 README，里面通常有一句"快速开始"命令。先拿那个跑通，证明 env 没问题。

### Step 3 — 本地烟雾跑（用上游脚本 + 小模型）

读上游 README 或 `examples/`，找到他们的复现脚本。比如：

| 方法 | 上游复现脚本（示例位置） |
|------|------------------------|
| GPTQ | `third_party/AutoGPTQ/examples/quantization/quant_with_alpaca.py` |
| AWQ  | `third_party/AutoAWQ/examples/quantize.py` + `examples/eval.py` |
| BiLLM | `third_party/BiLLM/run.py` |
| KIVI | `third_party/KIVI/quant_llama.py` + `third_party/KIVI/eval/ppl_eval.py` |

> 实际脚本名以你 vendor 那一刻 repo 里的为准。先 `ls third_party/<repo>/examples/` 看一眼。

**烟雾跑** = 用 TinyLlama-1.1B 跑同一个脚本，验证你的 env 不挂：

```powershell
cd F:\CODE\Quant\reproduce\GPTQ\third_party\AutoGPTQ
python examples/quantization/quant_with_alpaca.py `
    --pretrained_model_dir TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
    --quantized_model_dir ../../results/smoke/quantized `
    --bits 4 --group_size 128 `
    --num_samples 32                         # 跟着上游脚本的实际参数名
```

**判定**：脚本能跑完不挂，输出一个 quantized model 文件夹。**不**判定数字。

### Step 4 — Lab canonical 跑（用上游脚本 + LLaMA-2-7B）

把代码 sync 到 lab（`git push` / `rsync`），换成 canonical 参数：

```bash
cd ~/quant-reproduce/GPTQ/third_party/AutoGPTQ
python examples/quantization/quant_with_alpaca.py \
    --pretrained_model_dir meta-llama/Llama-2-7b-hf \
    --quantized_model_dir ../../results/quantized_w4g128 \
    --bits 4 --group_size 128 \
    --num_samples 128 \
    2>&1 | tee ../../results/canonical_w4g128_stdout.txt
```

跑完后通常上游脚本会同时跑 PPL / lm-eval。如果没有自动 eval，再单独跑：

```bash
# 例：再跑 PPL 评测（具体脚本看上游）
python examples/evaluation/ppl_eval.py \
    --model_path ../../results/quantized_w4g128 \
    --eval_dataset wikitext2 \
    2>&1 | tee ../../results/canonical_w4g128_ppl.txt
```

数字会在 stdout 里。把全部 stdout 重定向到 `<METHOD>/results/canonical_*.txt` 留底。

### Step 5 — 写 README 数字 + 元数据

- 从 `results/canonical_*.txt` 抽出 PPL / zero-shot / memory 数字。
- 在 `<METHOD>/README.md` 填数字表（实测 vs 论文）。
- 在 `<METHOD>/results/canonical_w4g128_meta.md` 手记一份元数据（不是 JSON，就一段 markdown）：
  ```markdown
  - 模型: meta-llama/Llama-2-7b-hf @ commit abc123
  - GPU: NVIDIA A100-SXM4-40GB
  - torch: 2.2.1+cu121
  - transformers: 4.42.0
  - auto-gptq: 0.7.1
  - 命令: python examples/quantization/quant_with_alpaca.py --pretrained_model_dir ...
  - 时间: 2026-05-15 14:32 → 15:47 (1h15m)
  ```
- Commit。

> Phase 2 源码笔记是另一步（按 [设计文档 §3.5](superpowers/specs/2026-05-09-quant-reproduce-design.md#35-phase-2-研读笔记-docsreportsmethodmd-大纲) 五节模板）写 `docs/reports/<method>.md`。Phase 1 不含。

---

## §5 GPTQ 完整 walkthrough（第一次）

下面是**第一次做 GPTQ** 的完整命令序列。AWQ / BiLLM / KIVI 后面三次都是同一套流程，只换上游脚本路径和参数名。

> **提醒**：本节全部是"跑上游 repo 自己的脚本"。**不**写 `repro.py`、**不**写 argparse、**不**用 `common/eval`。`common/` 是 Phase 3 才用。

### §5.1 Vendor 上游 repo + 写子目录骨架

```powershell
cd F:\CODE\Quant\reproduce\GPTQ
git submodule add https://github.com/AutoGPTQ/AutoGPTQ third_party/AutoGPTQ
```

子目录长这样（极简，全部产出文件都靠手写或上游 stdout）：

| 文件/目录 | 作用 |
|----------|------|
| `GPTQ/README.md` | 跑法（粘贴上游命令） + 数字表 + 对论文比 + troubleshooting |
| `GPTQ/env.yml` | conda env 定义（pytorch + auto-gptq pre-built wheel 或 -e third_party） |
| `GPTQ/third_party/AutoGPTQ/` | git submodule，**所有算法/CLI/eval 脚本都在这** |
| `GPTQ/results/` | 跑出来的产物（stdout 重定向、quantized checkpoint、手写 meta） |

#### `GPTQ/env.yml` 模板

```yaml
name: quant-gptq
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pytorch=2.2.*           # auto-gptq 0.7.x 与 torch 2.2 兼容良好
  - pytorch-cuda=12.1       # 与本机 NVIDIA 驱动对应（nvidia-smi 应 ≥ 12.1）
  - pip:
    - auto-gptq>=0.7,<1.0   # pip pre-built wheel；或 -e ./third_party/AutoGPTQ
    - sentencepiece         # LLaMA tokenizer 强制依赖
    # 其它依赖跟着 third_party/AutoGPTQ/requirements.txt 走（vendor 后再 pip install -r）
```

#### `GPTQ/README.md` 起步模板（先填占位，跑完 §5.5 再填实数）

```markdown
# GPTQ — LLaMA-2-7B 复现

## 上游
- Repo: https://github.com/AutoGPTQ/AutoGPTQ
- Vendor 路径: `third_party/AutoGPTQ`
- 用的脚本: `examples/quantization/quant_with_alpaca.py`（看 vendor 的实际目录）

## 跑法

### 烟雾跑（本地 12GB / TinyLlama-1.1B）
```bash
cd third_party/AutoGPTQ
python examples/quantization/quant_with_alpaca.py \
    --pretrained_model_dir TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --quantized_model_dir ../../results/smoke/quantized \
    --bits 4 --group_size 128 --num_samples 32
```

### Canonical（lab 96GB / LLaMA-2-7B）
```bash
cd third_party/AutoGPTQ
python examples/quantization/quant_with_alpaca.py \
    --pretrained_model_dir meta-llama/Llama-2-7b-hf \
    --quantized_model_dir ../../results/quantized_w4g128 \
    --bits 4 --group_size 128 --num_samples 128 \
    2>&1 | tee ../../results/canonical_w4g128_stdout.txt

# 然后跑评测（具体脚本看上游 examples/）
```

## 实测 vs 论文
| Config | Model | Metric | 实测 | 论文 anchor | 判定 |
|--------|-------|--------|------|------------|------|
| w4g128 | LLaMA-2-7B | WT2 PPL | _TBD_ | ≈ 5.69 | _TBD_ |
| w4g128 | LLaMA-2-7B | piqa    | _TBD_ | ≈ 0.78 | _TBD_ |
| w4g128 | LLaMA-2-7B | weights | _TBD_ | ≈ 3.7 GB | _TBD_ |

## Troubleshooting
（数字打不到 ±0.3 时的排查记录）
```

**完成后**：

```powershell
cd F:\CODE\Quant\reproduce
git add .gitmodules GPTQ/
git commit -m "scaffold(gptq): vendor AutoGPTQ + env.yml + README template"
```

### §5.2 建 `quant-gptq` conda env + 装上游依赖

```powershell
cd F:\CODE\Quant\reproduce
.\scripts\env_local.ps1 GPTQ
conda activate quant-gptq

# 装上游 repo 自己列的依赖
cd GPTQ\third_party\AutoGPTQ
pip install -r requirements.txt
# auto-gptq 主包已通过 env.yml 的 pip 段装好；如果想用源码版本：
# pip install -e .
```

**验证 env 没问题**：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.2.x+cu121 True

python -c "import auto_gptq; print(auto_gptq.__version__)"
# 期望: 0.7.x

# 跑一下上游 repo 自带的最简示例（看 README 第一段，通常是个 inference demo）
# 如果能跑出几句生成的文字，env OK。
```

### §5.3 本地烟雾跑（用上游 example，TinyLlama）

```powershell
conda activate quant-gptq
cd F:\CODE\Quant\reproduce\GPTQ\third_party\AutoGPTQ
python examples\quantization\quant_with_alpaca.py `
    --pretrained_model_dir TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
    --quantized_model_dir ..\..\results\smoke\quantized `
    --bits 4 --group_size 128 `
    --num_samples 32 `
    2>&1 | Tee-Object ..\..\results\smoke\stdout.txt
```

> 实际参数名以 `examples/quantization/quant_with_alpaca.py --help` 输出为准 —— 上游可能版本不同。如果上游的脚本名不同（比如 `examples/quantization/run.py`、`scripts/quantize.py`），相应替换。

**判定**：脚本能跑完不挂，`results/smoke/quantized/` 出来一个量化后的 checkpoint 文件夹。**不**关心 PPL 数字。

> Smoke 结果默认不进 git（`.gitignore` 屏蔽 `*/results/smoke/`）。

### §5.4 Lab canonical 跑（用上游 example + LLaMA-2-7B）

#### §5.4.1 把代码同步到 lab

**Option A — git push/pull**（推荐）：

```powershell
# 本地
cd F:\CODE\Quant\reproduce
git push origin master      # 假设已经 git remote add origin ...
```

```bash
# Lab 上
cd ~
git clone --recurse-submodules <你的远端 URL> quant-reproduce
# 注意 --recurse-submodules，把 third_party/AutoGPTQ 一并拉下来
cd quant-reproduce
```

> 如果之前 clone 时没带 `--recurse-submodules`：`git submodule update --init --recursive`。

**Option B — rsync**：

```bash
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='*/results/smoke' \
  user@local:F:/CODE/Quant/reproduce/  ~/quant-reproduce/
```

#### §5.4.2 Lab 上准备 env

```bash
cd ~/quant-reproduce
bash scripts/env_lab.sh GPTQ
conda activate quant-gptq

cd GPTQ/third_party/AutoGPTQ
pip install -r requirements.txt

# 预下载 LLaMA-2-7B（避免计入跑时）
huggingface-cli login    # 粘贴 HF token
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')"
# 14GB 下载到 $HF_HOME
```

#### §5.4.3 跑 canonical（量化 + 评测）

```bash
cd ~/quant-reproduce/GPTQ/third_party/AutoGPTQ

# 1. 量化（30–60 min）
python examples/quantization/quant_with_alpaca.py \
    --pretrained_model_dir meta-llama/Llama-2-7b-hf \
    --quantized_model_dir ../../results/quantized_w4g128 \
    --bits 4 --group_size 128 --num_samples 128 \
    2>&1 | tee ../../results/canonical_w4g128_quant_stdout.txt

# 2. 评测 PPL（看上游有什么评测脚本，比如 examples/benchmark/perplexity.py）
python examples/benchmark/perplexity.py \
    --model_name ../../results/quantized_w4g128 \
    --is_quantized \
    --dataset wikitext \
    2>&1 | tee ../../results/canonical_w4g128_ppl_stdout.txt

# 3. 评测 zero-shot（用 lm-eval-harness 直接跑量化后的 model）
python -m lm_eval \
    --model hf \
    --model_args pretrained=../../results/quantized_w4g128 \
    --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,openbookqa \
    --batch_size 1 \
    --output_path ../../results/canonical_w4g128_zeroshot.json
```

> 上游脚本名 / 参数名以你 vendor 时的实际为准。**这一节不是死命令**，是教你"看 upstream/examples/ 里有什么 → 用什么"的工作方式。

**完成标志**：`GPTQ/results/canonical_w4g128_*.txt` + `canonical_w4g128_zeroshot.json` 都生成，里面有 PPL / 6 项 zero-shot 数字。

#### §5.4.4 把 `results/` 拉回本地

```bash
# Lab 上
cd ~/quant-reproduce
git add GPTQ/results/canonical_w4g128_*.txt GPTQ/results/canonical_w4g128_zeroshot.json
git commit -m "data(gptq): canonical W4-g128 results on LLaMA-2-7B"
git push
```

```powershell
# 本地
cd F:\CODE\Quant\reproduce
git pull
ls GPTQ\results\
```

### §5.5 抽数字 + 写 README + 手记元数据

打开 `GPTQ/results/canonical_w4g128_ppl_stdout.txt` 等文件，从 stdout 里抠出 PPL / 6 项 zero-shot accuracy 数字（上游脚本通常会打印一行 `Perplexity: 5.71` 之类）。

填到 `GPTQ/README.md`：

```markdown
| Config | Model | Metric | 实测 | 论文 anchor | 判定 |
|--------|-------|--------|------|------------|------|
| w4g128 | LLaMA-2-7B | WT2 PPL | **5.71** | ≈ 5.69 | ✅ 在 ±0.3 内 |
| w4g128 | LLaMA-2-7B | piqa | 0.781 | ≈ 0.78 | ✅ |
| w4g128 | LLaMA-2-7B | weights GB | 3.71 | ≈ 3.7 | ✅ 真 INT4 packed |
```

**手记元数据**到 `GPTQ/results/canonical_w4g128_meta.md`（不写 JSON，markdown 一段就够）：

```markdown
# canonical_w4g128 元数据

- 模型: meta-llama/Llama-2-7b-hf @ commit `01c7f73d771dfac7d292323805ebc428287df4f9`
- GPU: NVIDIA A100-SXM4-40GB
- 环境: torch 2.2.1+cu121, transformers 4.42.0, auto-gptq 0.7.1
- 完整命令: 见 canonical_w4g128_quant_stdout.txt 第一行
- 时间: 2026-05-15 14:32 → 15:47 (1h15m)
- 论文 anchor 来源: AWQ paper Table 4（GPTQ 行）
```

> HF commit SHA：在 `~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/<sha>/` 看那个 sha，或 `huggingface-cli scan-cache`。

> 数字打不到 ±0.3 → §7 排查流程。**不要为了好看刷参数。**

Commit：

```powershell
git add GPTQ/README.md GPTQ/results/canonical_w4g128_meta.md
git commit -m "docs(gptq): fill in canonical numbers vs paper anchor + meta"
```

### §5.6 Phase 2 源码笔记

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

GPTQ 之后三个方法都按 §5 同样套路：vendor → 建 env → 跑上游 example → 抓数字 → 写 README。**唯一差异**是上游脚本路径、参数名、论文 anchor 数字。

### §6.1 AWQ

- **Vendor**：
  ```bash
  cd AWQ
  git submodule add https://github.com/casper-hansen/AutoAWQ third_party/AutoAWQ
  ```
- **`AWQ/env.yml`**：
  ```yaml
  name: quant-awq
  channels:
    - pytorch
    - nvidia
    - conda-forge
  dependencies:
    - python=3.10
    - pip
    - pytorch=2.3.*            # autoawq 0.2 要求 torch ≥ 2.2
    - pytorch-cuda=12.1
    - pip:
      - autoawq>=0.2           # 主算法（pip pre-built wheel）
      - autoawq-kernels        # 推理 kernel
      - sentencepiece
      # 其它依赖跟 third_party/AutoAWQ/requirements.txt 走
  ```
- **上游复现脚本**：通常在 `third_party/AutoAWQ/examples/` 下：
  - `examples/quantize.py` — 量化单个模型
  - `examples/eval.py` — PPL 评测（如果存在）
  - 或直接看 README 里"Quick Start"那段命令
- **跑法（粘贴到 `AWQ/README.md`）**：
  ```bash
  cd AWQ/third_party/AutoAWQ
  python examples/quantize.py \
      --model_path meta-llama/Llama-2-7b-hf \
      --quant_path ../../results/quantized_w4g128 \
      --w_bit 4 --q_group_size 128 \
      2>&1 | tee ../../results/canonical_w4g128_quant_stdout.txt
  # 然后跑评测：
  python -m lm_eval --model hf \
      --model_args pretrained=../../results/quantized_w4g128 \
      --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,openbookqa \
      --output_path ../../results/canonical_w4g128_zeroshot.json
  ```
  > 实际参数名以你 vendor 时的 `examples/quantize.py --help` 为准。
- **论文 anchor**: LLaMA-2-7B / W4-g128 / WT2 PPL ≈ 5.60（baseline 5.47，容差 ±0.3）。
- **跑时间**：~30 min on lab 96GB。
- **AWQ 的 quant_config 与 GPTQ 不同**（如果你看上游代码会注意到）：
  - `q_group_size` ≠ GPTQ 的 `group_size`
  - `w_bit` ≠ `bits`
  - `zero_point: True` ≠ `sym: False`
  - 多一个 `version`（推理 kernel 类型："GEMM" / "GEMV"）
  > 这些差异 Phase 2 笔记里展开，Phase 1 不用管。

### §6.2 BiLLM

- **Vendor**（**官方 repo 不在 PyPI，必须 vendor**）：
  ```bash
  cd BiLLM
  git submodule add https://github.com/Aaronhuang-778/BiLLM third_party/BiLLM
  ```
- **`BiLLM/env.yml`**：
  ```yaml
  name: quant-billm
  channels:
    - pytorch
    - nvidia
    - conda-forge
  dependencies:
    - python=3.10
    - pip
    - pytorch=2.1.*            # BiLLM 论文 repo 测过的版本
    - pytorch-cuda=12.1
    - pip: []
      # 实际依赖以 third_party/BiLLM/requirements.txt 为准；vendor 后用 pip install -r 装
  ```
- **vendor 后装依赖**：
  ```powershell
  conda activate quant-billm
  cd F:\CODE\Quant\reproduce\BiLLM\third_party\BiLLM
  pip install -r requirements.txt
  ```
- **上游复现脚本**：BiLLM 通常有一个总入口 `run.py`（看上游 README 实际叫什么）。**跑法**：
  ```bash
  cd BiLLM/third_party/BiLLM
  python run.py \
      --model meta-llama/Llama-2-7b-hf \
      --eval_ppl \
      --braq \
      --salient_metric hessian \
      2>&1 | tee ../../results/canonical_billm_stdout.txt
  ```
  > 上游参数风格继承自 GPTQ 原始 repo，`--braq` `--salient_metric` 等是 BiLLM 特有的。**以上游 README 给的命令为准。**
- **论文 anchor**: LLaMA-2-7B / 实测 PPL **20–60 都算"同量级"**（baseline 5.47，二值化容忍度大）—— 见 [设计文档 §3.4](superpowers/specs/2026-05-09-quant-reproduce-design.md#34-复现完成判定与论文同量级)。
- **跑时间**：1–4 小时（最长）；**不要在 12GB 本地跑 canonical**，OOM。
- **常见冲突**：BiLLM 的 `requirements.txt` 经常 pin 较老的 transformers，跟新版 lm-eval 可能不兼容 —— 可以让 BiLLM 跑出 PPL 后单独用别的 env（比如 quant-gptq）跑 lm-eval-harness 的 zero-shot 评测，间接绕开。

### §6.3 KIVI（流程最不同）

- **不需要 calibration**。KIVI 是 inference-time KV cache 量化（per-channel keys + per-token values, 2bit），跟权重量化是两个层面。
- **vendor + 编译 CUDA extension**（必须本机有 CUDA Toolkit + nvcc，§1.3）：
  ```bash
  cd KIVI
  git submodule add https://github.com/jy-yuan/KIVI third_party/KIVI
  ```
- **`KIVI/env.yml`**：
  ```yaml
  name: quant-kivi
  channels:
    - pytorch
    - nvidia
    - conda-forge
  dependencies:
    - python=3.10
    - pip
    - pytorch=2.1.*            # KIVI 官方 README 指定 2.1.x
    - pytorch-cuda=12.1        # 必须与下面 nvcc 12.1 一致
    - cuda-toolkit=12.1        # 提供 nvcc 用于编译自定义 kernel（lab 服务器上可走 module load 替代）
    - pip:
      - -r requirements.txt
  ```
- **vendor 后装依赖 + 编译 CUDA kernel**：
  ```bash
  conda activate quant-kivi
  cd ~/quant-reproduce/KIVI/third_party/KIVI
  pip install -r requirements.txt                  # 装 transformers / datasets / flash-attn 等
  cd quant && python setup_cuda.py install         # ← 编译，~5-10 分钟
  ```
  - **常见编译失败**：nvcc 版本与 PyTorch 自带 CUDA runtime 不匹配。检查：
    ```bash
    nvcc --version          # 编译器
    python -c "import torch; print(torch.version.cuda)"   # PyTorch 期望
    ```
    两者主版本号要一致（都是 12.1 或都是 11.8）。conda env 里 `pytorch-cuda=12.1` + `cuda-toolkit=12.1` 同 channel 装就一致。
- **上游复现脚本**：KIVI 的脚本通常在 `third_party/KIVI/` 根目录或 `eval/`：
  - `pred.py` / `example_chat.py` — 推理 demo
  - `eval/ppl_eval.py` — PPL 评测（具体名以 vendor 时为准）
- **跑法**：
  ```bash
  cd KIVI/third_party/KIVI
  python eval/ppl_eval.py \
      --model_name meta-llama/Llama-2-7b-hf \
      --k_bits 2 --v_bits 2 --k_group_size 32 --v_group_size 32 \
      --use_kivi True \
      2>&1 | tee ../../results/canonical_kivi2_stdout.txt
  ```
  > 上游脚本在 KIVI repo 里也叫各种名字，**以 vendor 时实际看到的为准**。KIVI 的 README 通常给一个 "Reproducing the paper" 段落。
- **KIVI 跑法的本质差异**：
  - **不需要** calibration（KIVI 是 inference-time KV cache 量化）
  - **不需要** 量化后保存 checkpoint（KIVI monkey-patch 模型的 attention 层，没有"量化产物"这个概念）
  - 所以脚本就是"加载模型 → 打补丁 → 跑 PPL"一步到位
- **关键评测差异**：还要测 KV cache 显存（不只权重）。上游脚本通常会自带；如果没有，自己 `nvidia-smi` 在跑 long-context PPL 时记一下峰值。Phase 3 的 `common/eval/memory.py` 会补 `measure_kv_cache_bytes`。
- **论文 anchor**: LLaMA-2-7B + KV-2bit / WT2 PPL 几乎不掉（容差 ±0.3）。

---

## §7 数字打不到目标的排查流程

按顺序排查（所有方法通用）：

### Step 1 — 模型 ID 与 HF commit SHA

打开 `meta_<config>.json` 看 `model` 字段，对照 HuggingFace 模型卡的当前 commit。HF 偶尔会换权重。LLaMA-2-7B **必须用** `meta-llama/Llama-2-7b-hf`（不是 NousResearch 等镜像，那个 tokenizer 略有差异）。

### Step 2 — Calibration 切片

确认你跑的命令里 calibration 数量、seq_len 跟论文一致。换 seed 重跑一次看波动多大：

```bash
# 修改上游 example 的 --seed / --num_samples 参数
python examples/quantization/quant_with_alpaca.py --seed 7 ...
```

差异 < 0.1 PPL = 正常波动；差异 > 0.5 PPL = calibration 路径有问题（数据集版本？切片逻辑？）。

### Step 3 — 官方 repo 的 issue 区

```
https://github.com/AutoGPTQ/AutoGPTQ/issues?q=reproduce
https://github.com/casper-hansen/AutoAWQ/issues?q=reproduce
https://github.com/Aaronhuang-778/BiLLM/issues?q=reproduce
https://github.com/jy-yuan/KIVI/issues?q=reproduce
```

搜 `"reproduce"` / `"PPL difference"`，看作者怎么回。

### Step 4 — 检查上游版本

看你 vendor 的 commit。`cd third_party/<repo>; git log -1` 对照论文发表时间。如果上游后续 commit 改过算法核心（比如 GPTQ 的 Cholesky 数值稳定性 patch），数字会有微小漂移。

```bash
# 退到论文当时的 commit 看看
cd third_party/AutoGPTQ
git log --oneline | grep -i "v0.7"
git checkout v0.7.0   # 或论文引用的 SHA
```

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
# Cell 1: clone 仓库（含 submodule）
!git clone --recurse-submodules https://github.com/你的账号/quant-reproduce.git
%cd quant-reproduce

# Cell 2: 装上游依赖（Colab 自带 CUDA torch，不要再装 torch）
!pip install auto-gptq>=0.7,<1.0 sentencepiece
!pip install -r GPTQ/third_party/AutoGPTQ/requirements.txt

# Cell 3: HF login
from huggingface_hub import login
login("hf_xxxxxxxx")  # 你的 token

# Cell 4: 预下载 LLaMA-2-7B（Colab 默认 HF cache 在 /root/.cache/huggingface）
import os
os.environ["HF_HOME"] = "/content/hf_cache"
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Cell 5: 跑上游 example（GPTQ 量化）
%cd GPTQ/third_party/AutoGPTQ
!python examples/quantization/quant_with_alpaca.py \
    --pretrained_model_dir meta-llama/Llama-2-7b-hf \
    --quantized_model_dir ../../results/quantized_w4g128 \
    --bits 4 --group_size 128 --num_samples 128 \
    2>&1 | tee ../../results/canonical_w4g128_quant_stdout.txt

# Cell 6: 跑评测（lm-eval-harness）
!python -m lm_eval \
    --model hf \
    --model_args pretrained=../../results/quantized_w4g128 \
    --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,openbookqa \
    --output_path ../../results/canonical_w4g128_zeroshot.json

# Cell 7: 把 results 下载到本地（保险措施，Colab session 会断）
from google.colab import files
files.download("../../results/canonical_w4g128_quant_stdout.txt")
files.download("../../results/canonical_w4g128_zeroshot.json")
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
| 8 | 上游 PPL 脚本输出 NaN | fp16 在长 seq 数值溢出 | 看上游脚本是否有 `--dtype fp32` 选项；或 issue 区搜"NaN" |
| 9 | BiLLM 跑到一半 OOM | 二值化中间张量大 | 减 batch；canonical 必须用 96GB lab，不要在 12GB 本地跑 |
| 10 | KIVI CUDA 编译失败 | nvcc 与 PyTorch CUDA 版不匹配 | `nvcc --version` 与 `python -c "import torch; print(torch.version.cuda)"` 主版本号要一致 |
| 11 | `pytest --collect-only` 报 `Unknown mark` | 标记没在 pyproject 注册 | 已在 pyproject.toml 注册 `slow` 和 `require_cuda`；自定义新标记要追加 |
| 12 | `pip install -e .` 在 Windows 报路径太长 | Windows 默认 MAX_PATH 260 | 注册表开 LongPathsEnabled，或装到短路径（D:\Q\repro） |
| 13 | `lm-eval-harness` 第一次跑卡很久不动 | 在下 dataset（PIQA ~10MB） | 等 / 设代理；走完一次后 cache 在 `~/.cache/huggingface/datasets` |
| 14 | git submodule 没拉下来上游代码 | clone 时漏了 `--recurse-submodules` | `git submodule update --init --recursive` |
| 15 | git push 出现 `LF will be replaced by CRLF` warning | Windows 默认 `core.autocrlf=true` | 不影响（git 内部存 LF，Linux 拉下来正常）；Warning 可忽略 |
| 16 | Colab session 断了 results 没保存 | runtime 超时 | 下次记得 §9.1 最后一个 Cell `files.download()` 取下来；或挂 Drive 持久化 |
| 17 | 上游 example 参数名跟我文档对不上 | 上游版本变了 | 用 `--help` 看实际参数；上游脚本 commit history 看改动 |

---

## §11 接下来你具体做什么

1. **现在**（已经完成）：framework 全部就绪，所有 `common/` 测试通过。**`common/` 暂时不动**，留给 Phase 3。
2. **下一步**：按 §5 走 GPTQ 完整流程：vendor → 建 env → 跑上游 example。**不写自己的 Python 代码**，用上游 repo 的复现脚本。
3. **第一里程碑**：跑出 GPTQ canonical 数字（PPL ≈ 5.69 ± 0.3） → tag `phase1-gptq-done`。Phase 2 源码笔记可以在 GPTQ 之后或者四个方法都跑完再写。
4. **告诉我数字** → 我写 Plan B (AWQ)。AWQ 流程跟 GPTQ 95% 重叠，只换上游脚本路径和参数。

GPTQ 做完之后的预期 commit 历史（**注意：没有 `repro.py` 相关 commit**）：

```
* (tag: phase1-gptq-done) docs: link GPTQ as completed in top README + initialize summary table
* docs(reports): add Phase 2 source-code study for GPTQ        ← 可选，可推迟到所有方法都做完
* docs(gptq): fill in canonical numbers vs paper anchor + meta  ← 抽 stdout 数字写表
* data(gptq): canonical W4-g128 results on LLaMA-2-7B          ← 上游 stdout 文件
* test(gptq): smoke run with TinyLlama passes (results gitignored)
* scaffold(gptq): vendor AutoGPTQ + env.yml + README template
```

只有 **2 个 commit 需要写代码**（scaffold = 写 env.yml + README + .gitmodules；docs = 填表+元数据）。其他都是跑完上游脚本拿到结果产物 + commit。

> Phase 3 才是真正"手搓"的阶段：那时你写自己的 unified pipeline，把 4 方法纳入同一 API，用 `common/` 做共享 eval。Phase 1 / 2 不沾。
>
> Plan A 的 Task 8–15 接下来会同步重写（旧版还假定你写 `repro.py`，新版改成"vendor + 跑上游"）。如果 Plan A 已重写但本文档某段还旧，以本文档为准。
